import os
import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd

import config
from dataset import ChromaDataset
from model import ChromaRegressor
from losses import MaskedHybridLoss
import visualizer

def maybe_ablate_rna(rna):
    if getattr(config, "RNA_ABLATION", False):
        return torch.zeros_like(rna)
    return rna

def get_splits():
    with open(os.path.join(config.PROCESSED_DIR, "cell_order.json"), 'r') as f:
        cells = json.load(f)
        
    np.random.seed(42)
    np.random.shuffle(cells)
    
    n = len(cells)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)
    
    train_cells = cells[:train_end]
    val_cells = cells[train_end:val_end]
    test_cells = cells[val_end:]
    
    return train_cells, val_cells, test_cells

def train_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0
    total_huber = 0
    total_pearson = 0
    
    pbar = tqdm(loader, desc="Training")
    for dna, rna, target, mask in pbar:
        dna, rna = dna.to(device), rna.to(device)
        rna = maybe_ablate_rna(rna)
        target, mask = target.to(device), mask.to(device)
        
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda'):
            pred = model(dna, rna)
            loss, huber, pearson = criterion(pred, target, mask)
            
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        total_huber += huber
        total_pearson += pearson
        
        pbar.set_postfix({'loss': f"{loss.item():.4f}", 'huber': f"{huber:.4f}", 'pearson_loss': f"{pearson:.4f}"})
        
    n = len(loader)
    return total_loss/n, total_huber/n, total_pearson/n

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    
    # Track predictions for correlation
    all_preds = []
    all_targets = []
    all_masks = []
    
    pbar = tqdm(loader, desc="Validating")
    for dna, rna, target, mask in pbar:
        dna, rna = dna.to(device), rna.to(device)
        rna = maybe_ablate_rna(rna)
        target, mask = target.to(device), mask.to(device)
        
        with torch.amp.autocast('cuda'):
            pred = model(dna, rna)
            loss, _, _ = criterion(pred, target, mask)
            
        total_loss += loss.item()
        
        all_preds.append(pred.cpu().numpy())
        all_targets.append(target.cpu().numpy())
        all_masks.append(mask.cpu().numpy())
        
    n = len(loader)
    
    # Calculate per-mark correlation
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)
    
    corrs = {}
    mean_corr = 0
    valid_marks = 0
    
    for m in range(config.NUM_MARKS):
        mark_name = config.TARGET_MARKS[m]
        # Only compute on samples where this mark exists
        valid_idx = all_masks[:, m] > 0.5
        
        if valid_idx.sum() > 2:
            p = all_preds[valid_idx, m]
            t = all_targets[valid_idx, m]
            
            corr = np.corrcoef(p, t)[0, 1]
            if not np.isnan(corr):
                corrs[mark_name] = corr
                mean_corr += corr
                valid_marks += 1
                
    if valid_marks > 0:
        mean_corr /= valid_marks
        
    return total_loss/n, corrs, mean_corr, all_preds, all_targets, all_masks

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if getattr(config, "RNA_ABLATION", False):
        print("RNA ablation enabled: RNA inputs will be zeroed.")
    
    # Setup Data
    train_cells, val_cells, test_cells = get_splits()
    print(f"Splits: {len(train_cells)} train, {len(val_cells)} val, {len(test_cells)} test")
    
    train_dataset = ChromaDataset(train_cells, "train")
    train_dataset.compute_target_stats()
    
    val_dataset = ChromaDataset(val_cells[:30], "val") # Sub-sample to 30 cells for 10x faster validation
    val_dataset.load_target_stats()
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE * 2,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    # Setup Model
    model = ChromaRegressor().to(device)
    criterion = MaskedHybridLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    
    # Warmup: linearly ramp LR from ~0 to target over WARMUP_STEPS,
    # then cosine-decay for the remaining training.
    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * config.EPOCHS
    warmup_steps = getattr(config, 'WARMUP_STEPS', 0)
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-3, total_iters=warmup_steps
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps]
    )
    scaler = torch.amp.GradScaler('cuda')
    
    start_epoch = 0
    best_corr = -1
    epochs_no_improve = 0
    
    # --- Check for Resume ---
    latest_path = os.path.join(config.RUNS_DIR, "latest_checkpoint.pt")
    if os.path.exists(latest_path):
        print(f"Found checkpoint! Resuming from {latest_path}")
        checkpoint = torch.load(latest_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_corr = checkpoint.get('best_corr', -1)
        epochs_no_improve = checkpoint.get('epochs_no_improve', 0)
        # Advance scheduler to match the step count from previous epochs
        for _ in range(start_epoch * steps_per_epoch):
            scheduler.step()
            
    # --- History Tracking ---
    history_path = os.path.join(config.RUNS_DIR, "history.json")
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history = json.load(f)
        n_logged = len(history.get('train_loss', []))
        history.setdefault('train_huber', [None] * n_logged)
        history.setdefault('train_pearson_loss', [None] * n_logged)
        if len(history['train_huber']) < n_logged:
            history['train_huber'] = [None] * (n_logged - len(history['train_huber'])) + history['train_huber']
        if len(history['train_pearson_loss']) < n_logged:
            history['train_pearson_loss'] = [None] * (n_logged - len(history['train_pearson_loss'])) + history['train_pearson_loss']
        history.setdefault('rna_ablation', getattr(config, "RNA_ABLATION", False))
    else:
        history = {
            'train_loss': [],
            'train_huber': [],
            'train_pearson_loss': [],
            'val_loss': [],
            'avg_pearson': [],
            'per_mark_pearson': {m: [] for m in config.TARGET_MARKS},
            'rna_ablation': getattr(config, "RNA_ABLATION", False)
        }
    
    # Training Loop
    for epoch in range(start_epoch, config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.EPOCHS}")
        
        train_loss, train_huber, train_pearson = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_corrs, mean_corr, val_preds, val_targets, val_masks = validate(model, val_loader, criterion, device)
        
        # Step scheduler once per epoch (SequentialLR handles warmup→cosine transition)
        for _ in range(steps_per_epoch):
            scheduler.step()
        
        # Update History
        history['train_loss'].append(train_loss)
        history['train_huber'].append(train_huber)
        history['train_pearson_loss'].append(train_pearson)
        history['val_loss'].append(val_loss)
        history['avg_pearson'].append(mean_corr)
        for mark in config.TARGET_MARKS:
            history['per_mark_pearson'][mark].append(val_corrs.get(mark, 0))
            
        with open(history_path, 'w') as f:
            json.dump(history, f)
            
        # Also save as CSV for easier reading

        df_dict = {
            'epoch': list(range(1, len(history['train_loss']) + 1)),
            'train_loss': history['train_loss'],
            'train_huber': history['train_huber'],
            'train_pearson_loss': history['train_pearson_loss'],
            'val_loss': history['val_loss'],
            'avg_pearson': history['avg_pearson'],
            'rna_ablation': [history.get('rna_ablation', False)] * len(history['train_loss'])
        }
        for mark in config.TARGET_MARKS:
            df_dict[f'pearson_{mark}'] = history['per_mark_pearson'][mark]
        
        pd.DataFrame(df_dict).to_csv(os.path.join(config.RUNS_DIR, "training_log.csv"), index=False)
            
        # Visualizations
        visualizer.save_epoch_visualizations(
            epoch, 
            history, 
            val_preds, 
            val_targets, 
            val_masks, 
            config.PLOTS_DIR,
            norm_mean=val_dataset.target_mean,
            norm_std=val_dataset.target_std
        )
        
        print(f"Train Loss: {train_loss:.4f} (Huber: {train_huber:.4f}, PearsonL: {train_pearson:.4f})")
        print(f"Val Loss: {val_loss:.4f} | Mean Val Corr: {mean_corr:.4f}")
        for mark, corr in val_corrs.items():
            print(f"  {mark}: {corr:.4f}")

        is_best = mean_corr > best_corr
        if is_best:
            best_corr = mean_corr
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        # Save latest checkpoint
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_corr': best_corr,
            'mean_corr': mean_corr,
            'epochs_no_improve': epochs_no_improve,
            'rna_ablation': getattr(config, "RNA_ABLATION", False)
        }
        torch.save(checkpoint_data, latest_path)
        
        # Save best model
        if is_best:
            print("Saving new best model...")
            torch.save(checkpoint_data, os.path.join(config.RUNS_DIR, "best_model.pt"))

        if epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break
            
if __name__ == "__main__":
    main()
