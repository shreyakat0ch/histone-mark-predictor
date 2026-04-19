import argparse
import os
import json
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, mean_squared_error, roc_auc_score

import config
from dataset import ChromaDataset
from model import ChromaRegressor
from train import get_splits
import visualizer

def safe_corr(x, y):
    if len(x) <= 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    corr = pearsonr(x, y)[0]
    if np.isnan(corr):
        return None
    return float(corr)

def evaluate_test_set(model_path=None, ablate_rna=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    if model_path is None:
        model_path = os.path.join(config.RUNS_DIR, "best_model.pt")
    if not os.path.exists(model_path):
        print(f"No trained model found at {model_path}!")
        return
        
    model = ChromaRegressor().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    if ablate_rna:
        print("RNA ablation enabled: RNA inputs will be zeroed during evaluation.")
    
    # Setup Data
    _, _, test_cells = get_splits()
    print(f"Evaluating on {len(test_cells)} test cell lines...")
    
    test_dataset = ChromaDataset(test_cells, "test")
    test_dataset.load_target_stats()
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE * 2,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )
    
    # Collect predictions
    all_preds = []
    all_targets = []
    all_masks = []
    
    print("Running inference on test set...")
    with torch.no_grad():
        for dna, rna, target, mask in test_loader:
            dna, rna = dna.to(device), rna.to(device)
            if ablate_rna:
                rna = torch.zeros_like(rna)
            
            # AMP inference
            with torch.cuda.amp.autocast():
                pred = model(dna, rna)
                
            all_preds.append(pred.cpu().numpy())
            all_targets.append(target.numpy())
            all_masks.append(mask.numpy())
            
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)
    window_meta = list(test_dataset.windows)
    cell_ids = np.array([cell for cell, _, _ in window_meta])
    chrom_ids = np.array([chrom for _, chrom, _ in window_meta])
    centers = np.array([center for _, _, center in window_meta], dtype=np.int64)
        
    print("\n--- Test Set Results ---")
    results = {
        'model_path': model_path,
        'rna_ablation': bool(ablate_rna),
        'marks': {}
    }
    prediction_rows = []
    
    for m in range(config.NUM_MARKS):
        mark_name = config.TARGET_MARKS[m]
        valid_idx = all_masks[:, m] > 0.5
        
        if valid_idx.sum() > 0:
            p = all_preds[valid_idx, m]
            t = all_targets[valid_idx, m]
            
            # Calculate correlations on normalized values (doesn't change correlation)
            p_corr = float(pearsonr(p, t)[0])
            s_corr = float(spearmanr(p, t)[0])
            
            # Un-normalize for real MSE (log1p scale)
            mean = test_dataset.target_mean[m]
            std = test_dataset.target_std[m]
            
            p_unnorm = p * std + mean
            t_unnorm = t * std + mean
            
            mse = float(mean_squared_error(t_unnorm, p_unnorm))

            y_true_present = (t_unnorm > 0).astype(np.int32)
            y_pred_present = (p_unnorm > config.PRESENCE_THRESHOLD).astype(np.int32)
            presence_accuracy = float((y_pred_present == y_true_present).mean())

            if np.unique(y_true_present).size == 2:
                auroc = float(roc_auc_score(y_true_present, p_unnorm))
                auprc = float(average_precision_score(y_true_present, p_unnorm))
            else:
                auroc = None
                auprc = None

            per_cell_pearson = {}
            valid_cells = cell_ids[valid_idx]
            for cell in sorted(set(valid_cells.tolist())):
                cell_idx = valid_cells == cell
                corr = safe_corr(p[cell_idx], t[cell_idx])
                if corr is not None:
                    per_cell_pearson[cell] = corr

            mean_per_cell_pearson = (
                float(np.mean(list(per_cell_pearson.values())))
                if per_cell_pearson else None
            )
            
            results['marks'][mark_name] = {
                'pearson': p_corr,
                'spearman': s_corr,
                'mse': mse,
                'presence_auroc': auroc,
                'presence_auprc': auprc,
                'presence_accuracy': presence_accuracy,
                'presence_threshold': config.PRESENCE_THRESHOLD,
                'mean_per_cell_pearson': mean_per_cell_pearson,
                'per_cell_pearson': per_cell_pearson,
                'samples': int(valid_idx.sum())
            }

            mark_cells = cell_ids[valid_idx]
            mark_chroms = chrom_ids[valid_idx]
            mark_centers = centers[valid_idx]
            for row_idx in range(len(p_unnorm)):
                prediction_rows.append({
                    'cell': mark_cells[row_idx],
                    'chrom': mark_chroms[row_idx],
                    'center': int(mark_centers[row_idx]),
                    'mark': mark_name,
                    'actual': float(t_unnorm[row_idx]),
                    'predicted': float(p_unnorm[row_idx]),
                    'present_true': int(y_true_present[row_idx]),
                    'present_pred': int(y_pred_present[row_idx]),
                })
            
            print(f"\n{mark_name}:")
            print(f"  Pearson:  {p_corr:.4f}")
            print(f"  Spearman: {s_corr:.4f}")
            print(f"  MSE:      {mse:.4f}")
            print(f"  Accuracy: {presence_accuracy:.4f} @ threshold {config.PRESENCE_THRESHOLD}")
            if auroc is not None:
                print(f"  AUROC:    {auroc:.4f}")
                print(f"  AUPRC:    {auprc:.4f}")
            else:
                print("  AUROC:    skipped (only one presence class)")
                print("  AUPRC:    skipped (only one presence class)")
            if mean_per_cell_pearson is not None:
                print(f"  Mean per-cell Pearson: {mean_per_cell_pearson:.4f}")
            else:
                print("  Mean per-cell Pearson: skipped")
            print(f"  Samples:  {int(valid_idx.sum())}")
            
    # Save results
    suffix = "_rna_ablation" if ablate_rna else ""
    out_path = os.path.join(config.RUNS_DIR, f"test_results{suffix}.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out_path}")

    if prediction_rows:
        pred_df = pd.DataFrame(prediction_rows)
        pred_csv_path = os.path.join(config.RUNS_DIR, f"heldout_predictions{suffix}.csv")
        pred_df.to_csv(pred_csv_path, index=False)
        print(f"Saved held-out predictions to {pred_csv_path}")

        heldout_plot_dir = os.path.join(config.PLOTS_DIR, f"heldout{suffix or '_baseline'}")
        visualizer.plot_heldout_distributions(pred_df, heldout_plot_dir)
        print(f"Saved held-out plots to {heldout_plot_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=None, help="Path to a trained checkpoint.")
    parser.add_argument("--ablate-rna", action="store_true", help="Zero RNA inputs during evaluation.")
    args = parser.parse_args()
    evaluate_test_set(model_path=args.weights, ablate_rna=args.ablate_rna)
