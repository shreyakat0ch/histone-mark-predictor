import torch
from torch.utils.data import DataLoader
from dataset import ChromaDataset
from train import get_splits
import config

def test_dataloader():
    train_cells, val_cells, test_cells = get_splits()
    # Test on a small subset of cells to be fast, but enough to hit edge cases
    dataset = ChromaDataset(train_cells[:5], "train")
    dataset.compute_target_stats()
    
    loader = DataLoader(
        dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False,
        num_workers=8, # Use multiple workers to trigger potential IPC issues
        pin_memory=True
    )
    
    print(f"Testing DataLoader with {len(dataset)} samples...")
    for i, (dna, rna, target, mask) in enumerate(loader):
        assert dna.shape[0] == target.shape[0], "Batch size mismatch"
        assert dna.shape[1:] == (4, config.WINDOW_SIZE), f"Invalid DNA shape: {dna.shape}"
        assert rna.shape[1:] == (config.RNA_INPUT_DIM,), f"Invalid RNA shape: {rna.shape}"
        assert target.shape[1:] == (config.NUM_MARKS,), f"Invalid target shape: {target.shape}"
        assert mask.shape[1:] == (config.NUM_MARKS,), f"Invalid mask shape: {mask.shape}"
        if i % 10 == 0:
            print(f"Passed batch {i}/{len(loader)}")
            
    print("DataLoader test passed successfully! No shape inconsistencies found.")

if __name__ == "__main__":
    test_dataloader()
