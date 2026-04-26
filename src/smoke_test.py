import torch
import config
from model import ChromaRegressor
from losses import MaskedHybridLoss

def run_smoke_test():
    print("Testing ChromaRegressor architecture...")
    
    # 1. Instantiate Model
    model = ChromaRegressor()
    
    # 2. Create Dummy Inputs
    B = 2
    # DNA: [B, 4, 4096]
    dna_seq = torch.randn(B, 4, config.WINDOW_SIZE)
    
    # RNA: [B, 2001]
    rna_feat = torch.randn(B, config.RNA_INPUT_DIM)
    
    # 3. Forward Pass
    print("Running forward pass...")
    out = model(dna_seq, rna_feat)
    
    print(f"Output shape: {out.shape} (Expected: [{B}, {config.NUM_MARKS}])")
    assert out.shape == (B, config.NUM_MARKS), "Output shape mismatch!"
    
    # 4. Test Loss
    print("Testing MaskedHybridLoss...")
    criterion = MaskedHybridLoss()
    target = torch.randn(B, config.NUM_MARKS)
    mask = torch.ones((B, config.NUM_MARKS), dtype=torch.float)
    # simulate some missing data
    if config.NUM_MARKS > 1:
        mask[0, 1] = 0.0
    
    loss, huber, pearson = criterion(out, target, mask)
    print(f"Hybrid Loss: {loss.item():.4f}")
    print(f"Huber Component: {huber:.4f}")
    print(f"Pearson Component: {pearson:.4f}")
    
    print("\nSmoke test passed successfully! Model architecture is sound.")

if __name__ == "__main__":
    run_smoke_test()
