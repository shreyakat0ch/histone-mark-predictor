import torch
import torch.nn as nn
import config

class MaskedHybridLoss(nn.Module):
    def __init__(self, alpha=config.LOSS_ALPHA):
        super().__init__()
        self.alpha = alpha
        self.huber = nn.HuberLoss(reduction='none')

    def forward(self, pred, target, mask):
        # pred, target, mask: [B, NUM_MARKS]
        
        # 1. Masked Huber Loss
        huber_loss = self.huber(pred, target) # [B, NUM_MARKS]
        masked_huber = (huber_loss * mask).sum() / (mask.sum() + 1e-8)
        
        # 2. Masked Pearson Loss
        # We calculate Pearson correlation per mark across the batch
        # shape: [B, NUM_MARKS]
        
        pearson_losses = []
        for m in range(pred.shape[1]):
            # Get valid samples for this mark in this batch
            m_mask = mask[:, m] > 0.5
            
            if m_mask.sum() < 2:
                # Need at least 2 points to compute correlation
                continue
                
            p = pred[m_mask, m]
            t = target[m_mask, m]
            
            p_mean = p.mean()
            t_mean = t.mean()
            
            p_var = p - p_mean
            t_var = t - t_mean
            
            cov = (p_var * t_var).sum()
            p_std = torch.sqrt((p_var**2).sum() + 1e-8)
            t_std = torch.sqrt((t_var**2).sum() + 1e-8)
            
            corr = cov / (p_std * t_std)
            # Loss is 1 - corr (so higher correlation = lower loss)
            pearson_losses.append(1.0 - corr)
            
        if not pearson_losses:
            masked_pearson = torch.tensor(0.0, device=pred.device, requires_grad=True)
        else:
            masked_pearson = torch.stack(pearson_losses).mean()
            
        # 3. Combine
        hybrid_loss = self.alpha * masked_huber + (1.0 - self.alpha) * masked_pearson
        
        return hybrid_loss, masked_huber.item(), masked_pearson.item()
