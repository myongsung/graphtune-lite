# graphtune/eval/metrics.py
import torch

def masked_mae_loss(pred, true, mask):
    """
    pred,true,mask: [B, T_out, N]
    """
    mask = (mask > 0).float()
    loss = torch.abs(pred - true) * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    denom = mask.sum()
    return loss.sum() / torch.clamp(denom, min=1.0)

def masked_rmse_loss(pred, true, mask):
    """
    pred,true,mask: [B, T_out, N]
    """
    mask = (mask > 0).float()
    se = (pred - true) ** 2 * mask
    se = torch.where(torch.isnan(se), torch.zeros_like(se), se)
    denom = mask.sum()
    mse = se.sum() / torch.clamp(denom, min=1.0)
    return torch.sqrt(mse)
