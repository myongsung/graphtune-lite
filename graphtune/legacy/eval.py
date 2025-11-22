import numpy as np
import torch

def evaluate_model(model, loader, scaler, device="cuda", is_bigst=False):
    model.eval()
    preds, trues, masks = [], [], []

    with torch.no_grad():
        for x, y, mask in loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            out = model(x)
            pred = out[0] if isinstance(out, tuple) else out
            preds.append(pred.cpu()); trues.append(y.cpu()); masks.append(mask.cpu())

    preds = torch.cat(preds, dim=0).numpy()
    trues = torch.cat(trues, dim=0).numpy()
    masks = torch.cat(masks, dim=0).numpy()

    preds_denorm = scaler.inverse_transform(preds)
    trues_denorm = scaler.inverse_transform(trues)

    diff = preds_denorm - trues_denorm
    m = masks > 0
    mae = np.abs(diff)[m].mean()
    rmse = np.sqrt((diff**2)[m].mean())
    return mae, rmse
