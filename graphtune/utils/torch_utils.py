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

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def param_size_mb(model, bytes_per_param=4):
    return count_trainable_params(model) * bytes_per_param / (1024**2)

def load_partial_state(target_model, source_state, exclude_prefixes=()):
    """
    서로 노드 수/버퍼가 달라도
    - 키가 존재하고
    - shape가 동일한 파라미터만 옮겨 심음.
    """
    tgt_state = target_model.state_dict()
    copied = 0
    for k, v in source_state.items():
        if any(k.startswith(p) for p in exclude_prefixes):
            continue
        if k in tgt_state and tgt_state[k].shape == v.shape:
            tgt_state[k] = v
            copied += 1
    target_model.load_state_dict(tgt_state)
    return copied
