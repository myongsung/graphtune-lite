import numpy as np
import torch

def create_sequences_with_start(data, T_in=12, T_out=12, stride=1):
    """
    data: (T, N)
    return X: (S, T_in, N), Y: (S, T_out, N), start_idx: (S,)
    """
    num_steps, _ = data.shape
    X, Y, start_idx = [], [], []
    for t in range(0, num_steps - T_in - T_out + 1, stride):
        X.append(data[t:t+T_in])
        Y.append(data[t+T_in:t+T_in+T_out])
        start_idx.append(t)
    return np.stack(X), np.stack(Y), np.array(start_idx, dtype=np.int64)

def create_sequences(data, T_in=12, T_out=12, stride=1):
    X, Y, _ = create_sequences_with_start(data, T_in, T_out, stride)
    return X, Y

class StandardScaler:
    """노드별 Z-score 스케일러 (train 기준)."""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return data * self.std + self.mean

def compute_mean_std_from_train(X_train, Y_train):
    """
    X_train: (S, T_in, N)
    Y_train: (S, T_out, N)
    0은 결측으로 보고 제외.
    """
    num_nodes = X_train.shape[2]
    train_all = np.concatenate(
        [X_train.reshape(-1, num_nodes), Y_train.reshape(-1, num_nodes)],
        axis=0
    )
    valid = (train_all != 0)
    valid_sum = valid.sum(axis=0)

    mean = np.zeros(num_nodes, dtype=np.float32)
    std  = np.ones(num_nodes, dtype=np.float32)
    for i in range(num_nodes):
        if valid_sum[i] > 0:
            vals = train_all[valid[:, i], i]
            mean[i] = vals.mean()
            std[i] = vals.std()
            if std[i] < 1e-6:
                std[i] = 1.0
    return mean.reshape(1, num_nodes), std.reshape(1, num_nodes)

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
