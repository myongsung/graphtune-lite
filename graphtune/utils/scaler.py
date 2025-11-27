# graphtune/utils/scaler.py

import numpy as np
import torch

class StandardScaler:
    """노드별 Z-score 스케일러 (train 기준)."""

    def __init__(self, mean, std):
        # mean, std 는 보통 numpy array (shape: [1, N])
        self.mean = mean
        self.std = std

    def _to_tensor_stats(self, data: torch.Tensor):
        """mean/std 를 data 와 같은 device/dtype 의 텐서로 변환."""
        mean = torch.as_tensor(self.mean, dtype=data.dtype, device=data.device)
        std = torch.as_tensor(self.std, dtype=data.dtype, device=data.device)
        return mean, std

    def transform(self, data):
        # torch.Tensor 인 경우 (학습/평가 시)
        if isinstance(data, torch.Tensor):
            mean, std = self._to_tensor_stats(data)
            return (data - mean) / std
        # numpy 인 경우 (데이터 전처리 단계)
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        # torch.Tensor 인 경우 (평가 시 y_hat, y)
        if isinstance(data, torch.Tensor):
            mean, std = self._to_tensor_stats(data)
            return data * std + mean
        # numpy 인 경우 (기존 코드와 동일)
        return data * self.std + self.mean


def compute_mean_std_from_train(X_train, Y_train):
    """
    X_train: (S, T_in, N)
    Y_train: (S, T_out, N)  또는 동일 shape, node별 평균/표준편차 계산
    """
    # 원래 코드 그대로 유지 (numpy 기반)
    # 두 시퀀스를 이어붙여서 node별로 통계 계산
    S, T_in, N = X_train.shape
    _, T_out, _ = Y_train.shape

    train_all = np.concatenate(
        [
            X_train.reshape(-1, N),
            Y_train.reshape(-1, N),
        ],
        axis=0,
    )
    valid = (train_all != 0)
    valid_sum = valid.sum(axis=0)

    mean = np.zeros(N, dtype=np.float32)
    std = np.ones(N, dtype=np.float32)
    for i in range(N):
        if valid_sum[i] > 0:
            vals = train_all[valid[:, i], i]
            mean[i] = vals.mean()
            std[i] = vals.std()
            if std[i] < 1e-6:
                std[i] = 1.0
    return mean.reshape(1, N), std.reshape(1, N)
