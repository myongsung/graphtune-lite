import numpy as np
# graphtune/utils/scaler.py
import torch  # ⬅ 꼭 추가

class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean   # 보통 numpy array
        self.std = std

    def _to_tensor_stats(self, data: torch.Tensor):
        """mean/std 를 data 와 같은 device/dtype 의 텐서로 변환."""
        mean = torch.as_tensor(self.mean, dtype=data.dtype, device=data.device)
        std = torch.as_tensor(self.std, dtype=data.dtype, device=data.device)
        return mean, std

    def transform(self, data):
        # torch.Tensor 인 경우 (학습 시점에는 이미 이렇게 쓰고 있을 수도 있음)
        if isinstance(data, torch.Tensor):
            mean, std = self._to_tensor_stats(data)
            return (data - mean) / std
        # numpy / float 등의 경우 (데이터 준비 단계)
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        # 평가 단계에서 y_hat, y 가 torch.Tensor (cuda) 인 상태로 들어옴
        if isinstance(data, torch.Tensor):
            mean, std = self._to_tensor_stats(data)
            return data * std + mean
        # numpy / float 대응
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
