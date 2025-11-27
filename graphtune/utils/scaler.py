import numpy as np

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
