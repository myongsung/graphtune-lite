import numpy as np
import torch
from torch.utils.data import Dataset

class SequenceDatasetWithMask(Dataset):
    def __init__(self, X_norm, Y_norm, mask):
        self.X = torch.from_numpy(X_norm).float()
        self.Y = torch.from_numpy(Y_norm).float()
        self.mask = torch.from_numpy(mask).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.mask[idx]

class BigSTDataset(Dataset):
    def __init__(self, X_norm, Y_norm, mask, start_idx,
                 time_in_day, week_id, T_in, num_nodes):
        self.X = X_norm
        self.Y = Y_norm
        self.mask = mask
        self.start_idx = start_idx
        self.time_in_day = time_in_day
        self.week_id = week_id
        self.T_in = T_in
        self.num_nodes = num_nodes

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x_speed = self.X[idx]  # [T_in,N]
        y = self.Y[idx]        # [T_out,N]
        m = self.mask[idx]
        s = int(self.start_idx[idx])

        t_idx = np.arange(s, s + self.T_in)
        tod = self.time_in_day[t_idx]
        week = self.week_id[t_idx]

        N = self.num_nodes
        tod_expand = np.repeat(tod.reshape(1, -1), N, axis=0)
        week_expand = np.repeat(week.reshape(1, -1), N, axis=0)

        speed = x_speed.T  # [N,T_in]
        feat = np.stack([speed, tod_expand, week_expand], axis=-1).astype(np.float32)
        return (
            torch.from_numpy(feat).float(),  # [N,T_in,3]
            torch.from_numpy(y).float(),     # [T_out,N]
            torch.from_numpy(m).float()
        )
