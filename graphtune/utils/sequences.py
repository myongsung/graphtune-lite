import numpy as np

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
