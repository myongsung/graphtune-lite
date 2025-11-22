import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .sources import resolve_dataset_key, ensure_local_file
from .graph import load_adj_and_coords
from .datasets import SequenceDatasetWithMask, BigSTDataset
from .time_features import make_time_features

# ⚠️ legacy __init__을 타면 사이드이펙트 있을 수 있으니 "직접 모듈"로 import
from graphtune.legacy.utils import (
    create_sequences_with_start,
    compute_mean_std_from_train,
    StandardScaler,
)

def prepare_dataset(
    dataset_name: str,
    data_dir="DATA",
    T_in=12,
    T_out=12,
    stride=1,
    batch_size=128,
    num_workers=2,
    for_bigst=False,
    source="auto",
    cache_dir=None,
    url_overrides=None,
    revision=None,
):
    """
    legacy prepare_dataset과 시그니처/동작 동일. :contentReference[oaicite:1]{index=1}
    """
    ds_key = resolve_dataset_key(dataset_name)

    url_overrides = url_overrides or {}
    h5_path  = ensure_local_file(ds_key, "h5",  data_dir, source, cache_dir,
                                url_overrides.get("h5"), revision)
    adj_path = ensure_local_file(ds_key, "adj", data_dir, source, cache_dir,
                                url_overrides.get("adj"), revision)
    loc_path = ensure_local_file(ds_key, "loc", data_dir, source, cache_dir,
                                url_overrides.get("loc"), revision)

    data = pd.read_hdf(h5_path).values.astype(np.float32)  # (T,N)
    num_steps, num_nodes = data.shape

    X, Y, start_idx = create_sequences_with_start(data, T_in, T_out, stride)
    num_samples = X.shape[0]

    train_len = int(num_samples * 0.7)
    val_len   = int(num_samples * 0.1)

    X_train, Y_train = X[:train_len], Y[:train_len]
    X_val,   Y_val   = X[train_len:train_len+val_len], Y[train_len:train_len+val_len]
    X_test,  Y_test  = X[train_len+val_len:], Y[train_len+val_len:]

    start_train = start_idx[:train_len]
    start_val   = start_idx[train_len:train_len+val_len]
    start_test  = start_idx[train_len+val_len:]

    mean, std = compute_mean_std_from_train(X_train, Y_train)
    scaler = StandardScaler(mean, std)

    def norm(arr):
        return scaler.transform(arr.reshape(-1, num_nodes)).reshape(arr.shape)

    X_train_n, Y_train_n = norm(X_train), norm(Y_train)
    X_val_n,   Y_val_n   = norm(X_val),   norm(Y_val)
    X_test_n,  Y_test_n  = norm(X_test),  norm(Y_test)

    mask_train = (Y_train != 0).astype(np.float32)
    mask_val   = (Y_val   != 0).astype(np.float32)
    mask_test  = (Y_test  != 0).astype(np.float32)

    time_in_day, week_id = make_time_features(num_steps)

    if for_bigst:
        train_ds = BigSTDataset(X_train_n, Y_train_n, mask_train, start_train,
                                time_in_day, week_id, T_in, num_nodes)
        val_ds   = BigSTDataset(X_val_n, Y_val_n, mask_val, start_val,
                                time_in_day, week_id, T_in, num_nodes)
        test_ds  = BigSTDataset(X_test_n, Y_test_n, mask_test, start_test,
                                time_in_day, week_id, T_in, num_nodes)
    else:
        train_ds = SequenceDatasetWithMask(X_train_n, Y_train_n, mask_train)
        val_ds   = SequenceDatasetWithMask(X_val_n, Y_val_n, mask_val)
        test_ds  = SequenceDatasetWithMask(X_test_n, Y_test_n, mask_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    A, coords, sensor_ids = load_adj_and_coords(adj_path, loc_path)

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "scaler": scaler,
        "A": A,
        "coords": coords,
        "num_nodes": num_nodes,
        "T_in": T_in,
        "T_out": T_out,
        "dataset_name": ds_key,
        "sensor_ids": sensor_ids,
    }
