import os
import pickle
import urllib.request
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from huggingface_hub import hf_hub_download

from .utils import (
    create_sequences_with_start,
    compute_mean_std_from_train,
    StandardScaler
)

STEPS_PER_DAY = 288  # 5분 간격
WEEK_NUM = 7


# ---------------------------------------------------------------------
# External data registry (AUTO download)
# ---------------------------------------------------------------------
DATA_SOURCES = {
    "metr-la": {
        # HF repo contains: metr-la.h5, adj_mx.pkl
        "hf_repo": "jimmygao3218/METRLA",
        "files": {
            "h5": "metr-la.h5",
            "adj": "adj_mx.pkl",
            "loc": "graph_sensor_locations.csv",
        },
        "urls": {
            # loc csv is tiny, fetch from DCRNN raw
            "graph_sensor_locations.csv":
                "https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/sensor_graph/graph_sensor_locations.csv",
        }
    },
    "pems-bay": {
        # Zenodo direct download for h5/adj
        "hf_repo": None,
        "files": {
            "h5": "pems-bay.h5",
            "adj": "adj_mx_bay.pkl",
            "loc": "graph_sensor_locations_bay.csv",
        },
        "urls": {
            "pems-bay.h5":
                "https://zenodo.org/records/4263971/files/pems-bay.h5?download=1",
            "adj_mx_bay.pkl":
                "https://zenodo.org/records/4263971/files/adj_mx_bay.pkl?download=1",
            "graph_sensor_locations_bay.csv":
                "https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/sensor_graph/graph_sensor_locations_bay.csv",
        }
    }
}


def _download_url(url, dst_path):
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    if not os.path.exists(dst_path):
        print(f"[download] {url} -> {dst_path}")
        urllib.request.urlretrieve(url, dst_path)
    return dst_path


def _ensure_local_file(dataset_key, kind, data_dir, source="auto",
                       cache_dir=None, url_override=None, revision=None):
    """
    kind: "h5" | "adj" | "loc"
    source:
      - "auto": urls 있으면 url, 없으면 hf_repo, 마지막으로 local
      - "hf": huggingface_hub에서 다운로드
      - "url": urls 또는 url_override로 다운로드
      - "local": data_dir에 있다고 가정
    """
    ds = DATA_SOURCES[dataset_key]
    filename = ds["files"][kind]
    local_path = os.path.join(data_dir, filename)

    # local already exists
    if os.path.exists(local_path) and source in ["auto", "local"]:
        return local_path

    # choose auto route
    if source == "auto":
        # 1) url if present
        url = url_override or ds["urls"].get(filename)
        if url:
            return _download_url(url, local_path)

        # 2) hf if repo present
        if ds.get("hf_repo"):
            return hf_hub_download(
                repo_id=ds["hf_repo"],
                filename=filename,
                repo_type="dataset",
                cache_dir=cache_dir,
                revision=revision,
            )

        # 3) fallback to local (but file missing)
        raise FileNotFoundError(
            f"{local_path} not found and no auto source for {dataset_key}:{filename}"
        )

    if source == "hf":
        if ds.get("hf_repo") is None:
            raise ValueError(f"{dataset_key} has no hf_repo. Use source='url' or 'auto'.")
        return hf_hub_download(
            repo_id=ds["hf_repo"],
            filename=filename,
            repo_type="dataset",
            cache_dir=cache_dir,
            revision=revision,
        )

    if source == "url":
        url = url_override or ds["urls"].get(filename)
        if url is None:
            raise ValueError(f"No URL for {dataset_key}:{filename}")
        return _download_url(url, local_path)

    if source == "local":
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"{local_path} not found. Put file manually.")
        return local_path

    raise ValueError(f"Unknown source={source}")


class SequenceDatasetWithMask(Dataset):
    """Baseline/HyperNet/DCRNN/DGCRN용: x=[T_in,N]"""
    def __init__(self, X_norm, Y_norm, mask):
        self.X = torch.from_numpy(X_norm).float()
        self.Y = torch.from_numpy(Y_norm).float()
        self.mask = torch.from_numpy(mask).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.mask[idx]


class BigSTDataset(Dataset):
    """BigST용: x=[N,T_in,3] (speed,tod,week)"""
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


def _infer_loc_columns(loc_df: pd.DataFrame):
    """
    위치 CSV의 컬럼명이 제각각일 수 있어서
    sensor_id / lat / lon 컬럼을 최대한 robust하게 추론.
    """
    # 컬럼 이름 정규화
    norm_map = {str(c).lower().strip(): c for c in loc_df.columns}

    # id 컬럼 후보
    id_candidates = ["sensor_id", "sensorid", "id", "station_id", "node_id"]
    lat_candidates = ["latitude", "lat", "y"]
    lon_candidates = ["longitude", "lon", "lng", "long", "x"]

    id_col = None
    lat_col = None
    lon_col = None

    for k in id_candidates:
        if k in norm_map:
            id_col = norm_map[k]; break

    for k in lat_candidates:
        if k in norm_map:
            lat_col = norm_map[k]; break

    for k in lon_candidates:
        if k in norm_map:
            lon_col = norm_map[k]; break

    # 헤더가 없거나 예상 못한 이름이면 first 3 columns fallback
    if id_col is None or lat_col is None or lon_col is None:
        if loc_df.shape[1] >= 3:
            cols = list(loc_df.columns[:3])
            id_col = id_col or cols[0]
            lat_col = lat_col or cols[1]
            lon_col = lon_col or cols[2]
        else:
            raise ValueError(
                f"Location file has <3 columns and cannot infer id/lat/lon: {loc_df.columns}"
            )

    return id_col, lat_col, lon_col


def load_adj_and_coords(adj_path, loc_path):
    """
    adj_pkl: (sensor_ids, sensor_id_to_ind, adj_mx)
    loc_csv: 다양한 헤더 가능. id/lat/lon을 자동 추론.
    coords는 sensor_ids 순서에 align.
    """
    with open(adj_path, "rb") as f:
        sensor_ids, sensor_id_to_ind, adj_mx = pickle.load(f, encoding="latin1")
    A = adj_mx.astype(np.float32)

    # 일단 일반 read
    loc_df = pd.read_csv(loc_path)

    # 혹시 헤더가 없어서 컬럼이 숫자로 들어온 경우(0,1,2...)도 처리
    if loc_df.columns.dtype == "int64" or all(str(c).isdigit() for c in loc_df.columns):
        # header=None로 다시 읽기 시도
        loc_df = pd.read_csv(loc_path, header=None)

    id_col, lat_col, lon_col = _infer_loc_columns(loc_df)

    # 타입 캐스팅
    loc_df[id_col] = loc_df[id_col].astype(int)
    loc_df[lat_col] = loc_df[lat_col].astype(float)
    loc_df[lon_col] = loc_df[lon_col].astype(float)

    id2coord = {
        int(row[id_col]): [float(row[lat_col]), float(row[lon_col])]
        for _, row in loc_df.iterrows()
    }

    coords_list = []
    for sid in sensor_ids:
        sid_int = int(sid)
        if sid_int not in id2coord:
            raise ValueError(f"sensor_id {sid_int} missing in {loc_path}")
        coords_list.append(id2coord[sid_int])
    coords = np.array(coords_list, dtype=np.float32)
    return A, coords, sensor_ids


def prepare_dataset(
    dataset_name: str,
    data_dir="DATA",
    T_in=12,
    T_out=12,
    stride=1,
    batch_size=128,
    num_workers=2,
    for_bigst=False,
    source="auto",        # default auto-download
    cache_dir=None,
    url_overrides=None,   # dict {"h5": "...", "adj": "...", "loc": "..."}
    revision=None,
):
    """
    return dict with loaders, scaler, A/coords, meta
    """
    dataset_name = dataset_name.lower()
    if dataset_name in ["metr-la", "metr", "la"]:
        ds_key = "metr-la"
    elif dataset_name in ["pems-bay", "bay", "pems"]:
        ds_key = "pems-bay"
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")

    url_overrides = url_overrides or {}
    h5_path  = _ensure_local_file(ds_key, "h5",  data_dir, source, cache_dir,
                                  url_overrides.get("h5"), revision)
    adj_path = _ensure_local_file(ds_key, "adj", data_dir, source, cache_dir,
                                  url_overrides.get("adj"), revision)
    loc_path = _ensure_local_file(ds_key, "loc", data_dir, source, cache_dir,
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

    all_idx = np.arange(num_steps)
    time_in_day = (all_idx % STEPS_PER_DAY) / float(STEPS_PER_DAY)
    day_idx = all_idx // STEPS_PER_DAY
    week_id = (day_idx % WEEK_NUM).astype(np.int64)

    if for_bigst:
        train_ds = BigSTDataset(X_train_n, Y_train_n, mask_train, start_train,
                                time_in_day.astype(np.float32), week_id, T_in, num_nodes)
        val_ds   = BigSTDataset(X_val_n, Y_val_n, mask_val, start_val,
                                time_in_day.astype(np.float32), week_id, T_in, num_nodes)
        test_ds  = BigSTDataset(X_test_n, Y_test_n, mask_test, start_test,
                                time_in_day.astype(np.float32), week_id, T_in, num_nodes)
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
