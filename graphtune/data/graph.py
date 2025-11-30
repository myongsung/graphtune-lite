import pickle
import numpy as np
import pandas as pd
from typing import Optional  # ← 추가

def _infer_loc_columns(loc_df: pd.DataFrame):
    norm_map = {str(c).lower().strip(): c for c in loc_df.columns}

    id_candidates  = ["sensor_id", "sensorid", "id", "station_id", "node_id"]
    lat_candidates = ["latitude", "lat", "y"]
    lon_candidates = ["longitude", "lon", "lng", "long", "x"]

    id_col = lat_col = lon_col = None
    for k in id_candidates:
        if k in norm_map:
            id_col = norm_map[k]; break
    for k in lat_candidates:
        if k in norm_map:
            lat_col = norm_map[k]; break
    for k in lon_candidates:
        if k in norm_map:
            lon_col = norm_map[k]; break

    if id_col is None or lat_col is None or lon_col is None:
        if loc_df.shape[1] >= 3:
            cols = list(loc_df.columns[:3])
            id_col  = id_col  or cols[0]
            lat_col = lat_col or cols[1]
            lon_col = lon_col or cols[2]
        else:
            raise ValueError(
                f"Location file has <3 columns and cannot infer id/lat/lon: {loc_df.columns}"
            )
    return id_col, lat_col, lon_col


def load_adj_and_coords(adj_path: str, loc_path: Optional[str]):
    """
    legacy load_adj_and_coords + loc_path=None 지원.
    """
    # 1) adj 로드 (원래 코드 그대로)
    with open(adj_path, "rb") as f:
        sensor_ids, sensor_id_to_ind, adj_mx = pickle.load(f, encoding="latin1")
    A = adj_mx.astype(np.float32)

    # 2) loc 파일이 아예 없는 경우 (예: songdo)
    if loc_path is None:
        coords = None
        return A, coords, sensor_ids

    # 3) 아래부터는 원래 코드 그대로 (metr-la / pems-bay 등)
    sensor_ids_int = []
    for sid in sensor_ids:
        try:
            sensor_ids_int.append(int(sid))
        except Exception:
            s = str(sid)
            digits = "".join(ch for ch in s if ch.isdigit())
            if digits == "":
                raise ValueError(f"Cannot parse sensor id: {sid}")
            sensor_ids_int.append(int(digits))

    loc_df = pd.read_csv(loc_path)

    if loc_df.columns.dtype == "int64" or all(str(c).isdigit() for c in loc_df.columns):
        loc_df = pd.read_csv(loc_path, header=None)

    id_col, lat_col, lon_col = _infer_loc_columns(loc_df)

    loc_df[id_col]  = loc_df[id_col].astype(int, errors="ignore")
    loc_df[lat_col] = loc_df[lat_col].astype(float, errors="ignore")
    loc_df[lon_col] = loc_df[lon_col].astype(float, errors="ignore")

    id2coord = {}
    for _, row in loc_df.iterrows():
        try:
            rid = int(row[id_col])
        except Exception:
            continue
        id2coord[rid] = [float(row[lat_col]), float(row[lon_col])]

    coords_list = []
    missing_ids = []
    for sid_int in sensor_ids_int:
        if sid_int in id2coord:
            coords_list.append(id2coord[sid_int])
        else:
            missing_ids.append(sid_int)
            coords_list.append(None)

    if len(missing_ids) > 0:
        if len(loc_df) == len(sensor_ids_int):
            print(f"[warn] {len(missing_ids)} sensor_ids missing in {loc_path}. "
                  f"Assuming loc file is already aligned by order.")
            coords_list = loc_df[[lat_col, lon_col]].to_numpy(dtype=np.float32).tolist()
            coords = np.array(coords_list, dtype=np.float32)
            return A, coords, sensor_ids

        print(f"[warn] {len(missing_ids)} sensor_ids missing in {loc_path}. "
              f"Filling missing coords with (0,0). Example missing: {missing_ids[:5]}")
        for i, c in enumerate(coords_list):
            if c is None:
                coords_list[i] = [0.0, 0.0]

    coords = np.array(coords_list, dtype=np.float32)
    return A, coords, sensor_ids
