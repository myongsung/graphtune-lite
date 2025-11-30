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
    legacy load_adj_and_coords 그대로.

    - adj_path: DCRNN 형식의 adj_mx.pkl (sensor_ids, sensor_id_to_ind, adj_mx)
    - loc_path: 센서 위치 CSV. None 이면 coords 없이 (coords=None) 반환.
    """
    with open(adj_path, "rb") as f:
        sensor_ids, sensor_id_to_ind, adj_mx = pickle.load(f, encoding="latin1")
    A = adj_mx.astype(np.float32)

    # 🔥 좌표 파일이 없는 데이터셋 (예: songdo)
    if loc_path is None:
        coords = None
        return A, coords, sensor_ids

    # ----- 아래부터는 원래 코드 유지 -----
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
    ...
    coords = np.array(coords_list, dtype=np.float32)
    return A, coords, sensor_ids