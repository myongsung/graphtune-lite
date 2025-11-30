import os
import urllib.request
from typing import Optional, Dict
from huggingface_hub import hf_hub_download

# legacy DATA_SOURCES 그대로
DATA_SOURCES = {
    "metr-la": {
        "hf_repo": "jimmygao3218/METRLA",
        "files": {
            "h5": "metr-la.h5",
            "adj": "adj_mx.pkl",
            "loc": "graph_sensor_locations.csv",
        },
        "urls": {
            "graph_sensor_locations.csv":
                "https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/sensor_graph/graph_sensor_locations.csv",
        }
    },
    "pems-bay": {
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
    },
       "songdo": {
        "hf_repo": None,  # 로컬 파일만 사용
        "files": {
            "h5": "songdo_full.h5",
            "adj": "adj_songdo_rulebased.pkl",
            # 실제 좌표는 없으니, 더미(loc) 파일 이름만 미리 지정
            "loc": None,   # ✅ 좌표 파일 없음
        },
        "urls": {},  # url/hf_hub로 받지 않고 항상 local만 쓸 것
    },
}

def resolve_dataset_key(dataset_name: str) -> str:
    name = dataset_name.lower()
    if name in ["metr-la", "metr", "la"]:
        return "metr-la"
    if name in ["pems-bay", "bay", "pems"]:
        return "pems-bay"
    if name in ["songdo", "songdo-full", "sd"]:
        return "songdo"
    raise ValueError(f"Unknown dataset_name: {dataset_name}")

def _download_url(url: str, dst_path: str) -> str:
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    if not os.path.exists(dst_path):
        print(f"[download] {url} -> {dst_path}")
        urllib.request.urlretrieve(url, dst_path)
    return dst_path

def ensure_local_file(
    dataset_key: str,
    kind: str,
    data_dir: str,
    source: str = "auto",
    cache_dir: Optional[str] = None,
    url_override: Optional[str] = None,
    revision: Optional[str] = None,
) -> Optional[str]:   # 🔥 Optional[str]
    """
    legacy _ensure_local_file 그대로, 단 loc 파일이 없는 경우를 허용.
    """
    ds = DATA_SOURCES[dataset_key]

    # 🔥 여기 추가: 파일 이름이 None이면 loc는 optional로 처리
    filename = ds["files"].get(kind)
    if filename is None:
        if kind == "loc":
            # Songdo처럼 loc가 아예 없는 데이터셋
            return None
        # h5/adj가 None이면 진짜 설정 문제
        raise ValueError(f"{dataset_key} has no file entry for kind={kind}")

    local_path = os.path.join(data_dir, filename)

    if os.path.exists(local_path) and source in ["auto", "local"]:
        return local_path

    if source == "auto":
        url = url_override or ds["urls"].get(filename)
        if url:
            return _download_url(url, local_path)

        if ds.get("hf_repo"):
            return hf_hub_download(
                repo_id=ds["hf_repo"],
                filename=filename,
                repo_type="dataset",
                cache_dir=cache_dir,
                revision=revision,
            )
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
