# graphtune/rag/traffic_docs.py
"""
traffic_docs
============

시계열 + 그래프 기반 교통 데이터를
RAG에서 사용할 수 있는 '텍스트 문서(TrafficDoc)' 형태로 변환하는 유틸리티.

현재 버전:
- train_loader 기준으로 각 노드(node)에 대한 전체 히스토리 통계를 계산
- 노드별로 1개씩 "전체 기간 요약" 문서를 생성

향후 확장 아이디어:
- 시간대(출근/퇴근), 요일별, 날씨 조건별로 더 세분화된 문서 생성
- 특정 기간(최근 7일, 최근 30일 등)만 따로 요약
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch


@dataclass
class TrafficDoc:
    """
    RAG 인덱스에 들어갈 기본 단위 문서.

    doc_id      : 고유 식별자 (예: "songdo_node_12_full")
    city        : 도시 이름 (예: "songdo", "pems-bay")
    node_index  : 노드 인덱스 (0-based)
    node_name   : 센서 ID 등 사람 친화적인 이름 (없으면 None)
    time_span   : 사람이 읽기 좋은 시간 범위 설명 (예: "full_history")
    summary_text: LLM 컨텍스트로 바로 사용할 수 있는 자연어 요약
    stats       : 수치 요약 (mean, std, max, count 등)
    """
    doc_id: str
    city: str
    node_index: int
    node_name: Optional[str]
    time_span: str
    summary_text: str
    stats: Dict[str, float]


def _compute_node_stats_from_loader(
    loader: torch.utils.data.DataLoader,
    scaler: Optional[Any] = None,
    max_batches: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    DataLoader에서 (batch, T, N) 형태의 시계열을 읽어
    노드별 히스토리 통계(mean, std, max, count)를 계산한다.

    - X(입력 시계열)만 사용 (y는 사용하지 않음)
    - scaler가 주어졌으면 inverse_transform을 적용해 원래 스케일에서 통계를 계산
    - 데이터 전체를 메모리에 올리지 않고, 배치 단위로 누적 계산
    """
    node_sum = None
    node_sumsq = None
    node_max = None
    total_count = 0

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        # batch는 (x, y, mask) 구조라고 가정
        if isinstance(batch, (list, tuple)) and len(batch) >= 1:
            x = batch[0]
        else:
            x = batch

        # 🔴 여기서부터는 GPU 텐서든 뭐든 일단 CPU numpy로 변환
        x_np = x.detach().cpu().numpy()

        # 🔧 scaler가 있으면 원래 단위로 inverse_transform
        if scaler is not None:
            try:
                # loaders.py에서 norm할 때와 동일하게 마지막 축을 node로 보고 flatten
                shape = x_np.shape            # (B, T, N)
                flat = x_np.reshape(-1, shape[-1])   # (B*T, N)
                flat_inv = scaler.inverse_transform(flat)  # (B*T, N)
                x_np = flat_inv.reshape(shape)      # 다시 (B, T, N)
            except Exception as e:
                # 한 번만 경고 찍고, 이후에는 그냥 정규화 스케일로 진행
                if batch_idx == 0:
                    print(f"[traffic_docs] warning: scaler.inverse_transform 실패, 정규화 스케일에서 통계 계산 ({e})")

        # 📊 통계 계산용으로 마지막 축(N)을 노드로 보고 나머지를 모두 펼치기
        if x_np.ndim < 2:
            continue

        if x_np.ndim == 2:
            flat_stats = x_np   # (B, N)
        else:
            flat_stats = x_np.reshape(-1, x_np.shape[-1])  # (..., N)

        if node_sum is None:
            N = flat_stats.shape[1]
            node_sum = flat_stats.sum(axis=0)
            node_sumsq = (flat_stats ** 2).sum(axis=0)
            node_max = flat_stats.max(axis=0)
        else:
            node_sum += flat_stats.sum(axis=0)
            node_sumsq += (flat_stats ** 2).sum(axis=0)
            node_max = np.maximum(node_max, flat_stats.max(axis=0))

        total_count += flat_stats.shape[0]

    if node_sum is None or total_count == 0:
        raise ValueError("[traffic_docs] loader에서 유효한 데이터를 찾지 못했습니다.")

    mean = node_sum / total_count
    var = node_sumsq / total_count - mean ** 2
    var = np.maximum(var, 1e-6)
    std = np.sqrt(var)

    stats = {
        "mean": mean,
        "std": std,
        "max": node_max,
        "count": total_count,
    }
    return stats


def build_node_level_docs_from_bundle(
    bundle: Dict[str, Any],
    city: str,
    max_batches: Optional[int] = None,
    time_span_label: str = "full_history",
) -> List[TrafficDoc]:
    """
    GraphTune의 dataset bundle로부터
    '노드별 전체 히스토리 요약 문서' 리스트를 생성한다.

    Parameters
    ----------
    bundle: dict
        prepare_dataset(...)가 반환하는 dict.
        다음 키들을 사용한다:
        - "train_loader": DataLoader
        - "scaler": StandardScaler (선택)
        - "num_nodes": int (선택, 없으면 adjacency로부터 추론)
        - "A": adjacency matrix (선택)
        - "sensor_ids": 센서 ID 리스트 (선택)

    city: str
        도시 이름(label). 예: "metr-la", "pems-bay", "songdo"

    max_batches: int, optional
        train_loader에서 통계를 계산할 때 사용할 최대 배치 수.
        None이면 전체 데이터 사용.

    time_span_label: str
        이 요약이 커버하는 시간 범위를 사람이 읽기 좋게 표현한 라벨.
        예: "full_history", "recent_30_days" 등.

    Returns
    -------
    docs: List[TrafficDoc]
        노드별로 1개씩 생성된 문서 리스트.
    """
    train_loader = bundle["train_loader"]
    scaler = bundle.get("scaler", None)
    num_nodes = bundle.get("num_nodes", None)
    A = bundle.get("A", None)
    sensor_ids = bundle.get("sensor_ids", None)

    # 노드 수 추론
    if num_nodes is None:
        if A is not None:
            num_nodes = A.shape[0]
        else:
            # 마지막 수단: train_loader에서 한 배치 꺼내서 추론
            first_batch = next(iter(train_loader))
            x = first_batch[0] if isinstance(first_batch, (list, tuple)) else first_batch
            num_nodes = x.shape[-1]
            print(f"[traffic_docs] num_nodes를 DataLoader에서 추론: {num_nodes}")

    # 노드별 통계 계산
    stats = _compute_node_stats_from_loader(train_loader, scaler=scaler, max_batches=max_batches)
    mean = stats["mean"]
    std = stats["std"]
    vmax = stats["max"]

    # 평균 기준으로 노드 혼잡도 순위 계산 (내림차순)
    order = np.argsort(-mean)  # 큰 값이 먼저
    node_rank = np.empty_like(order)
    node_rank[order] = np.arange(len(order))
    num_nodes_float = float(num_nodes)

    docs: List[TrafficDoc] = []

    for node_idx in range(num_nodes):
        node_mean = float(mean[node_idx])
        node_std = float(std[node_idx])
        node_max = float(vmax[node_idx])
        rank = int(node_rank[node_idx])
        # 상위 몇 % 혼잡 노드인지
        congestion_percentile = 100.0 * (num_nodes_float - rank) / num_nodes_float

        node_name: Optional[str] = None
        if sensor_ids is not None and node_idx < len(sensor_ids):
            node_name = str(sensor_ids[node_idx])

        node_label = node_name if node_name is not None else f"node_{node_idx}"

        # 한국어 기준의 간단한 요약 텍스트 (나중에 템플릿/프롬프트로 튜닝 가능)
        summary_lines = [
            f"{city} 교통 센서 {node_label}의 전체 관측 기간 요약입니다.",
            f"- 평균 교통량: 약 {node_mean:.1f} 단위",
            f"- 변동성(표준편차): 약 {node_std:.1f}",
            f"- 관측된 최대 값: {node_max:.1f}",
            f"- 다른 노드와 비교했을 때 혼잡도 상위 약 {congestion_percentile:.1f}% 수준에 해당합니다.",
        ]
        summary_text = "\n".join(summary_lines)

        doc_id = f"{city}_node_{node_idx}_{time_span_label}"

        doc = TrafficDoc(
            doc_id=doc_id,
            city=city,
            node_index=node_idx,
            node_name=node_name,
            time_span=time_span_label,
            summary_text=summary_text,
            stats={
                "mean": node_mean,
                "std": node_std,
                "max": node_max,
                "rank": float(rank),
                "congestion_percentile": congestion_percentile,
                "count": float(stats["count"]),
            },
        )
        docs.append(doc)

    return docs
