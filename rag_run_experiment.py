#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
rag_run_experiment.py

멀티시티 교통 데이터(METR-LA, PeMS-BAY, Songdo)를 활용해서

- GraphTune 데이터 파이프라인으로 bundle 로딩
- TrafficDoc로 각 노드의 과거 교통 패턴을 텍스트 문서로 변환
- TF-IDF 기반 간단한 retriever로 RAG 컨텍스트 구성
- bundle의 (x, y)에서 '미래 구간 y'를 사용한 naive forecast 요약 생성
- HuggingFace microsoft/phi-1_5로 자연어 답변 생성

까지 한 번에 실행하는 간단 데모.
"""

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer

# GraphTune-lite imports
from graphtune import prepare_dataset
from graphtune.rag.traffic_docs import TrafficDoc, build_node_level_docs_from_bundle


# -------------------------------------------------------------------------
# TF-IDF 기반 RAG retriever
# -------------------------------------------------------------------------


@dataclass
class TfidfRetriever:
    vectorizer: TfidfVectorizer
    doc_matrix: Any  # sparse matrix
    docs: List[TrafficDoc]


def build_tfidf_retriever(all_docs: List[TrafficDoc]) -> TfidfRetriever:
    corpus = [doc.summary_text for doc in all_docs]
    vectorizer = TfidfVectorizer()
    doc_matrix = vectorizer.fit_transform(corpus)
    return TfidfRetriever(vectorizer=vectorizer, doc_matrix=doc_matrix, docs=all_docs)


def retrieve_docs(
    retriever: TfidfRetriever,
    query: str,
    k: int = 5,
    city: Optional[str] = None,
) -> List[Tuple[TrafficDoc, float]]:
    """
    간단한 TF-IDF 기반 top-k retrieval.
    city가 지정되면 해당 도시 문서만 대상으로 검색.
    """
    docs = retriever.docs
    vectorizer = retriever.vectorizer
    doc_matrix = retriever.doc_matrix

    if city is not None:
        idxs = [i for i, d in enumerate(docs) if d.city == city]
    else:
        idxs = list(range(len(docs)))

    if not idxs:
        raise ValueError(f"No docs found for city={city}")

    sub_matrix = doc_matrix[idxs, :]

    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, sub_matrix)[0]  # (num_sub_docs,)

    top_idx_sorted = sims.argsort()[::-1][:k]
    results: List[Tuple[TrafficDoc, float]] = []
    for rank_idx in top_idx_sorted:
        doc_global_idx = idxs[rank_idx]
        results.append((docs[doc_global_idx], float(sims[rank_idx])))
    return results


# -------------------------------------------------------------------------
# Naive forecast summary (데이터의 (x, y)를 직접 사용)
# -------------------------------------------------------------------------


def build_naive_forecast_summary(
    bundle: Dict[str, Any],
    city: str,
    node_index: Optional[int] = None,
    horizon: int = 12,
) -> str:
    """
    GraphTune bundle의 test_loader에서 (x, y)를 사용해
    간단한 '미래 구간' 요약을 만든다.

    - x: 과거 입력 시계열 (B, T_in, N)
    - y: 미래 타깃 시계열 (B, T_out, N)
    - scaler: StandardScaler가 있으면 inverse_transform으로 원래 단위 복원

    여기서는:
      - test_loader의 마지막 배치, 마지막 샘플을 사용
      - 해당 node_index에 대해:
          * 과거 T_in 동안의 평균/마지막 값
          * 향후 horizon 동안의 평균/최대 값
    """

    test_loader = bundle["test_loader"]
    scaler = bundle.get("scaler", None)
    num_nodes = bundle.get("num_nodes", None)

    if num_nodes is None:
        # fallback: 첫 배치에서 추론
        first_batch = next(iter(test_loader))
        x0 = first_batch[0]
        num_nodes = x0.shape[-1]

    if node_index is None:
        node_index = 0
    if node_index < 0 or node_index >= num_nodes:
        node_index = 0

    last_batch = None
    for batch in test_loader:
        last_batch = batch
    if last_batch is None:
        return f"No test data available for {city}, so a forecast summary cannot be computed."

    if isinstance(last_batch, (list, tuple)) and len(last_batch) >= 2:
        x, y = last_batch[0], last_batch[1]
    else:
        return f"Unexpected batch structure for {city}, cannot build forecast summary."

    # x: (B, T_in, N), y: (B, T_out, N) 기준으로 처리
    x_last = x[-1]  # (T_in, N)
    y_last = y[-1]  # (T_out, N)

    x_np = x_last.detach().cpu().numpy()  # (T_in, N)
    y_np = y_last.detach().cpu().numpy()  # (T_out, N)

    if scaler is not None:
        try:
            x_flat = x_np.reshape(-1, x_np.shape[-1])  # (T_in, N)
            y_flat = y_np.reshape(-1, y_np.shape[-1])  # (T_out, N)
            x_denorm = scaler.inverse_transform(x_flat).reshape(x_np.shape)
            y_denorm = scaler.inverse_transform(y_flat).reshape(y_np.shape)
        except Exception as e:
            print(f"[Forecast] scaler.inverse_transform failed ({e}), using normalized scale.")
            x_denorm, y_denorm = x_np, y_np
    else:
        x_denorm, y_denorm = x_np, y_np

    hist = x_denorm[:, node_index]                  # (T_in,)
    fut = y_denorm[:horizon, node_index]            # (horizon,)

    hist_mean = float(hist.mean())
    hist_last = float(hist[-1])
    fut_mean = float(fut.mean())
    fut_max = float(fut.max())

    summary = (
        f"For city {city}, focusing on sensor index {node_index}, "
        f"the average traffic volume over the recent {len(hist)} input time steps "
        f"is about {hist_mean:.1f} units, with the last observed value around {hist_last:.1f}. "
        f"In the following {len(fut)} time steps (forecast horizon), "
        f"the ground-truth future trajectory in the held-out test set has an average "
        f"of about {fut_mean:.1f} units, with peaks up to roughly {fut_max:.1f} units. "
        f"This gives a rough indication of the expected traffic level for that node in the near future."
    )
    return summary


# -------------------------------------------------------------------------
# (옵션) run_experiment 결과 파일에서 BigST 성능 메타데이터 읽기
# -------------------------------------------------------------------------


def load_bigst_metrics_for_city(
    city: str,
    results_path: str = "results.json",
) -> Optional[Dict[str, float]]:
    """
    run_experiment.py가 저장한 results.json (S_uep 리포트)에서
    해당 도시 + bigst 모델의 성능 요약을 읽어온다.

    - zero_rmse: zero-shot RMSE (transfer 전)
    - test_rmse: fine-tuning 후 RMSE
    """
    if not os.path.exists(results_path):
        return None

    try:
        with open(results_path, "r") as f:
            results = json.load(f)
    except Exception:
        return None

    # results는 stage별 dict 리스트
    candidates = [
        r for r in results if r.get("dataset") == city and r.get("model") == "bigst"
    ]
    if not candidates:
        return None

    # 마지막 stage 기준으로 선택
    r = sorted(candidates, key=lambda x: x.get("stage", 0))[-1]

    return {
        "zero_rmse": r.get("zero_rmse"),
        "test_rmse": r.get("test_rmse"),
    }


def build_bigst_metrics_text(city: str, results_path: str = "results.json") -> str:
    metrics = load_bigst_metrics_for_city(city, results_path=results_path)
    if metrics is None:
        return ""

    zero_rmse = metrics.get("zero_rmse", None)
    test_rmse = metrics.get("test_rmse", None)

    parts = [f"GraphTune's BigST engine for city {city} has the following benchmark results:"]
    if zero_rmse is not None:
        parts.append(f"- Zero-shot RMSE (before fine-tuning): {zero_rmse:.4f}")
    if test_rmse is not None:
        parts.append(f"- Fine-tuned RMSE on the test set: {test_rmse:.4f}")
    if zero_rmse is not None and test_rmse is not None:
        gain = 100.0 * (zero_rmse - test_rmse) / max(zero_rmse, 1e-6)
        parts.append(f"- Relative improvement over zero-shot: {gain:.2f}%")

    return "\n".join(parts)


# -------------------------------------------------------------------------
# LLM (microsoft/phi-1_5) 로더 & RAG 답변 생성
# -------------------------------------------------------------------------


@dataclass
class PhiRAGConfig:
    model_name: str = "microsoft/phi-1_5"
    max_new_tokens: int = 160


@dataclass
class PhiRAGState:
    tokenizer: Any
    model: Any
    config: PhiRAGConfig


def load_phi_model(config: PhiRAGConfig) -> PhiRAGState:
    print(f"[phi] Loading model: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    return PhiRAGState(tokenizer=tokenizer, model=model, config=config)


def build_context_for_query(
    query: str,
    city: str,
    retriever: TfidfRetriever,
    rag_bundles: Dict[str, Dict[str, Any]],
    k: int = 3,
    horizon: int = 12,
    results_path: str = "results.json",
) -> str:
    """
    - TrafficDoc 기반 RAG: 해당 도시에서 query와 관련있는 노드 문서 top-k 검색
    - test_loader의 (x, y)를 활용한 naive forecast summary
    - (있다면) run_experiment 결과에서 BigST 성능 메타데이터 요약
    """
    # 1) 과거 TrafficDoc 기반 RAG
    retrieved = retrieve_docs(retriever, query, k=k, city=city)

    context_parts: List[str] = []

    for doc, score in retrieved:
        header = f"[Doc from {doc.city}, node_index={doc.node_index}, score={score:.3f}]"
        context_parts.append(header + "\n" + doc.summary_text)

    # 2) naive forecast 요약 (해당 도시 bundle 사용)
    bundle = rag_bundles[city]
    if retrieved:
        target_node = retrieved[0][0].node_index
    else:
        target_node = 0

    forecast_text = build_naive_forecast_summary(
        bundle,
        city=city,
        node_index=target_node,
        horizon=horizon,
    )
    context_parts.append("[Forecast]\n" + forecast_text)

    # 3) (옵션) BigST 벤치마크 성능 텍스트
    bigst_meta = build_bigst_metrics_text(city, results_path=results_path)
    if bigst_meta:
        context_parts.append("[Model performance]\n" + bigst_meta)

    context = "\n\n".join(context_parts)
    return context


def generate_answer_with_phi(
    query: str,
    city: str,
    rag_state: PhiRAGState,
    retriever: TfidfRetriever,
    rag_bundles: Dict[str, Dict[str, Any]],
    k: int = 3,
    horizon: int = 12,
    results_path: str = "results.json",
) -> str:
    context = build_context_for_query(
        query=query,
        city=city,
        retriever=retriever,
        rag_bundles=rag_bundles,
        k=k,
        horizon=horizon,
        results_path=results_path,
    )

    system_prompt = (
        "You are a helpful multi-city traffic assistant. "
        "You are given traffic history summaries (per sensor) and a naive forecast summary based on held-out data. "
        "Optionally, you may also see benchmark metrics from a separate GraphTune tuning pipeline. "
        "Use this information to answer the user's question accurately and concisely. "
        "Assume that all user questions are in English, and respond in English.\n"
    )

    prompt = (
        system_prompt
        + "\n[Context]\n"
        + context
        + "\n\n[User Question]\n"
        + query
        + "\n\n[Assistant Answer]\n"
    )

    tokenizer = rag_state.tokenizer
    model = rag_state.model
    max_new_tokens = rag_state.config.max_new_tokens

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    out_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "[Assistant Answer]" in out_text:
        out_text = out_text.split("[Assistant Answer]")[-1].strip()
    return out_text


# -------------------------------------------------------------------------
# main: end-to-end 파이프라인
# -------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="GraphTune-RAG demo with phi-1_5 (no training inside this script)")
    parser.add_argument(
        "--cities",
        type=str,
        default="metr-la,pems-bay,songdo",
        help="Comma-separated list of cities to load (metr-la,pems-bay,songdo).",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="DATA",
        help="Directory where traffic dataset files are stored.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="Which areas in LA tend to be most congested during the evening rush hour?",
        help="User question in English.",
    )
    parser.add_argument(
        "--city",
        type=str,
        default="metr-la",
        help="Target city for the query: one of metr-la, pems-bay, songdo.",
    )
    parser.add_argument(
        "--phi_model",
        type=str,
        default="microsoft/phi-1_5",
        help="Name of the HuggingFace causal LM to use.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=12,
        help="Forecast horizon (number of future time steps to summarize).",
    )
    parser.add_argument(
        "--results_path",
        type=str,
        default="results.json",
        help="Path to results.json produced by run_experiment.py (optional, for BigST metrics).",
    )
    args = parser.parse_args()

    city_list = [c.strip() for c in args.cities.split(",") if c.strip()]
    data_dir = args.data_dir

    print("=== GraphTune-RAG demo (phi-1_5) ===")
    print(f"cities = {city_list}")
    print(f"target city for query = {args.city}")
    print(f"query = {args.query}")

    # 1) 각 도시별 RAG용 데이터셋 준비 + TrafficDoc 생성
    rag_bundles: Dict[str, Dict[str, Any]] = {}
    all_docs: List[TrafficDoc] = []

    for city in city_list:
        print(f"\n[Data] Preparing RAG dataset for city={city}")
        bundle = prepare_dataset(
            dataset_name=city,
            data_dir=data_dir,
            source="local",
            T_in=12,
            T_out=12,
            batch_size=32,
        )
        bundle["dataset_name"] = city
        rag_bundles[city] = bundle

        print(f"[TrafficDoc] Building node-level docs for city={city}")
        docs = build_node_level_docs_from_bundle(bundle, city=city, max_batches=100)
        print(f"[TrafficDoc] {len(docs)} docs generated for {city}")
        all_docs.extend(docs)

    # 2) TF-IDF 기반 retriever 생성
    print("\n[Retriever] Building TF-IDF retriever over all docs...")
    retriever = build_tfidf_retriever(all_docs)
    print("[Retriever] Done.")

    # 3) phi-1_5 로드
    rag_config = PhiRAGConfig(model_name=args.phi_model)
    rag_state = load_phi_model(rag_config)

    # 4) 단일 질문에 대한 RAG + naive forecast + LLM 파이프라인 실행
    target_city = args.city
    if target_city not in rag_bundles:
        raise ValueError(f"target city {target_city} not in loaded cities {city_list}")

    print("\n=== Running RAG + Forecast (naive) + LLM ===")
    answer = generate_answer_with_phi(
        query=args.query,
        city=target_city,
        rag_state=rag_state,
        retriever=retriever,
        rag_bundles=rag_bundles,
        k=3,
        horizon=args.horizon,
        results_path=args.results_path,
    )

    print("\n[User]")
    print(args.query)
    print("\n[Assistant]")
    print(answer)


if __name__ == "__main__":
    main()
