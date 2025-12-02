#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
rag_run_experiment.py

멀티시티 교통 데이터(METR-LA, PeMS-BAY, Songdo)를 활용해서
- GraphTune의 예측 엔진(BigST)으로 미래 교통량을 예측하고
- TrafficDoc 기반 RAG로 과거 패턴을 검색한 뒤
- HuggingFace microsoft/phi-1_5 모델로 자연어 답변을 생성하는

간단한 RAG 추론 실험 파이프라인.

사용 예시 (Colab 등에서):

    %cd /content/graphtune-lite
    !python rag_run_experiment.py --city songdo --query "송도에서 금요일 저녁 6시에 얼마나 막힐지 알려줘."

"""

import argparse
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# GraphTune-lite imports
from graphtune import prepare_dataset, build_model, train_one_stage
from graphtune.config import DEFAULT_MODEL_KWARGS
from graphtune.rag.traffic_docs import TrafficDoc, build_node_level_docs_from_bundle


# -------------------------------------------------------------------------
# 간단한 RAG용 TF-IDF retriever
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

    # 서브코퍼스/매트릭스 구성
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
# 예측 엔진 (BigST) - GraphTune 기반
# -------------------------------------------------------------------------


def _prepare_bigst_input(x: torch.Tensor) -> torch.Tensor:
    """
    Ensure input shape for BigST is (B, N, T, D) with D >= 2.

    - If x is (B, T, N)           -> build (B, N, T, 2):
        * channel 0: original value
        * channel 1: simple time index in [0, 1]
    - If x is (B, T, N, D)        -> permute to (B, N, T, D), and if D == 1,
                                     append a time index channel.
    - If x is already (B, N, T, D)-> if D == 1, append time index channel.
    """
    if x.dim() == 3:
        # x: (B, T, N)
        B, T, N = x.size()
        # value channel: (B, N, T, 1)
        val = x.permute(0, 2, 1).unsqueeze(-1)  # (B, N, T, 1)

        # simple time index in [0, 1], shared across batch & nodes
        time_idx = torch.linspace(0.0, 1.0, steps=T, device=x.device)  # (T,)
        time_idx = time_idx.view(1, 1, T, 1).expand(B, N, T, 1)        # (B, N, T, 1)

        x4 = torch.cat([val, time_idx], dim=-1)  # (B, N, T, 2)
        return x4

    elif x.dim() == 4:
        # x: (B, T, N, D) or (B, N, T, D)
        B, d1, d2, D = x.size()
        x4 = x

        # Heuristic: if second dim looks like time (smaller than spatial dim),
        # we assume (B, T, N, D) and permute to (B, N, T, D).
        if d1 < d2:
            x4 = x4.permute(0, 2, 1, 3)  # (B, N, T, D)

        # Ensure we have at least 2 feature channels
        if x4.size(-1) == 1:
            B, N, T, _ = x4.size()
            time_idx = torch.linspace(0.0, 1.0, steps=T, device=x4.device)
            time_idx = time_idx.view(1, 1, T, 1).expand(B, N, T, 1)
            x4 = torch.cat([x4, time_idx], dim=-1)  # (B, N, T, 2)

        return x4

    else:
        raise ValueError(f"Unexpected input dimension for BigST: x.dim()={x.dim()}")


def _project_bigst_output(y_hat: torch.Tensor) -> torch.Tensor:
    """
    Project BigST output to (B, T_out, N) so it matches y.

    - If (B, N, T_out, D) -> squeeze D, then permute to (B, T_out, N)
    - If (B, N, T_out)     -> permute to (B, T_out, N)
    """
    if y_hat.dim() == 4:
        # (B, N, T_out, D)
        if y_hat.size(-1) == 1:
            y_hat = y_hat.squeeze(-1)
    if y_hat.dim() == 3:
        # (B, N, T_out) -> (B, T_out, N)
        y_hat = y_hat.permute(0, 2, 1)
    return y_hat


def train_bigst_for_bundle(
    bundle: Dict[str, Any],
    num_epochs: int = 3,
    lr: float = 1e-3,
    device: Optional[str] = None,
):
    """
    Simple custom training loop for BigST using the GraphTune dataloaders.

    - Uses masked MAE over y (B, T_out, N).
    - Only for demo: we don't track val metrics here.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(
        "bigst",
        bundle,
        **DEFAULT_MODEL_KWARGS["bigst"],
    ).to(device)

    train_loader = bundle["train_loader"]
    scaler = bundle["scaler"]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"[BigST] Simple training for dataset={bundle.get('dataset_name', 'unknown')} "
          f"(epochs={num_epochs}, lr={lr})")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        steps = 0

        for batch in train_loader:
            if isinstance(batch, (list, tuple)) and len(batch) >= 3:
                x, y, mask = batch
            else:
                # Fallback: assume (x, y)
                x, y = batch
                mask = torch.ones_like(y)

            x = x.to(device)          # (B, T_in, N) or (B, T_in, N, D)
            y = y.to(device)          # (B, T_out, N)
            mask = mask.to(device)    # same shape as y or y with extra dim

            x_in = _prepare_bigst_input(x)   # -> (B, N, T_in, D)
            y_hat = model(x_in)              # BigST output

            y_hat = _project_bigst_output(y_hat)  # -> (B, T_out, N)

            # Align mask shape
            if mask.dim() > y_hat.dim():
                # e.g., (B, T_out, N, 1) -> (B, T_out, N)
                mask = mask.squeeze(-1)
            mask = (mask > 0).float()

            diff = (y_hat - y) * mask
            loss = diff.abs().sum() / (mask.sum() + 1e-6)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            steps += 1

        avg_loss = total_loss / max(steps, 1)
        print(f"[BigST] epoch {epoch+1}/{num_epochs} train_masked_MAE={avg_loss:.4f}")

    print("[BigST] Training finished (demo).")
    return model


def simple_forecast_node(
    bundle: Dict[str, Any],
    model: torch.nn.Module,
    node_index: int,
    horizon: int = 12,
    device: Optional[str] = None,
) -> str:
    """
    Very simple demo:
    - Use the last batch / sample from test_loader
    - Predict the future horizon for the selected node
    - Return a short English summary of past and future statistics
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    test_loader = bundle["test_loader"]
    scaler = bundle["scaler"]

    model.eval()
    with torch.no_grad():
        last_batch = None
        for batch in test_loader:
            last_batch = batch
        if last_batch is None:
            return "No test data is available, so the forecast cannot be computed."

        if isinstance(last_batch, (list, tuple)) and len(last_batch) >= 3:
            x, y, mask = last_batch
        else:
            x, y = last_batch
            mask = torch.ones_like(y)

        x = x.to(device)  # (B, T_in, N) or (B, T_in, N, D)
        x_in = _prepare_bigst_input(x)  # -> (B, N, T_in, D)
        y_hat = model(x_in)             # BigST output

    # Use only the last sample in the last batch
    x_last = x[-1:]         # (1, T_in, N)  (original, for history)
    y_hat_last = y_hat[-1:] # e.g., (1, N, T_out, D?) or (1, N, T_out)

    # Project model output to (B, T_out, N)
    y_hat_last = _project_bigst_output(y_hat_last)  # -> (1, T_out, N)

    # Inverse scaling (scaler expects (..., N))
    x_np = x_last.detach().cpu().numpy().reshape(-1, x_last.shape[-1])
    y_np = y_hat_last.detach().cpu().numpy().reshape(-1, y_hat_last.shape[-1])

    x_denorm = scaler.inverse_transform(x_np).reshape(x_last.shape)
    y_denorm = scaler.inverse_transform(y_np).reshape(y_hat_last.shape)

    hist = x_denorm[0, :, node_index]             # (T_in,)
    fut = y_denorm[0, :horizon, node_index]       # (horizon,)

    hist_mean = float(hist.mean())
    hist_last = float(hist[-1])
    fut_mean = float(fut.mean())
    fut_max = float(fut.max())

    summary = (
        f"For the selected node (index {node_index}), "
        f"the average traffic volume over the last {len(hist)} time steps is about {hist_mean:.1f} units, "
        f"and the last observed value is {hist_last:.1f}. "
        f"The forecast model expects the next {horizon} time steps to have an average of {fut_mean:.1f} "
        f"and a maximum of about {fut_max:.1f} units of traffic volume."
    )
    return summary

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
    bundles: Dict[str, Dict[str, Any]],
    forecast_models: Dict[str, torch.nn.Module],
    k: int = 3,
    use_forecast: bool = True,
) -> str:
    # 1) 과거 TrafficDoc 기반 RAG
    retrieved = retrieve_docs(retriever, query, k=k, city=city)

    context_parts: List[str] = []

    for doc, score in retrieved:
        header = f"[Doc from {doc.city}, node_index={doc.node_index}, score={score:.3f}]"
        context_parts.append(header + "\n" + doc.summary_text)

    # 2) 예측 요약 추가 (예측 모델이 존재하는 경우에만)
    if use_forecast and city in forecast_models:
        top_doc, _ = retrieved[0]
        bundle = bundles[city]
        model = forecast_models[city]

        forecast_text = simple_forecast_node(bundle, model, node_index=top_doc.node_index)
        context_parts.append("[Forecast]\n" + forecast_text)

    context = "\n\n".join(context_parts)
    return context


def generate_answer_with_phi(
    query: str,
    city: str,
    rag_state: PhiRAGState,
    retriever: TfidfRetriever,
    bundles: Dict[str, Dict[str, Any]],
    forecast_models: Dict[str, torch.nn.Module],
    k: int = 3,
    use_forecast: bool = True,
) -> str:
    context = build_context_for_query(
        query=query,
        city=city,
        retriever=retriever,
        bundles=bundles,
        forecast_models=forecast_models,
        k=k,
        use_forecast=use_forecast,
    )

    # system_prompt는 영어지만, user 질문은 한국어로 넣어도 됨.
    # 답변도 한국어로 유도하고 싶으면 지시를 추가.
    system_prompt = (
        "You are a helpful multi-city traffic assistant. "
        "You are given traffic history summaries and model-based future forecasts. "
        "Use them to answer the user's question accurately and concisely. "
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
    parser = argparse.ArgumentParser(description="GraphTune-RAG demo with phi-1_5")
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
        "--epochs",
        type=int,
        default=3,
        help="Number of epochs to train the BigST forecast model (for demo, keep it small).",
    )
    parser.add_argument(
        "--phi_model",
        type=str,
        default="microsoft/phi-1_5",
        help="Name of the HuggingFace causal LM to use.",
    )

    parser.add_argument(
    "--query",
    type=str,
    default="How heavy will the traffic be around 6 pm in Songdo on a typical weekday?",
    help="User question in English.",
    )

    parser.add_argument(
        "--city",
        type=str,
        default="songdo",
        help="Target city for the query: one of metr-la, pems-bay, songdo.",
    )

    args = parser.parse_args()

    city_list = [c.strip() for c in args.cities.split(",") if c.strip()]
    data_dir = args.data_dir

    print("=== GraphTune-RAG demo ===")
    print(f"cities = {city_list}")
    print(f"target city for query = {args.city}")
    print(f"query = {args.query}")

    # 1) 각 도시별 데이터셋 준비 + TrafficDoc 생성
    bundles: Dict[str, Dict[str, Any]] = {}
    all_docs: List[TrafficDoc] = []

    for city in city_list:
        print(f"\n[Data] Preparing dataset for city={city}")
        bundle = prepare_dataset(
            dataset_name=city,
            data_dir=data_dir,
            source="local",
            T_in=12,
            T_out=12,
            batch_size=32,
        )
        bundle["dataset_name"] = city
        bundles[city] = bundle

        print(f"[TrafficDoc] Building node-level docs for city={city}")
        docs = build_node_level_docs_from_bundle(bundle, city=city, max_batches=100)
        print(f"[TrafficDoc] {len(docs)} docs generated for {city}")
        all_docs.extend(docs)

    # 2) TF-IDF 기반 retriever 생성
    print("\n[Retriever] Building TF-IDF retriever over all docs...")
    retriever = build_tfidf_retriever(all_docs)
    print("[Retriever] Done.")

    # 3) 각 도시별 BigST 예측 모델 학습 (간단히 몇 epoch만)
    forecast_models: Dict[str, torch.nn.Module] = {}
    for city in city_list:
        print(f"\n[Forecast] Training BigST for city={city}")
        model = train_bigst_for_bundle(bundles[city], num_epochs=args.epochs)
        forecast_models[city] = model

    # 4) phi-1_5 로드
    rag_config = PhiRAGConfig(model_name=args.phi_model)
    rag_state = load_phi_model(rag_config)

    # 5) 단일 질문에 대한 RAG+예측+LLM 파이프라인 실행
    target_city = args.city
    if target_city not in bundles:
        raise ValueError(f"target city {target_city} not in loaded cities {city_list}")

    print("\n=== Running RAG + Forecast + LLM ===")
    answer = generate_answer_with_phi(
        query=args.query,
        city=target_city,
        rag_state=rag_state,
        retriever=retriever,
        bundles=bundles,
        forecast_models=forecast_models,
        k=3,
        use_forecast=True,
    )

    print("\n[User]")
    print(args.query)
    print("\n[Assistant]")
    print(answer)


if __name__ == "__main__":
    main()
