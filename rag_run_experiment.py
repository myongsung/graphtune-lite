#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
rag_run_experiment.py

End-to-end demo of a multi-city traffic RAG system:

- Load traffic datasets (METR-LA, PEMS-BAY, Songdo) via GraphTune-lite.
- Build per-node TrafficDoc summaries (history-based text docs).
- Build a TF-IDF retriever over all TrafficDocs.
- Load trained BigST models from checkpoints (produced by run_experiment.py).
- For a user query about a target city, retrieve relevant docs,
  run BigST to obtain model-based forecasts for the most relevant sensor,
  and feed all of this as context to a small LLM (microsoft/phi-1_5).
- The LLM answers in English.

Assumptions:
- You have already run `run_experiment.py` with model=bigst and datasets
  including metr-la, pems-bay, songdo, so that checkpoints like
  `checkpoints/bigst_metr-la_stage0.pt` are available.
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
from graphtune import prepare_dataset, build_model
from graphtune.config import DEFAULT_MODEL_KWARGS
from graphtune.rag.traffic_docs import TrafficDoc, build_node_level_docs_from_bundle


# -------------------------------------------------------------------------
# TF-IDF based RAG retriever
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
    Simple TF-IDF top-k retrieval.
    If `city` is given, restrict to docs from that city.
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
# BigST checkpoint loading + model-based forecast summary
# -------------------------------------------------------------------------


def _find_latest_bigst_checkpoint(
    city: str,
    ckpt_dir: str = "checkpoints",
    prefix_model_name: str = "bigst",
) -> str:
    """
    Find the latest checkpoint file for BigST trained on a given city.
    Expected filename pattern: {model_name}_{city}_stage{stage_idx}.pt
    e.g. bigst_metr-la_stage0.pt, bigst_pems-bay_stage1.pt, ...
    """
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    prefix = f"{prefix_model_name}_{city}_stage"
    candidates: List[Tuple[int, str]] = []

    for fname in os.listdir(ckpt_dir):
        if not (fname.startswith(prefix) and fname.endswith(".pt")):
            continue
        # parse stage index
        try:
            middle = fname[len(prefix) :]  # e.g. "0.pt"
            stage_str = middle.replace(".pt", "").strip("_")
            stage_idx = int(stage_str)
            candidates.append((stage_idx, os.path.join(ckpt_dir, fname)))
        except Exception:
            continue

    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint found for city={city} in {ckpt_dir} "
            f"(looking for prefix '{prefix}*')."
        )

    # choose the highest stage index
    candidates.sort(key=lambda x: x[0])
    _, path = candidates[-1]
    return path


def load_bigst_model_for_city(
    city: str,
    data_dir: str = "DATA",
    ckpt_dir: str = "checkpoints",
    T_in: int = 12,
    T_out: int = 12,
    batch_size: int = 32,
    device: Optional[str] = None,
) -> Tuple[torch.nn.Module, Dict[str, Any], str]:
    """
    Load a BigST model + dataset bundle (for_bigst=True) for a given city
    using a checkpoint produced by run_experiment.py.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # BigST-friendly dataset
    bundle = prepare_dataset(
        dataset_name=city,
        data_dir=data_dir,
        source="local",
        T_in=T_in,
        T_out=T_out,
        batch_size=batch_size,
        for_bigst=True,
    )
    bundle["dataset_name"] = city

    # Build model with the same config as run_experiment
    model_kwargs = DEFAULT_MODEL_KWARGS["bigst"]
    model = build_model("bigst", bundle, **model_kwargs).to(device)

    # Load checkpoint
    ckpt_path = _find_latest_bigst_checkpoint(city, ckpt_dir=ckpt_dir)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    print(f"[BigST] Loaded checkpoint for city={city} from {ckpt_path}")
    return model, bundle, device


def _extract_node_sequence(
    t: torch.Tensor,
    node_index: int,
    num_nodes: int,
) -> torch.Tensor:
    """
    Extract a 1D time sequence (T,) for the given node_index from the last
    sample in the batch.

    Handles several possible shapes:
    - (B, N, T) or (B, T, N)
    - (B, N, T, D) or (B, T, N, D)
    """
    # We will always use the last sample in the batch
    if t.dim() == 4:
        # (B, ?, ?, D) with D as feature dim
        core = t[..., 0] if t.size(-1) > 1 else t.squeeze(-1)  # (B, ?, ?)
        B, d1, d2 = core.shape

        if d1 == num_nodes:
            # (B, N, T)
            seq = core[-1, node_index, :]  # (T,)
        elif d2 == num_nodes:
            # (B, T, N)
            seq = core[-1, :, node_index]  # (T,)
        else:
            # fallback: flatten assuming last dim is num_nodes
            flat = core[-1].reshape(-1, num_nodes)  # (T, N)
            seq = flat[:, node_index]
        return seq

    elif t.dim() == 3:
        # (B, ?, ?)
        B, d1, d2 = t.shape
        if d1 == num_nodes:
            # (B, N, T)
            seq = t[-1, node_index, :]  # (T,)
        elif d2 == num_nodes:
            # (B, T, N)
            seq = t[-1, :, node_index]  # (T,)
        else:
            # fallback
            flat = t[-1].reshape(-1, num_nodes)
            seq = flat[:, node_index]
        return seq

    else:
        # Unexpected; just flatten last sample
        return t[-1].view(-1)


def build_forecast_summary_with_bigst(
    bundle: Dict[str, Any],
    model: torch.nn.Module,
    city: str,
    node_index: int = 0,
    horizon: int = 12,
) -> str:
    """
    Use the trained BigST model to produce a *model-based* forecast summary
    for a given city and sensor (node_index).

    Steps:
    - Take the last batch of the test_loader.
    - Run model(x) to get y_hat (predicted future).
    - Extract the time sequence for the selected node.
    - Inverse-transform using bundle["scaler"] (if available).
    - Compute simple statistics (mean / last / max) and return as text.
    """
    test_loader = bundle["test_loader"]
    scaler = bundle.get("scaler", None)
    num_nodes = bundle.get("num_nodes", None)

    if num_nodes is None:
        # fallback: infer from first batch
        first_batch = next(iter(test_loader))
        x0 = first_batch[0]
        num_nodes = x0.shape[-2] if x0.dim() >= 3 else x0.shape[-1]

    if node_index < 0 or node_index >= num_nodes:
        node_index = 0

    # Grab last batch
    last_batch = None
    for batch in test_loader:
        last_batch = batch

    if last_batch is None:
        return f"No test data available for {city}, so a model-based forecast cannot be computed."

    if isinstance(last_batch, (list, tuple)) and len(last_batch) >= 2:
        x, _ = last_batch[0], last_batch[1]
    else:
        x = last_batch[0]

    device = next(model.parameters()).device
    x = x.to(device)

    # Run model
    with torch.no_grad():
        y_hat_raw = model(x)

    if isinstance(y_hat_raw, (tuple, list)):
        y_hat = y_hat_raw[0]
    else:
        y_hat = y_hat_raw

    # Bring tensors to CPU for shape processing & inverse-transform
    x_cpu = x.detach().cpu()
    y_hat_cpu = y_hat.detach().cpu()

    # Extract history (from x) and forecast (from y_hat) for this node
    hist_norm = _extract_node_sequence(x_cpu, node_index, num_nodes)  # (T_in,)
    fut_norm = _extract_node_sequence(y_hat_cpu, node_index, num_nodes)  # (T_out,)

    # Crop forecast to desired horizon
    if horizon is not None and fut_norm.numel() > horizon:
        fut_norm = fut_norm[:horizon]

    hist_np = hist_norm.numpy().reshape(-1)  # (T_in,)
    fut_np = fut_norm.numpy().reshape(-1)    # (T_out,)

    # Inverse-transform using scaler (if available).
    # GraphTune scalers are node-wise affine. We can embed our single-node vector
    # into a (T, num_nodes) matrix, then inverse_transform, then select the column.
    if scaler is not None:
        T_in = hist_np.shape[0]
        T_out = fut_np.shape[0]

        hist_mat = np.zeros((T_in, num_nodes), dtype=np.float32)
        fut_mat = np.zeros((T_out, num_nodes), dtype=np.float32)

        hist_mat[:, node_index] = hist_np
        fut_mat[:, node_index] = fut_np

        try:
            hist_denorm = scaler.inverse_transform(hist_mat)[:, node_index]
            fut_denorm = scaler.inverse_transform(fut_mat)[:, node_index]
        except Exception as e:
            print(f"[BigST Forecast] scaler.inverse_transform failed: {e}")
            print("[BigST Forecast] Falling back to normalized scale.")
            hist_denorm = hist_np
            fut_denorm = fut_np
    else:
        hist_denorm = hist_np
        fut_denorm = fut_np

    hist_mean = float(hist_denorm.mean())
    hist_last = float(hist_denorm[-1])
    fut_mean = float(fut_denorm.mean())
    fut_max = float(fut_denorm.max())

    summary = (
        f"For city {city}, focusing on sensor index {node_index}, "
        f"the BigST model predicts the next {len(fut_denorm)} time steps of traffic volume. "
        f"Over the recent input window, the average observed traffic volume is about {hist_mean:.1f} units, "
        f"with the last observed value around {hist_last:.1f}. "
        f"According to the model's forecast, the upcoming horizon has an average of about {fut_mean:.1f} "
        f"and peaks up to roughly {fut_max:.1f} units of traffic volume for this sensor."
    )
    return summary


# -------------------------------------------------------------------------
# (Optional) Read BigST performance metrics from run_experiment results.json
# -------------------------------------------------------------------------


def load_bigst_metrics_for_city(
    city: str,
    results_path: str = "results.json",
) -> Optional[Dict[str, float]]:
    """
    Read BigST performance metrics for a given city from results.json
    produced by run_experiment.py.

    - zero_rmse: zero-shot RMSE (transfer before fine-tuning)
    - test_rmse: fine-tuned RMSE on the test set
    """
    if not os.path.exists(results_path):
        return None

    try:
        with open(results_path, "r") as f:
            results = json.load(f)
    except Exception:
        return None

    # results is a list of per-stage dicts
    candidates = [
        r for r in results if r.get("dataset") == city and r.get("model") == "bigst"
    ]
    if not candidates:
        return None

    # pick the last stage for this city
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
# LLM (microsoft/phi-1_5) loader & RAG answer generation
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
    bigst_models: Optional[Dict[str, torch.nn.Module]] = None,
    bigst_bundles: Optional[Dict[str, Dict[str, Any]]] = None,
    k: int = 3,
    horizon: int = 12,
    results_path: str = "results.json",
) -> str:
    """
    Build the RAG context for a given query and city:

    - TrafficDoc-based RAG: retrieve top-k node-level history summaries.
    - BigST-based forecast summary for the most relevant node (if checkpoint exists).
    - Optional: BigST benchmark metrics from run_experiment results.json.
    """
    # 1) History-based RAG
    retrieved = retrieve_docs(retriever, query, k=k, city=city)

    context_parts: List[str] = []

    for doc, score in retrieved:
        header = f"[Doc from {doc.city}, node_index={doc.node_index}, score={score:.3f}]"
        context_parts.append(header + "\n" + doc.summary_text)

    # 2) Model-based forecast summary
    if retrieved:
        target_node = retrieved[0][0].node_index
    else:
        target_node = 0

    forecast_text: str
    if (
        bigst_models is not None
        and bigst_bundles is not None
        and city in bigst_models
        and city in bigst_bundles
    ):
        forecast_text = build_forecast_summary_with_bigst(
            bundle=bigst_bundles[city],
            model=bigst_models[city],
            city=city,
            node_index=target_node,
            horizon=horizon,
        )
    else:
        # Fallback: if BigST checkpoint is missing, do not crash.
        forecast_text = (
            f"A model-based forecast summary is not available for city {city} "
            f"(no BigST checkpoint found)."
        )

    context_parts.append("[Forecast]\n" + forecast_text)

    # 3) BigST benchmark metrics (if available)
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
    bigst_models: Optional[Dict[str, torch.nn.Module]] = None,
    bigst_bundles: Optional[Dict[str, Dict[str, Any]]] = None,
    k: int = 3,
    horizon: int = 12,
    results_path: str = "results.json",
) -> str:
    context = build_context_for_query(
        query=query,
        city=city,
        retriever=retriever,
        rag_bundles=rag_bundles,
        bigst_models=bigst_models,
        bigst_bundles=bigst_bundles,
        k=k,
        horizon=horizon,
        results_path=results_path,
    )

    system_prompt = (
        "You are a helpful multi-city traffic assistant. "
        "You are given traffic history summaries (per sensor), "
        "model-based forecasts from a trained BigST graph model, "
        "and optionally performance metrics from the GraphTune benchmarking pipeline. "
        "Use only the information in the context to answer the user's question. "
        "Do NOT invent new place names or districts that are not mentioned in the context. "
        "If you refer to congestion, be explicit whether a sensor is in the top X% of congestion. "
        "The user asks questions in English; answer in English.\n"
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
# main: end-to-end pipeline
# -------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="GraphTune-RAG demo with phi-1_5 and BigST model-based forecasts"
    )
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
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="checkpoints",
        help="Directory where BigST checkpoints (bigst_*_stage*.pt) are stored.",
    )
    args = parser.parse_args()

    city_list = [c.strip() for c in args.cities.split(",") if c.strip()]
    data_dir = args.data_dir

    print("=== GraphTune-RAG demo (phi-1_5 + BigST) ===")
    print(f"cities = {city_list}")
    print(f"target city for query = {args.city}")
    print(f"query = {args.query}")

    # 1) RAG datasets + TrafficDocs per city
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

    # 2) TF-IDF retriever over all docs
    print("\n[Retriever] Building TF-IDF retriever over all docs...")
    retriever = build_tfidf_retriever(all_docs)
    print("[Retriever] Done.")

    # 3) BigST models and bundles (for_bigst=True) per city
    bigst_models: Dict[str, torch.nn.Module] = {}
    bigst_bundles: Dict[str, Dict[str, Any]] = {}

    for city in city_list:
        try:
            m, b, dev = load_bigst_model_for_city(
                city,
                data_dir=data_dir,
                ckpt_dir=args.ckpt_dir,
                T_in=12,
                T_out=12,
                batch_size=32,
            )
            bigst_models[city] = m
            bigst_bundles[city] = b
        except FileNotFoundError as e:
            print(f"[BigST] {e}")
            print(f"[BigST] Forecasts for city={city} will fall back to a placeholder message.")

    # 4) Load phi-1_5
    rag_config = PhiRAGConfig(model_name=args.phi_model)
    rag_state = load_phi_model(rag_config)

    # 5) Run RAG + BigST forecast + LLM for a single query
    target_city = args.city
    if target_city not in rag_bundles:
        raise ValueError(f"target city {target_city} not in loaded cities {city_list}")

    print("\n=== Running RAG + BigST Forecast + LLM ===")
    answer = generate_answer_with_phi(
        query=args.query,
        city=target_city,
        rag_state=rag_state,
        retriever=retriever,
        rag_bundles=rag_bundles,
        bigst_models=bigst_models,
        bigst_bundles=bigst_bundles,
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
