# graphtune/core/scoring.py
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


def _trapz_auc(x: np.ndarray, y: np.ndarray) -> float:
    """Trapezoidal AUC, assumes x is sorted in [0,1]."""
    if len(x) < 2:
        return 0.0
    return float(np.trapz(y, x))


def _get_transfer_curve(sr: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      frac (budget fraction in [0,1]),
      rmse values at those fractions.
      
    Expected formats:
      sr["curve_rmse"] = [(frac, rmse), ...]
      OR fallback from zero_rmse/test_rmse.
    """
    if "curve_rmse" in sr and sr["curve_rmse"] is not None:
        pts = sr["curve_rmse"]
        pts = sorted(pts, key=lambda p: p[0])
        frac = np.array([p[0] for p in pts], dtype=np.float64)
        rmse = np.array([p[1] for p in pts], dtype=np.float64)
        return frac, rmse

    # fallback 2-point curve
    z = sr.get("zero_rmse", None)
    ft = sr.get("test_rmse", None)
    if z is None or z <= 0 or ft is None:
        return np.array([0.0, 1.0], dtype=np.float64), np.array([1.0, 1.0], dtype=np.float64)
    return np.array([0.0, 1.0], dtype=np.float64), np.array([z, ft], dtype=np.float64)


def compute_suep(
    stage_results: List[Dict[str, Any]],
    # main weights
    w_perf: float = 0.45,
    w_transfer_auc: float = 0.35,
    w_eff: float = 0.20,
    # explicit budget penalty weight
    w_budget: float = 0.25,
    # difficulty weighting hyperparam
    difficulty_alpha: float = 1.0,
    # explicit resource budgets (optional)
    budgets: Optional[Dict[str, float]] = None,
    eps: float = 1e-9,
) -> Dict[str, Any]:
    """
    Compute S_uep score across stages.

    StageScore_i =
        w_perf * PerfNorm_i
      + w_transfer_auc * TransferAUC_i
      + w_eff * EffNorm_i
      - w_budget * BudgetPenalty_i

    where:
      PerfNorm_i:
         = best_rmse / ft_rmse_i     (0..1+, higher better)

      TransferAUC_i:
         curve: frac∈[0,1] vs rmse(frac)
         improvement(frac) = (zero_rmse - rmse(frac)) / zero_rmse
         AUC = ∫ improvement(frac) d(frac)   (0..1)

      EffNorm_i:
         based on normalized resource use (mem/time/flops/trainable)
         EffNorm = 1 - min(mean(usage), 1)

      BudgetPenalty_i:
         only if budgets provided:
         penalty = mean(max(0, usage - 1))

    Final:
      S_uep = Σ difficulty_weight_i * StageScore_i
      difficulty_weight_i ∝ (difficulty_i)^difficulty_alpha
    """
    if len(stage_results) == 0:
        return {
            "S_uep": 0.0,
            "weights": {
                "w_perf": w_perf,
                "w_transfer_auc": w_transfer_auc,
                "w_eff": w_eff,
                "w_budget": w_budget,
                "difficulty_alpha": difficulty_alpha,
            },
            "per_stage": [],
        }

    # ---------- PerfNorm ----------
    ft_rmse = np.array([sr.get("test_rmse", 1.0) for sr in stage_results], dtype=np.float64)
    ft_rmse = np.maximum(ft_rmse, eps)
    best_rmse = float(np.min(ft_rmse))
    perf_norm = best_rmse / ft_rmse

    # ---------- TransferAUC ----------
    transfer_auc = np.zeros(len(stage_results), dtype=np.float64)
    for i, sr in enumerate(stage_results):
        zero = sr.get("zero_rmse", None)
        if zero is None or zero <= 0:
            transfer_auc[i] = 0.0
            continue
        frac, rmse = _get_transfer_curve(sr)
        rmse = np.maximum(rmse, eps)
        improvement = (zero - rmse) / zero
        improvement = np.clip(improvement, 0.0, 1.0)
        transfer_auc[i] = _trapz_auc(frac, improvement)

    # ---------- EffNorm & BudgetPenalty ----------
    peak_mem = np.array(
        [sr.get("dynamic_eff", {}).get("peak_mem_mb", 0.0) for sr in stage_results],
        dtype=np.float64,
    )
    time_sec = np.array(
        [sr.get("dynamic_eff", {}).get("train_time_sec", 0.0) for sr in stage_results],
        dtype=np.float64,
    )
    trainable_params = np.array(
        [sr.get("static_eff", {}).get("trainable_params", 0.0) for sr in stage_results],
        dtype=np.float64,
    )
    flops = np.array(
        [sr.get("flops", {}).get("flops", 0.0) for sr in stage_results],
        dtype=np.float64,
    )

    if budgets is None:
        # normalize by max within this experiment
        mem_n = peak_mem / max(float(np.max(peak_mem)), eps)
        time_n = time_sec / max(float(np.max(time_sec)), eps)
        param_n = trainable_params / max(float(np.max(trainable_params)), eps)
        flops_n = flops / max(float(np.max(flops)), eps)

        usage_mean = np.stack([mem_n, time_n, param_n, flops_n], axis=1).mean(axis=1)
        eff_norm = 1.0 - np.clip(usage_mean, 0.0, 1.0)
        budget_penalty = np.zeros_like(eff_norm)

    else:
        mem_b = budgets.get("peak_mem_mb", None)
        time_b = budgets.get("train_time_sec", None)
        param_b = budgets.get("trainable_params", None)
        flops_b = budgets.get("flops", None)

        comps = []
        if mem_b and mem_b > 0:   comps.append(peak_mem / mem_b)
        if time_b and time_b > 0: comps.append(time_sec / time_b)
        if param_b and param_b > 0: comps.append(trainable_params / param_b)
        if flops_b and flops_b > 0: comps.append(flops / flops_b)

        if len(comps) == 0:
            comps = [np.zeros_like(peak_mem)]

        usage_mean = np.stack(comps, axis=1).mean(axis=1)
        eff_norm = 1.0 - np.clip(usage_mean, 0.0, 1.0)
        budget_penalty = np.maximum(0.0, usage_mean - 1.0)

    # ---------- StageScore ----------
    stage_scores = (
        w_perf * perf_norm
        + w_transfer_auc * transfer_auc
        + w_eff * eff_norm
        - w_budget * budget_penalty
    )
    stage_scores = np.clip(stage_scores, -1.0, 2.0)

    # ---------- Difficulty Weights ----------
    diff = np.array([sr.get("difficulty", 1.0) for sr in stage_results], dtype=np.float64)
    diff = np.maximum(diff, eps)
    diff_w = diff ** difficulty_alpha
    diff_w = diff_w / diff_w.sum()

    S_uep = float(np.sum(diff_w * stage_scores))

    return {
        "S_uep": S_uep,
        "weights": {
            "w_perf": float(w_perf),
            "w_transfer_auc": float(w_transfer_auc),
            "w_eff": float(w_eff),
            "w_budget": float(w_budget),
            "difficulty_alpha": float(difficulty_alpha),
        },
        "per_stage": [
            {
                "stage": int(sr["stage"]),
                "dataset": sr["dataset"],
                "perf_norm": float(perf_norm[i]),
                "transfer_auc": float(transfer_auc[i]),
                "eff_norm": float(eff_norm[i]),
                "budget_penalty": float(budget_penalty[i]),
                "difficulty": float(diff[i]),
                "difficulty_weight": float(diff_w[i]),
                "stage_score": float(stage_scores[i]),
            }
            for i, sr in enumerate(stage_results)
        ],
    }
