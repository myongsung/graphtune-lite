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

    Expected optional field:
      sr["curve_rmse"] = [(frac, rmse), ...]
    Else fallback to:
      zero_rmse -> test_rmse with fractions [0,1]
    """
    if "curve_rmse" in sr and sr["curve_rmse"]:
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
    # budgets (None이면 stage 내부 max로 normalize)
    budgets: Optional[Dict[str, float]] = None,
    eps: float = 1e-9
) -> Dict[str, Any]:
    """
    Detailed Unified Efficiency–Elasticity score S_uep (GraphTune).

    Components per stage i:
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

      StageScore_i:
         = w_perf*PerfNorm_i
           + w_transfer_auc*TransferAUC_i
           + w_eff*EffNorm_i
           - w_budget*BudgetPenalty_i

    Difficulty weights:
      weight_i ∝ difficulty_i^alpha
      S_uep = Σ weight_i * StageScore_i
    """
    if len(stage_results) == 0:
        return {"S_uep": 0.0, "per_stage": [], "weights": {}}

    ft_rmses = np.array([sr["test_rmse"] for sr in stage_results], dtype=np.float64)

    # ---------- PerfNorm ----------
    rmse_ref = ft_rmses.min() if len(ft_rmses) else 1.0
    perf_norm = rmse_ref / (ft_rmses + eps)

    # ---------- Transfer AUC ----------
    transfer_auc = []
    for sr in stage_results:
        frac, rmse_curve = _get_transfer_curve(sr)
        z = sr.get("zero_rmse", None)
        if z is None or z <= 0:
            # Stage0 or missing zero-shot => no transfer credit
            transfer_auc.append(0.0)
            continue
        improv = (z - rmse_curve) / (z + eps)
        improv = np.clip(improv, 0.0, 1.0)
        auc = _trapz_auc(frac, improv)  # already normalized since frac in [0,1]
        transfer_auc.append(float(np.clip(auc, 0.0, 1.0)))
    transfer_auc = np.array(transfer_auc, dtype=np.float64)

    # ---------- Efficiency + BudgetPenalty ----------
    peak_mem = np.array([sr["dynamic_eff"].get("peak_mem_mb", 0.0) for sr in stage_results], dtype=np.float64)
    time_sec = np.array([sr["dynamic_eff"].get("train_time_sec", 0.0) for sr in stage_results], dtype=np.float64)
    trainable_params = np.array([sr["static_eff"].get("trainable_params", 0.0) for sr in stage_results], dtype=np.float64)
    flops = np.array([sr.get("flops", {}).get("flops", 0.0) for sr in stage_results], dtype=np.float64)

    if budgets is None:
        mem_ref = peak_mem.max() if peak_mem.max() > 0 else 1.0
        time_ref = time_sec.max() if time_sec.max() > 0 else 1.0
        param_ref = trainable_params.max() if trainable_params.max() > 0 else 1.0
        flops_ref = flops.max() if flops.max() > 0 else 1.0

        usage = np.stack([
            peak_mem / mem_ref,
            time_sec / time_ref,
            trainable_params / param_ref,
            flops / flops_ref,
        ], axis=1)
        usage_mean = usage.mean(axis=1)
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
            "w_perf": w_perf,
            "w_transfer_auc": w_transfer_auc,
            "w_eff": w_eff,
            "w_budget": w_budget,
            "difficulty_alpha": difficulty_alpha,
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
