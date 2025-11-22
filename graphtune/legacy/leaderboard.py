from typing import Any, Dict, List, Optional


def _fmt(x, nd=4):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def print_leaderboard(stage_results: List[Dict[str, Any]], suep_summary: Optional[Dict[str, Any]] = None):
    headers = [
        "stage", "dataset",
        "zero_RMSE", "ft_RMSE", "gain%",
        "params(M)", "trainable(M)",
        "peakMem(MB)", "time(s)",
        "FLOPs(G)",
        "AUC_tr", "Eff", "Penalty",
        "difficulty", "w_diff",
        "stage_score"
    ]

    lines = []
    for sr in stage_results:
        zero_rmse = sr.get("zero_rmse", None)
        ft_rmse = sr.get("test_rmse", None)
        gain = None
        if zero_rmse is not None and zero_rmse > 0 and ft_rmse is not None:
            gain = 100.0 * (zero_rmse - ft_rmse) / zero_rmse

        stat = sr.get("static_eff", {})
        dyn = sr.get("dynamic_eff", {})
        flops = sr.get("flops", {})

        row = [
            sr.get("stage"),
            sr.get("dataset"),
            _fmt(zero_rmse),
            _fmt(ft_rmse),
            _fmt(gain, nd=2),
            _fmt(stat.get("total_params", 0) / 1e6, nd=2),
            _fmt(stat.get("trainable_params", 0) / 1e6, nd=2),
            _fmt(dyn.get("peak_mem_mb", 0.0), nd=1),
            _fmt(dyn.get("train_time_sec", 0.0), nd=1),
            _fmt((flops.get("flops", 0.0) / 1e9) if flops else None, nd=2),

            _fmt(sr.get("transfer_auc"), nd=3),
            _fmt(sr.get("eff_norm"), nd=3),
            _fmt(sr.get("budget_penalty"), nd=3),

            _fmt(sr.get("difficulty"), nd=2),
            _fmt(sr.get("difficulty_weight"), nd=3),

            _fmt(sr.get("stage_score"), nd=4),
        ]
        lines.append(row)

    cols = list(zip(*([headers] + lines)))
    widths = [max(len(str(c)) for c in col) for col in cols]

    def format_row(row):
        return " | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row))

    sep = "-+-".join("-" * w for w in widths)

    print("\n[GraphTune-lite Leaderboard]")
    print(format_row(headers))
    print(sep)
    for row in lines:
        print(format_row(row))

    if suep_summary is not None:
        w = suep_summary["weights"]
        print(
            f"\nS_uep = {suep_summary.get('S_uep', 0.0):.4f} "
            f"(w_perf={w['w_perf']}, w_transfer_auc={w['w_transfer_auc']}, "
            f"w_eff={w['w_eff']}, w_budget={w['w_budget']}, "
            f"diff_alpha={w['difficulty_alpha']})"
        )
