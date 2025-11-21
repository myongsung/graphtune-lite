import argparse
import json
import numpy as np
import torch

from graphtune import prepare_dataset, build_model, load_partial_state
from graphtune.config import DEFAULT_MODEL_KWARGS
from graphtune.eval import evaluate_model
from graphtune.train import train_one_stage
from graphtune.efficiency import profile_model_static, estimate_flops
from graphtune.suep import compute_suep
from graphtune.leaderboard import print_leaderboard


def parse_list(arg, cast_fn=str):
    return [cast_fn(x) for x in arg.split(",") if x.strip()]


def compute_graph_difficulty(A_np: np.ndarray, num_nodes: int) -> float:
    """
    Simple, robust difficulty proxy:
      difficulty = N * log(1 + avg_degree)

    - N↑, degree↑일수록 그래프/도시 구조가 복잡 → 전이 난도↑
    """
    A_bin = (A_np > 0).astype(np.float32)
    avg_deg = float(A_bin.sum(axis=1).mean())
    diff = float(num_nodes * np.log1p(avg_deg))
    return diff


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True,
                   help="bigst | baseline | hypernet | dcrnn | dgcrn")
    p.add_argument("--datasets", type=str, required=True,
                   help="comma-separated order, e.g. metr-la,pems-bay")
    p.add_argument("--epochs", type=str, default="50",
                   help="comma-separated per stage, e.g. 50,30")
    p.add_argument("--lrs", type=str, default="0.001",
                   help="comma-separated per stage, e.g. 1e-3,5e-4")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--data_dir", type=str, default="DATA")

    p.add_argument("--data_source", type=str, default="auto",
                   choices=["auto", "hf", "url", "local"])
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--h5_urls", type=str, default=None)
    p.add_argument("--adj_urls", type=str, default=None)
    p.add_argument("--loc_urls", type=str, default=None)

    # ----- S_uep weights -----
    p.add_argument("--w_perf", type=float, default=0.45)
    p.add_argument("--w_transfer_auc", type=float, default=0.35)
    p.add_argument("--w_eff", type=float, default=0.20)
    p.add_argument("--w_budget", type=float, default=0.25)
    p.add_argument("--difficulty_alpha", type=float, default=1.0)

    # ----- budgets (optional) -----
    p.add_argument("--budget_mem_mb", type=float, default=None,
                   help="GPU peak memory budget (MB). None => normalize by max stage.")
    p.add_argument("--budget_time_sec", type=float, default=None,
                   help="Train wall-time budget (sec). None => normalize by max stage.")
    p.add_argument("--budget_flops_g", type=float, default=None,
                   help="Forward FLOPs budget per step (GFLOPs). None => normalize by max stage.")
    p.add_argument("--budget_trainable_m", type=float, default=None,
                   help="Trainable params budget (Millions). None => normalize by max stage.")

    # output
    p.add_argument("--out_json", type=str, default="results.json")
    p.add_argument("--out_csv", type=str, default="results.csv")

    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = args.model.lower()

    dataset_list = parse_list(args.datasets, str)
    epochs_list = parse_list(args.epochs, int)
    lrs_list = parse_list(args.lrs, float)
    if len(epochs_list) == 1:
        epochs_list *= len(dataset_list)
    if len(lrs_list) == 1:
        lrs_list *= len(dataset_list)

    h5_urls = parse_list(args.h5_urls, str) if args.h5_urls else [None] * len(dataset_list)
    adj_urls = parse_list(args.adj_urls, str) if args.adj_urls else [None] * len(dataset_list)
    loc_urls = parse_list(args.loc_urls, str) if args.loc_urls else [None] * len(dataset_list)

    budgets = None
    if any(v is not None for v in [args.budget_mem_mb, args.budget_time_sec, args.budget_flops_g, args.budget_trainable_m]):
        budgets = {
            "peak_mem_mb": args.budget_mem_mb,
            "train_time_sec": args.budget_time_sec,
            "flops": None if args.budget_flops_g is None else args.budget_flops_g * 1e9,
            "trainable_params": None if args.budget_trainable_m is None else args.budget_trainable_m * 1e6,
        }

    prev_state = None
    model = None
    all_stage_results = []

    for stage_idx, dname in enumerate(dataset_list):
        print("\n" + "=" * 80)
        print(f"[Stage {stage_idx}] dataset={dname}")

        is_bigst = (model_name == "bigst")
        bundle = prepare_dataset(
            dname,
            data_dir=args.data_dir,
            stride=args.stride,
            batch_size=args.batch_size,
            for_bigst=is_bigst,
            source=args.data_source,
            cache_dir=args.cache_dir,
            url_overrides={
                "h5": h5_urls[stage_idx],
                "adj": adj_urls[stage_idx],
                "loc": loc_urls[stage_idx],
            },
        )

        model_kwargs = DEFAULT_MODEL_KWARGS.get(model_name, {})
        model = build_model(model_name, bundle, **model_kwargs)

        reused = 0
        if prev_state is not None:
            exclude = ()
            if model_name in ["baseline", "hypernet"]:
                exclude = ("A_norm", "coords")
            elif model_name in ["dcrnn"]:
                exclude = ("W_fwd", "W_bwd")
            elif model_name in ["dgcrn"]:
                exclude = ("W_fwd_fix", "W_bwd_fix")

            reused = load_partial_state(model, prev_state, exclude_prefixes=exclude)
            print(f"  loaded partial state: copied {reused} tensors")

        # ----- zero-shot before finetune -----
        zero_mae = zero_rmse = None
        if stage_idx > 0:
            zero_mae, zero_rmse = evaluate_model(
                model, bundle["test_loader"], bundle["scaler"], device=device
            )
            print(f"  [Zero-shot] MAE={zero_mae:.4f} RMSE={zero_rmse:.4f}")

        # ----- static efficiency -----
        static_eff = profile_model_static(model)
        print(f"  [StaticEff] params={static_eff['total_params']:,} "
              f"trainable={static_eff['trainable_params']:,} "
              f"size={static_eff['param_size_mb']:.2f}MB "
              f"trainable_ratio={static_eff['trainable_ratio']:.3f}")

        # ----- FLOPs estimate (optional) -----
        flops_info = None
        try:
            first_batch = next(iter(bundle["train_loader"]))
            sample_x = first_batch[0]
            flops_info = estimate_flops(model, sample_x, device=device)
            if flops_info is not None:
                print(f"  [FLOPs] flops={flops_info['flops']/1e9:.2f}G "
                      f"macs={flops_info['macs']/1e9:.2f}G")
            else:
                print("  [FLOPs] thop not installed. (pip install thop to enable)")
        except Exception as e:
            print(f"  [FLOPs] failed to estimate: {e}")

        # ----- finetune/train -----
        model, dyn_eff = train_one_stage(
            model,
            bundle["train_loader"],
            bundle["val_loader"],
            bundle["scaler"],
            num_epochs=epochs_list[stage_idx],
            lr=lrs_list[stage_idx],
            device=device,
            name=f"{model_name}-{dname}",
            is_dcrnn=(model_name == "dcrnn"),
            profile=True
        )
        print(f"  [DynEff] time={dyn_eff['train_time_sec']:.1f}s "
              f"peak_mem={dyn_eff['peak_mem_mb']:.1f}MB")

        # ----- accuracy after finetune -----
        test_mae, test_rmse = evaluate_model(
            model, bundle["test_loader"], bundle["scaler"], device=device
        )
        print(f"  [Test] MAE={test_mae:.4f} RMSE={test_rmse:.4f}")

        # ----- transfer curve points -----
        # 지금은 2점(0-shot, finetune) curve를 기본으로 기록.
        curve_rmse = []
        if zero_rmse is not None:
            curve_rmse.append((0.0, float(zero_rmse)))
        curve_rmse.append((1.0, float(test_rmse)))

        # ----- graph difficulty -----
        difficulty = compute_graph_difficulty(bundle["A"], bundle["num_nodes"])
        print(f"  [Difficulty] {difficulty:.2f}")

        stage_result = {
            "stage": stage_idx,
            "dataset": bundle["dataset_name"],
            "model": model_name,
            "reused_tensors": int(reused),
            "zero_mae": None if zero_mae is None else float(zero_mae),
            "zero_rmse": None if zero_rmse is None else float(zero_rmse),
            "static_eff": static_eff,
            "dynamic_eff": dyn_eff,
            "flops": flops_info or {},
            "test_mae": float(test_mae),
            "test_rmse": float(test_rmse),
            "curve_rmse": curve_rmse,
            "difficulty": float(difficulty),
        }
        all_stage_results.append(stage_result)

        # ----- running S_uep + real-time leaderboard -----
        suep_summary = compute_suep(
            all_stage_results,
            w_perf=args.w_perf,
            w_transfer_auc=args.w_transfer_auc,
            w_eff=args.w_eff,
            w_budget=args.w_budget,
            difficulty_alpha=args.difficulty_alpha,
            budgets=budgets,
        )

        # push per-stage breakdown back for printing
        per_stage_map = {ps["stage"]: ps for ps in suep_summary["per_stage"]}
        for sr in all_stage_results:
            ps = per_stage_map[sr["stage"]]
            sr["perf_norm"] = ps["perf_norm"]
            sr["transfer_auc"] = ps["transfer_auc"]
            sr["eff_norm"] = ps["eff_norm"]
            sr["budget_penalty"] = ps["budget_penalty"]
            sr["difficulty_weight"] = ps["difficulty_weight"]
            sr["stage_score"] = ps["stage_score"]

        print_leaderboard(all_stage_results, suep_summary)

        prev_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print("\nAll stages done!")

    # save json/csv
    try:
        with open(args.out_json, "w") as f:
            json.dump(all_stage_results, f, indent=2)
        print(f"[Saved] {args.out_json}")
    except Exception as e:
        print("[Warn] Failed to save json:", e)

    try:
        import csv
        with open(args.out_csv, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "stage","dataset","model",
                    "zero_rmse","test_rmse",
                    "train_time_sec","peak_mem_mb",
                    "total_params","trainable_params","param_size_mb",
                    "flops","macs",
                    "transfer_auc","eff_norm","budget_penalty",
                    "difficulty","difficulty_weight",
                    "stage_score"
                ]
            )
            writer.writeheader()
            for sr in all_stage_results:
                st=sr["static_eff"]; dy=sr["dynamic_eff"]; fl=sr.get("flops",{})
                writer.writerow({
                    "stage": sr["stage"],
                    "dataset": sr["dataset"],
                    "model": sr["model"],
                    "zero_rmse": sr.get("zero_rmse"),
                    "test_rmse": sr["test_rmse"],
                    "train_time_sec": dy.get("train_time_sec"),
                    "peak_mem_mb": dy.get("peak_mem_mb"),
                    "total_params": st.get("total_params"),
                    "trainable_params": st.get("trainable_params"),
                    "param_size_mb": st.get("param_size_mb"),
                    "flops": fl.get("flops"),
                    "macs": fl.get("macs"),
                    "transfer_auc": sr.get("transfer_auc"),
                    "eff_norm": sr.get("eff_norm"),
                    "budget_penalty": sr.get("budget_penalty"),
                    "difficulty": sr.get("difficulty"),
                    "difficulty_weight": sr.get("difficulty_weight"),
                    "stage_score": sr.get("stage_score"),
                })
        print(f"[Saved] {args.out_csv}")
    except Exception as e:
        print("[Warn] Failed to save csv:", e)

    print(f"\nFinal S_uep = {suep_summary['S_uep']:.4f}")


if __name__ == "__main__":
    main()
