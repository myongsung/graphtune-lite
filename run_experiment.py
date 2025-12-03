import argparse
import json
import os
import numpy as np
import torch

# 최상위 API (v2 facade)
from graphtune import prepare_dataset, build_model, load_partial_state, train_one_stage

# 아직 legacy에 남아있는 것들은 legacy에서 직접 import
from graphtune.config import DEFAULT_MODEL_KWARGS
from graphtune.core.scoring import compute_suep
from graphtune.core.leaderboard import print_leaderboard

# v2 wrapper/서브패키지들
from graphtune.budget import make_budgeted_train_loader, LossGradientBudgetScheduler
from graphtune.efficiency.static import profile_model_static
from graphtune.efficiency.flops import estimate_flops
from graphtune.eval.evaluator import evaluate_model


def parse_list(arg, cast_fn=str):
    return [cast_fn(x) for x in arg.split(",") if x.strip()]


def compute_graph_difficulty(A_np: np.ndarray, num_nodes: int) -> float:
    """
    간단한 그래프 난이도 스코어:
      - adjacency를 binary로 보고, 평균 degree를 구한 뒤
      - num_nodes * log(1 + avg_deg) 형태로 사용
    """
    A_bin = (A_np > 0).astype(np.float32)
    avg_deg = float(A_bin.sum(axis=1).mean())
    return float(num_nodes * np.log1p(avg_deg))


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        type=str,
        required=True,
        help="bigst | baseline | hypernet | dcrnn | dgcrn | gemma3 (registry에 있는 이름)",
    )
    p.add_argument(
        "--datasets",
        type=str,
        required=True,
        help="comma-separated order, e.g. metr-la,pems-bay",
    )
    p.add_argument(
        "--epochs",
        type=str,
        default="50",
        help="comma-separated max epochs per stage, e.g. 50,30",
    )
    p.add_argument(
        "--lrs",
        type=str,
        default="0.001",
        help="comma-separated per stage, e.g. 1e-3,5e-4",
    )
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--data_dir", type=str, default="DATA")

    p.add_argument(
        "--data_source",
        type=str,
        default="auto",
        choices=["auto", "hf", "url", "local"],
    )
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--h5_urls", type=str, default=None)
    p.add_argument("--adj_urls", type=str, default=None)
    p.add_argument("--loc_urls", type=str, default=None)

    # ----- Few-shot / budget curve -----
    p.add_argument(
        "--fractions",
        type=str,
        default="0.1,0.3,1.0",
        help="budget fractions for transfer curve, e.g. 0.05,0.1,0.2,0.5,1.0",
    )
    p.add_argument(
        "--fewshot_mode",
        type=str,
        default="subset",
        choices=["subset", "steps", "both"],
        help="how to enforce low-budget training",
    )

    # ----- Loss-Gradient scheduler -----
    p.add_argument("--min_gain_rate", type=float, default=0.02)
    p.add_argument("--min_rel_improve", type=float, default=0.005)
    p.add_argument("--patience", type=int, default=1)

    # ----- S_uep weights -----
    p.add_argument("--w_perf", type=float, default=0.45)
    p.add_argument("--w_transfer_auc", type=float, default=0.35)
    p.add_argument("--w_eff", type=float, default=0.20)
    p.add_argument("--w_budget", type=float, default=0.25)
    p.add_argument("--difficulty_alpha", type=float, default=1.0)

    # ----- budgets (optional) -----
    p.add_argument("--budget_mem_mb", type=float, default=None)
    p.add_argument("--budget_time_sec", type=float, default=None)
    p.add_argument("--budget_flops_g", type=float, default=None)
    p.add_argument("--budget_trainable_m", type=float, default=None)

    # output
    p.add_argument("--out_json", type=str, default="results.json")
    p.add_argument("--out_csv", type=str, default="results.csv")

    # ----- Info-Budget Coreset (IBCS) -----
    p.add_argument(
        "--use_ibcs",
        action="store_true",
        help="if set, use InfoBudgetCoresetLayer for few-shot subset selection",
    )
    p.add_argument(
        "--ibcs_budget_mode",
        type=str,
        default="node",
        choices=["sample", "node"],
        help="기본 버전: sample 단위 코어셋 / node 단위 코어셋 중 선택",
    )
    p.add_argument(
        "--ibcs_budget_frac",
        type=float,
        default=1.0,
        help="difficulty 기반 global budget에 곱해줄 scale factor (0~1)",
    )

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

    # budget scheduler 템플릿 (각 stage에서 복제해서 사용)
    fractions = parse_list(args.fractions, float)
    scheduler_tpl = LossGradientBudgetScheduler(
        fractions=fractions,
        min_gain_rate=args.min_gain_rate,
        min_rel_improve=args.min_rel_improve,
        patience=args.patience,
    )

    # 전체 예산 (있으면 S_uep에서 penalty로 사용)
    budgets = None
    if any(
        v is not None
        for v in [
            args.budget_mem_mb,
            args.budget_time_sec,
            args.budget_flops_g,
            args.budget_trainable_m,
        ]
    ):
        budgets = {
            "peak_mem_mb": args.budget_mem_mb,
            "train_time_sec": args.budget_time_sec,
            "flops": None if args.budget_flops_g is None else args.budget_flops_g * 1e9,
            "trainable_params": None
            if args.budget_trainable_m is None
            else args.budget_trainable_m * 1e6,
        }

    prev_state = None
    model = None
    all_stage_results = []

    for stage_idx, dname in enumerate(dataset_list):
        print("\n" + "=" * 80)
        print(f"[Stage {stage_idx}] dataset={dname}")

        is_bigst = model_name == "bigst"
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

        # 모델 생성
        model_kwargs = DEFAULT_MODEL_KWARGS.get(model_name, {})
        model = build_model(model_name, bundle, **model_kwargs)

        # 이전 스테이지 가중치 일부 재사용 (multi-city transfer)
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

        # ----- static efficiency -----
        static_eff = profile_model_static(model)
        print(
            f"  [StaticEff] params={static_eff['total_params']:,} "
            f"trainable={static_eff['trainable_params']:,} "
            f"size={static_eff['param_size_mb']:.2f}MB "
            f"trainable_ratio={static_eff['trainable_ratio']:.3f}"
        )

        # ----- FLOPs (thop 기반) -----
        flops_info = None
        try:
            first_batch = next(iter(bundle["train_loader"]))
            sample_x = first_batch[0]
            flops_info = estimate_flops(model, sample_x, device=device)
            if flops_info is not None:
                print(
                    f"  [FLOPs] flops={flops_info['flops']/1e9:.2f}G "
                    f"macs={flops_info['macs']/1e9:.2f}G"
                )
            else:
                print("  [FLOPs] thop not installed. (pip install thop)")
        except Exception as e:
            print(f"  [FLOPs] failed: {e}")

        # ----- graph difficulty -----
        difficulty = compute_graph_difficulty(bundle["A"], bundle["num_nodes"])
        print(f"  [Difficulty] {difficulty:.2f}")

        # ============================
        # Stage 0: 전체 데이터로 pretrain
        # ============================
        if stage_idx == 0:
            model, dyn_eff = train_one_stage(
                model,
                bundle["train_loader"],
                bundle["val_loader"],
                bundle["scaler"],
                num_epochs=epochs_list[stage_idx],
                lr=lrs_list[stage_idx],
                device=device,
                name=f"{model_name}-{bundle['dataset_name']}",
                is_dcrnn=(model_name == "dcrnn"),
                profile=True,
            )
            test_mae, test_rmse = evaluate_model(
                model, bundle["test_loader"], bundle["scaler"], device=device
            )
            curve_rmse = [(1.0, float(test_rmse))]
            zero_mae = zero_rmse = None

        # ============================
        # Stage 1+: zero-shot → budget curve → few-shot fine-tune
        # ============================
        else:
            # (1) zero-shot: 이전 스테이지 가중치 그대로 평가
            zero_val_mae, zero_val_rmse = evaluate_model(
                model, bundle["val_loader"], bundle["scaler"], device=device
            )
            zero_mae, zero_rmse = evaluate_model(
                model, bundle["test_loader"], bundle["scaler"], device=device
            )
            print(
                f"  [Zero-shot] val_RMSE={zero_val_rmse:.4f} "
                f"test_RMSE={zero_rmse:.4f}"
            )

            curve_rmse = [(0.0, float(zero_rmse))]

            # (2) 예산 스케줄러 설정 (fractions curve)
            scheduler = LossGradientBudgetScheduler(
                fractions=scheduler_tpl.planned_fractions(),
                min_gain_rate=args.min_gain_rate,
                min_rel_improve=args.min_rel_improve,
                patience=args.patience,
            )

            spent_frac = 0.0
            prev_val_rmse = float(zero_val_rmse)

            # 스테이지 전체 dynamic efficiency 합산
            total_time = 0.0
            peak_mem = 0.0
            total_epochs = 0

            for target_frac in scheduler.planned_fractions():
                if target_frac <= spent_frac:
                    continue

                delta_frac = target_frac - spent_frac
                delta_epochs = max(
                    1, int(round(epochs_list[stage_idx] * delta_frac))
                )
                total_epochs += delta_epochs

                # budget fraction에 맞는 subset loader 생성
                ft_loader, max_batches = make_budgeted_train_loader(
                    bundle["train_loader"],
                    target_frac,
                    mode=args.fewshot_mode,
                    seed=42 + stage_idx,
                )

                # (3) 해당 budget 구간만큼 fine-tune
                model, dyn_eff_delta = train_one_stage(
                    model,
                    ft_loader,
                    bundle["val_loader"],
                    bundle["scaler"],
                    num_epochs=delta_epochs,
                    lr=lrs_list[stage_idx],
                    device=device,
                    name=f"{model_name}-{bundle['dataset_name']}-frac{target_frac:.2f}",
                    is_dcrnn=(model_name == "dcrnn"),
                    profile=True,
                    max_batches_per_epoch=max_batches,
                )

                total_time += dyn_eff_delta.get("train_time_sec", 0.0)
                peak_mem = max(peak_mem, dyn_eff_delta.get("peak_mem_mb", 0.0))

                # (4) 이 budget point에서의 val/test 성능
                val_mae, val_rmse = evaluate_model(
                    model, bundle["val_loader"], bundle["scaler"], device=device
                )
                test_mae, test_rmse = evaluate_model(
                    model, bundle["test_loader"], bundle["scaler"], device=device
                )

                spent_frac = target_frac
                curve_rmse.append((spent_frac, float(test_rmse)))

                print(
                    f"  [Budget {spent_frac:.2f}] "
                    f"val_RMSE={val_rmse:.4f} test_RMSE={test_rmse:.4f}"
                )

                # (5) 성능 개선이 작으면 조기 종료
                if not scheduler.should_continue(
                    prev_val_rmse, float(val_rmse), delta_frac
                ):
                    print("  [Scheduler] marginal gain too small → early stop.")
                    break

                prev_val_rmse = float(val_rmse)

            dyn_eff = {
                "train_time_sec": float(total_time),
                "peak_mem_mb": float(peak_mem),
                "epochs": int(total_epochs),
                "lr": float(lrs_list[stage_idx]),
            }

        # ----- stage 결과 정리 -----
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

        # (옵션) stage별 checkpoint 저장
        try:
            ckpt_dir = "checkpoints"
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(
                ckpt_dir,
                f"{model_name}_{bundle['dataset_name']}_stage{stage_idx}.pt",
            )
            torch.save(model.state_dict(), ckpt_path)
            print(f"[Saved checkpoint] {ckpt_path}")
        except Exception as e:
            print(f"[Warn] Failed to save checkpoint: {e}")

        # ----- S_uep 계산 + 리더보드 출력 -----
        suep_summary = compute_suep(
            all_stage_results,
            w_perf=args.w_perf,
            w_transfer_auc=args.w_transfer_auc,
            w_eff=args.w_eff,
            w_budget=args.w_budget,
            difficulty_alpha=args.difficulty_alpha,
            budgets=budgets,
        )

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

        # 다음 스테이지를 위한 state 저장 (CPU로 복사)
        prev_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print("\nAll stages done!")

    # ----- 결과 저장 (JSON / CSV) -----
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
                    "stage",
                    "dataset",
                    "model",
                    "zero_rmse",
                    "test_rmse",
                    "train_time_sec",
                    "peak_mem_mb",
                    "total_params",
                    "trainable_params",
                    "param_size_mb",
                    "flops",
                    "macs",
                    "transfer_auc",
                    "eff_norm",
                    "budget_penalty",
                    "difficulty",
                    "difficulty_weight",
                    "stage_score",
                ],
            )
            writer.writeheader()
            for sr in all_stage_results:
                st = sr["static_eff"]
                dy = sr["dynamic_eff"]
                fl = sr.get("flops", {})
                writer.writerow(
                    {
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
                    }
                )
        print(f"[Saved] {args.out_csv}")
    except Exception as e:
        print("[Warn] Failed to save csv:", e)

    print(f"\nFinal S_uep = {suep_summary['S_uep']:.4f}")


if __name__ == "__main__":
    main()
