import argparse
import torch

from graphtune import prepare_dataset, build_model, train_one_stage, load_partial_state
from graphtune.config import DEFAULT_MODEL_KWARGS
from graphtune.eval import evaluate_model

def parse_list(arg, cast_fn=str):
    return [cast_fn(x) for x in arg.split(",") if x.strip()]

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

    # data fetching
    p.add_argument("--data_source", type=str, default="auto",
                   choices=["auto","hf","url","local"])
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--h5_urls", type=str, default=None)
    p.add_argument("--adj_urls", type=str, default=None)
    p.add_argument("--loc_urls", type=str, default=None)

    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = args.model.lower()

    dataset_list = parse_list(args.datasets, str)
    epochs_list  = parse_list(args.epochs, int)
    lrs_list     = parse_list(args.lrs, float)
    if len(epochs_list) == 1:
        epochs_list = epochs_list * len(dataset_list)
    if len(lrs_list) == 1:
        lrs_list = lrs_list * len(dataset_list)

    h5_urls  = parse_list(args.h5_urls, str)  if args.h5_urls  else [None]*len(dataset_list)
    adj_urls = parse_list(args.adj_urls, str) if args.adj_urls else [None]*len(dataset_list)
    loc_urls = parse_list(args.loc_urls, str) if args.loc_urls else [None]*len(dataset_list)

    prev_state = None
    model = None

    for stage_idx, dname in enumerate(dataset_list):
        print("\n" + "="*80)
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

        if prev_state is not None:
            exclude = ()
            if model_name in ["baseline", "hypernet"]:
                exclude = ("A_norm", "coords")
            elif model_name in ["dcrnn"]:
                exclude = ("W_fwd", "W_bwd")
            elif model_name in ["dgcrn"]:
                exclude = ("W_fwd_fix", "W_bwd_fix")

            copied = load_partial_state(model, prev_state, exclude_prefixes=exclude)
            print(f"  loaded partial state: copied {copied} tensors")

        model = train_one_stage(
            model,
            bundle["train_loader"],
            bundle["val_loader"],
            bundle["scaler"],
            num_epochs=epochs_list[stage_idx],
            lr=lrs_list[stage_idx],
            device=device,
            name=f"{model_name}-{dname}",
            is_dcrnn=(model_name=="dcrnn")
        )

        test_mae, test_rmse = evaluate_model(
            model, bundle["test_loader"], bundle["scaler"], device=device
        )
        print(f"  [Test] MAE={test_mae:.4f} RMSE={test_rmse:.4f}")

        prev_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print("\nAll stages done!")

if __name__ == "__main__":
    main()
