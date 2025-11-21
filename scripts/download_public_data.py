import argparse
from graphtune.data import _ensure_local_file

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["metr-la", "pems-bay"])
    p.add_argument("--data_dir", default="DATA")
    p.add_argument("--source", default="auto", choices=["auto","hf","url","local"])
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--h5_url", default=None)
    p.add_argument("--adj_url", default=None)
    p.add_argument("--loc_url", default=None)
    args = p.parse_args()

    url_over = {"h5": args.h5_url, "adj": args.adj_url, "loc": args.loc_url}

    for kind in ["h5","adj","loc"]:
        _ensure_local_file(args.dataset, kind, args.data_dir, args.source,
                           cache_dir=args.cache_dir,
                           url_override=url_over.get(kind))

    print(f"Done. Files are in {args.data_dir}/")

if __name__ == "__main__":
    main()
