from __future__ import annotations

import argparse
from pathlib import Path
from shufflev2_tin.utils.config_loader import (
    load_config,
    derive_run_dir,
    validate_config,
    summarize,
    pretty,
)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Print merged config summary.")
    p.add_argument("--base", default="configs/base.yaml")
    p.add_argument("--system", default="configs/system/local_3060.yaml")
    # You’ll add these later:
    p.add_argument("--paths", default=None)
    p.add_argument("--data", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--optim", default=None)
    p.add_argument("--train", default=None)
    p.add_argument("--aug", default=None)
    return p.parse_args()

def main() -> None:
    args = parse_args()
    paths = [args.base, args.system]

    # Accept optional config files if provided
    for name in ["paths", "data", "model", "optim", "train", "aug"]:
        val = getattr(args, name)
        if val:
            paths.append(val)

    cfg = load_config(paths)
    validate_config(cfg)

    run_dir = derive_run_dir(cfg)
    summary = summarize(cfg)

    print("=== CONFIG SUMMARY ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print("run_dir:", run_dir.as_posix())

    # Uncomment if you want to see the final merged YAML:
    # print("\n=== MERGED YAML ===")
    # print(pretty(cfg))

if __name__ == "__main__":
    main()