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
    p = argparse.ArgumentParser(description="Merge YAML configs and print summary")
    p.add_argument("--base", default="shufflev2_tin/configs/base.yaml", help="Base config")
    p.add_argument("--paths", default="shufflev2_tin/configs/paths.yaml", help="Project paths config")
    p.add_argument("--data", default="shufflev2_tin/configs/data/tiny_imagenet.yaml", help="Dataset config")
    p.add_argument("--aug", default="shufflev2_tin/configs/aug/mild.yaml", help="Augmentation config")
    p.add_argument("--system", default="shufflev2_tin/configs/system/local_3060.yaml", help="System/hardware config")
    p.add_argument("--model", default=None, help="Model config (e.g. .../model/shufflenetv2_1p0.yaml)")
    # If you plan to optionally include these, define them:
    p.add_argument("--optim", default=None, help="Optimizer config")
    p.add_argument("--train", default=None, help="Training loop config")
    return p.parse_args()

def main() -> None:
    args = parse_args()

    # Fail fast if critical files are missing
    for must in [args.base, args.system]:
        if must and not Path(must).exists():
            raise FileNotFoundError(f"Required config not found: {must}")

    paths = [args.base, args.system]

    # Accept optional config files if provided
    for name in ["paths", "data", "model", "optim", "train", "aug"]:
        val = getattr(args, name, None)
        if val:
            if not Path(val).exists():
                raise FileNotFoundError(f"Config not found: {val}")
            paths.append(val)

    cfg = load_config(paths)
    validate_config(cfg)

    run_dir = derive_run_dir(cfg)
    smry = summarize(cfg)

    print("=== CONFIG SUMMARY ===")
    for k, v in smry.items():
        print(f"{k}: {v}")
    print("run_dir:", run_dir.as_posix())

    # print("\n=== MERGED YAML ===")
    # print(pretty(cfg))

if __name__ == "__main__":
    main()
