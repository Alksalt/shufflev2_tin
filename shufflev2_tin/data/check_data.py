from __future__ import annotations

import argparse
import torch
from shufflev2_tin.utils.config_loader import load_config, validate_config, summarize
from shufflev2_tin.utils.config_loader import derive_run_dir  # optional
from shufflev2_tin.data.tiny_imagenet import build_datasets, build_dataloaders

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="shufflev2_tin/configs/base.yaml")
    p.add_argument("--paths", default="shufflev2_tin/configs/paths.yaml")
    p.add_argument("--data", default="shufflev2_tin/configs/data/tiny_imagenet.yaml")
    p.add_argument("--aug", default="shufflev2_tin/configs/aug/mild.yaml")
    p.add_argument("--system", default="shufflev2_tin/configs/system/local_3060.yaml")
    return p.parse_args()

def main():
    args = parse_args()
    cfg = load_config([args.base, args.paths, args.data, args.aug, args.system])
    validate_config(cfg)
    info = summarize(cfg)
    print("CONFIG:", info)
    
    # subset logic (optional): use only N samples for speed if provided
    subset = cfg.get("subset", {"train": None, "val": None})
    train_ds, val_ds, class_map = build_datasets(
        cfg_data=cfg, cfg_paths=cfg, cfg_aug=cfg
    )
    
    # Applying subset slices if requested
    if subset.get("train"):
        train_ds.samples = train_ds.samples[: int(subset["train"])]
    if subset.get("val"):
        val_ds.samples = val_ds.samples[: int(subset["val"])]

    train_dl, val_dl = build_dataloaders(
        train_ds, val_ds, cfg_data=cfg, batch_size=cfg.get("batch_size", 256)
    )
    
    print(f"Train size: {len(train_ds)} | Val size: {len(val_ds)} | Classes: {len(class_map)}")
    xb, yb = next(iter(train_dl))
    print("Batch shapes:", tuple(xb.shape), tuple(yb.shape))
    print("Dtype/device:", xb.dtype, xb.device)

    # quick CUDA transfer test (no compute)
    if cfg.get("device", "cuda") == "cuda" and torch.cuda.is_available():
        xb_cuda = xb.cuda(non_blocking=True)
        print("Moved a batch to CUDA:", xb_cuda.shape)

if __name__ == "__main__":
    main()