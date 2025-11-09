# shufflev2_tin/scripts/train.py
from __future__ import annotations
import argparse
from pathlib import Path

from shufflev2_tin.utils.config_loader import load_config, deep_update, pretty
from shufflev2_tin.engine.train import Trainer

def main():
    p = argparse.ArgumentParser(description="Train ShuffleNetV2 on Tiny-ImageNet")
    p.add_argument("--base", default="shufflev2_tin/configs/base.yaml")
    p.add_argument("--paths", default="shufflev2_tin/configs/paths.yaml")
    p.add_argument("--data", default="shufflev2_tin/configs/data/tiny_imagenet.yaml")
    p.add_argument("--aug", default="shufflev2_tin/configs/aug/mild.yaml")
    p.add_argument("--system", default="shufflev2_tin/configs/system/local_3060.yaml")
    p.add_argument("--model", default="shufflev2_tin/configs/model/shufflenetv2_1p0.yaml")
    p.add_argument("--optim", default="shufflev2_tin/configs/optim/adamw.yaml")
    p.add_argument("--train", default="shufflev2_tin/configs/train/schedule_onecycle.yaml")
    args = p.parse_args()

    cfg = {}
    for pth in [args.base, args.paths, args.data, args.aug, args.system, args.model, args.optim, args.train]:
        cfg = deep_update(cfg, load_config(Path(pth)))

    print("=== MERGED CONFIG ===")
    print(pretty(cfg))

    trainer = Trainer(cfg)
    trainer.fit()

if __name__ == "__main__":
    main()