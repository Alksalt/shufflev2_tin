# shufflev2_tin/scripts/quick_sanity.py
from __future__ import annotations
from pathlib import Path
from shufflev2_tin.utils.config_loader import load_config, deep_update
from shufflev2_tin.data.tiny_imagenet import build_datasets, _resolve_split_dirs

CFG_PATHS = [
    "shufflev2_tin/configs/base.yaml",
    "shufflev2_tin/configs/paths.yaml",
    "shufflev2_tin/configs/data/tiny_imagenet.yaml",
    "shufflev2_tin/configs/aug/mild.yaml",
    "shufflev2_tin/configs/system/local_3060.yaml",
]

cfg = {}
for p in CFG_PATHS:
    cfg = deep_update(cfg, load_config(p))

root  = Path((cfg.get("paths") or cfg)["data_dir"]).expanduser().resolve()
split = (cfg.get("data") or cfg).get("split", {"train": "train", "val": "val"})
print("data_dir:", root)
print("split:", split)

train_dir, val_dir = _resolve_split_dirs(root, split)
print("resolved train_dir:", train_dir)
print("resolved val_dir:  ", val_dir)

train_ds, val_ds, _ = build_datasets(cfg_data=(cfg.get("data") or cfg),
                                     cfg_paths=(cfg.get("paths") or cfg),
                                     cfg_aug=(cfg.get("aug") or cfg))

print("train classes:", len(train_ds.classes))
print("val classes:  ", len(val_ds.classes))
print("class sets eq:", set(train_ds.classes) == set(val_ds.classes))
print("mapping eq:   ", train_ds.class_to_idx == val_ds.class_to_idx)
