# shufflev2_tin/data/tiny_imagenet.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ---------------------------------------------------------------------------
# Transform registry
# ---------------------------------------------------------------------------

_NAME_TO_TRANSFORM = {
    "RandomResizedCrop": transforms.RandomResizedCrop,
    "RandomHorizontalFlip": transforms.RandomHorizontalFlip,
    "ColorJitter": transforms.ColorJitter,
    "Resize": transforms.Resize,
    "CenterCrop": transforms.CenterCrop,
    "ToTensor": transforms.ToTensor,
    "Normalize": transforms.Normalize,
}

def _ensure_normalize_after_totensor(ops: List[transforms.Transform]) -> None:
    """
    Enforce order: ToTensor must appear before Normalize if Normalize is used.
    Raises ValueError when violated.
    """
    to_tensor_pos = None
    norm_pos = None
    for i, op in enumerate(ops):
        name = op.__class__.__name__
        if name == "ToTensor":
            to_tensor_pos = i
        elif name == "Normalize":
            norm_pos = i
    if norm_pos is not None:
        if to_tensor_pos is None or not (to_tensor_pos < norm_pos):
            raise ValueError(
                "Transform order invalid: Normalize requires ToTensor to come before it.\n"
                "Tip: put 'ToTensor' earlier in the list than 'Normalize'."
            )

def build_transform_list(cfg_list: List[Dict[str, Any]]) -> transforms.Compose:
    """
    Parse a list of {Name: {kwargs...}} into torchvision transforms.Compose.
    Example element: {"RandomResizedCrop": {"size": 64, "scale": [0.7, 1.0]}}

    Notes
    - Converts list values in kwargs to tuples (torchvision prefers tuples).
    - Enforces that Normalize (if present) comes after ToTensor.
    """
    if not isinstance(cfg_list, list):
        raise ValueError(f"Transforms config must be a list, got: {type(cfg_list)}")

    ops: List[transforms.Transform] = []
    for item in cfg_list:
        if not isinstance(item, dict) or len(item) != 1:
            raise ValueError(f"Bad transform entry: {item}")
        (name, kwargs), = item.items()
        if name not in _NAME_TO_TRANSFORM:
            raise KeyError(f"Unknown transform: {name}")
        cls = _NAME_TO_TRANSFORM[name]

        # small compatibility: torchvision expects tuples in some places
        if isinstance(kwargs, dict):
            kwargs = kwargs.copy()
            for k, v in list(kwargs.items()):
                if isinstance(v, list):
                    kwargs[k] = tuple(v)
        else:
            kwargs = {}

        ops.append(cls(**kwargs))

    _ensure_normalize_after_totensor(ops)
    return transforms.Compose(ops)

# ---------------------------------------------------------------------------
# Dataset builders
# ---------------------------------------------------------------------------

def _resolve_split_dirs(root: Path, split: Dict[str, str]) -> Tuple[Path, Path]:
    """
    Resolves train/val directories based on split mapping.
    If user points val to 'val_by_label' (a common reorg), we respect it.
    """
    train_dir = root / split.get("train", "train")
    val_dir = root / split.get("val", "val")
    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")
    if not val_dir.exists():
        raise FileNotFoundError(f"Val directory not found: {val_dir}")
    return train_dir, val_dir

def build_datasets(cfg_data: Dict[str, Any],
                   cfg_paths: Dict[str, Any],
                   cfg_aug: Dict[str, Any]):
    """
    Returns: train_ds, val_ds, class_to_idx
    Expects:
      cfg_paths["data_dir"] -> path to tiny-imagenet-200
      cfg_data["split"]["train"] -> e.g. "train"
      cfg_data["split"]["val"]   -> e.g. "val" or "val_by_label"
      cfg_aug has train_transforms / val_transforms lists
    """
    root = Path(cfg_paths["data_dir"]).expanduser().resolve()
    split = cfg_data.get("split", {"train": "train", "val": "val"})

    train_tf = build_transform_list(cfg_aug["train_transforms"])
    val_tf = build_transform_list(cfg_aug["val_transforms"])

    train_dir, val_dir = _resolve_split_dirs(root, split)

    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds = datasets.ImageFolder(val_dir, transform=val_tf)

    train_classes = train_ds.classes
    val_classes   = val_ds.classes
    if set(train_classes) != set(val_classes):
        missing_in_val   = sorted(set(train_classes) - set(val_classes))
        missing_in_train = sorted(set(val_classes) - set(train_classes))
        raise RuntimeError(
            f"Train/val class sets differ.\n"
            f"Missing in val: {missing_in_val[:5]} ...\n"
            f"Missing in train: {missing_in_train[:5]} ..."
        )

    if train_ds.class_to_idx != val_ds.class_to_idx:
        # Remap val targets to train's indices
        name_to_train = train_ds.class_to_idx
        name_to_val   = val_ds.class_to_idx
        idx_map = {name_to_val[name]: name_to_train[name] for name in name_to_val.keys()}

        val_samples = []
        for path, tgt in val_ds.samples:
            val_samples.append((path, idx_map[tgt]))
        val_ds.samples = val_samples
        if hasattr(val_ds, "targets"):
            val_ds.targets = [idx_map[t] for t in val_ds.targets]

        val_ds.class_to_idx = dict(name_to_train)
        val_ds.classes      = list(train_ds.classes)

    # Sanity: num_classes match check is done at model init / elsewhere if desired
    return train_ds, val_ds, train_ds.class_to_idx

# ---------------------------------------------------------------------------
# DataLoader builders
# ---------------------------------------------------------------------------

def build_dataloaders(
    train_ds, val_ds, cfg_data: Dict[str, Any], batch_size: int
) -> Tuple[DataLoader, DataLoader]:
    """
    Build DataLoaders using knobs from cfg_data.
    Reads: workers, pin_memory, persistent_workers, drop_last,
           loader.{shuffle_train, shuffle_val}
    """
    workers = int(cfg_data.get("workers", 0))
    pin_mem = bool(cfg_data.get("pin_memory", True))
    pers    = bool(cfg_data.get("persistent_workers", True))
    drop    = bool(cfg_data.get("drop_last", True))
    loader  = cfg_data.get("loader", {"shuffle_train": True, "shuffle_val": False})

    # Prefetch factor only valid when num_workers > 0
    prefetch_factor = cfg_data.get("prefetch_factor", None)
    dl_kwargs = {}
    if workers > 0 and prefetch_factor is not None:
        dl_kwargs["prefetch_factor"] = int(prefetch_factor)

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=loader.get("shuffle_train", True),
        num_workers=workers,
        pin_memory=pin_mem,
        persistent_workers=pers and workers > 0,
        drop_last=drop,
        **dl_kwargs,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=loader.get("shuffle_val", False),
        num_workers=workers,
        pin_memory=pin_mem,
        persistent_workers=pers and workers > 0,
        drop_last=False,
        **dl_kwargs,
    )
    return train_dl, val_dl

# ---------------------------------------------------------------------------
# Subset support (for smoke tests / tiny overfits)
# ---------------------------------------------------------------------------

def _maybe_subset(ds, n: Optional[int], seed: int = 0):
    """
    Return a Subset of length n if n is a positive int; otherwise return ds.
    Sampling is deterministic for reproducibility.
    """
    if n is None:
        return ds
    n = int(n)
    if n <= 0 or n >= len(ds):
        return ds
    g = torch.Generator()
    g.manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g)[:n].tolist()
    return Subset(ds, idx)

# ---------------------------------------------------------------------------
# One-call convenience wrapper
# ---------------------------------------------------------------------------

def build_datasets_and_loaders(cfg: Dict[str, Any]):
    """
    High-level helper that reads the merged cfg dict
    and returns ((train_ds, val_ds), (train_dl, val_dl)).
    """
    # Fallback to root-level keys if subtrees are missing
    cfg_paths = cfg.get("paths") or cfg
    cfg_data  = cfg.get("data")  or cfg
    cfg_aug   = cfg.get("aug")   or cfg
    cfg_train = cfg.get("train") or {}

    if "data_dir" not in cfg_paths:
        raise KeyError("cfg['paths']['data_dir'] (or root 'data_dir') is required")

    if "train_transforms" not in cfg_aug or "val_transforms" not in cfg_aug:
        raise KeyError("cfg['aug'] (or root) must contain 'train_transforms' and 'val_transforms'")

    train_ds, val_ds, class_to_idx = build_datasets(cfg_data, cfg_paths, cfg_aug)

    # Optional subset slicing for quick tests
    subset_cfg = cfg_data.get("subset", {})
    sub_train = subset_cfg.get("train", None)
    sub_val   = subset_cfg.get("val", None)
    seed      = int(subset_cfg.get("seed", 0))
    train_ds  = _maybe_subset(train_ds, sub_train, seed=seed)
    val_ds    = _maybe_subset(val_ds,   sub_val,   seed=seed)

    # Batch size priority: train.batch_size > data.batch_size > root.batch_size > default
    batch_size = (
        cfg_train.get("batch_size")
        or cfg_data.get("batch_size")
        or cfg.get("batch_size")
        or 128
    )
    batch_size = int(batch_size)

    train_dl, val_dl = build_dataloaders(train_ds, val_ds, cfg_data, batch_size)
    return (train_ds, val_ds), (train_dl, val_dl)
