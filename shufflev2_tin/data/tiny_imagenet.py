from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

_NAME_TO_TRANSFORM = {
    "RandomResizedCrop": transforms.RandomResizedCrop,
    "RandomHorizontalFlip": transforms.RandomHorizontalFlip,
    "ColorJitter": transforms.ColorJitter,
    "Resize": transforms.Resize,
    "CenterCrop": transforms.CenterCrop,
    "ToTensor": transforms.ToTensor,
    "Normalize": transforms.Normalize,
}

def build_transform_list(cfg_list: list[dict]) -> transforms.Compose:
    """
    Parse a list of {Name: {kwargs...}} into torchvision transforms.Compose.
    Example element: {"RandomResizedCrop": {"size": 64, "scale": [0.7, 1.0]}}
    """
    
    ops = []
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
    return transforms.Compose(ops)

def build_datasets(cfg_data: Dict[str, Any],
                   cfg_paths: Dict[str, Any],
                   cfg_aug: Dict[str, Any]):
    """
    Returns: train_ds, val_ds, class_to_idx
    Expects:
      cfg_paths["data_dir"] -> path to tiny-imagenet-200
      cfg_data["split"]["train"] -> "train"
      cfg_data["split"]["val"]   -> "val"
      cfg_aug has train_transforms / val_transforms lists
    """
    root = Path(cfg_paths["data_dir"])
    split = cfg_data.get("split", {"train": "train", "val": "val"})
    
    train_tf = build_transform_list(cfg_aug["train_transforms"])
    val_tf = build_transform_list(cfg_aug["val_transforms"])
    
    train_dir = root / split["train"]
    val_dir = root / split["val"]
    
    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds = datasets.ImageFolder(val_dir, transform=val_tf)
    
    # sanity: image size consistent with global config is checked elsewhere
    return train_ds, val_ds, train_ds.class_to_idx

def build_dataloaders(
    train_ds, val_ds, cfg_data: Dict[str, Any], batch_size: int
) -> Tuple[DataLoader, DataLoader]:
    """
    Build DataLoaders using knobs from cfg_data.
    Reads: workers, pin_memory, persistent_workers, drop_last, loader.{shuffle_train,shuffle_val}
    """
    workers = cfg_data.get("workers", 0)
    pin_mem = cfg_data.get("pin_memory", True)
    pers    = cfg_data.get("persistent_workers", True)
    drop    = cfg_data.get("drop_last", True)
    loader  = cfg_data.get("loader", {"shuffle_train": True, "shuffle_val": False})
    
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=loader.get("shuffle_train", True),
        num_workers=workers,
        pin_memory=pin_mem,
        persistent_workers=pers,
        drop_last=drop,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=loader.get("shuffle_val", False),
        num_workers=workers,
        pin_memory=pin_mem,
        persistent_workers=pers,
        drop_last=False,
    )
    return train_dl, val_dl