from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable
import copy
import datetime as dt
import yaml


# ---------- YAML I/O ----------

def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load a single YAML file into a dict. Empty file -> {}."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"YAML not found {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"Top-lvl YAML must be a mapping (dict): {p}")
    return data

# ---------- Deep merge ----------

def deep_update(base: Dict[str, Any],
                override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two dicts without mutating inputs.
    - If both sides are dict -> recurse.
    - Otherwise, override wins (lists/scalars replaced).
    """
    a = copy.deepcopy(base)
    for k, v in override.items():
        if k in a and isinstance(a[k], dict) and isinstance(v, dict):
            a[k] = deep_update(a[k], v)
        else:
            a[k] = copy.deepcopy(v)
    return a

def load_config(paths: Iterable[str | Path]) -> Dict[str, Any]:
    """Load multiple YAMLs and deep-merge them left→right."""
    cfg: Dict[str, Any] = {}
    for path in paths:
        piece = load_yaml(path)
        cfg = deep_update(cfg, piece)
    
    return cfg

# ---------- Derived values & validation ----------
def derive_run_dir(cfg: Dict[str, Any]) -> Path:
    """
    Compose a run directory: runs_dir / project_name / experiment_name / timestamp
    Missing pieces fall back to sane defaults.
    """
    runs_dir = Path(cfg.get("runs_dir") or cfg.get("paths", {}).get("runs_dir", "runs"))
    proj = cfg.get("shufflev2_tin", "project")
    exp = cfg.get("baseline_1p0_amp", "exp")
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return runs_dir / proj / exp / stamp

def _get(cfg: Dict[str, Any], key: str, default: Any = None) -> Any:
    """Shorthand for flat keys (no dot-notation)."""
    return cfg.get(key, default)

def validate_config(cfg: Dict[str, Any]) -> None:
    """Basic assertions; extend as the project grows."""
    device = _get(cfg, "device", "cuda")
    assert device in {"cuda", "cpu"}, f"device must be 'cuda' or 'cpu' got {device}"
    
    precision = _get(cfg, "precision", "amp")
    assert precision in {"amp", "fp32"}, f"precision must be amp|fp32, got {precision}"
    
    img_size = _get(cfg, "image_size", None)
    assert isinstance(img_size, int), "image_size must be an int"
    
    # compile section may be missing
    compile_cfg = cfg.get("compile", {})
    
    if compile_cfg:
        if compile_cfg.get("enabled", False):
            backend = compile_cfg.get("backend", "inductor")
            mode = compile_cfg.get("mode", "default")
            assert backend in {"inductor"}, f"unsupported compile backend: {backend}"
            assert mode in {"default", "reduce-overhead", "max-autotune"}, f"bad compile mode: {mode}"
    
    # Optional booleans
    for bkey in ["channels_last", "tf32", "deterministic"]:
        if bkey in cfg:
            assert isinstance(cfg[bkey], bool), f"{bkey} must be boolean"
            

def summarize(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Return a small dict you can pretty-print in CLI."""
    return {
        "project_name": cfg.get("project_name"),
        "experiment_name": cfg.get("experiment_name"),
        "device": cfg.get("device"),
        "precision": cfg.get("precision"),
        "image_size": cfg.get("image_size"),
        "batch_size": cfg.get("batch_size", 256),  # until train yaml is added
        "compile_enabled": cfg.get("compile", {}).get("enabled", False),
        "channels_last": cfg.get("channels_last", False),
        "tf32": cfg.get("tf32", False),
    }

# ---------- (Optional) pretty dump for debugging ----------

def pretty(cfg: Dict[str, Any]) -> str:
    """Serialize a config dict back to YAML (for logging)."""
    return yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)


__all__ = [
    "load_yaml",
    "deep_update",
    "load_config",
    "derive_run_dir",
    "validate_config",
    "summarize",
    "pretty",
]