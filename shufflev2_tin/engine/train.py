# shufflev2_tin/engine/train.py
from __future__ import annotations
import time
from pathlib import Path
from typing import Dict, Any, Tuple
from tqdm import tqdm
import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from shufflev2_tin.models import from_config
from shufflev2_tin.data.tiny_imagenet import build_datasets_and_loaders
from shufflev2_tin.utils.config_loader import derive_run_dir


# ---------- helpers / defaults ----------

def normalize_cfg(cfg: dict) -> dict:
    """Coerce legacy/root keys into places Trainer and data loaders expect; fill sensible defaults."""
    # ---- TRAIN SUBTREE ----
    cfg.setdefault("train", {})
    t = cfg["train"]

    # move commonly used roots under train
    for k in ("epochs", "label_smoothing", "criterion", "batch_size", "metrics"):
        if k not in t and k in cfg:
            t[k] = cfg[k]

    if "onecycle" in cfg and "onecycle" not in t:
        t["onecycle"] = dict(cfg["onecycle"])

    # runtime flags
    t.setdefault("amp", True)
    t.setdefault("compile", True)
    t.setdefault("channels_last", True)
    t.setdefault("cudnn_benchmark", True)
    t.setdefault("tf32", True)
    t.setdefault("log_interval", int(cfg.get("log_every", 50)))
    t.setdefault("ckpt_interval", int(cfg.get("checkpoint_every", 1)))
    t.setdefault("ckpt_keep_last", 3)

    # scheduler & warmup knobs
    t.setdefault("epochs", int(cfg.get("epochs", 80)))
    t.setdefault("pct_start", float(cfg.get("onecycle", {}).get("pct_start", 0.3)))
    t.setdefault("warmup_pct", 0.1)
    t.setdefault("max_lr_scale", 1.0)
    t.setdefault("grad_clip_norm", float(cfg.get("grad_clip_norm", 1.0)))
    t.setdefault("label_smoothing", float(cfg.get("label_smoothing", 0.0)))

    # ---- PATHS SUBTREE (compat with tiny_imagenet.py) ----
    paths = cfg.setdefault("paths", {})
    if not paths.get("data_dir") and cfg.get("data_dir"):
        paths["data_dir"] = cfg["data_dir"]
    for k in ("runs_dir", "ckpt_dir", "log_dir"):
        if not paths.get(k) and cfg.get(k):
            paths[k] = cfg[k]

    # ---- AUG SUBTREE (REQUIRED BY tiny_imagenet.py) ----
    aug = cfg.setdefault("aug", {})
    if "train_transforms" not in aug and "train_transforms" in cfg:
        aug["train_transforms"] = cfg["train_transforms"]
    if "val_transforms" not in aug and "val_transforms" in cfg:
        aug["val_transforms"] = cfg["val_transforms"]

    # ---- OPTIM ----
    cfg.setdefault("optim", {})
    ocfg = cfg["optim"]
    # prefer root optimizer keys if present
    ocfg.setdefault("name", cfg.get("optimizer", "adamw"))
    ocfg.setdefault("lr", float(cfg.get("lr", 1e-3)))
    ocfg.setdefault("weight_decay", float(cfg.get("weight_decay", 0.02)))
    ocfg.setdefault("betas", tuple(cfg.get("betas", (0.9, 0.999))))

    return cfg


def set_system_flags(cfg: Dict[str, Any]) -> None:
    """Backend toggles: cudnn benchmark, TF32 with new API if present."""
    tcfg = cfg.get("train", {})
    torch.backends.cudnn.benchmark = bool(tcfg.get("cudnn_benchmark", True))

    tf32 = bool(tcfg.get("tf32", True))
    # New API (PyTorch 2.9+)
    try:
        torch.backends.cuda.matmul.fp32_precision = "tf32" if tf32 else "ieee"
        torch.backends.cudnn.conv.fp32_precision = "tf32" if tf32 else "high"
    except Exception:
        # Fallback to legacy flags on older builds
        try:
            torch.backends.cuda.matmul.allow_tf32 = tf32
            torch.backends.cudnn.allow_tf32 = tf32
        except Exception:
            pass  # CPU-only etc.


def accuracy_top1(logits: torch.Tensor, targets: torch.Tensor) -> float:
    with torch.no_grad():
        preds = logits.argmax(dim=1)
        return (preds == targets).float().mean().item()


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor, label_smoothing: float = 0.0) -> torch.Tensor:
    if label_smoothing and label_smoothing > 0.0:
        return nn.functional.cross_entropy(logits, targets, label_smoothing=label_smoothing)
    return nn.functional.cross_entropy(logits, targets)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def save_ckpt(state: Dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, out_path)


# ---------- trainer ----------

class Trainer:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        cfg = normalize_cfg(cfg)
        self.cfg = cfg
        self.device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        set_system_flags(cfg)

        # Data
        (self.train_ds, self.val_ds), (self.train_dl, self.val_dl) = build_datasets_and_loaders(cfg)

        # Model
        model = from_config(cfg)

        # optional: sync model.dropout with train config if provided
        tcfg = cfg.get("train", {})
        if (
            "dropout" in cfg and cfg["dropout"] is not None
            and hasattr(model, "dropout")
            and isinstance(getattr(model, "dropout"), (nn.Dropout, nn.Identity))
        ):
            p = float(cfg["dropout"])
            model.dropout = nn.Dropout(p) if p > 0 else nn.Identity()

        if self.device.type == "cuda" and bool(tcfg.get("channels_last", True)):
            model = model.to(self.device, memory_format=torch.channels_last)
        else:
            model = model.to(self.device)

        # ----- robust torch.compile (avoid TritonMissing on Windows / no Triton) -----
        if bool(tcfg.get("compile", True)):
            backend = (self.cfg.get("compile", {}) or {}).get("backend", "inductor")
            mode = (self.cfg.get("compile", {}) or {}).get("mode", "default")

            supports_triton = False
            try:
                import triton  # noqa: F401
                supports_triton = True
            except Exception:
                supports_triton = False

            # If requested inductor but Triton is missing, fall back to a backend that doesn't need Triton
            if backend == "inductor" and not supports_triton:
                print("[compile] Triton not found -> falling back to backend='aot_eager'")
                backend = "aot_eager"

            try:
                model = torch.compile(model, backend=backend, mode=mode, fullgraph=False, dynamic=True)
            except Exception as e:
                # final safety net: run uncompiled
                print(f"[compile] disabled (reason: {e.__class__.__name__}: {e})")
                # leave model as eager

        self.model = model

        # Optimizer
        ocfg = cfg.get("optim", {})
        base_lr = float(ocfg.get("lr", 1e-3))
        wd = float(ocfg.get("weight_decay", 0.02))
        betas = tuple(ocfg.get("betas", (0.9, 0.999)))
        self.optim = AdamW(self.model.parameters(), lr=base_lr, weight_decay=wd, betas=betas)

        # Scheduler (OneCycle over total steps)
        epochs = int(tcfg.get("epochs", 80))
        steps_per_epoch = max(1, len(self.train_dl))
        total_steps = epochs * steps_per_epoch
        pct_start = float(tcfg.get("pct_start", 0.3))
        warmup_pct = float(tcfg.get("warmup_pct", 0.1))
        max_lr_scale = float(tcfg.get("max_lr_scale", 1.0))
        max_lr = base_lr * max_lr_scale
        div_factor = (1.0 / max(1e-8, warmup_pct)) if warmup_pct > 0 else 25.0

        self.sched = OneCycleLR(
            self.optim,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=pct_start,
            anneal_strategy="cos",
            div_factor=div_factor,
            final_div_factor=1e3,
        )

        # AMP / loss
        use_amp = bool(tcfg.get("amp", True)) and self.device.type == "cuda"
        self.scaler = GradScaler(device="cuda", enabled=use_amp)
        self.grad_clip_norm = float(tcfg.get("grad_clip_norm", 1.0) or 0.0)
        self.label_smoothing = float(tcfg.get("label_smoothing", 0.0) or 0.0)
        self.criterion = lambda logits, y: cross_entropy(logits, y, self.label_smoothing)

        # IO
        self.run_dir = Path(derive_run_dir(cfg))
        self.ckpt_dir = self.run_dir / "ckpts"
        self.log_interval = int(tcfg.get("log_interval", 50))
        self.ckpt_interval = int(tcfg.get("ckpt_interval", 1))
        self.ckpt_keep_last = int(tcfg.get("ckpt_keep_last", 3))
        self.best_path = self.ckpt_dir / "best.pt"

        # ---- summary ----
        params = count_params(self.model)
        print(
            f"[Init] Device={self.device}  Params={params/1e6:.3f}M  "
            f"Epochs={epochs}  Steps/Epoch={steps_per_epoch}  TotalSteps={total_steps}"
        )

    def _train_one_epoch(self, epoch: int) -> Tuple[float, float]:
        self.model.train()
        loss_sum = 0.0
        acc_sum = 0.0
        n_batches = 0

        for step, (xb, yb) in enumerate(self.train_dl, 1):
            if self.device.type == "cuda" and self.cfg["train"].get("channels_last", True):
                xb = xb.to(self.device, memory_format=torch.channels_last, non_blocking=True)
            else:
                xb = xb.to(self.device, non_blocking=True)
            yb = yb.to(self.device, non_blocking=True)

            self.optim.zero_grad(set_to_none=True)
            with autocast(self.device.type, enabled=self.scaler.is_enabled()):
                logits = self.model(xb)
                loss = self.criterion(logits, yb)

            self.scaler.scale(loss).backward()
            if self.grad_clip_norm and self.grad_clip_norm > 0:
                self.scaler.unscale_(self.optim)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

            self.scaler.step(self.optim)
            self.scaler.update()
            self.sched.step()

            loss_sum += float(loss.detach())
            acc_sum += accuracy_top1(logits, yb)
            n_batches += 1

            if step % self.log_interval == 0:
                avg_loss = loss_sum / n_batches
                avg_acc = acc_sum / n_batches
                print(f"[Epoch {epoch:03d}] step {step:05d}  loss={avg_loss:.4f}  acc@1={avg_acc*100:5.2f}%")

        return loss_sum / max(1, n_batches), acc_sum / max(1, n_batches)

    @torch.no_grad()
    def _validate(self, epoch: int) -> Tuple[float, float]:
        self.model.eval()
        loss_sum = 0.0
        acc_sum = 0.0
        n_batches = 0

        for xb, yb in self.val_dl:
            if self.device.type == "cuda" and self.cfg["train"].get("channels_last", True):
                xb = xb.to(self.device, memory_format=torch.channels_last, non_blocking=True)
            else:
                xb = xb.to(self.device, non_blocking=True)
            yb = yb.to(self.device, non_blocking=True)

            with autocast(device_type=self.device.type, enabled=self.scaler.is_enabled()):
                logits = self.model(xb)
                loss = cross_entropy(logits, yb, self.label_smoothing)

            loss_sum += float(loss)
            acc_sum += accuracy_top1(logits, yb)
            n_batches += 1

        return loss_sum / max(1, n_batches), acc_sum / max(1, n_batches)

    def fit(self) -> None:
        epochs = int(self.cfg["train"].get("epochs", 80))
        best_acc = -1.0

        for epoch in range(1, epochs + 1):
            tr_loss, tr_acc = self._train_one_epoch(epoch)
            va_loss, va_acc = self._validate(epoch)
            print(
                f"[Epoch {epoch:03d} DONE] "
                f"train_loss={tr_loss:.4f}  train_acc={tr_acc*100:5.2f}%  "
                f"val_loss={va_loss:.4f}  val_acc={va_acc*100:5.2f}%"
            )

            # periodic checkpoints
            if epoch % self.ckpt_interval == 0:
                path = self.ckpt_dir / f"epoch_{epoch:03d}.pt"
                save_ckpt(
                    {
                        "epoch": epoch,
                        "model": self.model.state_dict(),
                        "optim": self.optim.state_dict(),
                        "sched": self.sched.state_dict(),
                        "scaler": self.scaler.state_dict(),
                        "cfg": self.cfg,
                        "metrics": {
                            "train_loss": tr_loss,
                            "train_acc": tr_acc,
                            "val_loss": va_loss,
                            "val_acc": va_acc,
                        },
                    },
                    path,
                )

                # keep only last K epoch checkpoints
                ckpts = sorted(self.ckpt_dir.glob("epoch_*.pt"))
                if len(ckpts) > self.ckpt_keep_last:
                    for p in ckpts[:-self.ckpt_keep_last]:
                        try:
                            p.unlink()
                        except Exception:
                            pass

            # best checkpoint
            if va_acc > best_acc:
                best_acc = va_acc
                save_ckpt(
                    {
                        "epoch": epoch,
                        "model": self.model.state_dict(),
                        "cfg": self.cfg,
                        "metrics": {"val_acc": va_acc, "val_loss": va_loss},
                    },
                    self.best_path,
                )
                print(f"[BEST] val_acc={best_acc*100:5.2f}%  -> {self.best_path}")

        print(f"[Training complete] Best val_acc={best_acc*100:5.2f}%")
