# shufflev2_tin/models/shufflenetv2.py
from __future__ import annotations
from typing import Iterable, Tuple, Dict, Any, Optional

import torch
from torch import nn

# ---- helpers ---------------------------------------------------------------

def _make_activation(name: str) -> nn.Module:
    name = name.lower()
    if name in {"relu", "re_lu"}:
        return nn.ReLU(inplace=True)
    if name in {"relu6", "re_lu6"}:
        return nn.ReLU6(inplace=True)
    if name in {"hswish", "hardswish"}:
        return nn.Hardswish()
    if name in {"silu", "swish"}:
        return nn.SiLU(inplace=True)
    raise ValueError(f"Unsupported activation: {name}")

def _make_norm(name: str, num_features: int) -> nn.Module:
    name = name.lower()
    if name in {"bn", "batchnorm", "batch_norm"}:
        return nn.BatchNorm2d(num_features)
    if name in {"gn", "groupnorm", "group_norm"}:
        # 32 is a common default; must divide num_features
        groups = min(32, num_features)
        while num_features % groups != 0 and groups > 1:
            groups //= 2
        return nn.GroupNorm(groups, num_features)
    raise ValueError(f"Unsupported norm: {name}")

class ConvBNAct(nn.Sequential):
    """Pointwise / standard conv + BN + Act. Bias off (BN absorbs)."""
    def __init__(self, in_c: int, out_c: int, k: int = 1, s: int = 1, p: int = 0,
                 groups: int = 1, norm: str = "bn", act: str = "relu"):
        super().__init__(
            nn.Conv2d(in_c, out_c, k, s, p, groups=groups, bias=False),
            _make_norm(norm, out_c),
            _make_activation(act),
        )

class DepthwiseConv(nn.Sequential):
    """3×3 depthwise conv + BN (no activation here to match paper head)."""
    def __init__(self, channels: int, stride: int = 1, norm: str = "bn"):
        super().__init__(
            nn.Conv2d(channels, channels, 3, stride, 1, groups=channels, bias=False),
            _make_norm(norm, channels),
        )

def channel_shuffle(x: torch.Tensor, groups: int = 2) -> torch.Tensor:
    """Shuffle channels by groups: (N, C, H, W) -> (N, g, C/g, H, W) -> permute -> reshape."""
    n, c, h, w = x.shape
    assert c % groups == 0, "Channels must be divisible by groups"
    x = x.view(n, groups, c // groups, h, w)
    x = x.permute(0, 2, 1, 3, 4).contiguous()
    x = x.view(n, c, h, w)
    return x

# ---- building blocks -------------------------------------------------------

class ShuffleUnit(nn.Module):
    """Basic unit (stride=1). Split input channels, transform branch, concat, shuffle."""
    def __init__(self, in_channels: int, out_channels: int, *, norm: str = "bn", act: str = "relu"):
        super().__init__()
        # Split equally
        assert out_channels % 2 == 0, "out_channels should be even for split"
        self.out_channels = out_channels
        branch_channels = out_channels // 2

        # For stride=1 unit, in_channels == out_channels
        # x gets split into x1 (identity) and x2 (branch)
        self.branch = nn.Sequential(
            # 1×1 PW
            ConvBNAct(branch_channels, branch_channels, k=1, s=1, p=0, norm=norm, act=act),
            # 3×3 DW
            DepthwiseConv(branch_channels, stride=1, norm=norm),
            # 1×1 PW
            ConvBNAct(branch_channels, branch_channels, k=1, s=1, p=0, norm=norm, act=act),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = x.shape[1]
        assert c == self.out_channels, "Stride=1 units expect in_channels == out_channels"
        c_half = c // 2
        x1, x2 = x[:, :c_half, :, :], x[:, c_half:, :, :]
        out2 = self.branch(x2)
        out = torch.cat((x1, out2), dim=1)
        out = channel_shuffle(out, groups=2)
        return out

class ShuffleDownUnit(nn.Module):
    """Downsample unit (stride=2). Two branches; concat; shuffle."""
    def __init__(self, in_channels: int, out_channels: int, *, norm: str = "bn", act: str = "relu"):
        super().__init__()
        assert out_channels % 2 == 0, "out_channels should be even for split"
        branch_out = out_channels // 2

        # Branch 1: projection
        self.branch1 = nn.Sequential(
            DepthwiseConv(in_channels, stride=2, norm=norm),
            ConvBNAct(in_channels, branch_out, k=1, s=1, p=0, norm=norm, act=act),
        )

        # Branch 2: like basic branch but DW has stride=2
        self.branch2 = nn.Sequential(
            ConvBNAct(in_channels, branch_out, k=1, s=1, p=0, norm=norm, act=act),
            DepthwiseConv(branch_out, stride=2, norm=norm),
            ConvBNAct(branch_out, branch_out, k=1, s=1, p=0, norm=norm, act=act),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        out = channel_shuffle(out, groups=2)
        return out

# ---- network ---------------------------------------------------------------

_WIDTHS: Dict[float, Tuple[int, int, int, int, int]] = {
    # stem, stage2, stage3, stage4, conv5
    0.5:   (24,  48,  96, 192, 1024),
    1.0:   (24, 116, 232, 464, 1024),
    1.5:   (24, 176, 352, 704, 1024),
    2.0:   (24, 244, 488, 976, 2048),
}

class ShuffleNetV2(nn.Module):
    """
    ShuffleNetV2 backbone/head (Torch 2.x friendly).

    Args:
      num_classes: output classes
      width_mult: 0.5, 1.0, 1.5, 2.0 (only 1.0 used by default)
      stages: repeats per stage (e.g. [4, 8, 4] for 1.0×)
      out_channels: optional explicit channels [c1, c2, c3, c4, c5]
      dropout: head dropout
      norm: 'bn' or 'gn'
      act: activation name
      stem_stride: usually 2 for 64×64 inputs
    """
    def __init__(
        self,
        num_classes: int = 1000,
        width_mult: float = 1.0,
        *,
        stages: Iterable[int] = (4, 8, 4),
        out_channels: Optional[Iterable[int]] = None,
        dropout: float = 0.0,
        norm: str = "bn",
        act: str = "relu",
        stem_stride: int = 2,
        last_pool: str = "global_avg",
    ):
        super().__init__()

        if out_channels is None:
            if width_mult not in _WIDTHS:
                raise ValueError(f"Unsupported width_mult: {width_mult}")
            out_channels = _WIDTHS[width_mult]
        oc = list(out_channels)
        assert len(oc) == 5, "out_channels must be [c1, c2, c3, c4, c5]"
        c1, c2, c3, c4, c5 = oc
        s2, s3, s4 = list(stages)

        # stem
        self.stem = ConvBNAct(3, c1, k=3, s=stem_stride, p=1, norm=norm, act=act)

        # stages (each starts with a stride=2 down unit, then (n-1) stride=1 units)
        self.stage2 = self._make_stage(c1, c2, s2, norm=norm, act=act)
        self.stage3 = self._make_stage(c2, c3, s3, norm=norm, act=act)
        self.stage4 = self._make_stage(c3, c4, s4, norm=norm, act=act)

        # conv5 (1×1 PW + BN + act)
        self.conv5 = ConvBNAct(c4, c5, k=1, s=1, p=0, norm=norm, act=act)

        # head
        self.pool = nn.AdaptiveAvgPool2d(1) if last_pool == "global_avg" else nn.Identity()
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0.0 else nn.Identity()
        self.fc = nn.Linear(c5, num_classes)

        self._init_weights()

    def _make_stage(self, in_c: int, out_c: int, repeats: int, *, norm: str, act: str) -> nn.Sequential:
        blocks = [ShuffleDownUnit(in_c, out_c, norm=norm, act=act)]
        for _ in range(repeats - 1):
            blocks.append(ShuffleUnit(out_c, out_c, norm=norm, act=act))
        return nn.Sequential(*blocks)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# ---- small builders --------------------------------------------------------

def build_shufflenetv2(
    num_classes: int,
    width_mult: float = 1.0,
    *,
    stages: Iterable[int] = (4, 8, 4),
    out_channels: Optional[Iterable[int]] = None,
    dropout: float = 0.0,
    norm: str = "bn",
    act: str = "relu",
    stem_stride: int = 2,
    last_pool: str = "global_avg",
) -> ShuffleNetV2:
    return ShuffleNetV2(
        num_classes=num_classes,
        width_mult=width_mult,
        stages=stages,
        out_channels=out_channels,
        dropout=dropout,
        norm=norm,
        act=act,
        stem_stride=stem_stride,
        last_pool=last_pool,
    )

def from_config(cfg: Dict[str, Any]) -> ShuffleNetV2:
    mcfg = cfg.get("model", cfg)  # allow passing the merged cfg or only model part
    return build_shufflenetv2(
        num_classes=int(mcfg.get("num_classes", 1000)),
        width_mult=float(mcfg.get("width_mult", 1.0)),
        stages=tuple(mcfg.get("stages", (4, 8, 4))),
        out_channels=tuple(mcfg.get("out_channels")) if mcfg.get("out_channels") else None,
        dropout=float(mcfg.get("dropout", 0.0)),
        norm=str(mcfg.get("norm", "bn")),
        act=str(mcfg.get("act", "relu")),
        stem_stride=int(mcfg.get("stem_stride", 2)),
        last_pool=str(mcfg.get("last_pool", "global_avg")),
    )
