"""fotonet v2 shared building blocks and model builders (fotonet2n / Hi Bird)."""
from __future__ import annotations

import torch
import torch.nn as nn

from fotonet.nn.conv import Conv, RepConv
from fotonet.registry import alias

for _old, _new in (
    ("nn.conv@2", "nn.conv@1.0.0"),
    ("nn.repconv@2", "nn.repconv@1.0.0"),
    ("nn.sppf@2", "nn.sppf@1.0.0"),
):
    try:
        alias(_old, _new)
    except ValueError:
        pass

from .spec import GRAPH_VERSION, HYBRID_EXTRA_KEYS, LAYOUT_KEYS, get_spec

# State-dict prefixes that exist only while training. Stripped from deploy graphs and
# inference-only checkpoints.
TRAINING_ONLY_PREFIXES = ("head.o2m.", "head.aux_obj.", "head.o2o_groups.")


class RepBottleneck(nn.Module):
    """x + Conv3x3(RepConv3x3(x)). The middle width is ``e`` times the block width."""

    def __init__(self, channels: int, e: float = 0.5):
        super().__init__()
        mid = max(int(channels * e), 8)
        self.cv1 = RepConv(channels, mid, 3)
        self.cv2 = Conv(mid, channels, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv2(self.cv1(x))


class V2SPPF(nn.Module):
    """Spatial pyramid pooling with shortcut and optional cv1_act."""

    def __init__(self, c1: int, c2: int, kernel_size: int = 5, shortcut: bool = True, cv1_act: bool = False):
        super().__init__()
        hidden = c1 // 2
        self.shortcut = shortcut and (c1 == c2)
        self.reduce = Conv(c1, hidden, 1, act=cv1_act)
        self.pool = nn.MaxPool2d(kernel_size, stride=1, padding=kernel_size // 2)
        self.project = Conv(hidden * 4, c2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.reduce(x)
        p1 = self.pool(y)
        p2 = self.pool(p1)
        out = self.project(torch.cat((y, p1, p2, self.pool(p2)), dim=1))
        return x + out if self.shortcut else out


def build_v2_model(model_id: str = "fotonet2n", nc: int = 80, imgsz: int = 640, **kwargs):
    from .hybrid import V2HybridDetector

    key = str(model_id).strip().lower()
    return V2HybridDetector(model_id=key, nc=nc, imgsz=imgsz, **kwargs)


def build_from_config(model_id: str, config: dict, nc: int = None, imgsz: int = None):
    """Rebuild the v2 graph a checkpoint or live model describes."""
    from .hybrid import V2HybridDetector

    cfg = dict(config or {})
    key = str(model_id).strip().lower()
    spec = get_spec(key)
    if str(cfg.get("graph_version", GRAPH_VERSION)) != GRAPH_VERSION:
        raise ValueError(
            f"This checkpoint was built by v2 graph {cfg.get('graph_version')!r}; this package builds "
            f"{GRAPH_VERSION!r}."
        )
    return V2HybridDetector(
        key,
        nc=int(cfg.get("nc", spec.nc) if nc is None else nc),
        imgsz=int(cfg.get("imgsz", spec.imgsz) if imgsz is None else imgsz),
        reg_max=cfg.get("reg_max", spec.reg_max),
        aux_objectness=bool(cfg.get("aux_objectness", False)),
        **{k: cfg.get(k, getattr(spec, k)) for k in (*LAYOUT_KEYS, *HYBRID_EXTRA_KEYS)},
    )


__all__ = [
    "RepBottleneck",
    "TRAINING_ONLY_PREFIXES",
    "V2SPPF",
    "build_from_config",
    "build_v2_model",
]
