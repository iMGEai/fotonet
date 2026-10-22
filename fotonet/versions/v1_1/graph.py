"""fotonet v1.1 detector graph (fotonetn, fotonets, fotonetm, fotonetl).

Backbone: dense strided 3x3 downsamples feeding CSP-Star stages (1x1 split,
StarBlocks on half the channels, every intermediate kept, 1x1 merge). SPPF and
attention (PSA) at P5 for every tier, attention at P4 for m and l.
Neck: top-down then bottom-up pyramid with CSP-Star junctions.
Head: the deployed one-to-one (O2O) head reads detached neck features, so the
backbone learns only from the dense one-to-many (O2M) head. O2M, its
objectness branch, and any extra O2O groups exist only during training.

Every 3x3 RepConv carries training-only 1x1 and identity branches that fold
into one 3x3 at export, so they cost nothing deployed.

Eval returns [B, N, nc + 4]: raw class logits plus normalized cxcywh.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .spec import (
    architecture_fingerprint, get_spec, GRAPH_VERSION, LEVEL_STRIDES, validate_layout,
)

from fotonet.nn.conv import Conv, fuse_conv_bn as _fuse_conv_bn, RepConv  # noqa: F401
from fotonet.nn.blocks import CSPStar, SPPF, StarBlock  # noqa: F401
from fotonet.nn.attention import Attention, PSA, PSABlock  # noqa: F401
from fotonet.nn.backbones import Backbone, _stage  # noqa: F401
from fotonet.nn.necks import Neck  # noqa: F401
from fotonet.nn.heads import (  # noqa: F401
    _clip_norm_xywh, DetectHead, FEATURE_STRIDES, PredictorPair, _round8, TaskHead,
)


# State-dict prefixes that exist only while training. Stripped from deploy
# graphs and inference-only checkpoints.
TRAINING_ONLY_PREFIXES = ("head.o2m.", "head.aux_obj.", "head.o2o_groups.")


_LAYOUT_KEYS = (
    "stem_channels", "p2_channels", "widths", "depths", "neck_channels", "neck_depths",
    "head_hidden", "attn_p5", "attn_p4", "head_levels",
)


class V11Detector(nn.Module):
    """CSP-Star backbone with attention, bidirectional pyramid, decoupled dual head."""

    def __init__(
        self,
        model_id: str = "fotonetn",
        nc: int = 80,
        imgsz: int = 640,
        *,
        stem_channels=None,
        p2_channels=None,
        widths=None,
        depths=None,
        neck_channels=None,
        neck_depths=None,
        head_hidden=None,
        attn_p5=None,
        attn_p4=None,
        head_levels=None,
        reg_max=None,
        o2m_mult: float = 1.5,
        o2o_groups: int = 1,
        aux_objectness: bool = True,
    ):
        super().__init__()
        spec = get_spec(model_id)
        if reg_max is not None and int(reg_max) != 1:
            raise ValueError("v1.1 uses direct LTRB regression; reg_max must be 1.")
        overrides = {
            "stem_channels": stem_channels, "p2_channels": p2_channels, "widths": widths,
            "depths": depths, "neck_channels": neck_channels, "neck_depths": neck_depths,
            "head_hidden": head_hidden, "attn_p5": attn_p5, "attn_p4": attn_p4,
            "head_levels": head_levels,
        }
        layout = {}
        for key in _LAYOUT_KEYS:
            value = getattr(spec, key) if overrides[key] is None else overrides[key]
            layout[key] = tuple(int(v) for v in value) if isinstance(value, (tuple, list)) else int(value)
        validate_layout(nc=int(nc), reg_max=1, **layout)
        self.spec = spec
        self.model_id = spec.model_id
        self.graph_version = GRAPH_VERSION
        self.nc = int(nc)
        self.reg_max = 1
        self.imgsz = int(imgsz)
        self.backbone = Backbone(
            layout["stem_channels"], layout["p2_channels"], layout["widths"], layout["depths"],
            layout["attn_p5"], layout["attn_p4"],
        )
        self.neck = Neck(self.backbone.out_channels, layout["neck_channels"], layout["neck_depths"])
        self.head_levels = layout["head_levels"]
        self._level_index = [(3, 4, 5).index(level) for level in self.head_levels]
        self.head = DetectHead(
            [self.neck.out_channels[i] for i in self._level_index],
            nc=self.nc,
            hidden=layout["head_hidden"],
            strides=[LEVEL_STRIDES[level] for level in self.head_levels],
            imgsz=self.imgsz,
            o2m_mult=o2m_mult,
            o2o_groups=o2o_groups,
            aux_objectness=aux_objectness,
        )
        self.model_config = {
            "model_id": spec.model_id,
            "graph_version": GRAPH_VERSION,
            "nc": self.nc,
            "reg_max": self.reg_max,
            "imgsz": self.imgsz,
            **{key: list(value) if isinstance(value, tuple) else value for key, value in layout.items()},
            # Training-only head shape: not part of the deployed identity.
            "o2m_mult": float(o2m_mult),
            "o2o_groups": int(max(int(o2o_groups), 1)),
            "aux_objectness": bool(aux_objectness),
        }
        self.architecture_fingerprint = architecture_fingerprint(self.model_config)
        self.model_config["architecture_fingerprint"] = self.architecture_fingerprint
        self.apply(self._init_weights)
        self._zero_init_residuals()
        self.head.init_biases()
        self._fused = False

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _zero_init_residuals(self):
        """Residual branches start as identities: StarBlock's last BN and attention outputs."""
        for module in self.modules():
            if isinstance(module, StarBlock):
                nn.init.zeros_(module.p2.bn.weight)
            elif isinstance(module, PSABlock):
                nn.init.zeros_(module.attn.proj.weight)
                nn.init.zeros_(module.attn.proj.bias)
                nn.init.zeros_(module.ffn[1].weight)
                nn.init.zeros_(module.ffn[1].bias)

    @property
    def strides(self):
        return self.head.strides

    def set_imgsz(self, imgsz):
        if isinstance(imgsz, (tuple, list)):
            if int(imgsz[0]) != int(imgsz[1]):
                raise ValueError("v1.1 training image size must be square.")
            side = int(imgsz[0])
        else:
            side = int(imgsz)
        if side % FEATURE_STRIDES[-1]:
            raise ValueError(f"v1.1 image size must be divisible by {FEATURE_STRIDES[-1]}, got {side}.")
        self.imgsz = side
        self.head.imgsz = side
        self.model_config["imgsz"] = side
        return self

    def _features(self, x):
        feats = self.neck(self.backbone(x))
        return [feats[i] for i in self._level_index]

    def forward(self, x: torch.Tensor, use_o2m: bool = True):
        if self.training:
            height, width = int(x.shape[-2]), int(x.shape[-1])
            if height != width or height % FEATURE_STRIDES[-1]:
                raise RuntimeError(
                    f"v1.1 training expects a square image size divisible by "
                    f"{FEATURE_STRIDES[-1]}, got {height}x{width}."
                )
            return self.head(self._features(x), imgsz=(float(width), float(height)), use_o2m=use_o2m)

        stride = FEATURE_STRIDES[-1]
        tracing = torch.onnx.is_in_onnx_export() or torch.jit.is_tracing()
        if tracing:
            shape = torch._shape_as_tensor(x)
            pad_h = torch.remainder(-shape[2], stride)
            pad_w = torch.remainder(-shape[3], stride)
            x = F.pad(x, (0, pad_w, 0, pad_h), value=114.0 / 255.0)
            padded = torch._shape_as_tensor(x)
            orig_wh = shape[[3, 2]].to(dtype=x.dtype)
            padded_wh = padded[[3, 2]].to(dtype=x.dtype)
            raw = self.head(self._features(x), imgsz=padded_wh, use_o2m=False)
            scale = (padded_wh / orig_wh.clamp(min=1)).to(device=raw.device, dtype=raw.dtype)
            box_scale = torch.cat((scale, scale))          # [sw, sh, sw, sh]; CoreML cannot stack 0-d tensors
            boxes = raw[..., self.nc:] * box_scale
        else:
            height, width = int(x.shape[-2]), int(x.shape[-1])
            pad_h = (stride - height % stride) % stride
            pad_w = (stride - width % stride) % stride
            if pad_h or pad_w:
                x = F.pad(x, (0, pad_w, 0, pad_h), value=114.0 / 255.0)
            padded_h, padded_w = int(x.shape[-2]), int(x.shape[-1])
            raw = self.head(self._features(x), imgsz=(float(padded_w), float(padded_h)), use_o2m=False)
            boxes = raw[..., self.nc:]
            if pad_h or pad_w:
                boxes = boxes * boxes.new_tensor([
                    padded_w / width, padded_h / height, padded_w / width, padded_h / height,
                ])
        boxes = torch.nan_to_num(_clip_norm_xywh(boxes), nan=0.0, posinf=1.0, neginf=0.0)
        return torch.cat((raw[..., :self.nc], boxes), dim=-1)

    def strip_o2m_for_inference(self):
        self.head.strip_o2m_for_inference()
        return self

    def fuse(self):
        if self.training:
            raise RuntimeError("Call model.eval() before fusing Conv-BN modules.")
        if self._fused:
            return self
        for module in self.modules():
            if isinstance(module, RepConv):
                module.reparameterize()
        for module in self.modules():
            conv = getattr(module, "conv", None)
            bn = getattr(module, "bn", None)
            if isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d):
                module.conv = _fuse_conv_bn(conv, bn)
                module.bn = nn.Identity()
        self._fused = True
        return self


def build_v1_1_model(model_id: str = "fotonetn", nc: int = 80, imgsz: int = 640, **kwargs) -> V11Detector:
    return V11Detector(model_id=model_id, nc=nc, imgsz=imgsz, **kwargs)


def build_from_config(model_id: str, config: dict, nc: int = None, imgsz: int = None) -> V11Detector:
    cfg = dict(config or {})
    spec = get_spec(model_id)
    if str(cfg.get("graph_version", GRAPH_VERSION)) != GRAPH_VERSION:
        raise ValueError(
            f"This checkpoint was built by v1.1 graph {cfg.get('graph_version')!r}; this package builds "
            f"{GRAPH_VERSION!r}. Unreleased draft checkpoints do not load into the current layout."
        )
    if nc is None:
        nc = int(cfg.get("nc", spec.nc))
    if imgsz is None:
        imgsz = int(cfg.get("imgsz", spec.imgsz))
    return V11Detector(
        model_id,
        nc=int(nc),
        imgsz=int(imgsz),
        reg_max=cfg.get("reg_max", spec.reg_max),
        o2m_mult=float(cfg.get("o2m_mult", 1.5)),
        o2o_groups=int(cfg.get("o2o_groups", 1)),
        aux_objectness=bool(cfg.get("aux_objectness", True)),
        **{key: cfg.get(key) for key in _LAYOUT_KEYS},
    )


def fingerprint_for_config(model_id: str, config: dict) -> str:
    model = build_from_config(model_id, config)
    return model.architecture_fingerprint


__all__ = [
    "V11Detector",
    "DetectHead",
    "TaskHead",
    "Backbone",
    "Neck",
    "CSPStar",
    "StarBlock",
    "RepConv",
    "Attention",
    "PSA",
    "PSABlock",
    "SPPF",
    "Conv",
    "FEATURE_STRIDES",
    "TRAINING_ONLY_PREFIXES",
    "build_v1_1_model",
    "build_from_config",
    "fingerprint_for_config",
]
