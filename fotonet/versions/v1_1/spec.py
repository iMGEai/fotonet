"""fotonet v1.1 family specifications.

Architecture scales: fotonetn, fotonets, fotonetm, fotonetl. Each tier is
sized to land just under its YOLO27 cost gate (see ``targets.py``), fused,
training-only branches removed, 640x640.

Layout per tier:
- stem and P2 widths, P3/P4/P5 widths;
- CSP-Star depths for P2, P3, P4, P5 (a depth of 0 skips the stage block);
- neck widths and CSP-Star depths for the P3/P4/P5 junctions;
- deployed head hidden widths (regression, classification);
- attention blocks at P5 and P4;
- the pyramid levels the head predicts on.

Direct LTRB regression (reg_max is fixed at 1). The fingerprint is the built
layout, not the family name alone.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json


V11_MODEL_IDS = ("fotonetn", "fotonets", "fotonetm", "fotonetl")
LEVEL_STRIDES = {3: 8, 4: 16, 5: 32}

FINGERPRINT_KEYS = (
    "model_id",
    "stem_channels",
    "p2_channels",
    "widths",
    "depths",
    "neck_channels",
    "neck_depths",
    "head_hidden",
    "attn_p5",
    "attn_p4",
    "head_levels",
    "nc",
    "reg_max",
)


@dataclass(frozen=True)
class V11Spec:
    """One v1.1 detector profile."""

    model_id: str = "fotonetn"
    stem_channels: int = 16
    p2_channels: int = 48
    widths: tuple[int, int, int] = (64, 128, 256)
    depths: tuple[int, int, int, int] = (0, 2, 2, 1)
    neck_channels: tuple[int, int, int] = (64, 128, 256)
    neck_depths: tuple[int, int, int] = (1, 1, 1)
    head_hidden: tuple[int, int] = (48, 96)
    attn_p5: int = 1
    attn_p4: int = 0
    head_levels: tuple[int, ...] = (3, 4, 5)
    reg_max: int = 1
    nc: int = 80
    imgsz: int = 640

    def __post_init__(self) -> None:
        if self.model_id not in V11_MODEL_IDS:
            raise ValueError(
                f"Unknown v1.1 model id {self.model_id!r}; expected one of {V11_MODEL_IDS}."
            )
        validate_layout(
            stem_channels=self.stem_channels, p2_channels=self.p2_channels, widths=self.widths,
            depths=self.depths, neck_channels=self.neck_channels, neck_depths=self.neck_depths,
            head_hidden=self.head_hidden, attn_p5=self.attn_p5, attn_p4=self.attn_p4,
            head_levels=self.head_levels, nc=self.nc, reg_max=self.reg_max,
        )


def validate_layout(*, stem_channels, p2_channels, widths, depths, neck_channels, neck_depths,
                    head_hidden, attn_p5, attn_p4, head_levels, nc, reg_max):
    if len(widths) != 3 or len(neck_channels) != 3 or len(neck_depths) != 3:
        raise ValueError("v1.1 requires P3/P4/P5 widths, neck widths, and neck depths.")
    if len(depths) != 4:
        raise ValueError("v1.1 requires CSP-Star depths for P2, P3, P4, and P5.")
    if len(head_hidden) != 2:
        raise ValueError("v1.1 head_hidden is (regression width, classification width).")
    positive = (stem_channels, p2_channels, nc, *widths, *neck_channels, *head_hidden)
    if any(int(v) <= 0 for v in positive):
        raise ValueError("v1.1 widths and class counts must be positive.")
    if any(int(v) < 0 for v in (*depths, *neck_depths, attn_p5, attn_p4)):
        raise ValueError("v1.1 depths and attention block counts must be non-negative.")
    if any(int(v) < 1 for v in neck_depths):
        raise ValueError("v1.1 neck junctions need at least one CSP-Star block.")
    if any(int(c) % 4 for c in (*widths, *neck_channels)):
        raise ValueError("v1.1 stage and neck widths must be divisible by 4.")
    levels = tuple(int(v) for v in head_levels)
    if levels != tuple(sorted(set(levels))) or not set(levels) <= set(LEVEL_STRIDES) or len(levels) < 2:
        raise ValueError("v1.1 head_levels must be at least two sorted levels from (3, 4, 5).")
    if int(reg_max) != 1:
        raise ValueError("v1.1 uses direct LTRB regression; reg_max must be 1.")


# 1.1.0 was the unreleased all-StarBlock draft; its checkpoints do not load
# into this layout and are refused by fingerprint.
GRAPH_VERSION = "1.1.1"


def _load_family() -> dict[str, V11Spec]:
    """Read the official tier layouts from ``fotonet/config/models/v1_1/*.yaml``."""
    from fotonet.config.model_recipes import load_model_recipe

    family = {}
    for model_id in V11_MODEL_IDS:
        recipe = load_model_recipe(model_id, generation="v1.1")
        if recipe.get("model_id") != model_id or str(recipe.get("graph_version")) != GRAPH_VERSION:
            raise ValueError(f"model recipe for {model_id} does not match v1.1 graph {GRAPH_VERSION}.")
        layout = {key: tuple(value) if isinstance(value, list) else value
                  for key, value in dict(recipe["layout"]).items()}
        family[model_id] = V11Spec(model_id=model_id, **layout)
    return family


# The official tiers. Their layouts live in YAML; the fingerprint is unchanged
# because it hashes the layout values, not where they are written.
FAMILY: dict[str, V11Spec] = _load_family()


def get_spec(model_id: str) -> V11Spec:
    key = str(model_id).strip().lower()
    if key not in FAMILY:
        raise ValueError(
            f"Unknown v1.1 model id {model_id!r}; expected one of {sorted(FAMILY)}."
        )
    return FAMILY[key]


def _as_jsonable(value):
    if isinstance(value, (tuple, list)):
        return [_as_jsonable(item) for item in value]
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return int(value)
    return value


def architecture_fingerprint(config: dict) -> str:
    """SHA256 of the layout that was actually built.

    Image size is not part of the identity: the same weights run at any
    square resolution divisible by 32.
    """
    payload = {"graph_version": GRAPH_VERSION}
    for key in FINGERPRINT_KEYS:
        if key not in config:
            raise KeyError(f"v1.1 fingerprint config is missing {key!r}.")
        payload[key] = _as_jsonable(config[key])
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(blob).hexdigest()


def spec_config(spec: V11Spec) -> dict:
    return {key: getattr(spec, key) for key in FINGERPRINT_KEYS}


def graph_fingerprint(spec: V11Spec) -> str:
    """Fingerprint of an official family member at its default class count."""
    return architecture_fingerprint(spec_config(spec))


def fingerprint_matches(config: dict, stored: str) -> bool:
    if not stored or not isinstance(config, dict):
        return False
    try:
        return stored == architecture_fingerprint(config)
    except (KeyError, TypeError, ValueError):
        return False


__all__ = [
    "V11Spec",
    "FAMILY",
    "GRAPH_VERSION",
    "V11_MODEL_IDS",
    "LEVEL_STRIDES",
    "FINGERPRINT_KEYS",
    "get_spec",
    "graph_fingerprint",
    "architecture_fingerprint",
    "fingerprint_matches",
    "spec_config",
    "validate_layout",
]
