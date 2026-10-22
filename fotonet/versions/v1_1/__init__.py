"""fotonet v1.1 architecture family (fotonetn, fotonets, fotonetm, fotonetl).

CSP-Star CNN with attention at the deepest stages, direct LTRB regression, a
deployed one-to-one head, and training-only one-to-many supervision.
Eval output is [B, N, nc + 4] raw logits plus normalized cxcywh.
"""
from __future__ import annotations

from .graph import (
    Backbone,
    CSPStar,
    DetectHead,
    Neck,
    PSA,
    RepConv,
    SPPF,
    StarBlock,
    V11Detector,
    build_from_config,
    build_v1_1_model,
    fingerprint_for_config,
)
from .spec import (
    FAMILY,
    GRAPH_VERSION,
    V11_MODEL_IDS,
    V11Spec,
    get_spec,
    graph_fingerprint,
)

__all__ = [
    "FAMILY",
    "GRAPH_VERSION",
    "V11_MODEL_IDS",
    "V11Spec",
    "get_spec",
    "graph_fingerprint",
    "V11Detector",
    "DetectHead",
    "Backbone",
    "Neck",
    "CSPStar",
    "StarBlock",
    "RepConv",
    "PSA",
    "SPPF",
    "build_v1_1_model",
    "build_from_config",
    "fingerprint_for_config",
    "FotonetV11",
    "load_v11",
    "train",
]


def __getattr__(name: str):
    if name in ("FotonetV11", "load_v11"):
        from .api import FotonetV11, load_v11
        return {"FotonetV11": FotonetV11, "load_v11": load_v11}[name]
    if name in ("train", "train_cli"):
        from .engine import train, train_cli
        return {"train": train, "train_cli": train_cli}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
