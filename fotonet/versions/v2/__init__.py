"""fotonet v2 architecture family (fotonet2n / Hi Bird).

Hybrid CNN + Area-Attention (R-ELAN) + HyperACE/FullPAD-Lite + Top-300 GeoSA Query Refinement
detector. Trains with trainer profile ``2.0`` and shares the v1.1 loader, exporters, and
checkpoint contract. Eval output is single-pass NMS-free ``[B, 300, nc + 4]``: raw class logits
plus normalized cxcywh.
"""
from __future__ import annotations

from fotonet.registry import part

from .graph import (
    RepBottleneck,
    V2SPPF,
    build_from_config,
    build_v2_model,
)
from .hybrid import (
    AreaAttention,
    GeoSALayer,
    HybridStage0Branch,
    HyperACE,
    RELANBlock,
    UnifiedRELANStage,
    V2HybridBackbone,
    V2HybridDetector,
    V2HybridHead,
    V2HybridNeck,
)
from .spec import (
    FAMILY,
    GRAPH_VERSION,
    V2_MODEL_IDS,
    V2Spec,
    architecture_fingerprint,
    fingerprint_matches,
    get_spec,
)

# Public alias
V2Detector = V2HybridDetector

# name, version, kind, implementation
_BUILTIN = (
    ("nn.rep_bottleneck", "2.0.0", "block", RepBottleneck),
    ("nn.area_attention", "2.0.0", "block", AreaAttention),
    ("nn.relan", "2.0.0", "block", UnifiedRELANStage),
    ("nn.hyper_ace", "2.0.0", "block", HyperACE),
    ("nn.geosa", "2.0.0", "block", GeoSALayer),
    ("backbone.v2", "2.0.0", "backbone", V2HybridBackbone),
    ("neck.v2", "2.0.0", "neck", V2HybridNeck),
    ("head.v2", "2.0.0", "head", V2HybridHead),
)
for _name, _version, _kind, _obj in _BUILTIN:
    if getattr(_obj, "part_name", None) is None:
        part(_name, _version, _kind)(_obj)

__all__ = [
    "FAMILY",
    "GRAPH_VERSION",
    "V2_MODEL_IDS",
    "V2Spec",
    "get_spec",
    "architecture_fingerprint",
    "fingerprint_matches",
    "V2Detector",
    "V2HybridDetector",
    "V2HybridBackbone",
    "V2HybridNeck",
    "V2HybridHead",
    "AreaAttention",
    "RELANBlock",
    "UnifiedRELANStage",
    "HyperACE",
    "GeoSALayer",
    "HybridStage0Branch",
    "RepBottleneck",
    "V2SPPF",
    "build_v2_model",
    "build_from_config",
]
