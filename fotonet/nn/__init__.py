"""Shared, versioned model parts (blocks, backbones, necks, heads).

fotonet v1.1 is assembled from these; later generations reuse them. v1
(``fotonete``) keeps its own frozen copies and never imports this package.
The table below is the manifest of built-in parts and their versions; see
``fotonet.registry`` for the versioning rules.
"""
from fotonet.registry import part

from .attention import Attention, PSA, PSABlock
from .backbones import Backbone
from .blocks import CSPStar, SPPF, StarBlock
from .conv import Conv, fuse_conv_bn, RepConv
from .heads import DetectHead, PredictorPair, TaskHead
from .necks import Neck

# name, version, kind, implementation
_BUILTIN = (
    ("nn.conv", "1.0.0", "block", Conv),
    ("nn.repconv", "1.0.0", "block", RepConv),
    ("nn.starblock", "1.0.0", "block", StarBlock),
    ("nn.cspstar", "1.0.0", "block", CSPStar),
    ("nn.sppf", "1.0.0", "block", SPPF),
    ("nn.attention", "1.0.0", "block", Attention),
    ("nn.psa_block", "1.0.0", "block", PSABlock),
    ("nn.psa", "1.0.0", "block", PSA),
    ("backbone.cspstar", "1.0.0", "backbone", Backbone),
    ("neck.pan_cspstar", "1.0.0", "neck", Neck),
    ("head.task_towers", "1.0.0", "head", TaskHead),
    ("head.predictor_pair", "1.0.0", "head", PredictorPair),
    ("head.o2o_o2m", "1.0.0", "head", DetectHead),
)
for _name, _version, _kind, _obj in _BUILTIN:
    if getattr(_obj, "part_name", None) is None:
        part(_name, _version, _kind)(_obj)

__all__ = [
    "Attention", "Backbone", "CSPStar", "Conv", "DetectHead", "Neck", "PSA", "PSABlock", "PredictorPair",
    "RepConv", "SPPF", "StarBlock", "TaskHead", "fuse_conv_bn",
]
