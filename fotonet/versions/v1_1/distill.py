"""Moved to ``fotonet.train.distill``; this module keeps the old import path."""
from fotonet.train.distill import (  # noqa: F401
    CACHE_FORMAT, CODE_STRIDE, WEIGHT_STEPS, encode_labels, decode_labels, _iou_xywh,
    TeacherCache, build_cache,
)
