"""Moved to ``fotonet.train.assign``; this module keeps the old import path."""
from fotonet.train.assign import (  # noqa: F401
    DIRECT_ASSIGN_CELLS, xywh_to_xyxy_t, bbox_iou_t, _candidate_mask, unique_topk, _dense_targets,
    _Geometry, _o2o_targets, matcher_fn, o2o_group_targets,
)
