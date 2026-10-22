"""Moved to ``fotonet.train.losses``; this module keeps the old import path."""
from fotonet.train.losses import (  # noqa: F401
    bbox_ciou, bbox_siou, _box_iou_loss, _masked_box_loss, varifocal_loss, quality_focal_loss,
    o2o_branch_loss, loss_fn,
)
