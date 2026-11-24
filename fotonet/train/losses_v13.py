"""Detection loss for the fotonet v1.3 trainer: ``loss.dual_o2o_o2m@1.2`` plus self-distillation.

The one-to-many (O2M) head with NMS scores above the deployed one-to-one (O2O) head on the same
features, and its class scores rank boxes better. ``o2o_distill`` blends the O2M head's own detached
class probabilities into the O2O class target, on the O2O head's positive anchors only:

    target = (1 - o2o_distill) * target + o2o_distill * sigmoid(o2m logits)

Negative anchors keep a target of 0, so the push that removes duplicates does not change. The
teacher is the model's own training head: no outside weights. With ``o2o_distill`` 0, or on a step
where the O2M head did not run, this is exactly ``loss.dual_o2o_o2m@1.2``.
"""
from __future__ import annotations

import torch

from .losses_v2 import loss_fn_v2


def loss_fn_v13(
    preds_logits_o2o,
    preds_boxes_o2o,
    tgt_o2o,
    wgt_o2o,
    sb_o,
    sa_o,
    sg_o,
    rw_o,
    gtmap,
    preds_logits_o2m=None,
    preds_boxes_o2m=None,
    tgt_o2m=None,
    wgt_o2m=None,
    sb_m=None,
    sa_m=None,
    sg_m=None,
    rw_m=None,
    o2o_distill=0.0,
    **kwargs,
):
    """``loss.dual_o2o_o2m@1.2`` with the O2M scores blended into the O2O positives' class targets."""
    weight = float(o2o_distill)
    if weight > 0.0 and preds_logits_o2m is not None:
        teacher = preds_logits_o2m.detach().sigmoid()
        positive = (wgt_o2o > 0).unsqueeze(-1)
        tgt_o2o = torch.where(positive, (1.0 - weight) * tgt_o2o + weight * teacher, tgt_o2o)
    return loss_fn_v2(
        preds_logits_o2o, preds_boxes_o2o, tgt_o2o, wgt_o2o, sb_o, sa_o, sg_o, rw_o, gtmap,
        preds_logits_o2m, preds_boxes_o2m, tgt_o2m, wgt_o2m, sb_m, sa_m, sg_m, rw_m, **kwargs,
    )


__all__ = ["loss_fn_v13"]
