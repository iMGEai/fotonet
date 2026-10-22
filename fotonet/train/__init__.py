"""Shared, versioned training parts (losses, assigners, optimizers, EMA, schedules).

First used by fotonet v1.1; the v1.1 trainer imports them from here. The
table below is the manifest of built-in parts; see ``fotonet.registry``.
"""
from fotonet.registry import part

from .assign import matcher_fn, o2o_group_targets, unique_topk
from .ema import _EMA
from .losses import bbox_ciou, bbox_siou, loss_fn, o2o_branch_loss, quality_focal_loss, varifocal_loss
from .optim import build_v11_optimizer, MuonSGD
from .schedulers import _epoch_ramp, _lr_at

# name, version, kind, implementation
_BUILTIN = (
    ("assign.dual_o2o_o2m", "1.0.0", "assigner", matcher_fn),
    ("assign.o2o_groups", "1.0.0", "assigner", o2o_group_targets),
    ("assign.unique_topk", "1.0.0", "assigner", unique_topk),
    ("loss.dual_o2o_o2m", "1.0.0", "loss", loss_fn),
    ("loss.o2o_branch", "1.0.0", "loss", o2o_branch_loss),
    ("loss.varifocal", "1.0.0", "loss", varifocal_loss),
    ("loss.quality_focal", "1.0.0", "loss", quality_focal_loss),
    ("loss.siou", "1.0.0", "loss", bbox_siou),
    ("loss.ciou", "1.0.0", "loss", bbox_ciou),
    ("optim.v11", "1.0.0", "optimizer", build_v11_optimizer),
    ("optim.muon_sgd", "1.0.0", "optimizer", MuonSGD),
    ("ema.foreach", "1.0.0", "ema", _EMA),
    ("sched.lr", "1.0.0", "schedule", _lr_at),
    ("sched.epoch_ramp", "1.0.0", "schedule", _epoch_ramp),
)
for _name, _version, _kind, _obj in _BUILTIN:
    if getattr(_obj, "part_name", None) is None:
        part(_name, _version, _kind)(_obj)

__all__ = [
    "MuonSGD", "bbox_ciou", "bbox_siou", "build_v11_optimizer", "loss_fn", "matcher_fn", "o2o_branch_loss",
    "o2o_group_targets", "quality_focal_loss", "unique_topk", "varifocal_loss",
]
