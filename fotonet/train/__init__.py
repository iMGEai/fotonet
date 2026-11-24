"""Shared, versioned training parts (losses, assigners, optimizers, EMA, schedules).

First used by fotonet v1.1; the v1.1 trainer imports them from here. The
table below is the manifest of built-in parts; see ``fotonet.registry``.
"""
from fotonet.registry import alias, part

from fotonet.data.gpu_loader import mosaic_geometry_v1, mosaic_geometry_v2, mosaic_geometry_v3

from .assign import matcher_fn, o2o_group_targets, unique_topk
from .assign_v2 import matcher_fn_v2
from .ema import _EMA
from .ema_average import EmaAverager
from .forecast import forecast as _forecast_part
from .losses import bbox_ciou, bbox_siou, loss_fn, o2o_branch_loss, quality_focal_loss, varifocal_loss
from .losses_v2 import loss_fn_v2, o2o_branch_loss_v2
from .losses_v13 import loss_fn_v13
from .losses_hybrid import HybridTalCriterion
from .optim import build_v11_optimizer, MuAdamW, MuonSGD
from .probes import head_matrix, LivePreview
from .schedulers import _epoch_ramp, _lr_at, _lr_at_v2, _lr_at_v3
from .tal import task_aligned_assign

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
    ("aug.mosaic", "1.0.0", "augment", mosaic_geometry_v1),
    # The v1.2 trainer (fotonet/versions/v1_1/trainer/profiles.py, profile "1.2"), versioned by the
    # release they ship in.
    ("assign.dual_o2o_o2m", "1.2", "assigner", matcher_fn_v2),
    ("loss.dual_o2o_o2m", "1.2", "loss", loss_fn_v2),
    ("loss.o2o_branch", "1.2", "loss", o2o_branch_loss_v2),
    ("sched.lr", "1.2", "schedule", _lr_at_v2),
    ("aug.mosaic", "1.2", "augment", mosaic_geometry_v2),
    ("probe.head_matrix", "1.2", "probe", head_matrix),
    ("probe.live_preview", "1.2", "probe", LivePreview),
    ("probe.forecast", "1.2", "probe", _forecast_part),
    # The v1.3 trainer (profile "1.3"). Parts whose math did not change keep their version.
    ("loss.dual_o2o_o2m", "1.3", "loss", loss_fn_v13),
    ("aug.mosaic", "1.3", "augment", mosaic_geometry_v3),
    ("ema.average", "1.3", "ema", EmaAverager),
    ("sched.lr", "1.3", "schedule", _lr_at_v3),
    # The v2 trainer (profile "2.0", fotonet2n / Hi Bird).
    ("assign.tal", "2.0", "assigner", task_aligned_assign),
    ("loss.tal", "2.0", "loss", HybridTalCriterion),
    ("optim.muadamw", "2.0", "optimizer", MuAdamW),
)
for _name, _version, _kind, _obj in _BUILTIN:
    if getattr(_obj, "part_name", None) is None:
        part(_name, _version, _kind)(_obj)

# The names the v1.2 parts had before they took the release version. Run files and checkpoints
# from before the rename (the finished v1.2 nano run among them) name these.
_RENAMED = (
    ("assign.dual_o2o_o2m@2.0.0", "assign.dual_o2o_o2m@1.2"),
    ("loss.dual_o2o_o2m@2.0.0", "loss.dual_o2o_o2m@1.2"),
    ("loss.o2o_branch@2.0.0", "loss.o2o_branch@1.2"),
    ("sched.lr@2.0.0", "sched.lr@1.2"),
    ("aug.mosaic@2.0.0", "aug.mosaic@1.2"),
    ("probe.head_matrix@1.0.0", "probe.head_matrix@1.2"),
    ("probe.live_preview@1.0.0", "probe.live_preview@1.2"),
    ("probe.forecast@1.0.0", "probe.forecast@1.2"),
)
for _old, _new in _RENAMED:
    alias(_old, _new)

__all__ = [
    "HybridTalCriterion", "MuAdamW", "MuonSGD", "bbox_ciou", "bbox_siou", "build_v11_optimizer",
    "loss_fn", "matcher_fn", "o2o_branch_loss", "o2o_group_targets", "quality_focal_loss",
    "unique_topk", "varifocal_loss",
]
