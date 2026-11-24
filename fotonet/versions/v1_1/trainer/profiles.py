"""Trainer profiles: which part version fills each training slot, and the defaults that come with it.

A profile is a named set of part refs (``fotonet.registry``). ``"1.1"`` is the trainer the published
v1.1 weights were trained with; it stays the default for every recipe and checkpoint that does not
name a profile, and its parts never change. ``"1.2"`` is the v1.2 trainer
(``docs/internal/v1_2_trainer_plan.md``): a new deployed-head loss, a matcher whose one-to-one rule
no longer follows the quality ramp, the WSD decay shape, and the full-canvas mosaic. The model graph
is the same under both. A recipe can override single slots in a ``parts:`` section.
"""
from __future__ import annotations

from fotonet.registry import alias, resolve

PROFILES = {
    "1.1": {
        "assigner": "assign.dual_o2o_o2m@1.0.0",
        "loss": "loss.dual_o2o_o2m@1.0.0",
        "group_assigner": "assign.o2o_groups@1.0.0",
        "group_loss": "loss.o2o_branch@1.0.0",
        "lr_schedule": "sched.lr@1.0.0",
        "mosaic": "aug.mosaic@1.0.0",
    },
    "1.2": {
        "assigner": "assign.dual_o2o_o2m@1.2",
        "loss": "loss.dual_o2o_o2m@1.2",
        # unchanged since v1.1, so it keeps its version
        "group_assigner": "assign.o2o_groups@1.0.0",
        "group_loss": "loss.o2o_branch@1.2",
        "lr_schedule": "sched.lr@1.2",
        "mosaic": "aug.mosaic@1.2",
    },
    # The v1.3 trainer: self-distillation into the deployed head, the nine-image mosaic, and the mean
    # of the last EMA snapshots, and the WSS learning-rate schedule. The other parts' math did not
    # change, so they keep their versions.
    "1.3": {
        "assigner": "assign.dual_o2o_o2m@1.2",
        "loss": "loss.dual_o2o_o2m@1.3",
        "group_assigner": "assign.o2o_groups@1.0.0",
        "group_loss": "loss.o2o_branch@1.2",
        "lr_schedule": "sched.lr@1.3",
        "mosaic": "aug.mosaic@1.3",
        "ema_average": "ema.average@1.3",
    },
    # The v2 trainer (profile "2.0", made for fotonet2n / Hi Bird).
    "2.0": {
        "assigner": "assign.tal@2.0",
        "loss": "loss.tal@2.0",
        "group_assigner": "assign.o2o_groups@1.0.0",
        "group_loss": "loss.o2o_branch@1.2",
        "lr_schedule": "sched.lr@1.3",
        "mosaic": "aug.mosaic@1.3",
        "ema_average": "ema.average@1.3",
        "criterion": "loss.tal@2.0",
    },
}

for _old, _new in (
    ("optim.musgd@2", "optim.muon_sgd@1.0.0"),
    ("assign.tal@2", "assign.tal@2.0"),
    ("loss.tal@2", "loss.tal@2.0"),
    ("assign.tal@2.0.1", "assign.tal@2.0"),
    ("loss.tal@2.0.1", "loss.tal@2.0"),
):
    try:
        alias(_old, _new)
    except ValueError:
        pass

KINDS = {"assigner": "assigner", "loss": "loss", "group_assigner": "assigner", "group_loss": "loss",
         "lr_schedule": "schedule", "mosaic": "augment", "ema_average": "ema", "criterion": "loss"}
# Options whose default depends on the profile; a recipe or the command line still wins.
PROFILE_DEFAULTS = {
    "1.1": {},
    "1.2": {
        "o2o_cls": "bce",
        "o2o_prior_epochs": 6,
        "w_runner_up": 1.5,
        "probe_period": 5,
    },
    "1.3": {
        "o2o_cls": "bce",
        "o2o_prior_epochs": 6,
        "w_runner_up": 1.5,
        "probe_period": 5,
        "val_fused": True,
    },
    "2.0": {
        "optimizer": "muadamw",
        "lr0": 0.01,
        "lrf": 0.005,
        "weight_decay": 0.0005,
        "nbs": 64,
        "loss_scale": 1.0,
        "topk": 10,
        "o2o_candidates": 4,
        "alpha": 1.0,
        "beta": 6.0,
        "w_cls": 0.5,
        "w_box": 7.5,
        "w_l1": 1.5,
        "w_obj": 0.0,
        "w_dec": 1.0,
        "w_cdn": 0.5,
        "mal_gamma": 0.0,
        "prog_o2m_start": 1.0,
        "prog_o2m_end": 0.25,
        "prog_o2m_floor": 0.25,
        "prog_o2o_start": 0.6,
        "prog_total": 1.0,
        "o2o_body_grad_scale": 0.35,
        "val_fused": True,
        "channels_last": True,
        "amp_dtype": "bfloat16",
        "fused": True,
        "gpu_mixup": True,
        "gpu_copy_paste": True,
        "lr_mult": {"adamw": 0.15},
    },
}
DEFAULT_PROFILE = "1.1"


def profile_name(value):
    name = str(value if value not in (None, "") else DEFAULT_PROFILE).strip()
    if name.startswith("v"):
        name = name[1:]
    if name not in PROFILES:
        raise ValueError(f"unknown trainer profile {value!r}; expected one of {', '.join(sorted(PROFILES))}.")
    return name


def after_v11(info) -> bool:
    """Whether a part has the math that came after the v1.1 trainer, so the loop takes its newer path.

    Built-in parts from fotonet 1.2 on carry a release version (``1.2``); a custom part says so with a
    semver major of 2 or more.
    """
    return bool(info.release) or info.major >= 2


def release_at_least(info, release) -> bool:
    """Whether a built-in part carries fotonet release version ``release`` (``"1.3"``) or a later one."""
    if not info.release:
        return False
    return tuple(int(v) for v in info.version.split(".")) >= tuple(int(v) for v in str(release).split("."))


def resolve_parts(profile=None, overrides=None):
    """``{slot: PartInfo}`` for a profile with per-slot overrides (``{"loss": "loss.dual_o2o_o2m@1.0.0"}``)."""
    import fotonet.train  # noqa: F401  (registers the built-in training parts)

    refs = dict(PROFILES[profile_name(profile)])
    for slot, ref in dict(overrides or {}).items():
        if slot not in refs:
            raise ValueError(f"unknown parts slot {slot!r}; expected one of {', '.join(refs)}.")
        refs[slot] = str(ref)
    out = {}
    for slot, ref in refs.items():
        info = resolve(ref)
        if info.kind != KINDS[slot]:
            raise ValueError(f"parts.{slot} needs a {KINDS[slot]} part, got {info.ref} ({info.kind}).")
        out[slot] = info
    return out


__all__ = ["DEFAULT_PROFILE", "PROFILES", "PROFILE_DEFAULTS", "after_v11", "profile_name", "release_at_least",
           "resolve_parts"]
