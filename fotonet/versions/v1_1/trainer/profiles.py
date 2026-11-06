"""Trainer profiles: which part version fills each training slot, and the defaults that come with it.

A profile is a named set of part refs (``fotonet.registry``). ``"1.1"`` is the trainer the published
v1.1 weights were trained with; it stays the default for every recipe and checkpoint that does not
name a profile, and its parts never change. ``"1.2"`` is the v1.2 trainer
(``docs/internal/v1_2_trainer_plan.md``): a new deployed-head loss, a matcher whose one-to-one rule
no longer follows the quality ramp, the WSD decay shape, and the full-canvas mosaic. The model graph
is the same under both. A recipe can override single slots in a ``parts:`` section.
"""
from __future__ import annotations

from fotonet.registry import resolve

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
}
KINDS = {"assigner": "assigner", "loss": "loss", "group_assigner": "assigner", "group_loss": "loss",
         "lr_schedule": "schedule", "mosaic": "augment"}
# Options whose default depends on the profile; a recipe or the command line still wins.
PROFILE_DEFAULTS = {
    "1.1": {},
    "1.2": {
        "o2o_cls": "bce",
        "o2o_prior_epochs": 6,
        "w_runner_up": 1.5,
        "probe_period": 5,
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


__all__ = ["DEFAULT_PROFILE", "PROFILES", "PROFILE_DEFAULTS", "after_v11", "profile_name", "resolve_parts"]
