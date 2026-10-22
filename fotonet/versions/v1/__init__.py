"""fotonete v1 — the released, frozen architecture (version 1.0).

This package is the *versioned facade* for everything fotonete.  The
canonical implementation stays exactly where it shipped —
``fotonet.models.e`` (graph/spec/registry), ``fotonet.utils.loss``,
``fotonet.utils.matcher``, ``fotonet.training.protocols.v1`` and
``fotonet.config.recipes.fotonete_scratch`` — so every released
checkpoint, fingerprint, and command keeps working byte-identically:

    fotonet train model="C:\\...\\fotonet_last.pt" data="...coco.yaml" \\
        recipe=fotonete_scratch resume="C:\\...\\fotonet_last.pt"

runs this version, unchanged, through the original code paths.

Why a facade instead of a physical move: v1's modules are imported from
across the released package; relocating them would rewrite hundreds of
import paths (and re-serialize released machinery) for zero behavioral
gain.  The re-exports below are the stable version-1 API surface.
"""
from __future__ import annotations

from fotonet.models.e.graph import Detector, Backbone, Neck, Head  # noqa: F401
from fotonet.models.e.spec import (  # noqa: F401
    ExperimentSpec,
    get_experiment_spec,
)
from fotonet.models.e.registry import (  # noqa: F401
    available_models,
    model_generation,
    normalize_model_id,
)
from fotonet.utils.loss import DualLoss, get_loss  # noqa: F401
from fotonet.utils.matcher import (  # noqa: F401
    TaskAlignedAssigner,
    TaskAlignedO2OAssigner,
)
from fotonet.fast import FastPredictor  # noqa: F401

VERSION = "1.0"
RECIPE = "fotonete_scratch"

__all__ = [
    "VERSION", "RECIPE",
    "Detector", "Backbone", "Neck", "Head",
    "ExperimentSpec", "get_experiment_spec",
    "available_models", "model_generation", "normalize_model_id",
    "DualLoss", "get_loss",
    "TaskAlignedAssigner", "TaskAlignedO2OAssigner",
    "FastPredictor",
]
