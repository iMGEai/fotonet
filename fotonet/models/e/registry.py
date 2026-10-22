"""Canonical fotonete model registry."""

from __future__ import annotations

import os
from pathlib import Path
import re

import yaml

from .spec import EXPERIMENT_SCHEMA, experiment_fingerprint, get_experiment_spec


MODEL_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config" / "models"

# fotonete is one architecture, not a family of them. A trailing digit selects a
# *generation* of trained weights for that same architecture, not a different
# graph: "fotonete" is generation 1, "fotonete2" is generation 2, and so on.
# Each generation is published as its own v<generation>.0.0 GitHub release, so a
# generation can be re-pointed at newer weights later without stranding anyone
# who deliberately asked for an earlier one.
MODEL_ID_PATTERN = re.compile(r"^fotonete([2-9]|[1-9][0-9]+)?$")
BASE_MODEL_ID = "fotonete"
MODEL_IDS = (BASE_MODEL_ID,)


def available_models() -> tuple[str, ...]:
    return MODEL_IDS


def model_generation(model_id) -> int:
    """Return the weight generation a model id refers to (fotonete -> 1)."""
    match = MODEL_ID_PATTERN.match(str(model_id).strip().lower())
    if match is None:
        raise ValueError(
            f"Unknown model id '{model_id}'. Expected e.g. 'fotonete' or 'fotonete2'."
        )
    return int(match.group(1)) if match.group(1) else 1


def normalize_model_id(model_ref) -> str | None:
    if model_ref is None:
        return None
    raw = os.fspath(model_ref).strip().lower()
    stem = Path(raw).name
    for suffix in (".yaml", ".yml"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    if MODEL_ID_PATTERN.match(stem) is None:
        return None
    return stem


def is_model_ref(model_ref) -> bool:
    return normalize_model_id(model_ref) is not None


def load_model_config(model_ref) -> dict:
    model_id = normalize_model_id(model_ref)
    if model_id is None:
        raise ValueError(f"Unknown model {model_ref!r}. Use one of: {', '.join(MODEL_IDS)}.")
    # Every generation shares one graph definition, so they all read the same
    # config file. Only the weights differ between generations.
    path = MODEL_CONFIG_DIR / f"{BASE_MODEL_ID}.yaml"
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    config = normalize_model_config(data, model_id=model_id, source=str(path))
    # The graph is generation-independent; the id is not, so callers can tell
    # which weights they were handed.
    config["model_id"] = model_id
    return config


def normalize_model_config(data: dict, *, model_id: str | None = None, source=None) -> dict:
    if not isinstance(data, dict):
        raise TypeError("Model configuration must be a mapping.")
    forbidden = set(data) - {
        "nc", "profile", "p2", "reg_max", "quality_head", "architecture_schema"
    }
    if forbidden:
        raise ValueError(
            "Unsupported model configuration fields: " + ", ".join(sorted(forbidden))
        )
    if model_id is not None and normalize_model_id(model_id) is None:
        raise ValueError(f"Invalid model id {model_id!r}.")
    profile = str(data.get("profile", "e")).lower()
    spec = get_experiment_spec(profile)
    p2 = bool(data.get("p2", False))
    quality_head = bool(data.get("quality_head", False))
    if p2:
        raise ValueError("The fotonete graph has no P2 variant.")
    if quality_head:
        raise ValueError("The fotonete graph has no quality head.")
    schema = int(data.get("architecture_schema", EXPERIMENT_SCHEMA))
    if schema != EXPERIMENT_SCHEMA:
        raise ValueError(
            f"Unsupported architecture_schema={schema}; expected {EXPERIMENT_SCHEMA}."
        )
    reg_max = int(data.get("reg_max", 12))
    nc = int(data.get("nc", 80))
    return {
        "model_id": BASE_MODEL_ID,
        "source": source,
        "architecture_schema": schema,
        "architecture_fingerprint": experiment_fingerprint(reg_max=reg_max, nc=nc),
        "profile": spec.profile,
        "nc": nc,
        "p2": False,
        "reg_max": reg_max,
        "quality_head": False,
        "backbone_out_channels": list(spec.widths),
        "neck_out_channels": list(spec.neck_channels),
        "feature_strides": [8, 16, 32],
    }
