"""fotonet model-version registry.

Model id -> architecture version mapping:

    fotonete   -> 1.0  (``fotonet/versions/v1/`` facade; the canonical
                 implementation lives, unchanged, in ``fotonet.models.e``)
    fotonetn   -> 1.1  (``fotonet/versions/v1_1/``)
    fotonets   -> 1.1
    fotonetm   -> 1.1
    fotonetl   -> 1.1

The v1.1 family is fotonetn/s/m/l: a dense star-block detector with direct
box regression, a deployed one-to-one head, and a training-only one-to-many
head. See ``fotonet/versions/v1_1/``.
"""
from __future__ import annotations

VERSION_1_0 = "1.0"
VERSION_1_1 = "1.1"

MODEL_VERSIONS: dict[str, str] = {
    "fotonete": VERSION_1_0,
    "fotonetn": VERSION_1_1,
    "fotonets": VERSION_1_1,
    "fotonetm": VERSION_1_1,
    "fotonetl": VERSION_1_1,
}

V11_MODEL_IDS = ("fotonetn", "fotonets", "fotonetm", "fotonetl")


def resolve_version(model_id) -> str:
    """Return the architecture version string for a model id."""
    return MODEL_VERSIONS.get(str(model_id).strip().lower(), VERSION_1_0)


def is_v11_model(model_ref) -> bool:
    """True when a model reference names a v1.1 family model id.

    Checkpoint paths and ``fotonete`` resolve to False, so every v1
    entrypoint (including resume-from-checkpoint training) keeps its
    existing behavior untouched.
    """
    raw = str(model_ref)
    if raw.lower().endswith((".pt", ".onnx", ".engine", ".torchscript", ".yaml")):
        return False
    return raw.strip().lower() in V11_MODEL_IDS


__all__ = ["VERSION_1_0", "VERSION_1_1", "MODEL_VERSIONS", "V11_MODEL_IDS",
           "resolve_version", "is_v11_model"]
