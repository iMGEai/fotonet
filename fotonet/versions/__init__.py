"""fotonet model-version registry.

Model id -> architecture version mapping:

    fotonete   -> 1.0  (``fotonet/versions/v1/`` facade; the canonical
                 implementation lives, unchanged, in ``fotonet.models.e``)
    fotonetn   -> 1.1  (``fotonet/versions/v1_1/``)
    fotonets   -> 1.1
    fotonetm   -> 1.1
    fotonetl   -> 1.1
    fotonet2n  -> 2.0  (``fotonet/versions/v2/``; alpha, not released)

The v1.1 family is fotonetn/s/m/l: a dense star-block detector with direct
box regression, a deployed one-to-one head, and a training-only one-to-many
head. See ``fotonet/versions/v1_1/``.

The v2 family is named fotonet + version + size (fotonet2n). It keeps the v1.1
output contract and runs through the same trainer, loader and exporters; see
``fotonet/versions/v2/``.
"""
from __future__ import annotations

VERSION_1_0 = "1.0"
VERSION_1_1 = "1.1"
VERSION_2_0 = "2.0"

MODEL_VERSIONS: dict[str, str] = {
    "fotonete": VERSION_1_0,
    "fotonetn": VERSION_1_1,
    "fotonets": VERSION_1_1,
    "fotonetm": VERSION_1_1,
    "fotonetl": VERSION_1_1,
    "fotonetnr": VERSION_1_1,  # alpha layout, not released
    "fotonet2n": VERSION_2_0,
}

V11_MODEL_IDS = ("fotonetn", "fotonets", "fotonetm", "fotonetl")
V11_ALPHA_IDS = ("fotonetnr",)
V2_MODEL_IDS = ("fotonet2n",)
_FILE_SUFFIXES = (".pt", ".onnx", ".engine", ".torchscript", ".yaml")


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
    if raw.lower().endswith(_FILE_SUFFIXES):
        return False
    return raw.strip().lower() in V11_MODEL_IDS + V11_ALPHA_IDS


def is_v2_model(model_ref) -> bool:
    """True when a model reference names a v2 family model id (not a file)."""
    raw = str(model_ref)
    if raw.lower().endswith(_FILE_SUFFIXES):
        return False
    return raw.strip().lower() in V2_MODEL_IDS


def shares_v11_contract(graph_version) -> bool:
    """Whether a graph version uses the v1.1 output and artifact contract (v1.1 and v2 graphs do)."""
    return str(graph_version or "").startswith(("1.1", "2."))


__all__ = ["VERSION_1_0", "VERSION_1_1", "VERSION_2_0", "MODEL_VERSIONS", "V11_MODEL_IDS", "V11_ALPHA_IDS",
           "V2_MODEL_IDS", "resolve_version", "is_v11_model", "is_v2_model", "shares_v11_contract"]
