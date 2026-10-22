"""Public fotonet model API.

The ``Fotonet`` class lives in ``core``; each concern (loading, prediction,
training, export, ...) is a mixin module. Old imports from
``fotonet.engine.model`` keep working through these re-exports.
"""
from ._common import _IMAGE_SUFFIXES, _V11_TRAIN_ARGS, _VIDEO_SUFFIXES, UntrainedModelWarning, _records_explicit_arguments  # noqa: F401
from .core import Fotonet

__all__ = ["Fotonet", "UntrainedModelWarning"]
