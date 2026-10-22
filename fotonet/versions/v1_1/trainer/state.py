"""Run state: model resolution and weight transfer, freezing, EMA, RNG, run folders."""
from __future__ import annotations

import os
import random

import numpy as np
import torch

from ..api import (
    checkpoint_names, inference_state_dict, is_v11_checkpoint, load_checkpoint, load_weights,
)
from ..graph import build_from_config, V11Detector
from .options import _BACKBONE_ORDER, _HEAD_OPTION_KEYS

from fotonet.train.ema import _EMA  # noqa: F401


def transfer_weights(source, target):
    """Copy every tensor whose name and shape match from ``source`` into ``target``.

    Used to fine-tune onto a different class count (class predictors keep
    their fresh init) and to rebuild training-only heads (an inference-only
    source has none; they start fresh).
    """
    incoming = source.state_dict()
    state = target.state_dict()
    transferred, reinitialized = [], []
    for key, value in state.items():
        other = incoming.get(key)
        if other is not None and tuple(other.shape) == tuple(value.shape):
            state[key] = other.detach().to(device=value.device, dtype=value.dtype).clone()
            transferred.append(key)
        else:
            reinitialized.append(key)
    target.load_state_dict(state, strict=True)
    return {"transferred": transferred, "reinitialized": reinitialized}


def _resolve_training_model(model, model_id, nc, imgsz, head_options=None):
    """Return ``(model, transfer_report)`` ready to train ``nc`` classes.

    ``model`` may be a ``V11Detector``, a v1.1 checkpoint path (fine-tune init),
    or None to build ``model_id`` from scratch. ``model_id`` may also be a
    checkpoint path. ``head_options`` sets the training-only head (O2M width,
    O2O groups, objectness); a model built differently is rebuilt and its
    deployed weights transferred.
    """
    head_options = {key: value for key, value in dict(head_options or {}).items() if value is not None}
    if model is None or isinstance(model, (str, os.PathLike)):
        ref = os.fspath(model if model is not None else (model_id or "fotonetn"))
        if not os.path.isfile(ref):
            return V11Detector(str(ref).strip().lower(), nc=nc, imgsz=imgsz, **head_options), None
        ckpt = load_checkpoint(ref)
        if not is_v11_checkpoint(ckpt):
            raise SystemExit(f"{ref} is not a fotonet v1.1 checkpoint.")
        cfg = dict(ckpt.get("model_config") or {})
        model = build_from_config(cfg.get("model_id", "fotonetn"), cfg, imgsz=imgsz)
        load_weights(model, inference_state_dict(ckpt))
        model.class_names = checkpoint_names(ckpt, model.nc)
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a v1.1 detector, a v1.1 checkpoint path, or None.")
    wanted = {key: model.model_config.get(key) for key in _HEAD_OPTION_KEYS}
    wanted.update(head_options)
    same_head = all(model.model_config.get(key) == wanted[key] for key in _HEAD_OPTION_KEYS)
    if int(model.nc) == int(nc) and model.head.has_o2m and same_head:
        return model, None
    config = dict(model.model_config)
    config.update(wanted)
    target = build_from_config(model.model_id, config, nc=int(nc), imgsz=imgsz)
    report = transfer_weights(model, target)
    target.trained_weights = bool(getattr(model, "trained_weights", False))
    return target, report


def _freeze(model, spec):
    """Freeze the first N backbone modules (int) or modules by name prefix (list)."""
    if not spec:
        return []
    if isinstance(spec, (int, np.integer)) and not isinstance(spec, bool):
        prefixes = [f"backbone.{name}." for name in _BACKBONE_ORDER[: int(spec)]]
    else:
        prefixes = [str(p).rstrip(".") + "." for p in (spec if isinstance(spec, (list, tuple)) else [spec])]
    frozen = []
    for name, param in model.named_parameters():
        if any(name.startswith(prefix) for prefix in prefixes):
            param.requires_grad_(False)
            frozen.append(name)
    return frozen


def _run_dir(cfg, model_id, resuming):
    """Where this run writes. ``project``/``name`` auto-increment unless ``exist_ok``."""
    project, name = cfg.get("project"), cfg.get("name")
    if project is None and name is None:
        raw = str(cfg["out_dir"])
        if os.path.basename(os.path.normpath(raw)).lower() == model_id.lower():
            return raw
        return os.path.join(raw, model_id)
    path = os.path.join(str(project or cfg["out_dir"]), str(name or model_id))
    if resuming or bool(cfg.get("exist_ok")) or not os.path.exists(path) or not os.listdir(path):
        return path
    index = 2
    while os.path.exists(f"{path}{index}"):
        index += 1
    return f"{path}{index}"


def capture_rng_state(augmenter=None):
    """RNG state as tensors and plain values, so a checkpoint stays safe to load."""
    kind, keys, pos, has_gauss, cached = np.random.get_state()
    state = {
        "torch": torch.get_rng_state(),
        "numpy": {
            "kind": str(kind),
            "keys": torch.from_numpy(np.asarray(keys, dtype=np.int64)),
            "pos": int(pos),
            "has_gauss": int(has_gauss),
            "cached": float(cached),
        },
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if augmenter is not None:
        state["augmenter"] = augmenter.generator.get_state()
    return state


def restore_rng_state(state, augmenter=None):
    if not isinstance(state, dict):
        return
    if "torch" in state:
        torch.set_rng_state(state["torch"].cpu())
    numpy_state = state.get("numpy")
    if numpy_state:
        np.random.set_state((
            numpy_state["kind"],
            numpy_state["keys"].cpu().numpy().astype(np.uint32),
            int(numpy_state["pos"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached"]),
        ))
    if "python" in state:
        version, internal, gauss = state["python"]
        random.setstate((version, tuple(internal), gauss))
    if "cuda" in state and torch.cuda.is_available():
        cuda_states = [s.cpu() for s in state["cuda"]]
        if len(cuda_states) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(cuda_states)
    if augmenter is not None and "augmenter" in state:
        augmenter.generator.set_state(state["augmenter"].cpu())
