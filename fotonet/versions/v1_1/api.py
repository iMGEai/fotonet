"""Fotonet v1.1 checkpoint, export, and Python API."""
from __future__ import annotations

import copy
import json
import os

import torch

from fotonet.engine.model import Fotonet
from fotonet.registry import lockfile
from fotonet.utils.config import default_class_names

from .graph import FEATURE_STRIDES, TRAINING_ONLY_PREFIXES
from .spec import FAMILY, GRAPH_VERSION

V11_FORMAT = "fotonet-v1.1"


def load_checkpoint(path, map_location="cpu"):
    """Read a checkpoint with the tensor-only unpickler.

    Checkpoints hold tensors and plain Python values only. Arbitrary pickled
    objects are refused, so loading an untrusted file cannot run code.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception as exc:
        raise ValueError(
            f"Could not safely load {os.fspath(path)!r}; fotonet checkpoints must contain "
            "tensors and plain Python values only."
        ) from exc


def is_v11_checkpoint(ckpt):
    return isinstance(ckpt, dict) and ckpt.get("format") == V11_FORMAT


def _peek(path):
    """Return a v1.1 checkpoint, or None for anything else (including unsafe files)."""
    try:
        ckpt = load_checkpoint(path)
    except ValueError:
        return None
    return ckpt if is_v11_checkpoint(ckpt) else None


def checkpoint_names(ckpt, nc):
    """Class names stored in a checkpoint, or placeholders when it has none."""
    from fotonet.utils.config import normalize_class_schema

    raw = ckpt.get("names") if isinstance(ckpt, dict) else None
    if raw:
        try:
            _, names = normalize_class_schema(int(nc), raw, context="checkpoint")
            return dict(names)
        except (TypeError, ValueError):
            pass
    return default_class_names(int(nc))


def _cast_state(model, state):
    ref = next(model.parameters())
    out = {}
    for key, value in state.items():
        if not torch.is_tensor(value):
            continue
        value = value.detach()
        if value.is_floating_point():
            out[key] = value.to(device=ref.device, dtype=ref.dtype)
        else:
            out[key] = value.to(device=ref.device)
    return out


def inference_state_dict(ckpt):
    """Weights for prediction. An EMA snapshot wins over the live training weights."""
    base = dict(ckpt.get("model") or {})
    ema = ckpt.get("ema")
    if isinstance(ema, dict) and ema.get("params"):
        merged = dict(base)
        merged.update({key: value for key, value in ema["params"].items()})
        merged.update({key: value for key, value in (ema.get("buffers") or {}).items()})
        return merged
    return base


def _is_training_only(key):
    return str(key).startswith(TRAINING_ONLY_PREFIXES)


def load_weights(model, state):
    """Load a state dict, casting dtype and dropping training-only heads the file lacks.

    An inference-only checkpoint carries no O2M head, objectness, or extra
    O2O groups; the model is stripped to match and every deployed tensor must
    still load exactly.
    """
    if not any(_is_training_only(key) for key in state):
        model.strip_o2m_for_inference()
    model.load_state_dict(_cast_state(model, state), strict=True)
    # Training reads this: a model that starts from trained weights gets its
    # IoU-quality targets from the first epoch instead of ramping them in.
    model.trained_weights = True
    return model


def save_payload(model, path, *, epoch=None, ema=None, optimizer=None, best_map=None,
                 inference_only=False, extra=None, state=None, names=None, half=False):
    """Write a v1.1 checkpoint that ``load_checkpoint`` can read back safely.

    ``half`` stores the model's floating-point tensors as FP16 (half the file); the loader casts
    them back to the model's dtype. Meant for inference-only files such as released weights.
    """
    state = model.state_dict() if state is None else state
    if inference_only:
        state = {key: value for key, value in state.items() if not _is_training_only(key)}
    if half:
        state = {key: value.half() if torch.is_tensor(value) and value.is_floating_point() else value
                 for key, value in state.items()}
    if names is None:
        names = getattr(model, "class_names", None)
    payload = {
        "format": V11_FORMAT,
        "model_config": dict(model.model_config),
        "graph_version": model.graph_version,
        "architecture_fingerprint": model.architecture_fingerprint,
        "epoch": epoch,
        "best_map": best_map,
        "inference_only": bool(inference_only),
        "model": state,
        # Lockfile: the exact part versions this file was produced with.
        "parts": lockfile([model]),
    }
    if names:
        payload["names"] = {int(key): str(value) for key, value in dict(names).items()}
    if ema is not None:
        payload["ema"] = ema
    if optimizer is not None:
        payload["optimizer"] = optimizer
    if extra:
        extra = dict(extra)
        payload["parts"] = dict(sorted({**payload["parts"], **dict(extra.pop("parts", None) or {})}.items()))
        payload.update(extra)
    torch.save(payload, path)
    return path


def validate_v11_metadata(metadata, expected_format):
    """Check a v1.1 export sidecar. This contract does not use the v1 runtime fields."""
    if not isinstance(metadata, dict):
        raise RuntimeError("v1.1 artifact metadata must be a JSON object.")
    if not str(metadata.get("graph_version", "")).startswith("1.1"):
        raise RuntimeError("v1.1 artifact metadata has the wrong graph version.")
    if metadata.get("format") not in (None, expected_format):
        raise RuntimeError(
            f"Artifact metadata format={metadata.get('format')!r} does not match {expected_format!r}."
        )
    for field in ("model_id", "architecture_fingerprint", "nc", "imgsz", "strides"):
        if metadata.get(field) in (None, ""):
            raise RuntimeError(f"v1.1 artifact metadata is missing {field!r}.")
    imgsz = metadata["imgsz"]
    if not isinstance(imgsz, (list, tuple)) or len(imgsz) != 2:
        raise RuntimeError("v1.1 artifact imgsz must be [height, width].")
    strides = [int(value) for value in metadata["strides"]]
    if len(strides) < 2 or strides != sorted(set(strides)) or not set(strides) <= set(FEATURE_STRIDES):
        raise RuntimeError(f"v1.1 artifact strides must be a sorted subset of {list(FEATURE_STRIDES)}.")
    out = dict(metadata)
    out["nc"] = int(metadata["nc"])
    out["imgsz"] = [int(imgsz[0]), int(imgsz[1])]
    out["strides"] = strides
    out["batch"] = int(metadata.get("batch", 1))
    out["dynamic"] = bool(metadata.get("dynamic", False))
    return out


def _anchor_count(height, width, strides):
    top = max(FEATURE_STRIDES)
    padded_h = -(-int(height) // top) * top
    padded_w = -(-int(width) // top) * top
    return sum((padded_h // s) * (padded_w // s) for s in strides)


def v11_metadata(model_api, fmt, imgsz, batch, dynamic, half, extra=None):
    """Sidecar contract for a v1.1 artifact (what the v1.1 loader validates)."""
    h, w = (int(imgsz), int(imgsz)) if isinstance(imgsz, int) else (int(imgsz[0]), int(imgsz[1]))
    model = model_api.model
    names = getattr(model_api, "names", None)
    if isinstance(names, dict):
        names = {str(key): str(value) for key, value in names.items()}
    strides = [int(s) for s in getattr(getattr(model, "head", None), "strides", FEATURE_STRIDES)]
    data = {
        "graph_version": GRAPH_VERSION,
        "format": fmt,
        "model_id": str(model_api.model_id),
        "architecture_fingerprint": str(model_api.architecture_fingerprint),
        "nc": int(model_api.nc),
        "names": names,
        "imgsz": [h, w],
        "batch": int(batch),
        "dynamic": bool(dynamic),
        "half": bool(half),
        "strides": strides,
        "reg_max": 1,
        "anchor_count": _anchor_count(h, w, strides),
        "nms_free": True,
        "input_layout": "NCHW",
        "input_color": "RGB",
        "input_range": "[0,1]",
        "output": "[batch, anchors, nc + 4] raw logits + normalized cxcywh",
    }
    if extra:
        data.update(extra)
    return data


class _ExportProxy:
    """What the shared exporters need: a fused, O2M-free deploy graph plus identity."""

    is_v1_1 = True
    reg_max = 1
    use_p2 = False

    def __init__(self, model_api, device):
        self.device = torch.device(device)
        self.model_id = model_api.model.model_id
        self.architecture_fingerprint = model_api.model.architecture_fingerprint
        self.graph_version = model_api.model.graph_version
        self.nc = int(model_api.model.nc)
        self.names = dict(getattr(model_api, "names", None) or default_class_names(self.nc))
        self.model = _deploy_graph(model_api.model, self.device)


def deploy_for_inference(model, device=None, *, dtype=None):
    """Fused, O2M-stripped eval copy on ``device``; ``model`` is untouched."""
    raw = getattr(model, "_orig_mod", model)
    ref = next(raw.parameters(), None)
    out_dtype = (torch.float32 if ref is None else ref.dtype) if dtype is None else dtype
    target = (ref.device if ref is not None else torch.device("cpu")) if device is None else torch.device(device)
    fold_dtype = torch.float64 if out_dtype == torch.float64 else torch.float32
    deploy = copy.deepcopy(raw).to(device=target, dtype=fold_dtype).eval()
    deploy.strip_o2m_for_inference()
    deploy.fuse()
    return deploy.to(dtype=out_dtype) if out_dtype != fold_dtype else deploy


def _deploy_graph(model, device, imgsz=320):
    """Stripped and fused copy, checked to give the same output as the live model.

    The check runs in float64: in float32 on a GPU, convolutions use TF32 by
    default, and with trained weights its rounding alone can exceed any
    sensible tolerance, failing correct exports at random.
    """
    raw = getattr(model, "_orig_mod", model)
    ref = copy.deepcopy(raw).to(device).eval().double()
    deploy = deploy_for_inference(raw, device=device, dtype=torch.float64)
    probe = torch.rand(1, 3, imgsz, imgsz, device=device, dtype=torch.float64)
    with torch.no_grad():
        if not torch.allclose(ref(probe), deploy(probe), rtol=1e-6, atol=1e-6):
            raise RuntimeError("Stripping and folding the training graph changed the deployed output.")
    return deploy.float()


_FORMAT_ALIASES = {"ts": "torchscript", "jit": "torchscript", "engine": "tensorrt", "trt": "tensorrt"}
_DEFAULT_SUFFIX = {"onnx": "onnx", "torchscript": "torchscript", "tensorrt": "engine", "coreml": "mlpackage"}


def export_v11(model_api, path=None, format="onnx", imgsz=640, batch=1,
               dynamic=False, half=False, simplify=True, opset=17,
               device=None, verify=True, int8=False, **kwargs):
    """Export the deployed graph through the shared exporters with a v1.1 sidecar.

    Formats: ``onnx`` (optionally ``int8=True`` with ``calibration_data``),
    ``torchscript``, ``tensorrt`` (needs ``trtexec``), and ``coreml`` (needs
    coremltools). Returns ``{"artifact", "metadata" (dict), "metadata_path"}``.
    """
    from fotonet.export.backends import export_model

    fmt = _FORMAT_ALIASES.get(str(format).lower(), str(format).lower())
    if path is None:
        path = f"{model_api.model.model_id}.{_DEFAULT_SUFFIX.get(fmt, fmt)}"
    if getattr(model_api, "runtime_format", None) is not None:
        raise RuntimeError("Exported runtime artifacts cannot be re-exported; load the .pt checkpoint.")
    dev = torch.device(device) if device else torch.device(model_api.device)
    proxy = _ExportProxy(model_api, dev)
    result = export_model(
        proxy, path=os.fspath(path), format=fmt, imgsz=imgsz, batch=batch, dynamic=dynamic,
        half=half, simplify=simplify, int8=int8, opset=opset, device=dev, verify=verify, **kwargs,
    )
    meta_path = result["metadata"]
    with open(meta_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    return {"artifact": result["artifact"], "metadata": metadata, "metadata_path": meta_path}


class FotonetV11(Fotonet):
    """``Fotonet`` restricted to the v1.1 family.

    Loading, training, export, and saving all go through ``Fotonet`` so there is
    exactly one v1.1 loader. This class only refuses references that resolve to
    something other than a v1.1 model.
    """

    def __init__(self, model_ref=None, nc=None, device=None, task="detect"):
        super().__init__("fotonetn" if model_ref is None else model_ref, nc=nc, task=task, device=device)
        if not getattr(self, "is_v1_1", False):
            raise ValueError(
                f"'{model_ref}' is not a fotonet v1.1 model. Use one of {sorted(FAMILY)} "
                f"or a '{V11_FORMAT}' checkpoint/export."
            )


def load_v11(model_ref, device=None):
    return FotonetV11(model_ref, device=device)


__all__ = [
    "FotonetV11", "load_v11", "V11_FORMAT", "export_v11", "deploy_for_inference",
    "load_weights", "save_payload", "inference_state_dict", "validate_v11_metadata",
    "load_checkpoint", "is_v11_checkpoint", "checkpoint_names",
]
