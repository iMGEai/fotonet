"""Line-delimited JSON protocol: stdout capture, result/telemetry/error events, model-id checks."""

from __future__ import annotations

import json
import os
import sys
import traceback
from typing import Any, Dict, Optional


# Redirect default sys.stdout to sys.stderr so any stray library prints
# (e.g. from torch, pycocotools, onnx) never corrupt the line-delimited JSON protocol.
_REAL_STDOUT = sys.__stdout__
sys.stdout = sys.stderr


# Supported v1.1 family model identifiers
V11_MODEL_IDS = ("fotonetn", "fotonets", "fotonetm", "fotonetl")


def emit_json(payload: Dict[str, Any]) -> None:
    """Emit a single line of JSON to the actual stdout stream and flush immediately."""
    line = json.dumps(payload, ensure_ascii=False)
    _REAL_STDOUT.write(line + "\n")
    _REAL_STDOUT.flush()


def emit_telemetry(data: Dict[str, Any]) -> None:
    """Emit a progress/telemetry event to stdout."""
    emit_json({"type": "telemetry", "data": data})


def emit_result(data: Dict[str, Any], status: str = "ok") -> None:
    """Emit a completion result event to stdout."""
    emit_json({"type": "result", "status": status, "data": data})


def emit_error(message: str, traceback_str: Optional[str] = None) -> None:
    """Emit an error event to stdout."""
    payload = {"type": "error", "message": message}
    if traceback_str:
        payload["traceback"] = traceback_str
    emit_json(payload)


def validate_v11_model_id(model_ref: str) -> str:
    """Ensure that the model reference strictly belongs to Fotonet v1.1."""
    ref_lower = str(model_ref).strip().lower()
    if ref_lower in V11_MODEL_IDS:
        return ref_lower
    if os.path.isfile(model_ref) and model_ref.endswith(".pt"):
        from fotonet.versions.v1_1.api import _peek

        # Tensor-only read: an untrusted file is refused, never executed.
        if _peek(model_ref) is not None:
            return model_ref
    if "fotonete" in ref_lower:
        raise ValueError(
            "fotonete is a legacy architecture and is not supported in Fotonet Studio. "
            f"Fotonet Studio exclusively supports the Fotonet v1.1 family: {list(V11_MODEL_IDS)}."
        )
    if any(k in ref_lower for k in ("yolo", "rtdetr", "detectron")):
        raise ValueError(
            f"External model '{model_ref}' is not permitted. "
            f"Fotonet Studio exclusively supports the native Fotonet v1.1 family: {list(V11_MODEL_IDS)}."
        )
    raise ValueError(
        f"Unknown or unsupported model '{model_ref}'. "
        f"Expected one of {list(V11_MODEL_IDS)} or a valid Fotonet v1.1 checkpoint (.pt)."
    )
