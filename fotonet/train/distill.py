"""Offline teacher pseudo-labels for training (first used by fotonet v1.1).

A strong teacher runs once over the training images (``build_cache``); its
detections are stored compactly and safely loadable. During training, teacher
boxes that are confident enough and do not overlap any ground truth are added as
extra, down-weighted targets: objects the annotators missed stop being
trained as background, and the student sees more positives per image at no
per-step teacher cost.

Teacher rows travel through mosaic, letterbox, affine, and flips exactly like
ground truth because they *are* label rows. Their weight rides in the label
code: ``code = class + CODE_STRIDE * q`` with ``q`` in 1..1000 the weight in
thousandths; a plain ground-truth label has ``q = 0`` and weight 1.
``decode_labels`` splits classes and weights again right before assignment.

Teachers must be licensed for this use. Never distill from AGPL weights into
this Apache-2.0 project.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

CACHE_FORMAT = "fotonet-teacher-cache"
CODE_STRIDE = 10_000
WEIGHT_STEPS = 1000


def encode_labels(classes, weights):
    q = np.clip(np.round(np.asarray(weights, dtype=np.float64) * WEIGHT_STEPS), 1, WEIGHT_STEPS).astype(np.int64)
    return np.asarray(classes, dtype=np.int64) + CODE_STRIDE * q


def decode_labels(codes):
    """``(classes, weights)`` from label codes; plain labels decode to weight 1."""
    q = torch.div(codes, CODE_STRIDE, rounding_mode="floor")
    classes = codes - q * CODE_STRIDE
    weights = torch.where(q > 0, q.to(torch.float32) / WEIGHT_STEPS, torch.ones_like(q, dtype=torch.float32))
    return classes, weights


def _iou_xywh(a, b):
    """Pairwise IoU between [N,4] and [M,4] normalized cxcywh arrays."""
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), dtype=np.float32)
    a1, a2 = a[:, None, :2] - a[:, None, 2:] / 2, a[:, None, :2] + a[:, None, 2:] / 2
    b1, b2 = b[None, :, :2] - b[None, :, 2:] / 2, b[None, :, :2] + b[None, :, 2:] / 2
    wh = np.clip(np.minimum(a2, b2) - np.maximum(a1, b1), 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = a[:, 2] * a[:, 3]
    area_b = b[:, 2] * b[:, 3]
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


class TeacherCache:
    """Read side of a teacher cache: pseudo-label rows per training image."""

    def __init__(self, path, conf=0.35, weight=0.5, max_overlap=0.5, nc=None):
        from fotonet.versions.v1_1.api import load_checkpoint

        data = load_checkpoint(path)
        if not isinstance(data, dict) or data.get("format") != CACHE_FORMAT:
            raise ValueError(f"{path} is not a fotonet teacher cache.")
        self.conf = float(conf)
        self.weight = float(weight)
        self.max_overlap = float(max_overlap)
        self.teacher_nc = int(data.get("nc", 0)) or None
        if nc is not None and self.teacher_nc is not None and int(nc) != self.teacher_nc:
            raise ValueError(f"Teacher cache has nc={self.teacher_nc}; the dataset has nc={nc}.")
        self.names = data.get("names")
        self.rows = data["rows"].float().numpy()
        offsets = data["offsets"].numpy()
        self.index = {
            os.path.normcase(os.path.abspath(p)): (int(offsets[i]), int(offsets[i + 1]))
            for i, p in enumerate(data["paths"])
        }

    def __len__(self):
        return len(self.index)

    def pseudo_rows(self, image_path, gt_boxes, gt_labels):
        """``(boxes, codes)`` of teacher detections worth adding to this image's targets."""
        span = self.index.get(os.path.normcase(os.path.abspath(image_path)))
        if span is None or span[0] == span[1]:
            return np.zeros((0, 4), np.float32), np.zeros((0,), np.int64)
        rows = self.rows[span[0]:span[1]]
        rows = rows[rows[:, 1] >= self.conf]
        if len(rows) and len(gt_boxes):
            overlap = _iou_xywh(rows[:, 2:6], np.asarray(gt_boxes, dtype=np.float32)).max(axis=1)
            rows = rows[overlap < self.max_overlap]
        if not len(rows):
            return np.zeros((0, 4), np.float32), np.zeros((0,), np.int64)
        codes = encode_labels(rows[:, 0].astype(np.int64), rows[:, 1] * self.weight)
        return rows[:, 2:6].astype(np.float32), codes


def build_cache(teacher, data, out_path, imgsz=640, conf=0.05, max_det=100, batch=16, device=None,
                split="train", verbose=True):
    """Run ``teacher`` (any fotonet model reference) over a split and write the cache.

    Rows are ``[class, score, cx, cy, w, h]`` in normalized original-image
    coordinates. A low ``conf`` here keeps the choice of threshold for
    training time (``teacher_conf``).
    """
    from fotonet import Fotonet
    from fotonet.data.dataset import expand_image_sources
    from fotonet.utils.config import load_data_cfg

    cfg = load_data_cfg(os.fspath(data))
    if not cfg.get(split):
        raise ValueError(f"The data config has no {split!r} source.")
    paths = expand_image_sources(cfg[split])
    model = teacher if isinstance(teacher, Fotonet) else Fotonet(teacher, device=device)
    chunks, offsets = [], [0]
    for start in range(0, len(paths), int(batch)):
        results = model.predict(paths[start:start + int(batch)], imgsz=imgsz, conf=conf, max_det=max_det,
                                retain_images=False)
        for result in results:
            boxes = result.boxes_tensor.detach().float().cpu()
            scores = result.scores.detach().float().cpu().reshape(-1, 1)
            classes = result.classes.detach().float().cpu().reshape(-1, 1)
            rows = torch.cat((classes, scores, boxes), dim=1) if len(boxes) else torch.zeros(0, 6)
            chunks.append(rows.half())
            offsets.append(offsets[-1] + rows.shape[0])
        if verbose and (start // int(batch)) % 50 == 0:
            print(f"  [teacher] {min(start + int(batch), len(paths))}/{len(paths)}", file=sys.stderr, flush=True)
    payload = {
        "format": CACHE_FORMAT,
        "teacher": str(getattr(model, "loaded_weight_path", None) or getattr(model, "model_id", "")),
        "nc": int(model.nc),
        "names": {int(k): str(v) for k, v in dict(model.names).items()},
        "imgsz": int(imgsz) if not isinstance(imgsz, (list, tuple)) else list(imgsz),
        "conf": float(conf),
        "paths": [os.path.abspath(p) for p in paths],
        "offsets": torch.tensor(offsets, dtype=torch.int64),
        "rows": torch.cat(chunks, 0) if chunks else torch.zeros(0, 6, dtype=torch.float16),
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    torch.save(payload, out_path)
    return out_path


__all__ = [
    "CACHE_FORMAT", "CODE_STRIDE", "TeacherCache", "build_cache", "encode_labels", "decode_labels",
]
