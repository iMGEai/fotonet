"""Training data: datasets, collate, single-class view, loader plan and shutdown."""
from __future__ import annotations

import os
import random

import numpy as np
import torch

from fotonet.data.dataset import DetectionDataset, expand_image_sources
from fotonet.utils.config import load_data_cfg

from fotonet.data.gpu_loader import GeometryMixin

from .options import _M_TIERS


def _m_bucket(max_gt, cap, tiers=_M_TIERS):
    max_gt = int(max_gt)
    cap = int(cap)
    for tier in tiers:
        if tier > max_gt:
            return min(tier, cap)
    return cap


class _TrainDataset(GeometryMixin, DetectionDataset):
    """Training dataset: CPU mosaic/mixup path, or geometry-only for GPU decode.

    With a teacher cache, each image's targets gain the teacher's confident
    detections that do not overlap ground truth, encoded in the label code.
    """

    def __init__(self, *args, gpu_post=False, predecode=None, gpu_decode=False, teacher=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.gpu_post = bool(gpu_post)
        self.gpu_decode = bool(gpu_decode)
        self._predecode = predecode
        self.teacher = teacher

    def _get_label(self, idx):
        target = super()._get_label(idx)
        if self.teacher is None:
            return target
        boxes, codes = self.teacher.pseudo_rows(self.img_files[idx], target["boxes"], target["labels"])
        if not len(codes):
            return target
        merged = dict(target)
        merged["boxes"] = np.concatenate((target["boxes"], boxes.astype(np.float32)), 0)
        merged["labels"] = np.concatenate((target["labels"], codes.astype(np.int64)), 0)
        return merged

    def _read_image(self, idx):
        if self._predecode is None:
            return super()._read_image(idx)
        image = self._predecode.get(idx)
        return np.array(image, copy=True)

    def _geometric_sample(self, idx, allow_mixup):
        from fotonet.data.augmentations import (
            copy_paste,
            letterbox_resize,
            load_mosaic,
            mixup,
            sanitize_boxes_xywhn,
        )

        hyp = self.augment_hyp
        use_mosaic = self._mosaic_enabled() and np.random.random() < hyp.get("mosaic", 0.0)
        if use_mosaic:
            img, boxes, labels = load_mosaic(self, idx, self.imgsz)
        else:
            img, target = self.get_raw(idx)
            boxes, labels = target["boxes"], target["labels"]
            img, boxes, labels = letterbox_resize(img, self.imgsz, boxes, labels=labels)
        if self.augment and hyp.get("copy_paste", 0.0) > 0.0:
            img, boxes, labels = copy_paste(
                img, boxes, labels, p=hyp["copy_paste"], mode=hyp.get("copy_paste_mode", "flip"),
            )
        if (
            allow_mixup and self.augment and hyp.get("mixup", 0.0) > 0.0
            and np.random.random() < hyp["mixup"] and len(self.img_files) > 1
        ):
            other, other_boxes, other_labels = self._geometric_sample(self.sample_index(), False)
            img, boxes, labels = mixup(
                img, boxes, labels, other, other_boxes, other_labels,
                alpha=hyp.get("mixup_alpha", 32.0),
            )
        boxes, labels = sanitize_boxes_xywhn(boxes, labels)
        return img, boxes, labels

    def __getitem__(self, idx):
        if self.gpu_decode:
            return self.geometry_sample(idx)
        if not self.gpu_post:
            return super().__getitem__(idx)
        img, boxes, labels = self._geometric_sample(idx, True)
        chw = np.ascontiguousarray(img.transpose(2, 0, 1))
        return (
            torch.from_numpy(chw),
            {
                "boxes": torch.from_numpy(boxes.astype(np.float32, copy=False)),
                "labels": torch.from_numpy(labels.astype(np.int64, copy=False)),
            },
        )


def _collate(batch):
    imgs, targets = zip(*batch)
    return torch.stack(imgs, 0), list(targets)


class _SingleClassView(torch.utils.data.Dataset):
    """Validation view with every class folded into class 0 (``single_cls``)."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, target = self.dataset[index]
        target = dict(target)
        for key in ("labels", "eval_labels"):
            if key in target:
                value = target[key]
                target[key] = value * 0 if hasattr(value, "__mul__") else value
        return image, target


def _load_data_cfg(data):
    """Return ``(train, val, nc, names)`` through the shared data-config loader.

    Same path rules as v1: ``path`` is relative to the YAML file, ``train`` and
    ``val`` may be lists, and ``names`` is validated against ``nc``.
    """
    from fotonet.data.starter import resolve_data

    data = resolve_data(data)
    if isinstance(data, (str, os.PathLike)):
        path = os.path.abspath(os.path.expanduser(os.fspath(data)))
        if not os.path.isfile(path):
            raise SystemExit(f"Dataset config file not found: {path}")
        data = path
    cfg = load_data_cfg(data)
    train_source = cfg.get("train")
    if not train_source:
        raise SystemExit("The data config needs a 'train' source.")
    sources = train_source if isinstance(train_source, (list, tuple)) else [train_source]
    if any(isinstance(item, dict) or str(item).lower().endswith(".json") for item in sources):
        raise SystemExit("v1.1 training reads YOLO-format image folders; COCO JSON train sources are not supported.")
    names = cfg.get("names")
    nc = cfg.get("nc")
    nc = int(nc) if nc is not None else (len(names) if names else 80)
    return train_source, cfg.get("val") or None, nc, (dict(names) if names else None)


def _sample_fraction(source, fraction, seed=0):
    """A seeded image list holding ``fraction`` of the training images."""
    paths = expand_image_sources(source)
    count = max(1, int(round(len(paths) * float(fraction))))
    return sorted(random.Random(int(seed)).sample(paths, count))


def _epoch_data_plan(epoch, current_imgsz, wanted_imgsz, mosaic_closed, close_from, mosaic):
    """Decide whether the loader must be rebuilt this epoch.

    Returns ``(rebuild, mosaic_closed, mosaic_probability)``. Once mosaic is
    closed it stays closed, even when a later image-size change rebuilds the
    dataset.
    """
    close_now = (not mosaic_closed) and epoch >= close_from and float(mosaic) > 0.0
    closed = bool(mosaic_closed or close_now)
    rebuild = int(wanted_imgsz) != int(current_imgsz) or close_now
    return rebuild, closed, 0.0 if closed else float(mosaic)


def _close_loader(loader):
    """Shut down a DataLoader's persistent workers now instead of at garbage collection."""
    iterator = getattr(loader, "_iterator", None)
    if iterator is not None and hasattr(iterator, "_shutdown_workers"):
        try:
            iterator._shutdown_workers()
        except Exception:
            pass
        loader._iterator = None
