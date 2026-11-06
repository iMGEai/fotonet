"""Run-time probes (first used by the fotonet v1.2 trainer): head guard rails and live previews.

``probe.head_matrix``: on a fixed slice of the validation set (1,000 images by default), the deployed
one-to-one (O2O) head as it ships, the same head with class-wise NMS on top, the training-only
one-to-many (O2M) head with NMS, precision and recall at an operating confidence (IoU 0.5,
class-aware), and the duplicate rate: the share of confident O2O boxes that class-wise NMS at IoU 0.5
would remove. These are the numbers r2 was tuned by (recipe book, live tuning), computed in the
trainer instead of by scratchpad scripts. AP here is COCO-style (IoU 0.50:0.95, 101-point, 100
boxes per image, all sizes) over the slice, so it reads a little apart from full-val mAP.

``probe.live_preview``: every few minutes, one random validation image and one random training image
through the current EMA weights, with each image's own AP against its labels. Studio's training tab
shows both. Neither probe touches the training state, its random generators or its data order.
"""
from __future__ import annotations

import json
import os
import random
import time

import numpy as np
import torch

IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)
_RECALL_POINTS = np.linspace(0.0, 1.0, 101)


def _iou_matrix(a, b):
    """Pairwise IoU of xyxy arrays [N, 4] and [M, 4]."""
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), dtype=np.float64)
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0.0, None), axis=2)
    area_a = np.prod(np.clip(a[:, 2:] - a[:, :2], 0.0, None), axis=1)
    area_b = np.prod(np.clip(b[:, 2:] - b[:, :2], 0.0, None), axis=1)
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-12)


def _match(pred_xyxy, pred_scores, gt_xyxy, thresholds):
    """COCO greedy matching of one image and one class: [len(thresholds), N] true-positive flags."""
    order = np.argsort(-pred_scores, kind="stable")
    iou = _iou_matrix(pred_xyxy[order], gt_xyxy)
    tp = np.zeros((len(thresholds), len(order)), dtype=bool)
    for t, thr in enumerate(thresholds):
        taken = np.zeros(len(gt_xyxy), dtype=bool)
        for d in range(len(order)):
            if not len(gt_xyxy):
                break
            cand = np.where(taken, -1.0, iou[d])
            g = int(np.argmax(cand))
            if cand[g] >= thr:
                taken[g] = True
                tp[t, d] = True
    out = np.zeros_like(tp)
    out[:, order] = tp
    return out


class APAccumulator:
    """COCO-style AP over many images: add images one by one, then ``result()``."""

    def __init__(self, thresholds=IOU_THRESHOLDS):
        self.thresholds = np.asarray(thresholds, dtype=np.float64)
        self.scores = {}
        self.flags = {}
        self.n_gt = {}

    def add(self, pred_xyxy, pred_scores, pred_cls, gt_xyxy, gt_cls):
        pred_xyxy = np.asarray(pred_xyxy, dtype=np.float64).reshape(-1, 4)
        pred_scores = np.asarray(pred_scores, dtype=np.float64).reshape(-1)
        pred_cls = np.asarray(pred_cls, dtype=np.int64).reshape(-1)
        gt_xyxy = np.asarray(gt_xyxy, dtype=np.float64).reshape(-1, 4)
        gt_cls = np.asarray(gt_cls, dtype=np.int64).reshape(-1)
        for c in np.union1d(pred_cls, gt_cls):
            p = pred_cls == c
            g = gt_cls == c
            self.n_gt[int(c)] = self.n_gt.get(int(c), 0) + int(g.sum())
            if not p.any():
                continue
            flags = _match(pred_xyxy[p], pred_scores[p], gt_xyxy[g], self.thresholds)
            self.scores.setdefault(int(c), []).append(pred_scores[p])
            self.flags.setdefault(int(c), []).append(flags)

    def result(self):
        """``(ap50_95, ap50)`` as fractions, averaged over the classes that have ground truth; None without any."""
        per_class = []
        for c, n_gt in self.n_gt.items():
            if n_gt <= 0:
                continue
            if c not in self.scores:
                per_class.append(np.zeros(len(self.thresholds)))
                continue
            scores = np.concatenate(self.scores[c])
            flags = np.concatenate(self.flags[c], axis=1)
            order = np.argsort(-scores, kind="stable")
            flags = flags[:, order]
            tp = np.cumsum(flags, axis=1)
            fp = np.cumsum(~flags, axis=1)
            recall = tp / n_gt
            precision = tp / np.maximum(tp + fp, 1e-12)
            aps = []
            for t in range(len(self.thresholds)):
                envelope = np.maximum.accumulate(precision[t][::-1])[::-1]
                idx = np.searchsorted(recall[t], _RECALL_POINTS, side="left")
                sampled = np.where(idx < len(envelope), envelope[np.minimum(idx, len(envelope) - 1)], 0.0)
                aps.append(float(sampled.mean()))
            per_class.append(np.asarray(aps))
        if not per_class:
            return None, None
        table = np.stack(per_class)
        return float(table.mean()), float(table[:, 0].mean())


def operating_point(pred_xyxy, pred_scores, pred_cls, gt_xyxy, gt_cls, conf=0.25, iou=0.5):
    """``(true positives, predictions, ground truths)`` at confidence ``conf``, class-aware, IoU ``iou``."""
    keep = np.asarray(pred_scores) >= conf
    pred_xyxy, pred_scores, pred_cls = (np.asarray(v)[keep] for v in (pred_xyxy, pred_scores, pred_cls))
    gt_xyxy = np.asarray(gt_xyxy, dtype=np.float64).reshape(-1, 4)
    gt_cls = np.asarray(gt_cls, dtype=np.int64).reshape(-1)
    tp = 0
    for c in np.unique(pred_cls):
        p = pred_cls == c
        g = gt_cls == c
        if g.any():
            tp += int(_match(pred_xyxy[p].reshape(-1, 4), pred_scores[p], gt_xyxy[g], [iou])[0].sum())
    return tp, int(len(pred_scores)), int(len(gt_cls))


def image_report(pred_xyxy, pred_scores, pred_cls, gt_xyxy, gt_cls, conf=0.25):
    """One image's own numbers: AP50-95 and AP50 (None without labels), precision and recall at ``conf``."""
    acc = APAccumulator()
    acc.add(pred_xyxy, pred_scores, pred_cls, gt_xyxy, gt_cls)
    ap, ap50 = acc.result()
    tp, n_pred, n_gt = operating_point(pred_xyxy, pred_scores, pred_cls, gt_xyxy, gt_cls, conf=conf)
    return {
        "ap50_95": None if ap is None else round(100.0 * ap, 2),
        "ap50": None if ap50 is None else round(100.0 * ap50, 2),
        "precision": round(100.0 * tp / n_pred, 1) if n_pred else None,
        "recall": round(100.0 * tp / n_gt, 1) if n_gt else None,
        "tp": tp, "n_pred": n_pred, "n_gt": n_gt, "conf": conf,
    }


def letterbox(image, size):
    """Longest-side resize and centered 114 padding to ``size``; returns the batch tensor and the mapping."""
    import cv2

    h0, w0 = image.shape[:2]
    scale = min(size / h0, size / w0)
    nh, nw = max(int(round(h0 * scale)), 1), max(int(round(w0 * scale)), 1)
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[top:top + nh, left:left + nw] = resized
    tensor = torch.from_numpy(np.ascontiguousarray(canvas.transpose(2, 0, 1)))
    return tensor, (scale, left, top, w0, h0)


def decode(out, max_det=100, conf=0.001, nms_iou=None, candidates=300):
    """Detections from a v1.1 head output ``[B, N, nc + 4]`` (logits, normalized cxcywh).

    Returns per image ``(xyxy normalized [K, 4], scores [K], classes [K])`` as NumPy. With
    ``nms_iou`` the top ``candidates`` go through class-wise NMS first.
    """
    from torchvision.ops import batched_nms

    nc = out.shape[-1] - 4
    scores = out[..., :nc].float().sigmoid().flatten(1)
    boxes = out[..., nc:].float()
    k = min(int(candidates if nms_iou is not None else max_det), scores.shape[1])
    top, idx = scores.topk(k, dim=1)
    anchor, cls = idx // nc, idx % nc
    results = []
    for b in range(out.shape[0]):
        bx = boxes[b, anchor[b]]
        xyxy = torch.cat((bx[:, :2] - bx[:, 2:] * 0.5, bx[:, :2] + bx[:, 2:] * 0.5), 1)
        s, c = top[b], cls[b]
        keep = s >= conf
        xyxy, s, c = xyxy[keep], s[keep], c[keep]
        if nms_iou is not None and len(s):
            kept = batched_nms(xyxy, s, c, float(nms_iou))[:max_det]
            xyxy, s, c = xyxy[kept], s[kept], c[kept]
        else:
            xyxy, s, c = xyxy[:max_det], s[:max_det], c[:max_det]
        results.append((xyxy.cpu().numpy(), s.cpu().numpy(), c.cpu().numpy()))
    return results


def duplicate_count(xyxy, scores, cls, conf=0.3, iou=0.5):
    """``(boxes at or over conf, how many of them class-wise NMS at iou removes)``."""
    from torchvision.ops import batched_nms

    keep = scores >= conf
    n = int(keep.sum())
    if n == 0:
        return 0, 0
    b = torch.from_numpy(np.ascontiguousarray(xyxy[keep])).float()
    s = torch.from_numpy(np.ascontiguousarray(scores[keep])).float()
    c = torch.from_numpy(np.ascontiguousarray(cls[keep])).long()
    return n, n - int(batched_nms(b, s, c, float(iou)).numel())


def _gt_xyxy(target):
    """Ground truth of a raw sample as normalized xyxy and classes; teacher pseudo-label rows are left out."""
    boxes = np.asarray(target["boxes"], dtype=np.float64).reshape(-1, 4)
    labels = np.asarray(target["labels"], dtype=np.int64).reshape(-1)
    real = labels < 10_000
    boxes, labels = boxes[real], labels[real]
    xyxy = np.concatenate((boxes[:, :2] - boxes[:, 2:] * 0.5, boxes[:, :2] + boxes[:, 2:] * 0.5), 1)
    return xyxy, labels


def _to_original(xyxy, mapping, size):
    """Normalized letterboxed xyxy back to normalized original-image xyxy."""
    scale, left, top, w0, h0 = mapping
    px = xyxy * size
    px[:, [0, 2]] = (px[:, [0, 2]] - left) / scale / w0
    px[:, [1, 3]] = (px[:, [1, 3]] - top) / scale / h0
    return np.clip(px, 0.0, 1.0)


def _forward(model, batch, dev, amp, channels_last):
    x = batch.to(dev, non_blocking=True).float().div_(255.0)
    if channels_last:
        x = x.contiguous(memory_format=torch.channels_last)
    with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16 if dev.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16, enabled=bool(amp) and dev.type == "cuda"):
        return model(x)


class _swap_heads:
    """Point the eval forward at the training-only O2M towers for a moment (as dev-tools/v11_head_matrix.py)."""

    def __init__(self, model):
        self.head = model.head

    def __enter__(self):
        self.head.o2o, self.head.o2m = self.head.o2m, self.head.o2o
        return self

    def __exit__(self, *exc):
        self.head.o2o, self.head.o2m = self.head.o2m, self.head.o2o
        return False


@torch.no_grad()
def head_matrix(model, dataset, dev, imgsz, n_images=1000, batch=16, amp=True, channels_last=False,
                conf=0.25, nms_iou=0.65):
    """``probe.head_matrix@1.2``: guard-rail numbers of the deployed head on the first ``n_images`` of ``dataset``.

    ``model`` is an eval-mode v1.1 model holding the weights to probe (the trainer passes its EMA
    validation model). Returns a dict of percentages, or None when the model has no O2M head.
    """
    if getattr(model.head, "o2m", None) is None:
        return None
    t0 = time.perf_counter()
    count = min(int(n_images), len(dataset.img_files))
    acc = {"o2o": APAccumulator(), "o2o_nms": APAccumulator(), "o2m_nms": APAccumulator()}
    tp = n_pred = n_gt = dup_all = dup_removed = 0
    for start in range(0, count, int(batch)):
        images, maps, gts = [], [], []
        for i in range(start, min(start + int(batch), count)):
            image, target = dataset.get_raw(i)
            tensor, mapping = letterbox(image, imgsz)
            images.append(tensor)
            maps.append(mapping)
            gts.append(_gt_xyxy(target))
        x = torch.stack(images)
        out = _forward(model, x, dev, amp, channels_last)
        raw = decode(out)
        nms = decode(out, nms_iou=nms_iou)
        with _swap_heads(model):
            out_m = _forward(model, x, dev, amp, channels_last)
        nms_m = decode(out_m, nms_iou=nms_iou)
        for b, (gt_xyxy, gt_cls) in enumerate(gts):
            for key, dets in (("o2o", raw[b]), ("o2o_nms", nms[b]), ("o2m_nms", nms_m[b])):
                acc[key].add(_to_original(dets[0], maps[b], imgsz), dets[1], dets[2], gt_xyxy, gt_cls)
            xyxy, scores, cls = raw[b]
            t, p, g = operating_point(_to_original(xyxy, maps[b], imgsz), scores, cls, gt_xyxy, gt_cls, conf=conf)
            tp, n_pred, n_gt = tp + t, n_pred + p, n_gt + g
            a, r = duplicate_count(xyxy, scores, cls)
            dup_all, dup_removed = dup_all + a, dup_removed + r
    out = {"probe_images": count}
    for key, value in acc.items():
        ap, _ = value.result()
        out[f"probe_{key}"] = None if ap is None else round(100.0 * ap, 2)
    out["probe_precision"] = round(100.0 * tp / n_pred, 1) if n_pred else None
    out["probe_recall"] = round(100.0 * tp / n_gt, 1) if n_gt else None
    out["probe_dup_rate"] = round(100.0 * dup_removed / dup_all, 1) if dup_all else 0.0
    out["probe_seconds"] = round(time.perf_counter() - t0, 1)
    return out


def _atomic_json(path, payload):
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    os.replace(tmp, path)


class LivePreview:
    """``probe.live_preview@1.2``: a random val and train image through the current weights, every few minutes.

    Writes ``<run>/preview/preview.json`` (the newest pair; Studio reads it) and appends each pair to
    ``preview/history.jsonl``. Uses its own random generator, so the run's data order and
    augmentation draws are untouched.
    """

    def __init__(self, out_dir, train_dataset, val_dataset, names, imgsz, seed=0):
        self.dir = os.path.join(out_dir, "preview")
        os.makedirs(self.dir, exist_ok=True)
        self.train = train_dataset
        self.val = val_dataset
        self.names = {int(k): str(v) for k, v in dict(names or {}).items()}
        self.imgsz = int(imgsz)
        self.rng = random.Random(int(seed) + 12345)
        self.last = None
        self.failures = 0

    def due(self, minutes):
        if float(minutes or 0.0) <= 0.0 or self.failures >= 3:
            return False
        return self.last is None or time.time() - self.last >= float(minutes) * 60.0

    def _item(self, split, dataset, model, dev, amp, channels_last, conf):
        index = self.rng.randrange(len(dataset.img_files))
        image, target = dataset.get_raw(index)
        tensor, mapping = letterbox(image, self.imgsz)
        out = _forward(model, tensor.unsqueeze(0), dev, amp, channels_last)
        xyxy, scores, cls = decode(out, conf=0.05)[0]
        pred = _to_original(xyxy, mapping, self.imgsz)
        gt_xyxy, gt_cls = _gt_xyxy(target)
        report = image_report(pred, scores, cls, gt_xyxy, gt_cls, conf=conf)
        box = lambda b: [round(float(b[0]), 5), round(float(b[1]), 5), round(float(b[2] - b[0]), 5), round(float(b[3] - b[1]), 5)]
        return {
            "split": split,
            "image": os.path.abspath(dataset.img_files[index]),
            "width": int(mapping[3]),
            "height": int(mapping[4]),
            "labels": [{"cls": int(c), "name": self.names.get(int(c), str(int(c))), "box": box(b)} for b, c in zip(gt_xyxy, gt_cls)],
            "predictions": [
                {"cls": int(c), "name": self.names.get(int(c), str(int(c))), "conf": round(float(s), 4), "box": box(b)}
                for b, s, c in zip(pred, scores, cls)
            ],
            **report,
        }

    @torch.no_grad()
    def run(self, model, dev, epoch, step, steps, amp=True, channels_last=False, conf=0.25):
        """Write one preview pair; returns it, or None when it failed (three failures turn the preview off)."""
        t0 = time.perf_counter()
        try:
            items = [self._item(split, ds, model, dev, amp, channels_last, conf)
                     for split, ds in (("val", self.val), ("train", self.train)) if ds is not None and len(ds.img_files)]
        except Exception as exc:  # a preview must never stop training
            self.failures += 1
            self.last = time.time()
            return {"error": f"{type(exc).__name__}: {exc}"}
        self.last = time.time()
        payload = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "epoch": int(epoch), "step": int(step), "steps": int(steps),
            "seconds": round(time.perf_counter() - t0, 3), "items": items,
        }
        _atomic_json(os.path.join(self.dir, "preview.json"), payload)
        with open(os.path.join(self.dir, "history.jsonl"), "a", encoding="utf-8") as handle:
            handle.write(json.dumps({k: payload[k] for k in ("time", "epoch", "step")} | {
                "items": [{k: item[k] for k in ("split", "image", "ap50_95", "ap50", "precision", "recall")} for item in items],
            }) + "\n")
        return payload


__all__ = ["APAccumulator", "LivePreview", "decode", "duplicate_count", "head_matrix", "image_report",
           "letterbox", "operating_point"]
