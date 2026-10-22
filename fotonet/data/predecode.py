"""Pre-decoded image blob for training — the dataloader's fast path.

Why this exists
---------------
Training was measured at 624 s/epoch of which only ~259 s was GPU compute:
58% of the step was dataloading, H2D, and host syncs.  The dominant per-sample
cost is JPEG decode plus resize, and it is paid *every epoch* for a dataset
that never changes.  ``PredecodeCache`` pays it once.

What it does
------------
Decode every image a single time, downscale so the long side is at most
``max_side`` (aspect preserved, **no padding**), and concatenate the raw RGB
bytes into one contiguous ``images.u8`` blob plus an ``index.json`` of byte
offsets and shapes.  Per epoch a worker then does one ``mmap`` slice and one
``np.frombuffer`` — a memcpy from the page cache — instead of a JPEG decode.

Why aspect-preserved and unpadded
---------------------------------
Padding is deliberately *not* baked in.  Mosaic pastes whole images into a
canvas, so pre-padded images would smear gray letterbox bars into the mosaic
patches, and a fixed-size cache would break the progressive-resize schedule.
Storing unpadded pixels keeps ``letterbox_resize`` and ``load_mosaic``
bit-for-bit semantically identical to decoding from disk — only the pixel
source changes.  A downscale cap at 720 is a no-op for COCO (long side <= 640)
and bounds disk for larger corpora.

Wiring
------
``DetectionDataset._read_image`` is the single decode chokepoint used by both
the mosaic and non-mosaic paths, so a subclass that overrides it makes the
whole pipeline fast with no augmentation change::

    class _PredecodeDataset(_GpuAugDataset):
        def _read_image(self, idx):
            return self._predecode.get(idx)
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np


BLOB_NAME = "images.u8"
INDEX_NAME = "index.json"


def _decode(path):
    """Decode to RGB HWC uint8. Mirrors DetectionDataset._decode_image."""
    try:
        import cv2
    except ImportError:  # pragma: no cover - PIL fallback
        cv2 = None
    if cv2 is not None:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"))


def _resize_cap(img, max_side):
    """Downscale so max(h, w) <= max_side. No padding, aspect preserved."""
    if max_side is None or max_side <= 0:
        return img
    h, w = img.shape[:2]
    long_side = max(h, w)
    if long_side <= max_side:
        return img
    scale = max_side / float(long_side)
    new_w = max(int(round(w * scale)), 1)
    new_h = max(int(round(h * scale)), 1)
    try:
        import cv2

        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        return cv2.resize(img, (new_w, new_h), interpolation=interp)
    except ImportError:  # pragma: no cover
        from PIL import Image

        return np.asarray(Image.fromarray(img).resize((new_w, new_h)))


def _signature(img_files):
    """Identity of the source set: paths + sizes + mtimes. Cheap (one stat each)."""
    digest = hashlib.sha1()
    for path in img_files:
        digest.update(os.path.abspath(path).encode("utf-8", "replace"))
        try:
            stat = os.stat(path)
            digest.update(f"|{stat.st_size}|{stat.st_mtime_ns}".encode("ascii"))
        except OSError:
            digest.update(b"|missing")
    digest.update(f"|n={len(img_files)}".encode("ascii"))
    return digest.hexdigest()


class PredecodeCache:
    """Read side: an mmap'd blob of pre-decoded images."""

    def __init__(self, blob_path, offsets, shapes, dtype=np.uint8):
        self.blob_path = blob_path
        self.offsets = offsets
        self.shapes = shapes
        self._mm = None
        self._np = dtype

    def _open(self):
        if self._mm is None:
            self._mm = np.memmap(self.blob_path, dtype=self._np, mode="r")
        return self._mm

    def __len__(self):
        return len(self.offsets)

    def get(self, idx):
        """Return an HWC uint8 view into the blob (zero-copy)."""
        mm = self._open()
        h, w = self.shapes[idx]
        start = int(self.offsets[idx])
        count = int(h) * int(w) * 3
        return np.frombuffer(mm, dtype=self._np, count=count, offset=start).reshape(
            int(h), int(w), 3
        )


def cache_dir_for(root, imgsz_code):
    return os.path.join(str(root), f"predecode_{imgsz_code}")


def ensure_cache(img_files, root, *, max_side=720, workers=4, verbose=True,
                 cache_code="v1"):
    """Build the blob if absent/stale, then return a ready PredecodeCache.

    ``root`` is the directory that receives ``predecode_<code>/``.  Returns
    ``(cache, stats)`` where stats reports whether a rebuild happened.
    """
    out_dir = os.path.join(str(root), f"predecode_{cache_code}")
    os.makedirs(out_dir, exist_ok=True)
    blob_path = os.path.join(out_dir, BLOB_NAME)
    index_path = os.path.join(out_dir, INDEX_NAME)
    if not img_files:
        raise ValueError("ensure_cache needs at least one image")

    sig = _signature(img_files)
    if os.path.isfile(blob_path) and os.path.isfile(index_path):
        try:
            with open(index_path, "r", encoding="utf-8") as handle:
                index = json.load(handle)
            if (index.get("signature") == sig
                    and int(index.get("count", -1)) == len(img_files)
                    and int(index.get("max_side", -1)) == int(max_side or -1)):
                return (
                    PredecodeCache(blob_path, index["offsets"], index["shapes"]),
                    {"rebuilt": False, "images": len(img_files),
                     "bytes": os.path.getsize(blob_path)},
                )
        except (OSError, ValueError, KeyError):
            pass

    offsets, shapes = [], []
    total = 0
    order = list(range(len(img_files)))
    # Sequential build: JPEG decode is already fast per file and the blob is
    # one open handle; parallelism here would only add process startup.
    tmp_path = blob_path + ".tmp"
    with open(tmp_path, "wb") as blob:
        for n, idx in enumerate(order):
            img = _resize_cap(_decode(img_files[idx]), max_side)
            if img.dtype != np.uint8:
                img = img.astype(np.uint8, copy=False)
            img = np.ascontiguousarray(img)
            h, w = img.shape[:2]
            offsets.append(total)
            shapes.append([int(h), int(w)])
            blob.write(img.tobytes())
            total += int(h) * int(w) * 3
            if verbose and (n + 1) % 512 == 0:
                done = (n + 1) / max(len(img_files), 1) * 100.0
                print(f"  [predecode] {n + 1}/{len(img_files)} ({done:.0f}%)",
                      file=sys.stderr, flush=True)
    os.replace(tmp_path, blob_path)

    index = {
        "signature": sig,
        "count": len(img_files),
        "max_side": int(max_side or -1),
        "offsets": offsets,
        "shapes": shapes,
        "img_files": [os.path.abspath(p) for p in img_files],
    }
    with open(index_path, "w", encoding="utf-8") as handle:
        json.dump(index, handle)
    if verbose:
        print(f"  [predecode] built {len(img_files)} images, "
              f"{total / 1e6:.1f} MB -> {blob_path}", file=sys.stderr, flush=True)
    return (
        PredecodeCache(blob_path, offsets, shapes),
        {"rebuilt": True, "images": len(img_files), "bytes": total},
    )


__all__ = ["PredecodeCache", "ensure_cache", "cache_dir_for"]
