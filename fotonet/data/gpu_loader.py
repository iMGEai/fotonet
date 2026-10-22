"""GPU-assembly training loader (first used by fotonet v1.1).

The profile of the CPU loader showed 7 of 8 CPU threads busy on a 4-core box:
workers spent their time decoding JPEGs, pasting 1024 px mosaics, and running
the affine/HSV augmentation. Here a worker reads the image header and labels
and computes the mosaic or letterbox *geometry* and the transformed boxes
(cheap NumPy). Pixels take one of two routes (``jpeg_decoder``):

- ``"workers"``: the worker decodes, resizes, and crops out only the part of
  the image that lands in the canvas. Decode runs in parallel across workers.
- ``"nvjpeg"``: the worker ships the encoded bytes and the main process
  decodes the batch on the GPU. On GPUs without a hardware JPEG engine
  (consumer cards) nvJPEG still does Huffman decoding on one CPU thread, so
  this caps at roughly 600 images/s on a desktop CPU.

Either way the GPU pastes the parts into the canvas and augments the batch.

Pixel semantics match the CPU path: longest-side resize with bilinear
sampling, the same mosaic center and crop, the same 114 fill, EXIF orientation
applied, and the same box filters (``> 2 px`` sides, ``> 10%`` visible area).
Mixup and box copy-paste are not supported here; the engine falls back to the
CPU loader when they are enabled.
"""
from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn.functional as F

from fotonet.data.augmentations.boxes import clip_boxes_xyxy, sanitize_boxes_xywhn, xywhn_to_xyxy, xyxy_to_xywhn

_JPEG_MAGIC = b"\xff\xd8"
_ORIENTATION_TAG = 274


def image_header(path):
    """``(height, width, is_jpeg)`` after EXIF orientation, read from the header only."""
    from PIL import Image

    with Image.open(path) as image:
        width, height = image.size
        orientation = 1
        try:
            orientation = int(image.getexif().get(_ORIENTATION_TAG, 1) or 1)
        except Exception:
            orientation = 1
        is_jpeg = image.format == "JPEG"
    if orientation in (5, 6, 7, 8):
        width, height = height, width
    return int(height), int(width), bool(is_jpeg)


def _resized_shape(height, width, size):
    scale = min(size / height, size / width)
    return max(int(round(height * scale)), 1), max(int(round(width * scale)), 1)


class Part:
    """One source image pasted into a canvas: where it comes from and where it goes."""

    __slots__ = ("kind", "payload", "resized", "src", "dst")

    def __init__(self, kind, payload, resized, src, dst):
        self.kind = kind          # "jpeg": encoded bytes; "raw": decoded HWC uint8 RGB
        self.payload = payload
        self.resized = resized    # (h, w) after the longest-side resize
        self.src = src            # (y1, y2, x1, x2) in the resized image
        self.dst = dst            # (y1, y2, x1, x2) in the canvas


class GeometryMixin:
    """Mixed into the training dataset: emits parts and boxes instead of pixels."""

    gpu_decode = False
    jpeg_decoder = "nvjpeg"
    # Worker-decode mosaic partner buffer, in images per worker (0 = off). A
    # mosaic then decodes its first image fresh and takes the other three
    # from recently decoded, already resized images, so a sample costs about
    # one JPEG decode instead of four.
    mosaic_buffer = 0
    MOSAIC_BUFFER_WARM = 16

    def _header(self, idx):
        cache = self.__dict__.setdefault("_header_cache", {})
        if idx not in cache:
            cache[idx] = image_header(self.img_files[idx])
        return cache[idx]

    def _payload(self, idx, is_jpeg):
        path = self.img_files[idx]
        if is_jpeg and self.jpeg_decoder == "nvjpeg":
            data = np.fromfile(path, dtype=np.uint8)
            if data[:2].tobytes() == _JPEG_MAGIC:
                return "jpeg", torch.from_numpy(data)
        # Anything nvJPEG cannot take is decoded here, exactly like the CPU path.
        return "raw", self._decode_image(path)

    def _mosaic_pool(self):
        """This worker's partner buffer for the current image size, or None when off."""
        if int(self.mosaic_buffer or 0) <= 0 or self.jpeg_decoder == "nvjpeg":
            return None
        shared = getattr(self, "_sampling_shared_version", None)
        version = int(shared[0]) if shared is not None else None
        key = (int(self.imgsz), version)
        pool = self.__dict__.get("_mosaic_pool_state")
        if pool is None or pool["key"] != key:
            # New image size or a new sampling subset: start over.
            pool = {"key": key, "items": []}
            self.__dict__["_mosaic_pool_state"] = pool
        return pool["items"]

    def _remember(self, idx, resized, payload):
        items = self._mosaic_pool()
        if items is None:
            return
        entry = (int(idx), tuple(resized), payload)
        capacity = int(self.mosaic_buffer)
        if len(items) < capacity:
            items.append(entry)
        else:
            items[random.randrange(capacity)] = entry

    def _part(self, idx, is_jpeg, resized, src, dst, cached=None, remember=False):
        if cached is not None:
            payload = cached
        else:
            kind, payload = self._payload(idx, is_jpeg)
            if kind == "jpeg":
                return Part(kind, payload, resized, src, dst)
            # Decoded here: resize like the CPU path and send only the visible crop.
            import cv2

            h, w = resized
            if payload.shape[:2] != (h, w):
                payload = cv2.resize(payload, (w, h), interpolation=cv2.INTER_LINEAR)
            if remember:
                self._remember(idx, resized, payload)
        y1, y2, x1, x2 = src
        crop = np.ascontiguousarray(payload[y1:y2, x1:x2])
        return Part("raw", torch.from_numpy(crop), crop.shape[:2], (0, y2 - y1, 0, x2 - x1), dst)

    def _labels(self, idx):
        target = self._get_label(idx)
        return target["boxes"].astype(np.float32, copy=True), target["labels"].astype(np.int64, copy=True)

    def _letterbox_geometry(self, idx, size):
        h0, w0, is_jpeg = self._header(idx)
        nh, nw = _resized_shape(h0, w0, size)
        pad_top, pad_left = (size - nh) // 2, (size - nw) // 2
        part = self._part(idx, is_jpeg, (nh, nw), (0, nh, 0, nw), (pad_top, pad_top + nh, pad_left, pad_left + nw))
        boxes, labels = self._labels(idx)
        if len(boxes):
            # Same as letterbox_resize: exact scale on original coordinates,
            # not the rounded resized size.
            scale = min(size / h0, size / w0)
            xyxy = xywhn_to_xyxy(boxes, w0, h0)
            xyxy[:, :4] *= scale
            xyxy[:, [0, 2]] += pad_left
            xyxy[:, [1, 3]] += pad_top
            xyxy = clip_boxes_xyxy(xyxy, size, size)
            boxes, labels = sanitize_boxes_xywhn(xyxy_to_xywhn(xyxy, size, size), labels)
        return [part], boxes, labels

    def _mosaic_geometry_once(self, idx, size):
        pool = self._mosaic_pool()
        cached = [None] * 4
        candidates = None
        if pool is not None and len(pool) >= min(self.MOSAIC_BUFFER_WARM, int(self.mosaic_buffer)):
            # One entry per image: an image decoded twice sits in the pool twice.
            candidates = list({entry[0]: entry for entry in pool if entry[0] != int(idx)}.values())
        if candidates is not None and len(candidates) >= 3:
            # Three different images, none of them the fresh one.
            picks = random.sample(candidates, 3)
            indices = [idx] + [entry[0] for entry in picks]
            cached[1:] = [entry[2] for entry in picks]
        else:
            indices = [idx] + [int(self.sample_index()) for _ in range(3)]
        center_x = int(random.uniform(size * 0.5, size * 1.5))
        center_y = int(random.uniform(size * 0.5, size * 1.5))
        large = size * 2
        crop = size // 2
        # Part specs, not parts: an attempt whose boxes are all clipped away is
        # retried, so decoding waits until _mosaic_geometry keeps an attempt.
        specs, all_boxes, all_labels = [], [], []
        source_count = 0
        for slot, j in enumerate(indices):
            h0, w0, is_jpeg = self._header(j)
            h, w = _resized_shape(h0, w0, size)
            if slot == 0:
                x1a, y1a, x2a, y2a = max(center_x - w, 0), max(center_y - h, 0), center_x, center_y
                x1b, y1b = w - (x2a - x1a), h - (y2a - y1a)
            elif slot == 1:
                x1a, y1a, x2a, y2a = center_x, max(center_y - h, 0), min(center_x + w, large), center_y
                x1b, y1b = 0, h - (y2a - y1a)
            elif slot == 2:
                x1a, y1a, x2a, y2a = max(center_x - w, 0), center_y, center_x, min(large, center_y + h)
                x1b, y1b = w - (x2a - x1a), 0
            else:
                x1a, y1a, x2a, y2a = center_x, center_y, min(center_x + w, large), min(large, center_y + h)
                x1b, y1b = 0, 0
            boxes, labels = self._labels(j)
            source_count += len(labels)
            if x2a <= x1a or y2a <= y1a:
                continue
            # Only the part that lands inside the final crop window is pasted.
            vx1, vy1 = max(x1a, crop), max(y1a, crop)
            vx2, vy2 = min(x2a, crop + size), min(y2a, crop + size)
            if vx2 > vx1 and vy2 > vy1:
                src = (y1b + (vy1 - y1a), y1b + (vy2 - y1a), x1b + (vx1 - x1a), x1b + (vx2 - x1a))
                dst = (vy1 - crop, vy2 - crop, vx1 - crop, vx2 - crop)
                specs.append((j, is_jpeg, (h, w), src, dst, cached[slot], cached[slot] is None))
            if len(boxes):
                xyxy = xywhn_to_xyxy(boxes, w, h)
                xyxy[:, [0, 2]] += x1a - x1b
                xyxy[:, [1, 3]] += y1a - y1b
                all_boxes.append(xyxy)
                all_labels.append(labels)
        if not all_boxes:
            return specs, np.zeros((0, 4), np.float32), np.zeros((0,), np.int64), source_count
        boxes = np.concatenate(all_boxes, 0)
        labels = np.concatenate(all_labels, 0)
        boxes[:, [0, 2]] -= crop
        boxes[:, [1, 3]] -= crop
        area = np.maximum(boxes[:, 2] - boxes[:, 0], 0.0) * np.maximum(boxes[:, 3] - boxes[:, 1], 0.0)
        boxes = clip_boxes_xyxy(boxes, size, size)
        bw, bh = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
        visible = (bw * bh) / np.maximum(area, 1e-6)
        keep = (bw > 2.0) & (bh > 2.0) & (visible > 0.1) & np.isfinite(boxes).all(axis=1)
        boxes, labels = boxes[keep], labels[keep]
        if len(boxes):
            boxes, labels = sanitize_boxes_xywhn(xyxy_to_xywhn(boxes, size, size), labels)
        else:
            boxes, labels = np.zeros((0, 4), np.float32), np.zeros((0,), np.int64)
        return specs, boxes, labels, source_count

    def _mosaic_geometry(self, idx, size, max_retries=2):
        best = None
        for _ in range(max_retries + 1):
            specs, boxes, labels, source_count = self._mosaic_geometry_once(idx, size)
            if source_count == 0 or len(labels) > 0:
                best = (specs, boxes, labels)
                break
            if best is None:
                best = (specs, boxes, labels)
        specs, boxes, labels = best
        return [self._part(*spec) for spec in specs], boxes, labels

    def geometry_sample(self, idx):
        size = int(self.imgsz)
        use_mosaic = self._mosaic_enabled() and np.random.random() < self.augment_hyp.get("mosaic", 0.0)
        if use_mosaic:
            parts, boxes, labels = self._mosaic_geometry(idx, size)
        else:
            parts, boxes, labels = self._letterbox_geometry(idx, size)
        return {
            "parts": parts,
            "boxes": torch.from_numpy(np.asarray(boxes, dtype=np.float32).reshape(-1, 4)),
            "labels": torch.from_numpy(np.asarray(labels, dtype=np.int64).reshape(-1)),
        }


def collate_geometry(batch):
    return list(batch)


def _decode_cpu(data):
    import cv2

    image = cv2.imdecode(data.numpy(), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("could not decode image bytes")
    return torch.from_numpy(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).contiguous()


class GpuMosaicAssembler:
    """Decode a batch of parts on the GPU and paste them into uint8 canvases."""

    # Pinned upload buffers in flight at once; a buffer is reused only after the
    # copy that last read it has finished.
    STAGING_SLOTS = 3

    def __init__(self, device, fill=114):
        self.device = torch.device(device)
        self.fill = int(fill)
        self.cpu_fallbacks = 0
        self._side = None
        self._staging = [None] * self.STAGING_SLOTS
        self._staging_done = [None] * self.STAGING_SLOTS
        self._staging_next = 0

    def side_stream(self):
        """The one side stream DecodePrefetcher assembles on, for the whole run.

        The caching allocator keeps a separate pool per stream, so a new stream
        every epoch would strand a pool of canvases per epoch (about 70 MB at
        512 px) until the 32-stream pool wraps around.
        """
        if self._side is None:
            self._side = torch.cuda.Stream(device=self.device)
        return self._side

    def _staging_buffer(self, nbytes):
        slot = self._staging_next
        self._staging_next = (slot + 1) % self.STAGING_SLOTS
        done = self._staging_done[slot]
        if done is not None:
            done.synchronize()
        buffer = self._staging[slot]
        if buffer is None or buffer.numel() < nbytes:
            # Grow with headroom so random mosaic crop sizes settle on one buffer.
            buffer = torch.empty(int(nbytes * 1.25) + 1, dtype=torch.uint8, pin_memory=True)
            self._staging[slot] = buffer
        return slot, buffer[:nbytes]

    def _decode_all(self, payloads):
        from torchvision.io import ImageReadMode, decode_jpeg

        if not payloads:
            return []
        if self.device.type == "cuda":
            # torchvision's CUDA decode takes its output memory from the current
            # stream's pool but writes it on an internal stream that does not
            # wait for work already queued here. A block freed on the host whose
            # queued kernels have not run yet can be overwritten (NaN gradients
            # when the step frees activations; wrong pixels on a decode stream).
            # Drain the stream first; under DecodePrefetcher this blocks only
            # the producer thread. A blocking event sleeps instead of spinning
            # a CPU core while the GPU finishes the training step.
            drained = torch.cuda.Event(blocking=True)
            drained.record(torch.cuda.current_stream(self.device))
            drained.synchronize()
        try:
            return decode_jpeg(payloads, mode=ImageReadMode.RGB, device=self.device, apply_exif_orientation=True)
        except Exception:
            decoded = []
            for data in payloads:
                try:
                    decoded.append(decode_jpeg(
                        data, mode=ImageReadMode.RGB, device=self.device, apply_exif_orientation=True,
                    ))
                except Exception:
                    # CMYK and other JPEGs nvJPEG refuses take the CPU decoder.
                    self.cpu_fallbacks += 1
                    decoded.append(_decode_cpu(data).to(self.device, non_blocking=True))
            return decoded

    def _upload_raw(self, payloads):
        """HWC uint8 host images as CHW device views, with one host-to-device copy."""
        if not payloads:
            return []
        if self.device.type != "cuda":
            return [payload.permute(2, 0, 1) for payload in payloads]
        sizes = [payload.numel() for payload in payloads]
        # Reused pinned buffers: a fresh pinned tensor per batch at a random
        # size churns the pinned host cache.
        slot, staging = self._staging_buffer(sum(sizes))
        torch.cat([payload.reshape(-1) for payload in payloads], out=staging)
        flat = staging.to(self.device, non_blocking=True)
        done = torch.cuda.Event(blocking=True)
        done.record(torch.cuda.current_stream(self.device))
        self._staging_done[slot] = done
        return [chunk.view(payload.shape).permute(2, 0, 1)
                for chunk, payload in zip(flat.split(sizes), payloads)]

    @torch.no_grad()
    def __call__(self, samples, size):
        batch = len(samples)
        canvas = torch.full((batch, 3, size, size), self.fill, dtype=torch.uint8, device=self.device)
        jpeg = [part.payload for sample in samples for part in sample["parts"] if part.kind == "jpeg"]
        decoded = iter(self._decode_all(jpeg))
        raw = iter(self._upload_raw([part.payload for sample in samples for part in sample["parts"]
                                     if part.kind != "jpeg"]))
        for b, sample in enumerate(samples):
            for part in sample["parts"]:
                if part.kind == "jpeg":
                    image = next(decoded)
                else:
                    image = next(raw)
                h, w = part.resized
                if tuple(image.shape[-2:]) != (h, w):
                    image = F.interpolate(
                        image.unsqueeze(0).float(), size=(h, w), mode="bilinear", align_corners=False,
                    ).squeeze(0).round_().clamp_(0, 255).to(torch.uint8)
                y1b, y2b, x1b, x2b = part.src
                y1, y2, x1, x2 = part.dst
                canvas[b, :, y1:y2, x1:x2] = image[:, y1b:y2b, x1b:x2b]
        targets = [{"boxes": sample["boxes"], "labels": sample["labels"]} for sample in samples]
        return canvas, targets




_END = object()


class DecodePrefetcher:
    """Decode and assemble upcoming batches on a side CUDA stream in a thread.

    The training step runs on the default stream while the next ``depth``
    batches are decoded, resized, and pasted, so GPU decode leaves the critical
    path. Yields ``(uint8 canvas, targets)`` like the assembler.
    """

    def __init__(self, loader, assembler, size, depth=2):
        self.loader = loader
        self.assembler = assembler
        self.size = int(size)
        self.depth = max(int(depth), 1)

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        import queue
        import threading

        device = self.assembler.device
        side = self.assembler.side_stream()
        items = queue.Queue(maxsize=self.depth)
        stop = threading.Event()

        def produce():
            try:
                with torch.cuda.stream(side):
                    for batch in self.loader:
                        if stop.is_set():
                            break
                        canvas, targets = self.assembler(batch, self.size)
                        ready = torch.cuda.Event()
                        ready.record(side)
                        items.put((canvas, targets, ready))
            except BaseException as exc:  # surfaced on the consumer side
                items.put(exc)
            finally:
                items.put(_END)

        thread = threading.Thread(target=produce, name="fotonet-decode", daemon=True)
        thread.start()
        try:
            while True:
                item = items.get()
                if item is _END:
                    break
                if isinstance(item, BaseException):
                    raise item
                canvas, targets, ready = item
                current = torch.cuda.current_stream(device)
                current.wait_event(ready)
                canvas.record_stream(current)
                yield canvas, targets
        finally:
            stop.set()
            while thread.is_alive():
                try:
                    items.get(timeout=0.1)
                except queue.Empty:
                    pass
            thread.join(timeout=5)


__all__ = ["DecodePrefetcher", "GeometryMixin", "GpuMosaicAssembler", "Part", "collate_geometry", "image_header"]
