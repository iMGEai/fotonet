"""``Fotonet`` input sources, image conversion, and letterboxing."""
import io
import os
import math
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
try:
    import cv2
except Exception:
    cv2 = None
from ._common import _IMAGE_SUFFIXES, _VIDEO_SUFFIXES


class PreprocessMixin:
    """Input sources, image conversion, and letterboxing."""

    @staticmethod
    def _normalize_imgsz(imgsz):
        if isinstance(imgsz, int):
            h = w = int(imgsz)
        elif isinstance(imgsz, (tuple, list)) and len(imgsz) == 2:
            h, w = (int(imgsz[0]), int(imgsz[1]))
        else:
            raise TypeError("imgsz must be a positive integer or an (height, width) pair.")
        if h <= 0 or w <= 0:
            raise ValueError("imgsz dimensions must be positive.")
        return h, w

    def _resolve_predict_imgsz(self, imgsz):
        """Resolve a default inference size and enforce static artifact shape."""
        target = self._normalize_imgsz((640, 640) if imgsz is None else imgsz)
        if imgsz is None and self._runtime_imgsz is not None:
            target = self._runtime_imgsz
        if (
            self.runtime_format is not None
            and self._runtime_dynamic is False
            and self._runtime_imgsz is not None
            and tuple(target) != tuple(self._runtime_imgsz)
        ):
            raise ValueError(
                f"This static {self.runtime_format} artifact accepts imgsz={tuple(self._runtime_imgsz)}, "
                f"not {tuple(target)}. Re-export with dynamic=True for variable shapes."
            )
        return target

    @staticmethod
    def _is_url(value):
        return isinstance(value, str) and urlparse(value).scheme in {"http", "https"}

    @classmethod
    def _is_video_source(cls, source):
        if isinstance(source, int):
            return True
        if not isinstance(source, (str, Path)):
            return False
        value = os.fspath(source)
        parsed = urlparse(value)
        suffix = Path(parsed.path if parsed.scheme else value).suffix.lower()
        return suffix in _VIDEO_SUFFIXES

    @classmethod
    def _expand_sources(cls, source):
        if isinstance(source, (str, Path)):
            value = os.fspath(source)
            if os.path.isdir(value):
                files = [
                    str(path) for path in Path(value).rglob("*")
                    if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
                ]
                if not files:
                    raise FileNotFoundError(f"No supported images were found under '{value}'.")
                return sorted(files)
            return [value]
        if isinstance(source, (list, tuple)):
            expanded = []
            for item in source:
                if cls._is_video_source(item):
                    raise ValueError("Video/webcam inputs must be passed as one source, not mixed with image batches.")
                expanded.extend(cls._expand_sources(item))
            if not expanded:
                raise ValueError("source cannot be an empty list.")
            return expanded
        return [source]

    @staticmethod
    def _read_image_source(source):
        if isinstance(source, Path):
            source = os.fspath(source)
        if isinstance(source, str):
            if PreprocessMixin._is_url(source):
                request = Request(source, headers={"User-Agent": "fotonet/0.2"})
                with urlopen(request, timeout=20) as response:
                    return Image.open(io.BytesIO(response.read())).convert("RGB")
            if not os.path.isfile(source):
                raise FileNotFoundError(f"Image source not found: {source}")
            return Image.open(source).convert("RGB")
        if isinstance(source, (Image.Image, np.ndarray)):
            return source
        raise TypeError("source must be an image path/URL, directory, PIL image, NumPy array, tensor, video path, or webcam index.")

    @staticmethod
    def _orig_images_from_tensor(tensor, retain_images=True):
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 4:
            raise ValueError("Tensor source must have shape [B,3,H,W].")
        if not retain_images:
            return [None] * int(tensor.shape[0])
        value = tensor.detach().float().cpu()
        if value.ndim == 3:
            value = value.unsqueeze(0)
        if value.ndim != 4:
            raise ValueError("Tensor source must have shape [B,3,H,W].")
        value_min, value_max = PreprocessMixin._tensor_value_range(value)
        if not math.isfinite(value_min) or not math.isfinite(value_max) or value_min < 0.0 or value_max > 255.0:
            raise ValueError("Tensor image values must be finite and in [0,1] or [0,255].")
        if value_max <= 1.0:
            value = value * 255.0
        images = [
            image.permute(1, 2, 0).clamp(0, 255).byte().numpy().copy()
            for image in value
        ]
        return images

    @staticmethod
    def _tensor_value_range(tensor):
        """Return a tensor's min/max through one host transfer when it lives on CUDA."""
        value_min, value_max = torch.aminmax(tensor.detach())
        values = torch.stack((value_min, value_max)).cpu().tolist()
        return float(values[0]), float(values[1])

    @staticmethod
    def _normalize_tensor_input(tensor, *, input_range_known=False):
        """Validate and normalize public tensor input without repeated CUDA scalar reads."""
        if input_range_known:
            return tensor

        value_min, value_max = PreprocessMixin._tensor_value_range(tensor)
        if not math.isfinite(value_min) or not math.isfinite(value_max) or value_min < 0.0 or value_max > 255.0:
            raise ValueError("Tensor image values must be finite and in [0,1] or [0,255].")
        if not tensor.is_floating_point():
            # Integer image tensors conventionally encode pixel intensities in
            # [0, 255], including the degenerate all-zero/all-one cases.
            return tensor.float().div_(255.0)
        if value_max > 1.0:
            return tensor.float().div_(255.0)
        return tensor

    @staticmethod
    def _as_rgb_uint8(img, *, bgr=False):
        arr = np.asarray(img)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.ndim != 3 or arr.shape[-1] not in (3, 4):
            raise ValueError("Image arrays must have shape [H,W], [H,W,3], or [H,W,4].")
        arr = arr[..., :3]
        if np.issubdtype(arr.dtype, np.floating):
            if not np.isfinite(arr).all() or arr.min() < 0.0 or arr.max() > 255.0:
                raise ValueError("Floating image arrays must contain finite values in [0,1] or [0,255].")
            if arr.max() <= 1.0:
                arr = arr * 255.0
            arr = np.rint(arr).astype(np.uint8)
        elif arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if bgr:
            arr = arr[..., ::-1]
        return np.ascontiguousarray(arr)

    @classmethod
    def _preprocess_rgb_array(cls, arr, imgsz, device=None):
        """One RGB letterbox adapter shared by PIL, NumPy, and BGR entry points."""
        target_h, target_w = cls._normalize_imgsz(imgsz)
        arr = cls._as_rgb_uint8(arr)
        orig_h, orig_w = arr.shape[:2]
        gain = min(target_h / max(orig_h, 1), target_w / max(orig_w, 1))
        new_w = max(int(round(orig_w * gain)), 1)
        new_h = max(int(round(orig_h * gain)), 1)
        if cv2 is not None:
            resized = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            resized = np.asarray(Image.fromarray(arr).resize((new_w, new_h), Image.BILINEAR))
        canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        pad_w = (target_w - new_w) // 2
        pad_h = (target_h - new_h) // 2
        canvas[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
        tensor = torch.from_numpy(np.ascontiguousarray(canvas.transpose(2, 0, 1))).float().div_(255.0).unsqueeze(0)
        if device is not None:
            tensor = tensor.to(device, non_blocking=True)
        return tensor, {
            "input_w": float(target_w),
            "input_h": float(target_h),
            "gain": float(gain),
            "pad_w": float(pad_w),
            "pad_h": float(pad_h),
            "orig_w": float(orig_w),
            "orig_h": float(orig_h),
        }

    @staticmethod
    def _preprocess_image(img, imgsz, device):
        """Letterbox image to the training geometry and return scale metadata."""
        if not isinstance(img, Image.Image):
            return PreprocessMixin._preprocess_rgb_array(img, imgsz, device)
        return PreprocessMixin._preprocess_rgb_array(np.asarray(img.convert("RGB")), imgsz, device)

    @staticmethod
    def _preprocess_numpy_rgb(img, imgsz, device=None):
        """Fast NumPy/OpenCV letterbox path for already-decoded RGB frames."""
        return PreprocessMixin._preprocess_rgb_array(img, imgsz, device)

    @staticmethod
    def _preprocess_numpy_bgr(img, imgsz, device=None):
        """Fast letterbox path for OpenCV BGR frames, converting to RGB only after resize."""
        return PreprocessMixin._preprocess_rgb_array(PreprocessMixin._as_rgb_uint8(img, bgr=True), imgsz, device)

    @staticmethod
    def _scale_boxes_from_letterbox(boxes, meta):
        """Map normalized xywh boxes from letterboxed square input back to the original image."""
        if meta is None or boxes.numel() == 0:
            return boxes
        input_w = float(meta.get("input_w", meta.get("imgsz")))
        input_h = float(meta.get("input_h", meta.get("imgsz")))
        gain = max(float(meta["gain"]), 1e-9)
        pad_w = float(meta["pad_w"])
        pad_h = float(meta["pad_h"])
        orig_w = max(float(meta["orig_w"]), 1.0)
        orig_h = max(float(meta["orig_h"]), 1.0)

        x, y, w, h = boxes.unbind(-1)
        x1 = (x - w * 0.5) * input_w
        y1 = (y - h * 0.5) * input_h
        x2 = (x + w * 0.5) * input_w
        y2 = (y + h * 0.5) * input_h

        x1 = ((x1 - pad_w) / gain).clamp(0, orig_w)
        y1 = ((y1 - pad_h) / gain).clamp(0, orig_h)
        x2 = ((x2 - pad_w) / gain).clamp(0, orig_w)
        y2 = ((y2 - pad_h) / gain).clamp(0, orig_h)

        cx = ((x1 + x2) * 0.5) / orig_w
        cy = ((y1 + y2) * 0.5) / orig_h
        bw = (x2 - x1).clamp_min(0) / orig_w
        bh = (y2 - y1).clamp_min(0) / orig_h
        return torch.stack((cx, cy, bw, bh), -1).clamp(0, 1)

    @staticmethod
    def _clip_normalized_xywh(boxes):
        """Clip normalized boxes through xyxy, preserving valid edge geometry."""
        if boxes.numel() == 0:
            return boxes
        x, y, w, h = boxes.unbind(-1)
        x1 = (x - w * 0.5).clamp(0, 1)
        y1 = (y - h * 0.5).clamp(0, 1)
        x2 = (x + w * 0.5).clamp(0, 1)
        y2 = (y + h * 0.5).clamp(0, 1)
        return torch.stack(((x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1), dim=-1)

    @staticmethod
    def _letterbox_tensor_batch(tensor, target_h, target_w):
        """Letterbox a normalized NCHW batch without a CPU image round-trip."""
        if tensor.ndim != 4 or tensor.shape[1] != 3:
            raise ValueError("Tensor source must have shape [B,3,H,W].")
        orig_h, orig_w = (int(tensor.shape[-2]), int(tensor.shape[-1]))
        gain = min(target_h / max(orig_h, 1), target_w / max(orig_w, 1))
        new_h = max(int(round(orig_h * gain)), 1)
        new_w = max(int(round(orig_w * gain)), 1)
        resized = F.interpolate(tensor, size=(new_h, new_w), mode="bilinear", align_corners=False)
        canvas = tensor.new_full((tensor.shape[0], 3, target_h, target_w), 114.0 / 255.0)
        pad_w = (target_w - new_w) // 2
        pad_h = (target_h - new_h) // 2
        canvas[..., pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
        meta = {
            "input_w": float(target_w),
            "input_h": float(target_h),
            "gain": float(gain),
            "pad_w": float(pad_w),
            "pad_h": float(pad_h),
            "orig_w": float(orig_w),
            "orig_h": float(orig_h),
        }
        return canvas, [dict(meta) for _ in range(int(tensor.shape[0]))]
