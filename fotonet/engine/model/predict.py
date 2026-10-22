"""``Fotonet`` inference: images, batches, tensors, video, class filters, saving."""
import os
from pathlib import Path
import torch
import numpy as np
try:
    import cv2
except Exception:
    cv2 = None
from fotonet.engine.results import Results


class PredictMixin:
    """Inference: images, batches, tensors, video, class filters, saving."""

    def prepare_for_inference(self, device=None, half=False, strip_o2m=False, fuse=True,
                              matmul_precision=None, channels_last=False):
        """Prepare an isolated deploy graph without mutating the train graph.

        Fusion, O2M stripping, and FP16 conversion happen on a deep copy.  If
        the user subsequently calls :meth:`train`, the original graph is
        restored before any optimizer or loss state is constructed.
        """
        device = device or self.device
        if self.runtime_format is None and (fuse or strip_o2m or half):
            self._ensure_deployment_copy()
        self.model.to(device).eval()
        if strip_o2m and self.runtime_format is None:
            self.strip_o2m_for_inference()
        if fuse and self.runtime_format is None:
            raw = self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model
            raw.fuse()
        if half and torch.device(device).type == "cuda":
            self.model.half()
        if channels_last and torch.device(device).type == "cuda" and self.runtime_format is None:
            # cuDNN's tensor-core convolution kernels want NHWC. Predict feeds
            # contiguous NCHW, so without this the weights are the only thing
            # in NHWC and every layer pays a layout transpose instead.
            self.model.to(memory_format=torch.channels_last)
            self._inference_channels_last = True
        if matmul_precision is not None:
            if matmul_precision not in {"highest", "high", "medium"}:
                raise ValueError("matmul_precision must be 'highest', 'high', 'medium', or None.")
            torch.set_float32_matmul_precision(matmul_precision)
        self._infer_cache = None
        return self

    def _infer_cache_key(self, device):
        raw = getattr(self.model, "_orig_mod", self.model)
        first = next(raw.parameters(), None) if hasattr(raw, "parameters") else None
        params = getattr(self, "_infer_params", None)
        versions = sum(p._version for p in params) if params and params[0] is first else (
            None if first is None else first._version
        )
        return (
            id(self.model), id(raw), device, self.runtime_format,
            bool(getattr(raw, "_fused", False)),
            bool(getattr(getattr(raw, "head", None), "has_o2m", False)),
            bool(getattr(self, "_inference_channels_last", False)),
            None if first is None else (id(first), first.device, first.dtype, versions),
        )

    def _prepare_predict_model(self, device=None):
        """Return ``(infer_model, model_dtype)`` on ``device``, caching the fused v1.1 deploy graph."""
        target = torch.device(device or self.device)
        if target.type == "cuda" and target.index is None:
            target = torch.device("cuda", torch.cuda.current_device())
        cache = getattr(self, "_infer_cache", None)
        if cache is not None and cache[0] == self._infer_cache_key(target):
            if getattr(self.model, "training", False):
                self.model.eval()
            return cache[1], cache[2]

        raw = getattr(self.model, "_orig_mod", self.model)
        first = next(raw.parameters(), None) if hasattr(raw, "parameters") else None
        if device is None and first is not None and first.device.type != target.type:
            self.device = str(first.device)
            target = first.device
        if self.runtime_format is None and getattr(self, "is_v1_1", False) and not getattr(raw, "_fused", False):
            from fotonet.versions.v1_1.api import deploy_for_inference

            self.model.eval()
            infer_model = deploy_for_inference(self.model, device=target)
            if getattr(self, "_inference_channels_last", False) and target.type == "cuda":
                infer_model = infer_model.to(memory_format=torch.channels_last)
        else:
            self.model.to(target).eval()
            infer_model = self.model

        self._infer_params = tuple(raw.parameters()) if hasattr(raw, "parameters") else ()
        infer_raw = getattr(infer_model, "_orig_mod", infer_model)
        infer_param = next(infer_raw.parameters(), None) if hasattr(infer_raw, "parameters") else None
        model_dtype = (
            infer_param.dtype if infer_param is not None else getattr(infer_model, "input_torch_dtype", torch.float32)
        )
        self._infer_cache = (self._infer_cache_key(target), infer_model, model_dtype)
        return infer_model, model_dtype

    def _predict_video(self, source, imgsz, conf, max_det, device, retain_images=True, vid_stride=1, prepared=None):
        if cv2 is None:
            raise RuntimeError("Video/webcam prediction requires opencv-python.")
        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Unable to open video/webcam source: {source}")
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                for _ in range(max(int(vid_stride), 1) - 1):
                    if not capture.grab():
                        break
                yield self._predict_batch_impl(
                    [frame], imgsz, conf, device=device, bgr=True, max_det=max_det,
                    retain_images=retain_images, prepared=prepared,
                )[0]
        finally:
            capture.release()

    def predict(
        self,
        source,
        imgsz=None,
        conf=0.25,
        max_det=300,
        device=None,
        batch=16,
        stream=False,
        retain_images=None,
        classes=None,
        vid_stride=1,
        save=False,
        save_txt=False,
        project=None,
        name=None,
        exist_ok=False,
        **kwargs,
    ):
        """Run detection and return a list of Results, or an iterator when ``stream=True``.

        ``retain_images`` defaults to ``True`` for decoded images and ``False``
        for tensor input.  The tensor default avoids a full GPU-to-CPU image
        copy just to make plotting available; pass ``True`` when pixels are
        needed for ``plot()``, ``save()``, or crops.

        ``classes`` keeps only those class ids or names. ``stream=True`` yields
        results one by one for any source (folders, lists, video, webcam).
        ``vid_stride`` skips frames of a video. ``save``/``save_txt`` write the
        plotted image and YOLO-format labels to ``project/name`` (default
        ``runs/predict/exp``, auto-incremented unless ``exist_ok``).
        """
        if kwargs:
            import difflib

            known = ["source", "imgsz", "conf", "max_det", "device", "batch", "stream", "retain_images",
                     "classes", "vid_stride", "save", "save_txt", "project", "name", "exist_ok"]
            hints = []
            for key in sorted(kwargs):
                close = difflib.get_close_matches(key, known, n=1)
                hints.append(key + (f" (did you mean '{close[0]}'?)" if close else ""))
            raise TypeError(f"Unsupported predict option(s): {', '.join(hints)}")
        class_mask = self._class_mask(classes)
        saver = self._result_saver(save, save_txt, project, name, exist_ok)
        if save and retain_images is False:
            raise ValueError("save=True needs the pixels; do not combine it with retain_images=False.")
        if saver is not None and retain_images is None and isinstance(source, torch.Tensor) and save:
            retain_images = True
        # Eager paths (tensor input) compute here; lazy ones inside the generator.
        previous = self.__dict__.get("_active_class_mask")
        self._active_class_mask = class_mask
        try:
            results = self._predict_results(source, imgsz, conf, max_det, device, batch, stream,
                                            retain_images, int(vid_stride))
        finally:
            self._active_class_mask = previous
        flow = self._with_class_mask(results, class_mask, saver)
        return flow if stream else list(flow)

    def _predict_results(self, source, imgsz, conf, max_det, device, batch, stream, retain_images,
                         vid_stride=1):
        """The original predict path; returns a list or (for streams) an iterator."""
        target_h, target_w = self._resolve_predict_imgsz(imgsz)
        conf = float(conf)
        if not 0.0 <= conf <= 1.0:
            raise ValueError("conf must be in [0, 1].")
        max_det = max(int(max_det), 1)
        prepared = self._prepare_predict_model(device)
        device = torch.device(device or self.device)
        batch = max(int(batch or 1), 1)
        retain_images_arg = retain_images
        retain_images = True if retain_images is None else bool(retain_images)

        if self._is_video_source(source):
            return self._predict_video(
                source, (target_h, target_w), conf, max_det, device, retain_images=retain_images,
                vid_stride=vid_stride, prepared=prepared,
            )

        if isinstance(source, torch.Tensor):
            tensor = source.unsqueeze(0) if source.ndim == 3 else source
            if tensor.ndim != 4 or tensor.shape[1] != 3:
                raise ValueError("Tensor source must have shape [3,H,W] or [B,3,H,W].")
            # Tensor sources intentionally default to no host copy.  Explicit
            # retain_images=True restores visualization-compatible behavior.
            tensor_retain_images = False if retain_images_arg is None else bool(retain_images_arg)
            return self._predict_tensor(
                tensor.to(device),
                self._orig_images_from_tensor(tensor, retain_images=tensor_retain_images),
                conf,
                orig_shapes=[tuple(int(v) for v in tensor.shape[-2:])] * int(tensor.shape[0]),
                max_det=max_det,
                input_range_known=False,
                # An explicit imgsz letterboxes like decoded images; without one a
                # tensor runs at its own size (callers that preprocess themselves).
                letterbox_to=(target_h, target_w) if imgsz is not None else None,
                prepared=prepared,
            )

        sources = self._expand_sources(source)

        def batches():
            for start in range(0, len(sources), batch):
                chunk = sources[start:start + batch]
                outputs = self._predict_batch_impl(
                    chunk, (target_h, target_w), conf, device=device,
                    max_det=max_det, retain_images=retain_images, prepared=prepared,
                )
                for item, result in zip(chunk, outputs):
                    if isinstance(item, (str, Path)):
                        result.path = os.fspath(item)
                    yield result

        return batches()

    def _class_mask(self, classes):
        """Bool [nc] tensor for a ``classes=`` filter (ids or names), or None."""
        if classes is None:
            return None
        wanted = classes if isinstance(classes, (list, tuple, set)) else [classes]
        by_name = {str(v).lower(): int(k) for k, v in dict(self.names or {}).items()}
        mask = torch.zeros(int(self.nc), dtype=torch.bool)
        for item in wanted:
            if isinstance(item, str) and not item.isdigit():
                if item.lower() not in by_name:
                    raise ValueError(f"Unknown class name {item!r}; known: {sorted(by_name)}")
                mask[by_name[item.lower()]] = True
            else:
                index = int(item)
                if not 0 <= index < int(self.nc):
                    raise ValueError(f"Class id {index} is outside 0..{int(self.nc) - 1}.")
                mask[index] = True
        return mask

    def _with_class_mask(self, results, mask, saver):
        """Apply the class filter while each result is computed, then save it."""
        iterator = iter(results)
        index = 0
        while True:
            previous = self.__dict__.get("_active_class_mask")
            self._active_class_mask = mask
            try:
                result = next(iterator)
            except StopIteration:
                return
            finally:
                self._active_class_mask = previous
            if saver is not None:
                saver(result, index)
            index += 1
            yield result

    @staticmethod
    def _increment_dir(path, exist_ok=False):
        path = Path(path)
        if exist_ok or not path.exists():
            return path
        index = 2
        while Path(f"{path}{index}").exists():
            index += 1
        return Path(f"{path}{index}")

    def _result_saver(self, save, save_txt, project, name, exist_ok):
        if not (save or save_txt):
            return None
        save_dir = self._increment_dir(Path(project or "runs/predict") / (name or "exp"), exist_ok)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.last_save_dir = str(save_dir)

        def saver(result, index):
            stem = Path(getattr(result, "path", None) or f"{index:06d}").stem
            if save:
                result.save(str(save_dir / f"{stem}.jpg"))
            if save_txt:
                result.save_txt(str(save_dir / "labels" / f"{stem}.txt"), save_conf=True)
            result.save_dir = str(save_dir)

        return saver

    def _predict_batch(self, sources, imgsz, conf=0.25, device=None, max_det=300,
                       retain_images=True):
        return self._predict_batch_impl(
            sources, imgsz, conf, device=device, bgr=False, max_det=max_det,
            retain_images=retain_images,
        )

    def predict_bgr(self, source, imgsz=None, conf=0.25, max_det=300, device=None,
                    batch=16, retain_images=True):
        """Run RGB-model inference from BGR OpenCV frames and return a list of Results."""
        target_h, target_w = self._resolve_predict_imgsz(imgsz)
        prepared = self._prepare_predict_model(device)
        device = torch.device(device or self.device)
        batch = max(int(batch or 1), 1)
        sources = list(source) if isinstance(source, (list, tuple)) else [source]
        results = []
        for start in range(0, len(sources), batch):
            results.extend(
                self._predict_batch_impl(
                    sources[start:start + batch], (target_h, target_w), conf, device=device,
                    bgr=True, max_det=max_det, retain_images=bool(retain_images),
                    prepared=prepared,
                )
            )
        return results

    def _predict_batch_impl(self, sources, imgsz, conf=0.25, device=None, bgr=False,
                            max_det=300, retain_images=True, prepared=None):
        if not sources:
            return []
        device = torch.device(device or self.device)
        if (device.type == "cuda" and cv2 is not None
                and not os.environ.get("FOTONET_NO_FAST_PATH")
                and all(isinstance(s, np.ndarray) and s.dtype == np.uint8
                        and s.ndim == 3 and s.shape[2] == 3 for s in sources)):
            return self._predict_bgr_fast(
                sources, imgsz, conf, device, max_det=max_det,
                retain_images=retain_images, bgr=bgr, prepared=prepared,
            )
        tensors = []
        orig_imgs = []
        orig_shapes = []
        metas = []
        for source in sources:
            img = source if bgr else self._read_image_source(source)
            tensor, meta = self._preprocess_numpy_bgr(img, imgsz, device=None) if bgr else self._preprocess_image(img, imgsz, device=None)
            tensors.append(tensor)
            image_array = np.asarray(img)
            orig_shapes.append(tuple(int(v) for v in image_array.shape[:2]))
            if retain_images:
                if bgr:
                    orig_imgs.append(
                        cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
                        if cv2 is not None and image_array.dtype == np.uint8
                        else image_array[..., ::-1].copy()
                    )
                else:
                    orig_imgs.append(img.copy() if isinstance(img, np.ndarray) else img)
            else:
                orig_imgs.append(None)
            metas.append(meta)
        batch_tensor = torch.cat(tensors, 0).to(device, non_blocking=True)
        return self._predict_tensor(
            batch_tensor, orig_imgs, conf, letterbox_meta=metas, orig_shapes=orig_shapes,
            max_det=max_det, input_range_known=True, prepared=prepared,
        )

    def _predict_bgr_fast(self, sources, imgsz, conf, device, max_det=300,
                          retain_images=True, bgr=True, prepared=None):
        """CUDA fast path for BGR and RGB uint8 NumPy frames: reusable pinned
        canvas and on-device normalize (plus BGR->RGB flip when ``bgr=True``)."""
        from fotonet.fast import _letterbox_params

        target_h, target_w = imgsz
        stage = getattr(self, "_fast_stage", None)
        if stage is None or stage["canvas"].shape != (len(sources), target_h, target_w, 3):
            pinned = torch.full(
                (len(sources), target_h, target_w, 3), 114, dtype=torch.uint8, pin_memory=True
            )
            stage = {
                "canvas": pinned.numpy(),
                "pinned": pinned,
                "geometry": [None] * len(sources),
            }
            self._fast_stage = stage
        canvas = stage["canvas"]
        pinned = stage["pinned"]
        geometry = stage.get("geometry")
        if geometry is None or len(geometry) != len(sources):
            geometry = [None] * len(sources)
            stage["geometry"] = geometry
        metas = []
        orig_imgs = []
        orig_shapes = []
        for slot, img in enumerate(sources):
            if not img.flags.c_contiguous:
                img = np.ascontiguousarray(img)
            orig_h, orig_w, gain, new_h, new_w, pad_h, pad_w = _letterbox_params(
                img.shape, target_h, target_w)
            geom = (new_h, new_w, pad_h, pad_w)
            view = canvas[slot]
            if geometry[slot] != geom:
                view.fill(114)
                geometry[slot] = geom
            cv2.resize(img, (new_w, new_h),
                       dst=view[pad_h:pad_h + new_h, pad_w:pad_w + new_w],
                       interpolation=cv2.INTER_LINEAR)
            metas.append({
                "input_w": float(target_w),
                "input_h": float(target_h),
                "gain": float(gain),
                "pad_w": float(pad_w),
                "pad_h": float(pad_h),
                "orig_w": float(orig_w),
                "orig_h": float(orig_h),
            })
            orig_shapes.append((int(orig_h), int(orig_w)))
            orig_imgs.append(
                (cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if bgr else img.copy())
                if retain_images else None
            )

        infer_model, model_dtype = prepared or self._prepare_predict_model(device)
        gpu = pinned.to(device, non_blocking=True).permute(0, 3, 1, 2)
        if bgr:
            gpu = gpu.flip(1)
        tensor = gpu.to(torch.float32).div_(255.0)
        if model_dtype != torch.float32:
            tensor = tensor.to(model_dtype)
        return self._predict_tensor(
            tensor, orig_imgs, conf, letterbox_meta=metas,
            orig_shapes=orig_shapes, max_det=max_det, input_range_known=True,
            prepared=(infer_model, model_dtype),
        )

    def _predict_tensor(
        self,
        tensor,
        orig_img,
        conf=0.25,
        letterbox_meta=None,
        orig_shapes=None,
        force_list=True,
        max_det=300,
        input_range_known=False,
        letterbox_to=None,
        prepared=None,
    ):
        infer_model, model_dtype = prepared or self._prepare_predict_model(tensor.device)
        tensor = self._normalize_tensor_input(tensor, input_range_known=input_range_known)
        if tensor.dtype != model_dtype:
            tensor = tensor.to(dtype=model_dtype)
        if (letterbox_to is not None and letterbox_meta is None
                and tuple(int(v) for v in tensor.shape[-2:]) != tuple(int(v) for v in letterbox_to)):
            tensor, letterbox_meta = self._letterbox_tensor_batch(tensor, *letterbox_to)

        # A fixed-shape artifact cannot consume arbitrary direct tensors.  For
        # the public tensor API we letterbox on-device and carry the inverse
        # transform into Results; decoded image sources already have this shape
        # because ``_resolve_predict_imgsz`` selected the artifact contract.
        if (
            self.runtime_format is not None
            and self._runtime_dynamic is False
            and self._runtime_imgsz is not None
            and tuple(int(value) for value in tensor.shape[-2:]) != tuple(self._runtime_imgsz)
        ):
            if letterbox_meta is not None:
                raise ValueError(
                    f"This static {self.runtime_format} artifact expects HxW={tuple(self._runtime_imgsz)}; "
                    "decoded inputs must be preprocessed at that size."
                )
            tensor, letterbox_meta = self._letterbox_tensor_batch(tensor, *self._runtime_imgsz)

        if getattr(self, "_inference_channels_last", False):
            tensor = tensor.contiguous(memory_format=torch.channels_last)
        with torch.inference_mode():
            out = infer_model(tensor)
        self._sync_runtime_output_schema(out)
        nc = int(infer_model.head.nc)
        pred_logits = out[:, :, :nc]
        pred_boxes  = out[:, :, nc:]

        if pred_logits.ndim == 2:
            pred_logits = pred_logits.unsqueeze(0)
            pred_boxes = pred_boxes.unsqueeze(0)

        batched_orig = orig_img if isinstance(orig_img, list) else [orig_img]
        batched_meta = letterbox_meta if isinstance(letterbox_meta, list) else [letterbox_meta] * pred_logits.shape[0]
        batched_shapes = orig_shapes if isinstance(orig_shapes, list) else [orig_shapes] * pred_logits.shape[0]
        results = []
        for index in range(pred_logits.shape[0]):
            boxes, scores, classes = self._postprocess_single(
                pred_logits[index],
                pred_boxes[index],
                conf=conf,
                letterbox_meta=batched_meta[index],
                max_det=max_det,
            )
            results.append(
                Results(
                    batched_orig[index], boxes, scores, classes, names=self.names,
                    orig_shape=batched_shapes[index],
                )
            )

        return results

    def _postprocess_single(self, pred_logits, pred_boxes, conf=0.25,
                            letterbox_meta=None, max_det=300):
        num_anchors, num_classes = pred_logits.shape
        topk = min(int(max_det), num_anchors)
        class_mask = self.__dict__.get("_active_class_mask")
        if class_mask is not None:
            pred_logits = pred_logits.masked_fill(
                ~class_mask.to(device=pred_logits.device), float("-inf")
            )

        # Two-stage NMS-free selection. Reducing each anchor to its argmax class
        # first would cap the image at one object per anchor, so two genuinely
        # overlapping objects of different classes on the same cell could never
        # both be reported. Stage one narrows to the strongest `topk` anchors
        # on raw logits (sigmoid is strictly monotonic), stage two applies
        # sigmoid on the CPU shortlist and ranks (anchor, class) pairs.
        _, anchor_idx = pred_logits.amax(-1).topk(topk)
        short = torch.cat(
            (pred_logits.index_select(0, anchor_idx), pred_boxes.index_select(0, anchor_idx)),
            dim=-1,
        ).float().cpu()
        probs = short[:, :num_classes].sigmoid()
        short_boxes = short[:, num_classes:]

        scores, pair_idx = probs.flatten().topk(topk)
        mask = scores > conf
        scores = scores[mask]
        pair_idx = pair_idx[mask]
        rows = pair_idx.div(num_classes, rounding_mode="floor")
        classes = pair_idx.remainder(num_classes)
        boxes = short_boxes[rows]

        if letterbox_meta is not None:
            return self._scale_boxes_from_letterbox(boxes, letterbox_meta), scores, classes
        # Tensor input has no letterbox inverse transform. Clamp it too, so
        # right/bottom stride padding never leaks out-of-image coordinates.
        return self._clip_normalized_xywh(boxes), scores, classes

    def __call__(self, source, **kwargs):
        return self.predict(source, **kwargs)
