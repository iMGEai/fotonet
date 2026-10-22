"""``Fotonet`` cOCO-style validation."""
import torch
from fotonet.utils.config import (
    class_names_are_default,
    default_class_names,
    load_data_cfg,
    normalize_class_schema,
)


def collate_detection_batch(items):
    """Module level so DataLoader workers started by spawn (Windows, macOS) can pickle it."""
    images, targets = zip(*items)
    if isinstance(images[0], dict):
        common_keys = set(images[0]).intersection(*(set(image) for image in images[1:]))
        return {
            key: torch.stack([image[key] for image in images], 0)
            for key in common_keys
        }, list(targets)
    return torch.stack(images, 0), list(targets)


class ValidateMixin:
    """COCO-style validation."""

    def val(
        self,
        data,
        imgsz=None,
        batch=8,
        conf=0.0,
        max_det=100,
        decode_max_det=300,
        operating_conf=0.25,
        operating_iou=0.50,
        workers=0,
        annotation_policy="fix",
        allow_missing_labels=False,
        source_recursive=True,
        progress=False,
        progress_interval=100,
    ):
        """Evaluate an explicit validation source using the trainer's COCO protocol.

        ``conf=0`` retains every finite detection so COCO maxDets performs the
        only score-ranked truncation.  A nonzero ``conf`` is explicitly labeled
        as a score-floor protocol. ``operating_conf`` and ``operating_iou``
        define the separately reported P/R point. A ``val`` entry is mandatory:
        validation never falls back to ``train``. A static ONNX/TorchScript
        artifact uses its exported input HxW when ``imgsz`` is omitted.
        """
        from torch.utils.data import DataLoader

        from fotonet.data.dataset import build_detection_dataset
        from fotonet.engine.validation import evaluate_detection_model

        from fotonet.data.starter import resolve_data

        cfg = load_data_cfg(resolve_data(data))
        if "val" not in cfg:
            raise ValueError(
                "Fotonet.val() requires an explicit 'val' source in the data config; "
                "it will not evaluate the training set."
            )
        if getattr(self, "_runtime_schema_pending", False):
            raise RuntimeError(
                "This metadata-less dynamic ONNX artifact has a symbolic output width, so its class count "
                "is not known yet. Run predict() once to resolve the runtime schema, or restore the export "
                "metadata/re-export before validating."
            )
        if not hasattr(self.model, "head") or not hasattr(self.model.head, "nc"):
            raise RuntimeError("Validation requires a detector runtime with a head.nc output contract.")
        model_nc = int(self.model.head.nc)
        configured_nc = cfg.get("nc")
        if configured_nc is not None and int(configured_nc) != model_nc:
            raise ValueError(
                f"Dataset nc={int(configured_nc)} does not match model nc={model_nc}. "
                "Load a compatible checkpoint/model instead of mutating the model during validation."
            )
        configured_names = cfg.get("names")
        model_names = None
        if configured_names is not None:
            _, configured_names = normalize_class_schema(
                model_nc,
                configured_names,
                context="data config",
            )
            raw_model_names = getattr(self, "names", None)
            if raw_model_names is not None:
                _, model_names = normalize_class_schema(
                    model_nc,
                    raw_model_names,
                    context="model",
                )
            if (
                model_names is not None
                and not class_names_are_default(model_names, model_nc)
                and model_names != configured_names
            ):
                raise ValueError(
                    "Dataset class names/order do not match the loaded model checkpoint. "
                    "Use a data config with the same class mapping."
                )
        batch = int(batch)
        if batch < 1:
            raise ValueError("batch must be positive")
        workers = int(workers)
        if workers < 0:
            raise ValueError("workers must be non-negative")

        runtime_format = getattr(self, "runtime_format", None)
        runtime_dynamic = getattr(self, "_runtime_dynamic", None)
        runtime_imgsz = getattr(self, "_runtime_imgsz", None)
        if runtime_format is None:
            eval_imgsz = self._normalize_imgsz(640 if imgsz is None else imgsz)
        elif runtime_dynamic is False:
            if runtime_imgsz is None:
                raise RuntimeError(
                    f"Cannot validate this static {runtime_format} artifact without its input-shape metadata. "
                    "Re-export it with FOTO-NET so the adjacent metadata file is produced."
                )
            eval_imgsz = tuple(int(value) for value in runtime_imgsz)
            if imgsz is not None and self._normalize_imgsz(imgsz) != eval_imgsz:
                raise ValueError(
                    f"This static {runtime_format} artifact accepts imgsz={eval_imgsz}, "
                    f"not {self._normalize_imgsz(imgsz)}. Re-export with dynamic=True for another validation shape."
                )
        else:
            if runtime_dynamic is None and runtime_imgsz is None:
                raise RuntimeError(
                    f"Cannot determine the input contract for '{getattr(self, 'loaded_weight_path', None)}'. "
                    "Keep the export metadata sidecar or load a native PyTorch checkpoint."
                )
            eval_imgsz = self._normalize_imgsz(640 if imgsz is None else imgsz)

        if progress:
            print("[validate] building validation dataset...", flush=True)
        dataset = build_detection_dataset(
            cfg["val"],
            imgsz=eval_imgsz,
            augment=False,
            cache_labels=True,
            cache_to_ram=False,
            num_classes=model_nc,
            annotation_policy=annotation_policy,
            allow_missing_labels=allow_missing_labels,
            source_recursive=source_recursive,
            coco_images=cfg.get("val_images", cfg.get("coco_images")),
        )
        loader = DataLoader(
            dataset,
            batch_size=batch,
            shuffle=False,
            num_workers=workers,
            pin_memory=torch.device(self.device).type == "cuda",
            collate_fn=collate_detection_batch,
            persistent_workers=workers > 0,
        )
        if progress:
            print(
                f"[validate] dataset ready: images={len(dataset)} batches={len(loader)}",
                flush=True,
            )
        metrics = evaluate_detection_model(
            self.model,
            loader,
            self.device,
            model_nc,
            conf=conf,
            coco_max_dets=max_det,
            max_det=max(int(decode_max_det), int(max_det)),
            operating_conf=operating_conf,
            operating_iou=operating_iou,
            amp=False,
            class_names=(
                configured_names
                if configured_names is not None
                else getattr(self, "names", default_class_names(model_nc))
            ),
            progress=progress,
            progress_interval=progress_interval,
        )
        # Maintain the historic spelling without making it a separate metric.
        metrics["mAP50-95"] = metrics["mAP50_95"]
        from fotonet.engine.metrics import Metrics

        result = Metrics(metrics)
        result.names = dict(configured_names or getattr(self, "names", {}) or {})
        return result
