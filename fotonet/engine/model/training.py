"""``Fotonet`` training entry points for v1 and v1.1 models."""
import os
from fotonet.models.resolve import is_model_ref
from fotonet.utils.config import default_class_names, load_data_cfg, normalize_class_schema
from ._common import _V11_TRAIN_ARGS, _records_explicit_arguments


class TrainingMixin:
    """Training entry points for v1 and v1.1 models."""

    def train_from_recipe(self, data, recipe="fotonete_scratch", **overrides):
        """Train from a packaged/resolved recipe, with explicit overrides.

        This method does not execute a hidden recipe: it returns the same value
        as :meth:`train` and prints the source recipe before any training begins.
        """
        if getattr(self, "is_v1_1", False):
            return self._train_v11(data, recipe=recipe, options=overrides)
        from fotonet.config.recipes import load_training_recipe

        resolved = load_training_recipe(recipe)
        recipe_model = resolved.get("model")
        v11_ids = ("fotonetn", "fotonets", "fotonetm", "fotonetl")
        if recipe_model and not (is_model_ref(recipe_model) or recipe_model in v11_ids):
            raise ValueError(f"Recipe '{resolved['name']}' names unsupported model '{recipe_model}'.")
        settings = dict(resolved["settings"])
        settings["recipe_name"] = resolved["name"]
        settings.update(overrides)
        return self.train(data=data, **settings)

    def _train_v11(self, data, recipe=None, options=None):
        """Run the v1.1 engine, then load the trained weights and class names."""
        from fotonet.versions.v1_1.engine import train as v11_train

        if self.runtime_format is not None:
            raise RuntimeError("Exported runtime artifacts cannot be trained; load the .pt checkpoint.")
        self._restore_training_model()
        options = dict(options or {})
        options.setdefault("device", str(self.device))
        if self._callbacks and "callbacks" not in options:
            options["callbacks"] = {event: list(fns) for event, fns in self._callbacks.items()}
        result = v11_train(
            model=self.model, model_id=self.model_id, data=data, recipe=recipe, **options,
        )
        trained = None
        if isinstance(result, dict):
            trained = result.get("best_checkpoint") or result.get("last_checkpoint")
        if trained and os.path.isfile(trained):
            self._load_v11_checkpoint(trained, self._safe_torch_load(trained))
        return result

    def _v11_train_options(self, call_args, kwargs):
        """Translate an explicit ``Fotonet.train`` call into v1.1 engine options."""
        explicit = getattr(self, "_explicit_arguments", frozenset(call_args))
        options = {}
        unsupported = []
        for name in sorted(explicit):
            if name in {"data", "resume", "weights", "recipe_name"}:
                continue
            if name in _V11_TRAIN_ARGS:
                if call_args[name] is not None:
                    options[_V11_TRAIN_ARGS[name]] = call_args[name]
            else:
                unsupported.append(name)
        if unsupported:
            raise TypeError(
                f"{', '.join(unsupported)}: v1 training option(s) not supported by v1.1 model "
                f"'{self.model_id}'. Supported: {', '.join(sorted(_V11_TRAIN_ARGS))}, plus v1.1 "
                "engine options as keyword arguments."
            )
        resume, weights = call_args.get("resume"), call_args.get("weights")
        if isinstance(resume, (str, os.PathLike)):
            options["resume"] = os.fspath(resume)
        elif resume and weights:
            options["resume"] = os.fspath(weights)
        elif resume:
            options["resume"] = True
        options.update(kwargs)
        return options

    @_records_explicit_arguments
    def train(
        self, data, epochs=100, imgsz=640, batch=16, lr0=0.01, lrf=0.01,
        nbs=128, warmup_epochs=3, frozen_epochs=0, save_dir=".", val_split=None,
        val_period=1, val_batch=None, workers=None, pin_memory=True,
        cache_to_ram=True, ram_cache_images=1024, amp=True,
        amp_init_scale=65536.0, val_amp=None, cos_lr=True,
        lr_scheduler="Cosine", lr_drop_factor=0.92, lr_drop_patience=5,
        lr_drop_threshold=0.001, lr_drop_min_lr=1e-5, weights=None,
        wsd_decay_shape="linear", wsd_decay_fraction=0.25,
        wsd_decay_start_epoch=None, wsd_ema_reanchor=True,
        wsd_ema_decay_start=0.99, wsd_ema_ramp_epochs=20.0,
        resume=False, pretrained=False, compile_model=False, save_period=-1,
        slim_best=True, best_metric="mAP50_95", save_last=True,
        optimizer="sgd", momentum=0.937, weight_decay=0.0005,
        augment_hyp=None, augmentation_passes=None, loss_hyp=None,
        matcher_hyp=None, imgsz_schedule=None, val_subset_size=0,
        full_val_after=1.0, cache_labels=True, disk_cache_images=False,
        disk_cache_dir=None, unfreeze_backbone_at=None, coco_max_dets=100,
        val_max_det=300, val_conf=0.0, operating_conf=0.25, operating_iou=0.50,
        amp_dtype="float16", channels_last=True, cudnn_benchmark=True,
        grad_clip_norm=10.0, recipe_name=None,
        annotation_policy="fix", allow_missing_labels=False,
        source_recursive=True, train_dataset=None, _training_run_id=None,
        sampling_hyp=None, afss_orbit_hyp=None, art_hyp=None,
        distill=None, distill_teacher=None, distill_weight=0.0,
        distill_labels_dir=None, allow_unscored_pseudo_labels=False,
        profile=False, profiling_hyp=None, epoch_cut=1, cuda_graphs=False,
        **kwargs,
    ):
        """Train or resume the uniform production protocol."""
        if getattr(self, "is_v1_1", False):
            call_args = {
                name: value for name, value in locals().items()
                if name not in {"self", "kwargs"}
            }
            options = self._v11_train_options(call_args, kwargs)
            return self._train_v11(data, recipe=recipe_name, options=options)
        self._restore_training_model()
        self._assert_trainable_graph()
        epochs = int(epochs)
        if epochs < 1:
            raise ValueError("epochs must be a positive integer")
        max_stride = max(int(stride) for stride in self.model.head.strides)
        for label, size in [("imgsz", imgsz)]:
            if int(size) <= 0 or int(size) % max_stride:
                raise ValueError(f"{label} must be positive and divisible by {max_stride}")
        if imgsz_schedule:
            for index, item in enumerate(imgsz_schedule):
                size = item.get("imgsz", item.get("size")) if isinstance(item, dict) else item[1]
                if int(size) <= 0 or int(size) % max_stride:
                    raise ValueError(f"imgsz_schedule[{index}] must be divisible by {max_stride}")
        if isinstance(resume, (str, os.PathLike)):
            weights, resume = os.fspath(resume), True
        weights = os.fspath(weights) if weights is not None else None
        if resume and pretrained:
            raise ValueError("resume and pretrained are mutually exclusive")
        locked_off = {
            "sampling_hyp": sampling_hyp not in (None, {}, {"strategy": "uniform", "seed": 0, "dataset_mismatch": "error"}),
            "afss_orbit_hyp": bool((afss_orbit_hyp or {}).get("enabled", False)),
            "art_hyp": bool((art_hyp or {}).get("enabled", False)),
            "distill": distill not in (None, False),
            "distill_teacher": distill_teacher not in (None, False, ""),
            "distill_weight": float(distill_weight or 0.0) != 0.0,
            "distill_labels_dir": distill_labels_dir not in (None, False, ""),
            "allow_unscored_pseudo_labels": bool(allow_unscored_pseudo_labels),
            "profile": bool(profile) or bool((profiling_hyp or {}).get("enabled", False)),
            "epoch_cut": int(epoch_cut) != 1,
            "cuda_graphs": bool(cuda_graphs),
        }
        enabled = [name for name, value in locked_off.items() if value]
        if enabled:
            raise ValueError("Removed training options cannot be enabled: " + ", ".join(enabled))
        if unfreeze_backbone_at is not None:
            frozen_epochs = max(int(unfreeze_backbone_at), 0)

        data_config = load_data_cfg(data)
        data_nc, data_names = data_config.get("nc"), data_config.get("names")
        resume_checkpoint = None
        if resume:
            checkpoint_path = weights or self.loaded_weight_path
            if not checkpoint_path:
                raise ValueError("resume requires weights='.../fotonet_last.pt'")
            resume_checkpoint = self._safe_torch_load(checkpoint_path)
            config = self._checkpoint_config(resume_checkpoint)
            if not self._checkpoint_is_full_training_resume(resume_checkpoint):
                raise ValueError(self._resume_checkpoint_error(checkpoint_path))
            candidate = resume_checkpoint.get("checkpoint_format") is None
            if resume_checkpoint.get("training_protocol") != self.TRAINING_PROTOCOL and not candidate:
                raise ValueError("Unsupported or missing training_protocol")
            if candidate:
                states = (resume_checkpoint.get("sampling_state"), resume_checkpoint.get("afss_orbit_state"), resume_checkpoint.get("art_state"))
                if any(value is not None for value in states):
                    raise ValueError("Active-run migration requires controller-free state")
                if any((resume_checkpoint.get(key) or {}).get("enabled", False) for key in ("afss_orbit_hyp", "art_hyp")):
                    raise ValueError("Active-run migration rejects enabled adaptive controllers")
            checkpoint_nc, checkpoint_names = normalize_class_schema(
                resume_checkpoint.get("nc"), resume_checkpoint.get("names"),
                context="resume checkpoint",
            )
            if data_nc is not None and int(data_nc) != checkpoint_nc:
                raise ValueError("Resume checkpoint class count does not match the dataset")
            if data_names is not None and checkpoint_names is not None and data_names != checkpoint_names:
                raise ValueError("Resume checkpoint class order does not match the dataset")
            config["nc"] = checkpoint_nc
            self._apply_checkpoint_config(config)
            self.names = dict(checkpoint_names or data_names or default_class_names(checkpoint_nc))
            self.model.load_state_dict(self._checkpoint_state_dict(resume_checkpoint), strict=True)
            optimizer = resume_checkpoint.get("optimizer_name", optimizer)
            self.loaded_weight_path = checkpoint_path
        elif pretrained:
            checkpoint_path = weights or self.loaded_weight_path
            if not checkpoint_path:
                raise ValueError("pretrained requires a supported checkpoint path")
            checkpoint = self._safe_torch_load(checkpoint_path)
            config = self._checkpoint_config(checkpoint)
            checkpoint_nc, checkpoint_names = normalize_class_schema(
                checkpoint.get("nc"), checkpoint.get("names"), context="pretrained checkpoint"
            )
            if data_nc is not None and int(data_nc) != checkpoint_nc:
                raise ValueError("Pretrained checkpoint class count does not match the dataset")
            config["nc"] = checkpoint_nc
            self._apply_checkpoint_config(config)
            self.names = dict(checkpoint_names or data_names or default_class_names(checkpoint_nc))
            state = self._checkpoint_inference_state_dict(checkpoint)
            missing = self._load_model_state(state, inference_only=self._checkpoint_is_inference_only(checkpoint))
            self._ensure_training_o2m_heads(state, mirror=True)
            if missing:
                self._mirror_o2o_to_o2m()
            self._loaded_missing_o2m = False
            self.loaded_weight_path = checkpoint_path
        else:
            if data_nc is not None and int(data_nc) != self.nc:
                if self.loaded_weight_path:
                    # Fine-tune onto new classes: keep every tensor whose shape
                    # still fits, start only the class predictors fresh.
                    kept, total = self._rebuild_for_classes(int(data_nc))
                    print(f"[INFO] Fine-tuning from '{self.loaded_weight_path}': "
                          f"{kept}/{total} tensors kept, nc={self.nc}.")
                else:
                    self.nc = int(data_nc)
                    self.model = self._new_model()
            if getattr(self, "_loaded_missing_o2m", False):
                # Release checkpoints ship without the training-only O2M heads.
                self._mirror_o2o_to_o2m()
                self._loaded_missing_o2m = False
            if data_names is not None:
                self.names = dict(data_names)

        dataset = train_dataset
        if dataset is not None and val_split not in (None, 0, 0.0):
            raise ValueError("train_dataset cannot be combined with val_split")
        if dataset is None and val_split not in (None, 0, 0.0):
            from fotonet.data.dataset import build_detection_dataset
            if "train" not in data_config:
                raise ValueError("val_split requires a train source")
            dataset = build_detection_dataset(
                data_config["train"], imgsz=imgsz, augment=True,
                cache_labels=cache_labels, disk_cache_images=disk_cache_images,
                disk_cache_dir=disk_cache_dir, cache_to_ram=cache_to_ram,
                ram_cache_images=ram_cache_images, augment_hyp=augment_hyp,
                num_classes=self.nc, annotation_policy=annotation_policy,
                allow_missing_labels=allow_missing_labels, source_recursive=source_recursive,
                coco_images=data_config.get("train_images", data_config.get("coco_images")),
            )
        from fotonet.engine.trainer import Trainer
        trainer = Trainer(
            self.model, data_config, epochs=epochs, imgsz=imgsz, lr0=lr0, lrf=lrf,
            batch=batch, val_batch=val_batch, nbs=nbs, warmup_epochs=warmup_epochs,
            save_dir=save_dir, val_split=val_split, val_period=val_period,
            workers=workers, pin_memory=pin_memory, cache_to_ram=cache_to_ram,
            ram_cache_images=ram_cache_images, amp=amp, amp_init_scale=amp_init_scale,
            val_amp=val_amp, cos_lr=cos_lr, lr_scheduler=lr_scheduler,
            lr_drop_factor=lr_drop_factor, lr_drop_patience=lr_drop_patience,
            lr_drop_threshold=lr_drop_threshold, lr_drop_min_lr=lr_drop_min_lr,
            wsd_decay_shape=wsd_decay_shape, wsd_decay_fraction=wsd_decay_fraction,
            wsd_decay_start_epoch=wsd_decay_start_epoch,
            wsd_ema_reanchor=wsd_ema_reanchor,
            wsd_ema_decay_start=wsd_ema_decay_start,
            wsd_ema_ramp_epochs=wsd_ema_ramp_epochs,
            resume_ckpt=resume_checkpoint, compile_model=compile_model,
            save_period=save_period, slim_best=slim_best, best_metric=best_metric,
            save_last=save_last, optimizer=optimizer, momentum=momentum,
            weight_decay=weight_decay, augment_hyp=augment_hyp,
            augmentation_passes=augmentation_passes, loss_hyp=loss_hyp,
            matcher_hyp=matcher_hyp, imgsz_schedule=imgsz_schedule,
            val_subset_size=val_subset_size, full_val_after=full_val_after,
            cache_labels=cache_labels, disk_cache_images=disk_cache_images,
            disk_cache_dir=disk_cache_dir, coco_max_dets=coco_max_dets,
            val_max_det=val_max_det, amp_dtype=amp_dtype,
            channels_last=channels_last, cudnn_benchmark=cudnn_benchmark,
            grad_clip_norm=grad_clip_norm, recipe_name=recipe_name,
            val_conf=val_conf, operating_conf=operating_conf,
            operating_iou=operating_iou, annotation_policy=annotation_policy,
            allow_missing_labels=allow_missing_labels, source_recursive=source_recursive,
            _training_run_id=_training_run_id,
        )
        return trainer.train(dataset, frozen_epochs=frozen_epochs)
