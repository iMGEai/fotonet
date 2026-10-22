"""``Fotonet`` load checkpoints and runtime artifacts, keep model identity, save-side state."""
import json
import os
import copy
from pathlib import Path
import torch
from fotonet.models.e.graph import Detector
from fotonet.models.resolve import load_model_config
from fotonet.models.e.spec import EXPERIMENT_SCHEMA, experiment_fingerprint
from fotonet.utils.config import default_class_names, normalize_class_schema
from ._common import UntrainedModelWarning


class LoadingMixin:
    """Load checkpoints and runtime artifacts, keep model identity, save-side state."""

    def _accept_v11(self, model_path, nc):
        """Build a v1.1 model before any v1 identity is applied. Return False to keep the v1 path."""
        if model_path is None:
            return False
        from fotonet.versions.v1_1.api import validate_v11_metadata
        from fotonet.versions.v1_1.graph import V11Detector
        from fotonet.versions.v1_1.spec import FAMILY, get_spec

        path = os.fspath(model_path)
        key = str(path).strip().lower()
        if key in FAMILY:
            spec = get_spec(key)
            self.is_v1_1 = True
            self.model_id = key
            self.nc = spec.nc if nc is None else int(nc)
            weights = None
            if getattr(self, "_pretrained", True):
                from fotonet.models.release import resolve_v11_release

                weights = resolve_v11_release(key)
            if weights is not None:
                self._load_v11_checkpoint(str(weights), self._safe_torch_load(str(weights)))
                if int(self.model.nc) != int(self.nc):
                    # Explicit class count: keep the pretrained backbone, fresh class predictors.
                    from fotonet.versions.v1_1.engine import transfer_weights

                    target = V11Detector(key, nc=self.nc, imgsz=spec.imgsz).to(self.device)
                    transfer_weights(self.model, target)
                    self.model = target.eval()
                    self._finish_v11_identity()
                return True
            if getattr(self, "_pretrained", True):
                import warnings

                warnings.warn(
                    f"No published weights for '{key}' yet: this model is untrained and its "
                    "predictions are meaningless. Train it, load a checkpoint, or pass "
                    "pretrained=False to build an untrained model without this warning.",
                    UntrainedModelWarning,
                    stacklevel=3,
                )
            self.model = V11Detector(key, nc=self.nc, imgsz=spec.imgsz).to(self.device).eval()
            self._finish_v11_identity()
            return True
        if not os.path.isfile(path):
            return False
        suffix = Path(path).suffix.lower()
        if suffix == ".pt":
            # One safe read serves both versions: a v1 checkpoint is handed to
            # the v1 loader below instead of being read a second time.
            try:
                checkpoint = self._safe_torch_load(path)
            except ValueError:
                return False
            from fotonet.versions.v1_1.api import is_v11_checkpoint

            if not is_v11_checkpoint(checkpoint):
                self._pending_checkpoint = (path, checkpoint)
                return False
            self._load_v11_checkpoint(path, checkpoint, nc)
            return True
        if suffix in {".onnx", ".torchscript", ".ts"}:
            meta_path = path + ".metadata.json"
            if not os.path.isfile(meta_path):
                return False
            try:
                with open(meta_path, "r", encoding="utf-8") as handle:
                    raw = json.load(handle)
            except (OSError, json.JSONDecodeError):
                return False
            if not str(raw.get("graph_version", "")).startswith("1.1"):
                return False
            metadata = validate_v11_metadata(raw, "onnx" if suffix == ".onnx" else "torchscript")
            self._load_v11_runtime(path, metadata)
            return True
        return False

    def _load_v11_checkpoint(self, path, checkpoint, nc=None):
        """Build the stored v1.1 graph, verify its fingerprint, and load its weights."""
        from fotonet.versions.v1_1.api import checkpoint_names, inference_state_dict, load_weights
        from fotonet.versions.v1_1.graph import build_from_config
        from fotonet.versions.v1_1.spec import fingerprint_matches, get_spec

        cfg = dict(checkpoint.get("model_config") or {})
        model_id = cfg.get("model_id", "fotonetn")
        ckpt_nc = int(cfg.get("nc", get_spec(model_id).nc))
        if nc is not None and int(nc) != ckpt_nc:
            raise ValueError(f"Checkpoint nc={ckpt_nc} does not match requested nc={nc}.")
        model = build_from_config(model_id, cfg, nc=ckpt_nc).to(self.device).eval()
        if not fingerprint_matches(model.model_config, checkpoint.get("architecture_fingerprint")):
            raise ValueError(f"Checkpoint fingerprint does not match the built {model_id} graph.")
        state = inference_state_dict(checkpoint)
        if not state:
            raise ValueError("v1.1 checkpoint carries no weights.")
        load_weights(model, state)
        self.model = model
        self.is_v1_1 = True
        self.model_id = model_id
        self.nc = model.nc
        self.loaded_weight_path = path
        self.loaded_epoch = checkpoint.get("epoch")
        self._finish_v11_identity(checkpoint_names(checkpoint, model.nc))

    def _finish_v11_identity(self, names=None):
        self.names = dict(names) if names else default_class_names(self.nc)
        self.model.class_names = dict(self.names)
        self.reg_max = self.model.reg_max
        self.use_p2 = False
        self.graph_version = self.model.graph_version
        self._model_cfg = self.model.model_config
        self.architecture_fingerprint = self.model.architecture_fingerprint
        self.model_path = getattr(self, "model_path", None)
        self._infer_cache = None

    def _load_v11_runtime(self, model_path, metadata):
        suffix = Path(model_path).suffix.lower()
        if suffix == ".onnx":
            from fotonet.engine.runtimes import ONNXRuntime
            runtime = ONNXRuntime(model_path, int(metadata["nc"]), self.device, strides=metadata["strides"])
            self.runtime_format = "onnx"
        else:
            from fotonet.engine.runtimes import TorchScriptRuntime
            runtime = TorchScriptRuntime(model_path, int(metadata["nc"]), self.device)
            self.runtime_format = "torchscript"
        runtime.head.nc = int(metadata["nc"])
        self.model = runtime
        self.nc = int(metadata["nc"])
        try:
            _, artifact_names = normalize_class_schema(
                self.nc, metadata.get("names"), context="runtime artifact metadata"
            )
        except (TypeError, ValueError):
            artifact_names = None
        self.names = dict(artifact_names) if artifact_names else default_class_names(self.nc)
        self.model_id = str(metadata.get("model_id"))
        self.is_v1_1 = True
        self.reg_max = 1
        self.use_p2 = False
        self.architecture_fingerprint = str(metadata.get("architecture_fingerprint"))
        self.graph_version = str(metadata.get("graph_version"))
        self._runtime_imgsz = tuple(int(v) for v in metadata["imgsz"])
        self._runtime_dynamic = bool(metadata.get("dynamic", False))
        self._runtime_schema_pending = False
        self.loaded_weight_path = model_path

    def _load_runtime_artifact(self, model_path):
        """Load an exported inference artifact with its adjacent metadata contract."""
        metadata_path = Path(model_path).with_suffix(Path(model_path).suffix + ".metadata.json")
        suffix = Path(model_path).suffix.lower()
        if suffix == ".onnx":
            from fotonet.engine.runtimes import ONNXRuntime, validate_artifact_metadata

            if not metadata_path.is_file():
                raise RuntimeError(
                    f"ONNX artifact is missing required metadata sidecar '{metadata_path}'. Re-export it."
                )
            try:
                with open(metadata_path, "r", encoding="utf-8") as handle:
                    metadata = json.load(handle)
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"Could not read ONNX metadata '{metadata_path}'.") from exc
            metadata = validate_artifact_metadata(metadata, "onnx")
            runtime = ONNXRuntime(
                model_path,
                metadata.get("nc", self.nc),
                self.device,
                strides=metadata.get("strides"),
            )
            # ONNX preserves its output width, so class count and input shape
            # can still be recovered when a sidecar was not copied alongside
            # the artifact.
            if runtime.output_nc is not None:
                if "nc" in metadata and int(metadata["nc"]) != runtime.output_nc:
                    raise RuntimeError(
                        f"ONNX metadata nc={int(metadata['nc'])} does not match the graph output nc="
                        f"{runtime.output_nc}. Restore the matching sidecar or re-export the artifact."
                    )
                metadata = {**metadata, "nc": runtime.output_nc}
            graph_imgsz = getattr(runtime, "input_imgsz", None)
            if graph_imgsz is not None:
                declared_imgsz = tuple(metadata["imgsz"]) if "imgsz" in metadata else None
                if declared_imgsz is not None and declared_imgsz != tuple(graph_imgsz):
                    raise RuntimeError(
                        f"ONNX metadata imgsz={declared_imgsz} does not match graph input HxW={tuple(graph_imgsz)}. "
                        "Restore the matching sidecar or re-export the artifact."
                    )
                metadata = {**metadata, "imgsz": list(graph_imgsz)}
            graph_batch = getattr(runtime, "input_batch", None)
            if graph_batch is not None:
                if "batch" in metadata and int(metadata["batch"]) != int(graph_batch):
                    raise RuntimeError(
                        f"ONNX metadata batch={int(metadata['batch'])} does not match graph input batch="
                        f"{int(graph_batch)}. Restore the matching sidecar or re-export the artifact."
                    )
                metadata = {**metadata, "batch": int(graph_batch)}
            graph_dynamic = getattr(runtime, "dynamic_input", None)
            if graph_dynamic is not None:
                if "dynamic" in metadata and metadata["dynamic"] != bool(graph_dynamic):
                    raise RuntimeError(
                        f"ONNX metadata dynamic={metadata['dynamic']} does not match the graph input contract "
                        f"(dynamic={bool(graph_dynamic)}). Restore the matching sidecar or re-export the artifact."
                    )
                metadata = {**metadata, "dynamic": bool(graph_dynamic)}
            # A dynamic ONNX graph often leaves its output width symbolic.
            # When the signed sidecar supplies both class count and stride
            # geometry, that schema is complete enough to preallocate the
            # CUDA I/O-binding output. Leaving ``schema_pending`` true here
            # silently forces every GPU request through CPU copies.
            if "nc" in metadata and "strides" in metadata:
                runtime.set_output_schema(metadata["nc"])
            self.runtime_format = "onnx"
        else:
            from fotonet.engine.runtimes import TorchScriptRuntime, validate_artifact_metadata

            runtime = TorchScriptRuntime(model_path, self.nc, self.device)
            if runtime.metadata:
                # The archive is authoritative: a stale sidecar must not make
                # runtime preprocessing disagree with the serialized graph.
                metadata = runtime.metadata
            elif metadata_path.is_file():
                try:
                    with open(metadata_path, "r", encoding="utf-8") as handle:
                        metadata = json.load(handle)
                except (OSError, json.JSONDecodeError) as exc:
                    raise RuntimeError(
                        f"Could not read TorchScript runtime metadata '{metadata_path}'. "
                        "Re-export the artifact to embed its contract."
                    ) from exc
                metadata = validate_artifact_metadata(metadata, "torchscript")
            else:
                raise RuntimeError(
                    "TorchScript artifact is missing both embedded FOTO-NET metadata and its "
                    "'.metadata.json' sidecar. A traced module does not reliably retain its "
                    "class count or static input HxW. Re-export it with "
                    "Fotonet(...).export(format='torchscript', path='model.torchscript')."
                )
            self.runtime_format = "torchscript"
        try:
            self.nc, artifact_names = normalize_class_schema(
                metadata.get("nc", getattr(runtime.head, "nc", self.nc)),
                metadata.get("names"),
                context="runtime artifact metadata",
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"Runtime artifact has an invalid class schema: {exc}") from exc
        runtime.head.nc = self.nc
        self.model = runtime
        artifact_imgsz = metadata.get("imgsz")
        if isinstance(artifact_imgsz, (list, tuple)) and len(artifact_imgsz) == 2:
            self._runtime_imgsz = tuple(int(value) for value in artifact_imgsz)
        self._runtime_dynamic = bool(metadata["dynamic"]) if "dynamic" in metadata else None
        self.names = dict(artifact_names) if artifact_names is not None else default_class_names(self.nc)
        if self._runtime_imgsz is None:
            self._runtime_imgsz = getattr(self.model, "input_imgsz", None)
        if self._runtime_dynamic is None:
            self._runtime_dynamic = getattr(self.model, "dynamic_input", None)
        # A symbolic ONNX output is still fully specified when its sidecar
        # supplies both class count and stride geometry. Without either, do
        # not let validation assume the constructor's default COCO schema.
        self._runtime_schema_pending = bool(
            getattr(self.model, "schema_pending", False)
            and ("nc" not in metadata or "strides" not in metadata)
        )
        self.loaded_weight_path = model_path

    def _sync_runtime_output_schema(self, output):
        """Adopt the graph's class count once a symbolic runtime output resolves."""
        if self.runtime_format is None or not torch.is_tensor(output):
            return
        if output.ndim != 3:
            raise RuntimeError(
                f"Runtime artifact returned invalid output shape {tuple(output.shape)}; "
                "expected raw [B,N,nc+4]."
            )
        output_nc = int(output.shape[-1]) - 4
        if output_nc <= 0:
            raise RuntimeError(
                f"Runtime artifact returned invalid output shape {tuple(output.shape)}; "
                "expected raw [B,N,nc+4]."
            )
        if output_nc == self.nc:
            self._runtime_schema_pending = False
            return
        self.nc = output_nc
        self.model.head.nc = output_nc
        expected_keys = set(range(output_nc))
        if not isinstance(self.names, dict) or set(self.names) != expected_keys:
            self.names = default_class_names(output_nc)
        self._runtime_schema_pending = False

    def _apply_model_cfg(self, cfg, update_nc=True):
        self._model_cfg = dict(cfg)
        if update_nc:
            self.nc, _ = normalize_class_schema(cfg.get("nc", self.nc), context="model config")
        self.model_id = str(cfg["model_id"])
        self.architecture_schema = int(cfg["architecture_schema"])
        self.architecture_fingerprint = str(cfg["architecture_fingerprint"])
        self.profile = str(cfg["profile"])
        self.use_p2 = bool(cfg["p2"])
        self.reg_max = int(cfg["reg_max"])
        self.quality_head = bool(cfg.get("quality_head", getattr(self, "quality_head", False)))

    def _new_model(self):
        model = Detector(
            nc=self.nc,
            reg_max=self.reg_max,
        ).to(self.device)
        self.architecture_fingerprint = model.architecture_fingerprint
        model.model_config = {
            "model_id": self.model_id,
            "architecture_schema": self.architecture_schema,
            "architecture_fingerprint": self.architecture_fingerprint,
            "profile": self.profile,
            "nc": int(self.nc),
            "p2": bool(self.use_p2),
            "reg_max": int(self.reg_max),
            "quality_head": bool(self.quality_head),
            "backbone_out_channels": list(model.backbone_out_channels),
            "neck_out_channels": list(model.neck_out_channels),
            "feature_strides": list(model.feature_strides),
        }
        return model

    @classmethod
    def _checkpoint_config(cls, checkpoint):
        """Validate the sole supported production checkpoint identity."""
        if not isinstance(checkpoint, dict):
            raise ValueError("Checkpoint must be a self-identifying tensor-only mapping.")
        if checkpoint.get("checkpoint_format") == cls.CHECKPOINT_FORMAT:
            required = (
                "architecture_schema", "architecture_fingerprint", "model_id",
                "model_config", "nc",
            )
            missing = [key for key in required if checkpoint.get(key) is None]
            if missing:
                raise ValueError("Checkpoint is missing required identity: " + ", ".join(missing))
            config = load_model_config(checkpoint["model_id"])
            config["nc"] = int(checkpoint["nc"])
            expected = experiment_fingerprint(
                reg_max=config["reg_max"], nc=config["nc"]
            )
            if int(checkpoint["architecture_schema"]) != config["architecture_schema"]:
                raise ValueError("Unsupported checkpoint architecture_schema.")
            if str(checkpoint["architecture_fingerprint"]) != expected:
                raise ValueError("Checkpoint architecture_fingerprint does not match its model identity.")
            embedded = checkpoint["model_config"]
            if not isinstance(embedded, dict):
                raise ValueError("Checkpoint model_config must be a mapping.")
            for key in ("model_id", "architecture_schema", "profile", "p2", "reg_max", "quality_head"):
                if embedded.get(key) != config.get(key):
                    raise ValueError(f"Checkpoint has conflicting model_config.{key} metadata.")
            return config

        # The pre-fotonete lineages (schema 1 and 2: the removed n/s/m/l/x
        # graphs) changed tensor shapes and are deliberately rejected rather
        # than partially loaded.
        legacy_schema = checkpoint.get("architecture_schema")
        if legacy_schema is not None and int(legacy_schema) < EXPERIMENT_SCHEMA:
            raise ValueError(
                f"Checkpoint uses architecture_schema={int(legacy_schema)}, but this build is "
                f"schema {EXPERIMENT_SCHEMA}. The graph changed shape; retrain rather than "
                "attempting to port the weights."
            )
        raise ValueError(
            "Unsupported checkpoint identity. Only self-identifying checkpoint_format=1 "
            f"artifacts at architecture_schema={EXPERIMENT_SCHEMA} are accepted."
        )

    def _restore_training_model(self):
        """Restore the untouched training graph after a deploy-only transform."""
        self._infer_cache = None
        if self._training_model_before_deploy is not None:
            self.model = self._training_model_before_deploy.to(self.device)
            self._training_model_before_deploy = None
            self._inference_channels_last = False

    def _assert_trainable_graph(self):
        """Reject deploy-fused native graphs that cannot be faithfully unfused."""
        raw = self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model
        if bool(getattr(raw, "_fused", False)):
            raise RuntimeError(
                "This model contains a fused deployment graph and cannot be trained safely. "
                "Reload an unfused training checkpoint, or use the original model before fusion."
            )
        return self.model

    def _ensure_deployment_copy(self):
        """Switch inference to an isolated copy, retaining the trainable graph."""
        self._infer_cache = None
        if self.runtime_format is not None or self._training_model_before_deploy is not None:
            return self.model
        try:
            deploy = copy.deepcopy(self.model)
        except Exception as exc:
            raise RuntimeError(
                "Could not create an isolated deployment copy. Reload the model and export without "
                "prepare_for_inference() rather than fusing the live training graph."
            ) from exc
        self._training_model_before_deploy = self.model
        self.model = deploy
        return self.model

    def _safe_torch_load(self, path):
        """Load supported checkpoints without enabling pickle execution."""
        pending = self.__dict__.pop("_pending_checkpoint", None)
        if pending is not None and pending[0] == os.fspath(path):
            return pending[1]
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise ValueError(
                f"Could not safely load {path!r}; supported checkpoints must be tensor-only."
            ) from exc

    @staticmethod
    def _checkpoint_state_dict(checkpoint):
        if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
            raise ValueError("Checkpoint is missing its model tensor mapping.")
        return checkpoint["model"]

    @classmethod
    def _checkpoint_inference_state_dict(cls, checkpoint):
        state = checkpoint.get("ema_state") if isinstance(checkpoint, dict) else None
        return state if isinstance(state, dict) else cls._checkpoint_state_dict(checkpoint)

    @staticmethod
    def _checkpoint_is_inference_only(checkpoint):
        return bool(
            checkpoint.get("inference_only", False)
            or checkpoint.get("stripped_o2m", False)
            or checkpoint.get("has_o2m") is False
        )

    @staticmethod
    def _checkpoint_is_full_training_resume(checkpoint):
        required = (
            "model", "ema_state", "optimizer_state", "optimizer_name",
            "scheduler_state", "lr_scheduler", "scheduler_step_unit",
            "scaler_state", "epoch", "global_step", "optimizer_step_count",
            "rng_state", "training_run_id", "nc", "names", "model_config",
        )
        return isinstance(checkpoint, dict) and all(key in checkpoint for key in required)

    @staticmethod
    def _resume_checkpoint_error(weights_path):
        return (
            f"Cannot resume from {weights_path!r}: use a full fotonet training checkpoint "
            "containing model, optimizer, scheduler, scaler, EMA, and epoch state."
        )

    def _apply_checkpoint_config(self, config):
        self._apply_model_cfg(config, update_nc=True)
        self.names = default_class_names(self.nc)
        self.model = self._new_model()

    def _load_model_state(self, state, *, inference_only=False):
        if not isinstance(state, dict):
            raise ValueError("Checkpoint model state must be a tensor mapping.")
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        allowed_missing = (
            "head.cls_o2m.", "head.reg_o2m.", "head.quality_o2m.",
            "head.cls_adapter_o2m.", "head.reg_adapter_o2m.",
        )
        illegal_missing = [key for key in missing if not key.startswith(allowed_missing)]
        if unexpected or illegal_missing or (missing and not inference_only):
            raise ValueError(
                "Checkpoint tensors do not exactly match the declared graph: "
                f"missing={illegal_missing or missing}, unexpected={unexpected}."
            )
        return missing

    def load_weights(self, weights_path):
        """Strict-load a supported self-identifying checkpoint."""
        if self.runtime_format is not None:
            raise RuntimeError("Exported runtime artifacts cannot accept PyTorch checkpoints.")
        path = os.fspath(weights_path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        checkpoint = self._safe_torch_load(path)
        from fotonet.versions.v1_1.api import is_v11_checkpoint

        if is_v11_checkpoint(checkpoint):
            self._load_v11_checkpoint(path, checkpoint)
            return True
        config = self._checkpoint_config(checkpoint)
        self._apply_checkpoint_config(config)
        state = self._checkpoint_inference_state_dict(checkpoint)
        inference_only = self._checkpoint_is_inference_only(checkpoint)
        self._load_model_state(state, inference_only=inference_only)
        self.loaded_weight_path = path
        self._infer_cache = None
        return True

    def _ensure_training_o2m_heads(self, source_state=None, mirror=False):
        raw = self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model
        if bool(getattr(raw.head, "has_o2m", True)):
            return
        state = source_state or raw.state_dict()
        self.model = self._new_model()
        self._load_model_state(state, inference_only=True)
        if mirror:
            self._mirror_o2o_to_o2m()

    def freeze_backbone(self, freeze=True):
        """Freeze/unfreeze backbone weights (useful for first N epochs)."""
        raw = self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model
        for p in raw.backbone.parameters():
            p.requires_grad_(not freeze)
        state = "frozen" if freeze else "unfrozen"
        print(f"[INFO] Backbone {state}.")

    def add_callback(self, event, function):
        """Run ``function(payload)`` at a training event (v1.1 models).

        Events: on_train_start, on_epoch_end (the metrics row), on_val_end,
        on_train_end (the result dict).
        """
        from fotonet.versions.v1_1.engine import _CALLBACK_EVENTS

        if event not in _CALLBACK_EVENTS:
            raise ValueError(f"Unknown callback event {event!r}; expected one of {_CALLBACK_EVENTS}.")
        if not getattr(self, "is_v1_1", False):
            raise ValueError("Training callbacks are available for v1.1 models.")
        self._callbacks.setdefault(event, []).append(function)
        return self

    def _mirror_o2o_to_o2m(self):
        """Rebuild training-only O2M heads when loading slim inference checkpoints."""
        state = self.model.state_dict()
        for key in list(state.keys()):
            if "head.cls_o2m." in key:
                src = key.replace("head.cls_o2m.", "head.cls_o2o.")
            elif "head.reg_o2m." in key:
                src = key.replace("head.reg_o2m.", "head.reg_o2o.")
            elif "head.quality_o2m." in key:
                src = key.replace("head.quality_o2m.", "head.quality_o2o.")
            else:
                continue
            if src in state and state[src].shape == state[key].shape:
                state[key].copy_(state[src])
        self.model.load_state_dict(state)

    def strip_o2m_for_inference(self):
        """Remove O2M heads from an isolated inference graph.

        The original graph is retained for a later ``train()`` call, matching
        the non-mutating deploy behavior of ``prepare_for_inference()``.
        """
        if self.runtime_format is None:
            self._ensure_deployment_copy()
        raw = self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model
        if hasattr(raw, "strip_o2m_for_inference"):
            raw.strip_o2m_for_inference()
            raw.eval()
        self._infer_cache = None
        return self

    def to(self, device):
        """Move the model to ``device`` and update the default device."""
        from fotonet.utils.general import check_device

        self.device = check_device(device)
        if self.model is not None:
            self.model.to(self.device)
        self._infer_cache = None
        return self

    def load(self, model_path):
        """Load one explicit production checkpoint; identity guessing is prohibited."""
        if self.runtime_format is not None:
            raise RuntimeError("Cannot load a native checkpoint into an exported runtime")
        path = os.fspath(model_path)
        checkpoint = self._safe_torch_load(path)
        from fotonet.versions.v1_1.api import is_v11_checkpoint

        if is_v11_checkpoint(checkpoint):
            self._load_v11_checkpoint(path, checkpoint)
            return self
        config = self._checkpoint_config(checkpoint)
        checkpoint_nc, checkpoint_names = normalize_class_schema(
            checkpoint.get("nc"), checkpoint.get("names"), context="checkpoint"
        )
        config["nc"] = checkpoint_nc
        self._apply_checkpoint_config(config)
        self.names = dict(checkpoint_names or default_class_names(checkpoint_nc))
        state = self._checkpoint_inference_state_dict(checkpoint)
        self._load_model_state(state, inference_only=self._checkpoint_is_inference_only(checkpoint))
        self.loaded_weight_path = path
        self._infer_cache = None
        return self
