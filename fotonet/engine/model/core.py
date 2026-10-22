"""The public ``Fotonet`` class, composed from one mixin per concern."""
import os
from pathlib import Path
from fotonet.models.resolve import available_models, is_model_ref, load_model_config
from fotonet.models.release import resolve_release_checkpoint
from fotonet.utils.general import check_device
from fotonet.utils.config import default_class_names, load_model_cfg, normalize_class_schema

from .loading import LoadingMixin
from .info import InfoMixin
from .training import TrainingMixin
from .preprocess import PreprocessMixin
from .predict import PredictMixin
from .track import TrackMixin
from .validate import ValidateMixin
from .export import ExportMixin


class Fotonet(
    LoadingMixin, InfoMixin, TrainingMixin, PreprocessMixin, PredictMixin, TrackMixin, ValidateMixin,
    ExportMixin,
):
    """
    NMS-free object detection with one self-identifying production graph.

    Use ``Fotonet('fotonete')`` for the verified v1 release checkpoint or
    ``Fotonet('path/to.pt')`` for a supported local checkpoint. Use
    ``Fotonet()`` for an untrained architecture.
    """
    MODELS = (*available_models(), "fotonetn", "fotonets", "fotonetm", "fotonetl")
    CHECKPOINT_FORMAT = 1
    TRAINING_PROTOCOL = 1

    def __init__(self, model_path=None, nc=None, task="detect", device=None, pretrained=True):
        # Preserve the zero-argument device hook used by downstream callers;
        # only pass an argument when the caller explicitly chose a device.
        self.device = check_device() if device is None else check_device(device)
        self.task = str(task).lower()
        if self.task != "detect":
            raise ValueError("fotonet supports task='detect' only.")
        self.nc, _ = normalize_class_schema(80 if nc is None else nc, context="model")
        self.model  = None
        self._model_cfg = None
        self.model_path = model_path
        self.loaded_weight_path = None
        self.feature_init_path = None
        self.runtime_format = None
        self._runtime_imgsz = None
        self._runtime_dynamic = None
        self._runtime_schema_pending = False
        # Deployment transforms are applied to a clone.  This keeps the graph
        # with BatchNorm/O2M heads intact if the caller later invokes train().
        self._training_model_before_deploy = None
        self._inference_channels_last = False
        self._infer_cache = None
        self.is_v1_1 = False
        self._callbacks = {}
        self._pretrained = bool(pretrained)
        if self._accept_v11(model_path, nc):
            return
        self._apply_model_cfg(load_model_config("fotonete"), update_nc=False)
        self.names = default_class_names(self.nc)

        if model_path is None:
            self.model = self._new_model()
            return

        model_path = os.fspath(model_path)
        path_lower = model_path.lower()

        if os.path.isfile(model_path) and path_lower.endswith((".yaml", ".yml")):
            cfg = load_model_cfg(model_path)
            self._model_cfg = cfg
            self._apply_model_cfg(cfg, update_nc=nc is None)
            self.names = default_class_names(self.nc)
            self.model = self._new_model()

        elif os.path.isfile(model_path) and path_lower.endswith((".onnx", ".torchscript", ".ts")):
            self._load_runtime_artifact(model_path)

        elif os.path.isfile(model_path) and path_lower.endswith((".engine", ".mlmodel", ".mlpackage")):
            raise RuntimeError(
                f"'{Path(model_path).suffix}' runtime loading is not available in this Python build. "
                "Use the native TensorRT/CoreML runtime integration for that platform, or load the ONNX/TorchScript artifact."
            )

        elif os.path.isfile(model_path):
            self.model = self._new_model()
            self.load(model_path)
            self.loaded_weight_path = model_path

        elif is_model_ref(path_lower):
            checkpoint_path = resolve_release_checkpoint(path_lower)
            self._apply_model_cfg(load_model_config(path_lower), update_nc=nc is None)
            self.names = default_class_names(self.nc)
            self.model = self._new_model()
            self.load(checkpoint_path)
            self.loaded_weight_path = str(checkpoint_path)
            return

        else:
            v11_ids = ("fotonetn", "fotonets", "fotonetm", "fotonetl")
            valid = ", ".join((*available_models(), *v11_ids))
            raise ValueError(f"Unknown model '{model_path}'. Use one of: {valid}, a YAML file, or a checkpoint.")
