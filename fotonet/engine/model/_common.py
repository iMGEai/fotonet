"""Helpers shared by the ``Fotonet`` mixins."""
import functools
import inspect

_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
_VIDEO_SUFFIXES = {".avi", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"}

# ``Fotonet.train`` arguments that a v1.1 model understands, mapped to the v1.1
# engine's option names. Any other explicitly passed v1 argument is rejected
# for a v1.1 model instead of being silently ignored.
_V11_TRAIN_ARGS = {
    "epochs": "epochs",
    "imgsz": "imgsz",
    "batch": "batch",
    "lr0": "lr0",
    "lrf": "lrf",
    "nbs": "nbs",
    "warmup_epochs": "warmup_epochs",
    "save_dir": "out_dir",
    "val_period": "val_period",
    "val_batch": "val_batch",
    "workers": "workers",
    "optimizer": "optimizer",
    "momentum": "momentum",
    "weight_decay": "weight_decay",
    "grad_clip_norm": "grad_clip",
    "best_metric": "best_metric",
    "lr_scheduler": "lr_scheduler",
    "wsd_decay_fraction": "wsd_decay_fraction",
    "imgsz_schedule": "imgsz_schedule",
    "compile_model": "compile",
    "augment_hyp": "augment_hyp",
}


def _records_explicit_arguments(method):
    """Remember which parameters the caller actually passed.

    Defaults stay exactly as declared (v1 behavior is untouched); the v1.1 path
    uses the record so v1 default values never override v1.1 engine defaults.
    """
    signature = inspect.signature(method)

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        bound = signature.bind_partial(self, *args, **kwargs)
        previous = self.__dict__.get("_explicit_arguments")
        self._explicit_arguments = frozenset(
            name for name in bound.arguments if name not in {"self", "kwargs"}
        )
        try:
            return method(self, *args, **kwargs)
        finally:
            if previous is None:
                self.__dict__.pop("_explicit_arguments", None)
            else:
                self._explicit_arguments = previous

    return wrapper


class UntrainedModelWarning(UserWarning):
    """A model was built by name but has no trained weights."""
