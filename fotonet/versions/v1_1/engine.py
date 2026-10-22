"""fotonet v1.1 training engine.

One trainer. The knobs it reads are the knobs that run:

- data: GPU mosaic assembly (``gpu_decode``) with JPEG decode in the workers
  or on the GPU (``jpeg_decoder``), GPU affine/HSV/flip augmentation,
  progressive image sizes, mosaic close,
  ``fraction``, ``single_cls``, teacher pseudo-labels (``teacher_cache``);
- model: the deployed O2O head trains on detached features (``o2o_detach``)
  while the training-only O2M head, its objectness branch (``w_obj``), and
  optional extra O2O groups (``o2o_groups``) supervise the backbone;
- assignment: ``o2o_metric`` ("consistent" or "self"), sparse candidates
  (``matcher_candidates``);
- optimization: AdamW (fused on CUDA) or the fixed MuonSGD, WSD, cosine, or a
  pure decay schedule (``lr_scheduler="decay"``, used by ``finish``), EMA,
  ``freeze``, a wall-clock ``time`` budget, ``patience``;
- bookkeeping: exact resume, a sparse-then-dense validation schedule with the
  fast COCO evaluator, run folders (``project``/``name``), callbacks,
  TensorBoard, ``images_seen`` in every metrics row, and the step profiler.

The step itself does not read device values on the host except once per
optimizer step (the gradient-norm check) and every 50 steps for the progress
bar.
"""
from __future__ import annotations

import os

import torch

from .api import V11_FORMAT, is_v11_checkpoint, load_checkpoint

from .trainer.options import (  # noqa: F401  (old import paths)
    _BACKBONE_ORDER, _CALLBACK_EVENTS, _check_options, _COMPILE_MIN_STEPS, _EXTRA_OPTIONS,
    _HEAD_OPTION_KEYS, _M_TIERS, _recipe_overrides, _require_square, _resolve_compile, DEFAULTS,
    select_device,
)
from .trainer.schedule import (  # noqa: F401  (old import paths)
    _epoch_ramp, _final_val_epochs, _lr_at, _o2m_enabled, _parse_schedule, _Schedule, _should_validate,
)
from .trainer.data import (  # noqa: F401  (old import paths)
    _close_loader, _collate, _epoch_data_plan, _load_data_cfg, _m_bucket, _sample_fraction,
    _SingleClassView, _TrainDataset,
)
from .trainer.state import (  # noqa: F401  (old import paths)
    _EMA, _freeze, _resolve_training_model, _run_dir, capture_rng_state, restore_rng_state,
    transfer_weights,
)
from .trainer.callbacks import (  # noqa: F401  (old import paths)
    _callbacks, _fire,
)
from .trainer.step import (  # noqa: F401  (old import paths)
    _bank_grads, _EagerFallback, _restore_grads, _shape_key,
)
from .trainer.loop import (  # noqa: F401  (old import paths)
    _train,
)


def train(model=None, model_id="fotonetn", data="coco.yaml", recipe=None, imgsz=None,
          resume=None, **kwargs):
    """Train a v1.1 model.

    ``model`` is a v1.1 detector, a v1.1 checkpoint path to fine-tune from, or
    None to build ``model_id`` from scratch. Unknown options raise ``TypeError``.
    A different class count than the model's re-initializes only the class
    predictors.
    """
    threads = torch.get_num_threads()
    try:
        return _train(model, model_id, data, recipe, imgsz, resume, **kwargs)
    finally:
        torch.set_num_threads(threads)


def train_cli(model_id="fotonetn", data="coco.yaml", recipe=None, imgsz=None, **kwargs):
    print(f"fotonet train | model={model_id} data={data}" + (f" recipe={recipe}" if recipe else ""))
    return train(model_id=model_id, data=data, recipe=recipe, imgsz=imgsz, **kwargs)


def finish(run, decay_epochs=10, data=None, **kwargs):
    """Turn any stable-phase checkpoint into a finished model with a short decay.

    Loads the EMA weights of ``run`` (a run folder or a ``*_last.pt``), starts a
    fresh optimizer, and trains ``decay_epochs`` with the learning rate decaying
    from ``lr0`` to ``lr0 * lrf`` and mosaic closed throughout. Writes to
    ``<run folder>/finish``.
    """
    path = os.fspath(run)
    if os.path.isdir(path):
        candidates = [os.path.join(path, f) for f in os.listdir(path) if f.endswith("_last.pt")]
        if not candidates:
            raise SystemExit(f"No *_last.pt in {path}.")
        path = candidates[0]
    ckpt = load_checkpoint(path)
    if not is_v11_checkpoint(ckpt):
        raise SystemExit(f"{path} is not a fotonet v1.1 checkpoint.")
    if data is None:
        raise SystemExit("finish needs data=<data.yaml> (the dataset the run trained on).")
    options = {
        "epochs": int(decay_epochs),
        "lr_scheduler": "decay",
        "warmup_epochs": 0,
        "close_mosaic": int(decay_epochs),
        # A stable-phase checkpoint is past the quality ramp; keep it finished.
        "quality_start_epoch": 1,
        "quality_warmup_epochs": 0,
        "out_dir": os.path.join(os.path.dirname(os.path.abspath(path)), "finish"),
    }
    options.update(kwargs)
    return train(model=path, data=data, **options)


__all__ = ["train", "train_cli", "finish", "select_device", "V11_FORMAT", "DEFAULTS"]
