"""Trainer options: defaults, validation, recipe overrides, device and compile choice."""
from __future__ import annotations

import difflib

import torch

from fotonet.utils.console import log_event

from ..graph import FEATURE_STRIDES


DEFAULTS = {
    # Trainer profile: which part version fills each training slot (trainer/profiles.py). "1.1" is
    # the trainer of the published v1.1 weights; "1.2" the v1.2 trainer. Same model either way.
    "trainer": "1.1",
    "epochs": 120,
    "batch": 16,
    "nbs": 128,
    "imgsz": 640,
    "imgsz_schedule": None,
    "workers": 4,
    "optimizer": "adamw",
    "lr_mult": None,
    "fused": None,
    "amp_dtype": None,
    "lr0": 1e-3,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.05,
    "warmup_epochs": 5,
    "ema_decay": 0.99985,
    "ema_tau": 2000.0,
    # "wsd" (warmup, stable, decay), "wss" (warmup, stable, decay, stable, decay; sched.lr@1.3),
    # "cosine", or "decay" (decay only; finish).
    "lr_scheduler": "wsd",
    "wsd_decay_fraction": 0.20,
    # WSD decay length in epochs; 0 = use wsd_decay_fraction. 300 of 500 with warmup 5 gives
    # 5 warmup + 195 stable + 300 decay.
    "wsd_decay_epochs": 0,
    # sched.lr@1.2 only: the decay's shape, "cosine" (v1), "linear" or "sqrt" (1 - sqrt).
    "wsd_shape": "cosine",
    # WSS only: the last epoch of the first stable phase, of the first decay and of the second stable
    # phase; the second decay ends the run. 0 takes 50%, 60% and 80% of the run: 250, 300 and 400 of
    # 500 give 245 + 50 + 100 + 100 epochs after warmup.
    "wss_stable_end": 0,
    "wss_decay_end": 0,
    "wss_settle_end": 0,
    # WSS only: the second stable phase's learning rate, as a fraction of lr0.
    "wss_settle": 0.4,
    "grad_clip": 10.0,
    "close_mosaic": 15,
    "mmax": 160,
    "topk": 8,
    "o2o_topk": 1,
    "alpha": 0.5,
    "beta": 6.0,
    "o2o_beta": 2.0,
    "o2o_center_prior": 5.0,
    "radius": 2.5,
    "slack_cells": 0.75,
    # "consistent": O2O scored with the O2M alignment metric (YOLOv10);
    # "self": deployed-head score x IoU^o2o_beta x center prior.
    "o2o_metric": "consistent",
    # 0 = assign over every anchor; K > 0 = only the K nearest legal anchors per GT.
    "matcher_candidates": 0,
    "o2m_every_n_steps": 1,
    "o2m_start_epoch": 1,
    "o2m_disable_epoch": 0,
    # The deployed head reads detached features; only O2M trains the backbone.
    "o2o_detach": True,
    # Straight-through gradient scale from deployed O2O/GeoSA head back into neck/backbone when o2o_detach=False.
    "o2o_body_grad_scale": 1.0,
    "o2m_mult": 1.5,
    "o2o_groups": 1,
    "w_group": 1.0,
    "predecode": False,
    # Images per data worker kept for mosaic partners (worker JPEG decode
    # only). 0 = off: every mosaic decodes four images.
    "mosaic_buffer": 0,
    "predecode_dir": None,
    "predecode_max_side": 720,
    "quality_start_epoch": 15,
    "quality_warmup_epochs": 30,
    # O2M (and consistent O2O) matching blends in a center prior early on.
    # 0 = fade it out together with the quality ramp (1 - quality); N > 0 =
    # fade it out linearly over the first N epochs instead.
    "center_prior_epochs": 0,
    "w_box": 7.5,
    "w_dfl": 0.0,
    "w_o2o": 1.0,
    "w_cons": 0.0,
    "w_obj": 0.5,
    # Negative-loss multiplier on runner-up anchors (O2M positive, O2O
    # negative): the anchors that become duplicates. 1.0 = off.
    "w_runner_up": 1.0,
    # Negative weight of the deployed head's varifocal loss, alpha * p^gamma. The defaults are
    # varifocal's own. A lower gamma pushes confident wrong boxes (duplicates) down harder, while
    # background, where p is near zero, stays almost free.
    "o2o_neg_alpha": 0.75,
    "o2o_neg_gamma": 2.0,
    # loss.dual_o2o_o2m@1.2 only: the deployed head's class loss, "bce" (every negative at full
    # strength, as YOLOv10/YOLO26), "mal" (DEIM) or "vfl" (v1), and the power on its IoU target.
    "o2o_cls": "vfl",
    "o2o_q_gamma": 1.0,
    # assign.dual_o2o_o2m@1.2 only: the one-to-one rule's own center-prior fade, in epochs. -1 follows
    # the O2M prior as v1 does (fix F8); 0 = no prior; N > 0 = fades out over the first N epochs.
    "o2o_prior_epochs": -1,
    # assign.dual_o2o_o2m@1.2 only: the O2M positives' target, "iou" (v1) or "tal" (task-aligned).
    "o2m_target": "iou",
    # loss.dual_o2o_o2m@1.3 only: how much of the O2M head's class score goes into the deployed head's
    # targets on its positive anchors. 0 = none.
    "o2o_distill": 0.0,
    "w_distill": 0.0,
    "w_dup": 0.0,
    "w_distill_box": 2.0,
    "dup_iou_thresh": 0.5,
    # loss.tal@2.0 (trainer 2.0). Gains of the class, L1, decoder,
    # CDN, and MAL gamma terms (w_box is shared); the one-to-one rule picks o2o_candidates anchors per
    # object and keeps o2o_topk; tal_small widens the centre test for objects smaller than the first stride.
    "w_cls": 0.5,
    "w_l1": 1.5,
    "w_dec": 1.0,
    "w_cdn": 0.5,
    "mal_gamma": 0.0,
    "o2o_candidates": 7,
    "tal_small": True,
    # The head balance over the run: the O2M weight goes linearly from prog_o2m_start to prog_o2m_end,
    # and the O2O weight goes from prog_o2o_start to prog_total.
    "prog_o2m_start": 0.8,
    "prog_o2m_end": 0.1,
    "prog_o2m_floor": 0.0,
    "prog_o2o_start": 0.6,
    "prog_total": 1.0,
    # Optional decoder context mode override for fotonet2n ("edge_sample" or "hyper_xattn").
    "decoder_context": None,
    # Multiplies the total loss. SGD takes its step size from it (set it to nbs); AdamW ignores it.
    "loss_scale": 1.0,
    # aug.mosaic@1.3 only: the share of mosaics that use nine images, inside the part of the run
    # between the fractions mosaic_dense_from and mosaic_dense_to.
    "mosaic_dense": 0.0,
    "mosaic_dense_from": 0.0,
    "mosaic_dense_to": 1.0,
    # ema.average@1.3 only: average the EMA weights of the last N snapshots, taken every M epochs up
    # to the final epoch, and write <model>_avg.pt. 0 = off.
    "ema_average_last": 0,
    "ema_average_every": 5,
    # Validate a fused copy of the deployed graph instead of the training graph (faster, same boxes).
    "val_fused": False,
    # Train only the deployed head on the features of a finished model: the rest stays frozen, with
    # its BatchNorm statistics held.
    "head_refit": False,
    # Multiplies the scheduled learning rate; a live-tunable knob that leaves the schedule's shape alone.
    "lr_scale": 1.0,
    # Read live_settings.yaml in the output folder at every epoch start (see trainer/live.py).
    "live_tuning": True,
    "iou_type": "siou",
    "teacher_cache": None,
    "teacher_conf": 0.35,
    "teacher_weight": 0.5,
    "out_dir": "runs/train",
    "project": None,
    "name": None,
    "exist_ok": False,
    "max_images": 0,
    "fraction": 1.0,
    "single_cls": False,
    # COCO train sources with the v2 criterion: anchors inside a crowd region are neither object nor
    # background for that region's class. False trains them as background, as text labels do.
    "crowd_ignore": True,
    "freeze": 0,
    "patience": 0,
    # Wall-clock budget in hours; the epoch count is refitted after epoch 1.
    "time": 0.0,
    "val_period": 1,
    # Validate every epoch over the last N epochs; "auto" = the decay window.
    "val_final_epochs": "auto",
    "val_batch": 32,
    "val_workers": 2,
    "val_amp": True,
    "fast_val": True,
    "best_metric": "mAP50_95",
    # torch.compile of the model, matcher, and loss (1.4x images/s on the 4060
    # for fotonetn, ~5 min to compile). "auto": on for CUDA runs long enough
    # to repay the compile time.
    "compile": "auto",
    "compile_augment": True,
    "gpu_augment": True,
    "gpu_decode": True,
    "gpu_mixup": True,
    "gpu_copy_paste": True,
    # Where JPEGs are decoded on the GPU-assembly path: "workers" (parallel
    # CPU decode of only the visible crop), "nvjpeg" (GPU), or "auto"
    # (workers when there are at least two of them).
    "jpeg_decoder": "auto",
    "channels_last": True,
    "tensorboard": False,
    "device": "auto",
    # Profile this many training steps (after 10 warmup steps) into
    # <run>/profile.json; 0 disables the profiler.
    "profile_steps": 0,
    # Guard rails every N epochs on the first probe_images validation images (probe.head_matrix):
    # o2o, o2o + NMS and o2m + NMS AP, precision and recall at 0.25, duplicate rate. 0 = off.
    "probe_period": 0,
    "probe_images": 1000,
    # A random val and train image through the current weights every N minutes, for Studio
    # (probe.live_preview; <run>/preview/preview.json). 0 = off.
    "preview_minutes": 5.0,
    # The final-mAP forecast after each validation (probe.forecast; <run>/forecast.json).
    "forecast": True,
    "nc": None,
}


_M_TIERS = (16, 32, 64, 96, 128, 160)


_EXTRA_OPTIONS = ("augment_hyp", "callbacks", "knob_schedule", "parts")


_CALLBACK_EVENTS = ("on_train_start", "on_epoch_end", "on_val_end", "on_train_end")


# Freeze order for an integer ``freeze``: backbone modules, input to output.
_BACKBONE_ORDER = (
    "stem", "p2", "stage2", "down3", "stage3", "down4", "stage4", "attn4",
    "down5", "stage5", "sppf", "attn5",
)


_HEAD_OPTION_KEYS = ("o2m_mult", "o2o_groups", "aux_objectness", "decoder_context")


def select_device(device=None) -> torch.device:
    if device is not None and str(device).lower() not in {"auto", "none", ""}:
        req = str(device).strip().lower()
        if req.startswith("cuda") and not torch.cuda.is_available():
            log_event("CUDA requested but not available; falling back to CPU.", level="warn")
            return torch.device("cpu")
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _require_square(imgsz):
    if isinstance(imgsz, (tuple, list)):
        if len(imgsz) != 2 or int(imgsz[0]) != int(imgsz[1]):
            raise SystemExit(f"v1.1 training image size must be square, got {imgsz!r}.")
        side = int(imgsz[0])
    else:
        side = int(imgsz)
    if side <= 0 or side % FEATURE_STRIDES[-1]:
        raise SystemExit(
            f"v1.1 image size must be a positive multiple of {FEATURE_STRIDES[-1]}, got {side}."
        )
    return side


def _check_options(options):
    """Reject unknown training options instead of silently ignoring them."""
    allowed = sorted(set(DEFAULTS) | set(_EXTRA_OPTIONS))
    unknown = sorted(set(options) - set(allowed))
    if not unknown:
        return
    hints = []
    for key in unknown:
        close = difflib.get_close_matches(str(key), allowed, n=1)
        hints.append(f"'{key}'" + (f" (did you mean '{close[0]}'?)" if close else ""))
    raise TypeError("Unknown v1.1 training option(s): " + ", ".join(hints) + ".")


def _recipe_overrides(recipe):
    if not recipe:
        return {}
    from fotonet.config.recipes import load_training_recipe

    settings = dict(load_training_recipe(recipe)["settings"])
    out = {}
    for key in DEFAULTS:
        if key in settings and settings[key] is not None:
            out[key] = settings[key]
    for section in ("loss_hyp", "matcher_hyp", "o2m_hyp", "optim_hyp"):
        prefix = section[:-4]
        for key, value in dict(settings.get(section) or {}).items():
            for name in (key, f"{prefix}_{key}"):
                if name in DEFAULTS and value is not None:
                    out[name] = value
                    break
    out["augment_hyp"] = dict(settings.get("augment_hyp") or {})
    out["knob_schedule"] = list(settings.get("knob_schedule") or [])
    out["parts"] = dict(settings.get("parts") or {})
    return out


# Below this many planned steps the ~5 minute compile does not pay for itself.
_COMPILE_MIN_STEPS = 20_000


def _resolve_compile(value, dev, planned_steps):
    if isinstance(value, str):
        text = value.strip().lower()
        if text == "auto":
            if dev.type != "cuda" or not hasattr(torch, "compile") or planned_steps < _COMPILE_MIN_STEPS:
                return False
            try:
                import triton  # noqa: F401
            except ImportError:
                return False
            return True
        if text in ("true", "1", "yes", "on"):
            return True
        if text in ("false", "0", "no", "off", ""):
            return False
        raise ValueError(f"compile must be true, false, or 'auto', got {value!r}.")
    return bool(value)
