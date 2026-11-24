"""Schedules: learning rate (WSD, WSS, cosine, linear, decay) and epoch ramps."""

from __future__ import annotations

import math


def _epoch_ramp(epoch_index, start_epoch, warmup_epochs):
    epoch = float(epoch_index) + 1.0
    start = max(float(start_epoch), 1.0)
    warmup = max(float(warmup_epochs), 0.0)
    if epoch < start:
        return 0.0
    if warmup <= 0:
        return 1.0
    return max(0.0, min(1.0, (epoch - start + 1.0) / warmup))


def _lr_at(gstep, warmup_steps, decay_start, total_steps, lr0, lrf, kind):
    kind = str(kind).lower()
    if kind == "decay":
        prog = min(max(gstep / max(total_steps, 1), 0.0), 1.0)
        return lr0 * (lrf + (1.0 - lrf) * 0.5 * (1.0 + math.cos(math.pi * prog)))
    if gstep < warmup_steps:
        return lr0 * (gstep + 1) / max(warmup_steps, 1)
    if kind == "cosine":
        prog = (gstep - warmup_steps) / max(total_steps - warmup_steps, 1)
        prog = min(max(prog, 0.0), 1.0)
        return lr0 * (lrf + (1.0 - lrf) * 0.5 * (1.0 + math.cos(math.pi * prog)))
    if gstep < decay_start:
        return lr0
    prog = (gstep - decay_start) / max(total_steps - decay_start, 1)
    prog = min(max(prog, 0.0), 1.0)
    return lr0 * (lrf + (1.0 - lrf) * 0.5 * (1.0 + math.cos(math.pi * prog)))


WSD_SHAPES = ("cosine", "linear", "sqrt")


def _lr_at_v2(gstep, warmup_steps, decay_start, total_steps, lr0, lrf, kind, shape="cosine"):
    """``sched.lr@1`` plus the shape of the WSD decay: "cosine" (v1), "linear" or "sqrt" (1 - sqrt).

    Hägele et al. (NeurIPS 2024) find the (1 - sqrt) cooldown best for short cooldowns (10-20% of
    the steps); for a long decay a cosine is the safer shape. Every other phase is ``_lr_at``.
    """
    shape = str(shape).lower()
    if shape not in WSD_SHAPES:
        raise ValueError(f"wsd_shape must be one of {', '.join(WSD_SHAPES)}, got {shape!r}.")
    if str(kind).lower() != "wsd" or gstep < decay_start or shape == "cosine":
        return _lr_at(gstep, warmup_steps, decay_start, total_steps, lr0, lrf, kind)
    prog = (gstep - decay_start) / max(total_steps - decay_start, 1)
    prog = min(max(prog, 0.0), 1.0)
    frac = 1.0 - prog if shape == "linear" else 1.0 - math.sqrt(prog)
    return lr0 * (lrf + (1.0 - lrf) * frac)


def _lr_at_v3(gstep, warmup_steps, decay_start, total_steps, lr0, lrf, kind, shape="cosine", wss=None):
    """``sched.lr@1.2`` plus WSS, the Weird Settling Scheduler.

    WSS runs warmup, stable at ``lr0``, a cosine down to ``lr0 * settle``, stable there, then a second
    cosine down to ``lr0 * lrf``. The first cosine starts at ``decay_start``; ``wss`` holds the rest as
    ``(first_decay_end, second_decay_start, settle)``, boundaries in steps. Every other schedule is
    ``_lr_at_v2``.
    """
    if str(kind).lower() != "wss":
        return _lr_at_v2(gstep, warmup_steps, decay_start, total_steps, lr0, lrf, kind, shape)
    if gstep < warmup_steps:
        return lr0 * (gstep + 1) / max(warmup_steps, 1)
    if gstep < decay_start:
        return lr0
    first_decay_end, second_decay_start, settle = wss
    if gstep < first_decay_end:
        prog = (gstep - decay_start) / max(first_decay_end - decay_start, 1)
        return lr0 * (settle + (1.0 - settle) * 0.5 * (1.0 + math.cos(math.pi * prog)))
    if gstep < second_decay_start:
        return lr0 * settle
    prog = min((gstep - second_decay_start) / max(total_steps - second_decay_start, 1), 1.0)
    return lr0 * (lrf + (settle - lrf) * 0.5 * (1.0 + math.cos(math.pi * prog)))
