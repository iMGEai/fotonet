"""Schedules: learning rate (WSD, cosine, linear, decay) and epoch ramps."""

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
