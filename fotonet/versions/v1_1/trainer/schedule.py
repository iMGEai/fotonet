"""Learning-rate, image-size, quality-ramp, validation, and O2M schedules."""
from __future__ import annotations

import math

from .options import _require_square

from fotonet.train.schedulers import _epoch_ramp, _lr_at  # noqa: F401


def _parse_schedule(schedule, epochs):
    if not schedule:
        return None
    pairs = []
    for entry in schedule:
        values = list(entry)
        if len(values) != 2:
            raise SystemExit(f"imgsz_schedule entries must be [fraction, size], got {entry!r}.")
        first, second = float(values[0]), float(values[1])
        if first <= 1.0 < second:
            pairs.append((first, _require_square(int(second))))
        elif second <= 1.0 < first:
            pairs.append((second, _require_square(int(first))))
        else:
            raise SystemExit(f"imgsz_schedule entry {entry!r} must pair a fraction with a size.")
    total = sum(frac for frac, _ in pairs)
    if total <= 0:
        raise SystemExit("imgsz_schedule fractions must sum to a positive value.")
    cuts, acc = [], 0.0
    for frac, size in pairs:
        acc += frac
        cuts.append((acc / total, size))

    def imgsz_for_epoch(epoch):
        pos = (epoch - 1) / max(int(epochs), 1)
        for bound, size in cuts:
            if pos < bound:
                return size
        return cuts[-1][1]

    return imgsz_for_epoch


def _o2m_enabled(epoch, step, cfg):
    if int(epoch) < int(cfg["o2m_start_epoch"]):
        return False
    disable = int(cfg["o2m_disable_epoch"] or 0)
    if disable and int(epoch) > disable:
        return False
    every = int(cfg["o2m_every_n_steps"] or 0)
    if every <= 1:
        return True
    return int(step) % every == 0


def _should_validate(epoch, epochs, period, final_epochs):
    return (
        epoch == epochs
        or epoch % max(int(period), 1) == 0
        or epoch > epochs - max(int(final_epochs), 0)
    )


def _final_val_epochs(setting, epochs, kind, decay_fraction):
    if setting not in (None, "auto"):
        return max(int(setting), 0)
    if str(kind).lower() == "wsd":
        return int(math.ceil(epochs * min(max(float(decay_fraction), 0.0), 1.0)))
    return 0


class _Schedule:
    """Step-level schedule; rebuilt when a time budget changes the epoch count."""

    def __init__(self, cfg, epochs, steps_per_epoch):
        self.epochs = int(epochs)
        self.kind = str(cfg["lr_scheduler"]).lower()
        self.total_steps = self.epochs * steps_per_epoch
        warm = 0 if self.kind == "decay" else int(cfg["warmup_epochs"])
        self.warmup_steps = warm * steps_per_epoch
        fraction = min(max(float(cfg["wsd_decay_fraction"]), 0.01), 0.95)
        self.decay_start = self.warmup_steps + int((self.total_steps - self.warmup_steps) * (1.0 - fraction))
        self.close_from = self.epochs - int(cfg["close_mosaic"]) + 1
        self.imgsz_for = _parse_schedule(cfg.get("imgsz_schedule"), self.epochs)
        self.val_final = _final_val_epochs(cfg.get("val_final_epochs"), self.epochs, self.kind, fraction)

    def lr(self, gstep, lr0, lrf):
        return _lr_at(gstep, self.warmup_steps, self.decay_start, self.total_steps, lr0, lrf, self.kind)

    def phase(self, epoch, gstep, warmup_epochs):
        if self.kind == "decay":
            return "decay"
        if epoch <= int(warmup_epochs):
            return "warmup"
        return "decay" if (self.kind == "cosine" or gstep >= self.decay_start) else "stable"
