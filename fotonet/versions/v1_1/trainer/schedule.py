"""Learning-rate, image-size, quality-ramp, validation, and O2M schedules."""
from __future__ import annotations

import math

from .options import _require_square

from fotonet.train.schedulers import _epoch_ramp, _lr_at, _lr_at_v3  # noqa: F401


# WSS boundaries left at 0 take these shares of the run.
WSS_SHARES = (0.5, 0.6, 0.8)


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
    if str(kind).lower() in ("wsd", "wss"):
        return int(math.ceil(epochs * min(max(float(decay_fraction), 0.0), 1.0)))
    return 0


class _Schedule:
    """Step-level schedule; rebuilt when a time budget changes the epoch count.

    ``lr_fn`` is the run's ``sched.lr`` part: ``_lr_at`` (v1, the default), ``_lr_at_v2``, which
    also takes ``wsd_shape``, or ``_lr_at_v3``, which adds WSS. ``wsd_decay_epochs`` > 0 sets the
    decay length in epochs instead of ``wsd_decay_fraction``.
    """

    def __init__(self, cfg, epochs, steps_per_epoch, lr_fn=None):
        self.epochs = int(epochs)
        self.steps_per_epoch = int(steps_per_epoch)
        self.kind = str(cfg["lr_scheduler"]).lower()
        self.total_steps = self.epochs * steps_per_epoch
        warm = 0 if self.kind == "decay" else int(cfg["warmup_epochs"])
        self.warmup_steps = warm * steps_per_epoch
        fraction = min(max(float(cfg["wsd_decay_fraction"]), 0.01), 0.95)
        decay_epochs = int(cfg.get("wsd_decay_epochs") or 0)
        if decay_epochs > 0 and self.kind == "wsd":
            decay_epochs = min(decay_epochs, max(self.epochs - warm, 1))
            self.decay_start = (self.epochs - decay_epochs) * steps_per_epoch
            fraction = decay_epochs / max(self.epochs, 1)
        else:
            self.decay_start = self.warmup_steps + int((self.total_steps - self.warmup_steps) * (1.0 - fraction))
        self.wss = None
        if self.kind == "wss":
            fraction = self._build_wss(cfg, warm)
        self.close_from = self.epochs - int(cfg["close_mosaic"]) + 1
        self.imgsz_for = _parse_schedule(cfg.get("imgsz_schedule"), self.epochs)
        self.val_final = _final_val_epochs(cfg.get("val_final_epochs"), self.epochs, self.kind, fraction)
        self.lr_fn = lr_fn if lr_fn is not None else (_lr_at_v3 if self.kind == "wss" else _lr_at)
        if self.kind == "wss" and self.lr_fn is not _lr_at_v3:
            raise SystemExit("lr_scheduler wss needs the sched.lr@1.3 part (trainer 1.3).")
        self.shape = str(cfg.get("wsd_shape") or "cosine").lower()
        self.dense = (float(cfg.get("mosaic_dense") or 0.0), float(cfg.get("mosaic_dense_from") or 0.0),
                      float(cfg.get("mosaic_dense_to", 1.0)))

    def _build_wss(self, cfg, warm):
        """Set the WSS boundaries from ``wss_*_end`` and ``wss_settle``; returns the second decay's share of the run."""
        keys = ("wss_stable_end", "wss_decay_end", "wss_settle_end")
        stages = [int(cfg.get(key) or 0) or int(round(share * self.epochs)) for key, share in zip(keys, WSS_SHARES)]
        if stages != sorted(stages):
            raise SystemExit("wss_stable_end, wss_decay_end and wss_settle_end must not fall: the end of the first "
                             f"stable phase, of the first decay and of the second stable phase; got {stages}.")
        settle = float(cfg["wss_settle"])
        if not 0.0 < settle <= 1.0:
            raise SystemExit(f"wss_settle is a fraction of lr0 in (0, 1], got {settle!r}.")
        # A shorter run (a time budget, a smoke test) keeps the order and loses the late phases.
        first, second, third = (min(max(stage, warm), self.epochs) for stage in stages)
        self.decay_start = first * self.steps_per_epoch
        self.wss = (second * self.steps_per_epoch, third * self.steps_per_epoch, settle)
        return (self.epochs - third) / max(self.epochs, 1)

    def wss_info(self):
        """WSS in epochs for run_info: the ends of the first stable phase, the first decay and the second stable phase, and the settle share."""
        if self.wss is None:
            return None
        spe = max(self.steps_per_epoch, 1)
        return {"stable_end": self.decay_start // spe, "decay_end": self.wss[0] // spe, "settle_end": self.wss[1] // spe, "settle": self.wss[2]}

    def dense_p(self, epoch):
        """The share of nine-image mosaics at ``epoch`` (aug.mosaic@1.3); 0 outside its window."""
        share, start, stop = self.dense
        position = (int(epoch) - 1) / max(self.epochs, 1)
        return share if share > 0.0 and start <= position < stop else 0.0

    def lr(self, gstep, lr0, lrf):
        if self.lr_fn is _lr_at:
            return _lr_at(gstep, self.warmup_steps, self.decay_start, self.total_steps, lr0, lrf, self.kind)
        if self.kind == "wss":
            return self.lr_fn(gstep, self.warmup_steps, self.decay_start, self.total_steps, lr0, lrf, self.kind, self.shape, self.wss)
        return self.lr_fn(gstep, self.warmup_steps, self.decay_start, self.total_steps, lr0, lrf, self.kind, self.shape)

    def decay_epoch(self):
        """The first epoch of the decay (1-based), or None for schedules without one."""
        if self.kind == "decay":
            return 1
        if self.kind not in ("wsd", "wss"):
            return None
        return int(self.decay_start // max(self.steps_per_epoch, 1)) + 1

    def plan(self, lr0, lrf, side, mosaic, samples=16):
        """Per-epoch plan: mean learning rate, image size and mosaic probability (forecast, run_info)."""
        lrs, sizes, mos = [], [], []
        spe = max(self.steps_per_epoch, 1)
        for e in range(self.epochs):
            start = e * spe
            lrs.append(sum(self.lr(start + int((i + 0.5) * spe / samples), lr0, lrf) for i in range(samples)) / samples)
            sizes.append(int(self.imgsz_for(e + 1)) if self.imgsz_for else int(side))
            mos.append(0.0 if e + 1 >= self.close_from else float(mosaic))
        return lrs, sizes, mos

    def phase(self, epoch, gstep, warmup_epochs):
        if self.kind == "decay":
            return "decay"
        if epoch <= int(warmup_epochs):
            return "warmup"
        if self.kind == "wss":
            first_decay_end, second_decay_start, _ = self.wss
            return "decay" if self.decay_start <= gstep < first_decay_end or gstep >= second_decay_start else "stable"
        return "decay" if (self.kind == "cosine" or gstep >= self.decay_start) else "stable"
