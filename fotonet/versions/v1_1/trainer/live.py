"""Live tuning: change a running v1.1 training's settings between epochs.

At the start of every epoch the trainer reads ``live_settings.yaml`` (or ``live_settings.json``)
in its output folder. Every listed setting whose value differs from what the run is using is
checked, applied before the epoch starts, logged, and appended to ``live_changes.jsonl`` next to
it. A resumed run applies the file again at its first epoch, so the file is the source of truth.

Only non-destructive settings can change this way: loss weights, assignment, ramps, the learning
rate, weight decay, momentum, EMA, augmentation, the image-size schedule, workers, validation and
patience. Settings that would change the model, the shape of the optimizer state, the dataset, the
batch or the length of the schedule are refused, and the refusal is logged too.

A recipe's ``knob_schedule`` changes the same settings at fixed epochs (``KnobSchedule``), so a run
that was tuned by hand can be written down as a recipe and replayed.
"""
from __future__ import annotations

import json
import math
import os
import time

from fotonet.data.augmentations.hyp import build_augment_hyp

from .options import DEFAULTS

LIVE_KEYS = frozenset({
    # loss
    "w_box", "w_o2o", "w_cons", "w_obj", "w_runner_up", "w_group", "o2o_neg_alpha", "o2o_neg_gamma",
    "iou_type",
    # assignment
    "alpha", "beta", "topk", "o2o_topk", "o2o_beta", "o2o_center_prior", "radius", "slack_cells",
    "o2o_metric", "matcher_candidates",
    # ramps and the auxiliary head
    "quality_start_epoch", "quality_warmup_epochs", "center_prior_epochs",
    "o2m_every_n_steps", "o2m_start_epoch", "o2m_disable_epoch", "o2o_detach",
    # optimizer, learning-rate schedule and EMA
    "lr_scale", "lr0", "lrf", "weight_decay", "momentum", "grad_clip", "warmup_epochs",
    "wsd_decay_fraction", "lr_scheduler", "ema_decay", "ema_tau",
    # data
    "imgsz_schedule", "close_mosaic", "workers",
    # validation and stopping
    "val_period", "patience", "fast_val", "val_amp",
})
# Settings that rebuild the learning-rate and data schedule when they change.
SCHEDULE_KEYS = frozenset({"warmup_epochs", "wsd_decay_fraction", "lr_scheduler", "imgsz_schedule", "close_mosaic"})
CHOICES = {
    "o2o_metric": {"consistent", "self"},
    "iou_type": {"siou", "ciou"},
    "lr_scheduler": {"wsd", "cosine", "decay"},
}


def _coerce(value, current):
    """Convert ``value`` to the type of ``current``; raise ValueError if it does not fit."""
    if isinstance(current, bool):
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"true", "1", "yes", "on"}:
            return True
        if text in {"false", "0", "no", "off"}:
            return False
        raise ValueError(f"expected true or false, got {value!r}")
    if isinstance(current, int) and not isinstance(value, bool):
        number = float(value)
        if not number.is_integer():
            raise ValueError(f"expected a whole number, got {value!r}")
        return int(number)
    if isinstance(current, float) or current is None and isinstance(value, (int, float)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"expected a finite number, got {value!r}")
        return number
    if isinstance(current, str):
        return str(value).strip()
    return value


def _checked(key, value, current):
    """The value a live or scheduled change sets ``key`` to, and ``current`` in the same type.

    Raises ValueError when the value does not fit the setting.
    """
    if key == "imgsz_schedule":
        new = [[float(f), int(s)] for f, s in value] if value else None
    else:
        # Type from the trainer defaults: command-line overrides may have arrived as text.
        ref = DEFAULTS.get(key, current)
        new = _coerce(value, ref)
        try:
            current = _coerce(current, ref)
        except (TypeError, ValueError):
            pass
    if key in CHOICES and new not in CHOICES[key]:
        raise ValueError(f"must be one of {sorted(CHOICES[key])}")
    if isinstance(new, (int, float)) and not isinstance(new, bool) and new < 0:
        raise ValueError("must not be negative")
    return new, current


class LiveTuner:
    """Reads the live settings file at each epoch start and applies what changed."""

    def __init__(self, out_dir, cfg, aug_hyp, log):
        self.paths = [os.path.join(out_dir, "live_settings.yaml"), os.path.join(out_dir, "live_settings.json")]
        self.changes_path = os.path.join(out_dir, "live_changes.jsonl")
        self.cfg, self.aug_hyp, self.log = cfg, aug_hyp, log
        self._seen = None
        # Settings the file holds right now; a knob schedule leaves them alone.
        self.pinned = frozenset()

    def _read(self):
        for path in self.paths:
            if os.path.isfile(path):
                stamp = (path, os.stat(path).st_mtime_ns)
                with open(path, encoding="utf-8-sig") as handle:  # Windows editors may add a BOM
                    text = handle.read()
                if path.endswith(".json"):
                    data = json.loads(text) if text.strip() else {}
                else:
                    import yaml
                    data = yaml.safe_load(text) or {}
                if not isinstance(data, dict):
                    raise ValueError("the file must hold a mapping of setting: value")
                return stamp, data
        return None, {}

    def _record(self, epoch, rows):
        with open(self.changes_path, "a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps({"epoch": epoch, "time": time.strftime("%Y-%m-%d %H:%M:%S"), **row}) + "\n")

    def poll(self, epoch):
        """Apply changes from the file. Returns ``{key: (old, new)}`` for what was applied."""
        try:
            stamp, data = self._read()
        except Exception as exc:   # a half-saved or malformed file: keep the current settings
            self.log(f"live: epoch {epoch}: could not read live settings ({exc}); nothing changed", level="warn")
            return {}
        if stamp is not None and stamp == self._seen:
            return {}
        self._seen = stamp
        applied, rows = {}, []
        aug_changes = {}
        pinned = set()
        for key, value in data.items():
            key = str(key)
            if key in self.aug_hyp and key != "close_mosaic":
                aug_changes[key] = value
                continue
            if key not in LIVE_KEYS:
                rows.append({"key": key, "value": value, "status": "refused",
                             "reason": "not a live setting (it would change the model, data, batch or schedule length)"})
                continue
            try:
                new, current = _checked(key, value, self.cfg.get(key))
            except (TypeError, ValueError) as exc:
                rows.append({"key": key, "value": value, "status": "refused", "reason": str(exc)})
                continue
            pinned.add(key)
            if new != current:
                self.cfg[key] = new
                applied[key] = (current, new)
        if aug_changes:
            merged = build_augment_hyp({**self.aug_hyp, **aug_changes})
            for key in aug_changes:
                if merged.get(key) != self.aug_hyp.get(key):
                    applied[key] = (self.aug_hyp.get(key), merged.get(key))
                    self.aug_hyp[key] = merged[key]
        for key, (old, new) in applied.items():
            rows.append({"key": key, "old": old, "new": new, "status": "applied"})
            self.log(f"live: epoch {epoch}: {key} {old} -> {new}")
        for row in rows:
            if row["status"] == "refused":
                self.log(f"live: epoch {epoch}: refused {row['key']}={row['value']!r}: {row['reason']}", level="warn")
        if rows:
            self._record(epoch, rows)
        self.pinned = frozenset(pinned)
        return applied


class KnobSchedule:
    """Settings that change at fixed epochs: the recipe's ``knob_schedule``.

    Each entry is ``{epoch: N, setting: value, ...}``; its values hold from epoch N on and are
    applied at the epoch start, the same way a live change is. Only live settings can be
    scheduled. A resumed run gets the values its epoch would have had, so a run that was tuned by
    hand while it trained can be written down and replayed. A setting that live_settings.yaml
    holds overrides the schedule for as long as it stays in that file.
    """

    def __init__(self, entries, cfg, log):
        self.cfg, self.log = cfg, log
        self.steps = []
        seen = set()
        for entry in entries or ():
            if not isinstance(entry, dict) or "epoch" not in entry:
                raise ValueError(f"knob_schedule: every entry needs an epoch, got {entry!r}")
            epoch = entry["epoch"]
            if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
                raise ValueError(f"knob_schedule: epoch must be a whole number from 1, got {epoch!r}")
            if epoch in seen:
                raise ValueError(f"knob_schedule: epoch {epoch} is listed twice")
            seen.add(epoch)
            values = {}
            for key, value in entry.items():
                if key == "epoch":
                    continue
                if key not in LIVE_KEYS:
                    raise ValueError(f"knob_schedule: {key} cannot be scheduled; only live settings can")
                try:
                    values[key] = _checked(key, value, cfg.get(key))[0]
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"knob_schedule: epoch {epoch}: {key}: {exc}") from None
            self.steps.append((epoch, values))
        self.steps.sort(key=lambda step: step[0])
        # What each scheduled setting is before its first step: the recipe's own value.
        self.base = {}
        for _epoch, values in self.steps:
            for key in values:
                if key not in self.base:
                    try:
                        self.base[key] = _checked(key, cfg.get(key), cfg.get(key))[0]
                    except (TypeError, ValueError):
                        self.base[key] = cfg.get(key)

    def values_at(self, epoch):
        out = dict(self.base)
        for start, values in self.steps:
            if start > epoch:
                break
            out.update(values)
        return out

    def apply(self, epoch, skip=()):
        """Set the values epoch ``epoch`` should train with. Returns ``{key: (old, new)}`` for what changed."""
        applied = {}
        for key, new in self.values_at(epoch).items():
            if key in skip:
                continue
            current = self.cfg.get(key)
            try:
                current = _checked(key, current, current)[0]
            except (TypeError, ValueError):
                pass
            if new != current:
                self.cfg[key] = new
                applied[key] = (current, new)
                self.log(f"schedule: epoch {epoch}: {key} {current} -> {new}")
        return applied


__all__ = ["KnobSchedule", "LiveTuner", "LIVE_KEYS", "SCHEDULE_KEYS"]
