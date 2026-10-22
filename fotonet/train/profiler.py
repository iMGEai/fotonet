"""Training-step profiler (first used by the fotonet v1.1 engine).

Opt-in (``profile_steps=N``). It answers where a step's time goes and how
busy the machine is:

- per-section GPU time (decode, augment, forward, assign+loss, backward, optimizer)
  from CUDA events, or host time on CPU;
- host time spent waiting for the dataloader;
- wall-clock step time and images per second;
- busy CPU cores (psutil) and GPU utilization (``nvidia-smi``), sampled once
  per window;
- peak GPU memory.

Sections are delimited by marks at the boundaries, so the engine calls
``begin_step()``, then ``mark(name)`` after each phase, then ``end_step()``.
Events are resolved once per window, so profiling adds one device sync per
``window`` steps rather than one per section. When disabled every call returns
immediately.
"""
from __future__ import annotations

import json
import os
import shutil
import statistics
import subprocess
import time

import torch

# "decode" appears only with the GPU-decode loader.
SECTIONS = ("decode", "augment", "forward", "assign_loss", "backward", "optimizer")


def _percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[index]


def _stats(values):
    if not values:
        return {"median": None, "p90": None, "mean": None}
    return {
        "median": round(statistics.median(values), 4),
        "p90": round(_percentile(values, 0.9), 4),
        "mean": round(statistics.fmean(values), 4),
    }


def _gpu_utilization(device):
    """Percent GPU utilization from nvidia-smi, or None when unavailable."""
    if device.type != "cuda" or shutil.which("nvidia-smi") is None:
        return None
    index = device.index if device.index is not None else torch.cuda.current_device()
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits", "-i", str(index)],
            capture_output=True, text=True, timeout=2.0, check=True,
        ).stdout.strip()
        return float(out.splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


class StepProfiler:
    """Profile ``steps`` training steps after ``warmup`` steps, then stop."""

    def __init__(self, device, steps=0, warmup=10, window=50):
        self.device = torch.device(device)
        self.steps = max(int(steps or 0), 0)
        self.warmup = max(int(warmup), 0)
        self.window = max(int(window), 1)
        self.enabled = self.steps > 0
        self.cuda = self.enabled and self.device.type == "cuda"
        self._seen = 0
        self._recorded = 0
        self._images = 0
        self._pending = []
        self._marks = None
        self._step_start = None
        self._last_end = None
        self._recording = False
        self._first_recorded_start = None
        self._last_recorded_end = None
        self.section_ms = {name: [] for name in SECTIONS}
        self.data_wait_ms = []
        self.step_ms = []
        self.cpu_busy_cores = []
        self.gpu_util = []
        self._psutil = None
        if self.enabled:
            try:
                import psutil

                self._psutil = psutil
                psutil.cpu_percent(percpu=True)   # prime the interval counter
            except ImportError:
                self._psutil = None
            if self.cuda:
                torch.cuda.reset_peak_memory_stats(self.device)

    @property
    def active(self):
        return self.enabled and self._recorded < self.steps

    @property
    def finished(self):
        return self.enabled and self._recorded >= self.steps

    def _stamp(self):
        if self.cuda:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            return event
        return time.perf_counter()

    def begin_step(self):
        if not self.active:
            return
        now = time.perf_counter()
        self._recording = self._seen >= self.warmup
        if self._recording:
            if self._last_end is not None:
                self.data_wait_ms.append((now - self._last_end) * 1000.0)
            if self._first_recorded_start is None:
                self._first_recorded_start = now
            self._step_start = now
            self._marks = [("start", self._stamp())]

    def mark(self, name):
        if self._recording and self._marks is not None:
            self._marks.append((name, self._stamp()))

    def end_step(self, images):
        if not self.active:
            return
        self._seen += 1
        if self._recording and self._marks is not None:
            now = time.perf_counter()
            self.step_ms.append((now - self._step_start) * 1000.0)
            self._last_recorded_end = now
            self._pending.append(self._marks)
            self._images += int(images)
            self._recorded += 1
            if len(self._pending) >= self.window or self._recorded >= self.steps:
                self._flush()
        self._marks = None
        self._recording = False
        self._last_end = time.perf_counter()

    def _flush(self):
        if not self._pending:
            return
        if self.cuda:
            self._pending[-1][-1][1].synchronize()
        for marks in self._pending:
            for (_, start), (name, end) in zip(marks, marks[1:]):
                if name not in self.section_ms:
                    continue
                if self.cuda:
                    self.section_ms[name].append(start.elapsed_time(end))
                else:
                    self.section_ms[name].append((end - start) * 1000.0)
        self._pending = []
        if self._psutil is not None:
            per_core = self._psutil.cpu_percent(percpu=True)
            self.cpu_busy_cores.append(sum(per_core) / 100.0)
        utilization = _gpu_utilization(self.device)
        if utilization is not None:
            self.gpu_util.append(utilization)

    def summary(self):
        self._flush()
        wall = None
        if self._first_recorded_start is not None and self._last_recorded_end is not None:
            wall = max(self._last_recorded_end - self._first_recorded_start, 1e-9)
        sections = {name: _stats(values) for name, values in self.section_ms.items()}
        device_total = sum(s["median"] or 0.0 for s in sections.values())
        for stats in sections.values():
            stats["share"] = round((stats["median"] or 0.0) / device_total, 4) if device_total else None
        out = {
            "device": str(self.device),
            "timer": "cuda_events" if self.cuda else "host",
            "steps": self._recorded,
            "warmup_steps": self.warmup,
            "images": self._images,
            "images_per_s": None if wall is None else round(self._images / wall, 3),
            "step_ms": _stats(self.step_ms),
            "data_wait_ms": _stats(self.data_wait_ms),
            "sections_ms": sections,
            "cpu_busy_cores": None if not self.cpu_busy_cores else round(statistics.fmean(self.cpu_busy_cores), 3),
            "cpu_logical_cores": os.cpu_count(),
            "gpu_util_pct": None if not self.gpu_util else round(statistics.fmean(self.gpu_util), 2),
            "gpu_peak_mem_mb": (
                round(torch.cuda.max_memory_allocated(self.device) / 2**20, 1) if self.cuda else None
            ),
        }
        return out

    def write(self, path):
        summary = self.summary()
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        return summary


def format_summary(summary):
    parts = [f"{summary['images_per_s']} img/s", f"step {summary['step_ms']['median']} ms"]
    parts.append(f"data wait {summary['data_wait_ms']['median']} ms")
    for name, stats in summary["sections_ms"].items():
        if stats["median"] is not None:
            parts.append(f"{name} {stats['median']} ms")
    if summary["cpu_busy_cores"] is not None:
        parts.append(f"CPU {summary['cpu_busy_cores']}/{summary['cpu_logical_cores']} cores")
    if summary["gpu_util_pct"] is not None:
        parts.append(f"GPU {summary['gpu_util_pct']}%")
    return "profile: " + ", ".join(parts)


__all__ = ["SECTIONS", "StepProfiler", "format_summary"]
