"""Cost targets for the v1.1 family.

This is the one place to re-baseline when a peer publishes new numbers. A peer
row holds the published, rounded figures; the gate is 1% under the *lowest*
true value that rounds to the published one, so passing it means "at least 1%
under the peer" whatever the unrounded value is.

Counting rule, same as the peers publish: Conv/BatchNorm folded, training-only
branches removed, 640x640 input, GFLOPs = 2 x multiply-accumulates.
"""
from __future__ import annotations

from dataclasses import dataclass

MARGIN = 0.01
PARAMS_ROUNDING_M = 0.05   # published params are rounded to 0.1 M
GFLOPS_ROUNDING = 0.05     # published FLOPs are rounded to 0.1 B
IMGSZ = 640


@dataclass(frozen=True)
class PeerRow:
    name: str
    params_m: float
    gflops: float
    map50_95: float
    note: str = ""


# Ultralytics YOLO27 preview table (fused model, unused branch removed).
# https://docs.ultralytics.com/models/yolo27/ - retrieved 2026-10-15. Weights
# and code are unreleased and the page says figures may change: re-check and
# re-measure on our own harness when they ship. n/s accuracy is reported for
# their default one-to-many head with NMS; ours is NMS-free.
YOLO27 = {
    "n": PeerRow("YOLO27n", 3.0, 7.2, 42.6, "NMS head by default"),
    "s": PeerRow("YOLO27s", 11.8, 28.2, 50.1, "NMS head by default"),
    "m": PeerRow("YOLO27m", 22.8, 65.0, 56.1),
    "l": PeerRow("YOLO27l", 72.3, 165.3, 60.5),
}

# YOLO26 end-to-end (NMS-free) accuracy, the v1.1 ship gate.
# https://docs.ultralytics.com/models/yolo26/ - retrieved 2026-10-15.
YOLO26_E2E_MAP = {"n": 40.1, "s": 47.8, "m": 52.5, "l": 54.4}

# Measured locally (fused, thop at 640): the only peer we can run today. Used to
# calibrate our counter against the published counting rule.
YOLO26N_MEASURED = {"params": 2_408_932, "gflops_thop": 5.36}

TIER_OF = {"fotonetn": "n", "fotonets": "s", "fotonetm": "m", "fotonetl": "l"}


def gate(tier, peers=YOLO27, margin=MARGIN):
    """Return ``(max_params, max_gflops)`` for a tier: at least ``margin`` under the peer."""
    row = peers[str(tier)]
    max_params = (1.0 - margin) * (row.params_m - PARAMS_ROUNDING_M) * 1e6
    max_gflops = (1.0 - margin) * (row.gflops - GFLOPS_ROUNDING)
    return int(max_params), round(max_gflops, 4)


def tier_of(model_id):
    key = str(model_id).strip().lower()
    if key not in TIER_OF:
        raise ValueError(f"No cost target for {model_id!r}; expected one of {sorted(TIER_OF)}.")
    return TIER_OF[key]


__all__ = [
    "MARGIN", "IMGSZ", "PeerRow", "YOLO27", "YOLO26_E2E_MAP", "YOLO26N_MEASURED",
    "TIER_OF", "gate", "tier_of",
]
