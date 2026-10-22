"""Deploy-cost measurement for the v1.1 family against the peer gates.

Counts what ships: a deep copy of the model with the training-only O2M head
stripped and Conv/BatchNorm folded, at 640x640 with batch 1.

Two FLOP counts are taken and the larger one is gated:

- ``gflops_thop``: thop on the fused module, the method behind the published
  peer numbers and our site benchmark. It only sees ``nn.Module`` calls, so a
  functional matmul (attention) is invisible to it.
- ``gflops_full``: ``torch.utils.flop_counter`` over ATen ops, which also
  counts matmuls. Calibrated at runtime so it always reports 2 x MACs.

Run ``python -m fotonet.versions.v1_1.budget`` for the family table, or add
``--json`` for machine-readable output.
"""
from __future__ import annotations

import copy
import json
import sys
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

from .targets import IMGSZ, YOLO27, gate, tier_of


@dataclass
class BudgetReport:
    model_id: str
    tier: str
    params: int
    gflops_thop: float | None
    gflops_full: float
    gflops: float
    gate_params: int
    gate_gflops: float
    peer: str
    peer_params_m: float
    peer_gflops: float

    @property
    def passes(self):
        return self.params <= self.gate_params and self.gflops <= self.gate_gflops

    @property
    def params_under_peer(self):
        return 1.0 - self.params / (self.peer_params_m * 1e6)

    @property
    def gflops_under_peer(self):
        return 1.0 - self.gflops / self.peer_gflops

    def to_dict(self):
        out = asdict(self)
        out.update(
            passes=self.passes,
            params_under_peer=round(self.params_under_peer, 5),
            gflops_under_peer=round(self.gflops_under_peer, 5),
        )
        return out


def deploy_copy(model):
    """Fused, O2M-stripped, eval-mode copy; the caller's model is untouched."""
    deploy = copy.deepcopy(model).cpu().float().eval()
    if hasattr(deploy, "strip_o2m_for_inference"):
        deploy.strip_o2m_for_inference()
    if hasattr(deploy, "fuse"):
        deploy.fuse()
    remaining = sum(isinstance(module, nn.BatchNorm2d) for module in deploy.modules())
    if remaining:
        raise RuntimeError(f"Fusion left {remaining} BatchNorm2d modules; the count would be wrong.")
    return deploy


def count_params(module):
    return int(sum(p.numel() for p in module.parameters()))


_FULL_COUNTER_SCALE = None


def _full_counter_scale():
    """Ratio that turns the ATen counter's conv figure into 2 x MACs.

    Torch versions have reported conv FLOPs as MACs or as 2 x MACs; a conv with
    a known MAC count settles which one this build uses.
    """
    global _FULL_COUNTER_SCALE
    if _FULL_COUNTER_SCALE is None:
        from torch.utils.flop_counter import FlopCounterMode

        conv = nn.Conv2d(8, 16, 3, padding=1, bias=False)
        x = torch.zeros(1, 8, 10, 10)
        expected = 2 * (8 * 16 * 9 * 10 * 10)
        with torch.no_grad(), FlopCounterMode(display=False) as counter:
            conv(x)
        _FULL_COUNTER_SCALE = expected / float(counter.get_total_flops())
    return _FULL_COUNTER_SCALE


def full_gflops(module, imgsz=IMGSZ):
    """GFLOPs (2 x MACs) of every conv and matmul in one forward pass."""
    from torch.utils.flop_counter import FlopCounterMode

    x = torch.zeros(1, 3, int(imgsz), int(imgsz))
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        module(x)
    return counter.get_total_flops() * _full_counter_scale() / 1e9


def thop_gflops(module, imgsz=IMGSZ):
    """GFLOPs (2 x MACs) as thop reports them, or None when thop is not installed."""
    try:
        from thop import profile
    except ImportError:
        return None
    probe = copy.deepcopy(module)   # thop attaches counters to the module it profiles
    x = torch.zeros(1, 3, int(imgsz), int(imgsz))
    with torch.no_grad():
        macs, _ = profile(probe, inputs=(x,), verbose=False)
    return 2.0 * float(macs) / 1e9


def measure_module(deploy, imgsz=IMGSZ):
    """``(params, gflops_thop, gflops_full)`` for an already-fused deploy module."""
    return count_params(deploy), thop_gflops(deploy, imgsz), full_gflops(deploy, imgsz)


def measure(model, model_id=None, imgsz=IMGSZ, peers=YOLO27):
    """Measure a v1.1 model (or family id) against its tier's peer gate."""
    if isinstance(model, str):
        from .graph import V11Detector

        model_id = model_id or model
        model = V11Detector(model, nc=80, imgsz=imgsz)
    model_id = str(model_id or getattr(model, "model_id"))
    tier = tier_of(model_id)
    params, g_thop, g_full = measure_module(deploy_copy(model), imgsz)
    gflops = max(g_full, g_thop) if g_thop is not None else g_full
    max_params, max_gflops = gate(tier, peers)
    row = peers[tier]
    return BudgetReport(
        model_id=model_id, tier=tier, params=params,
        gflops_thop=None if g_thop is None else round(g_thop, 4),
        gflops_full=round(g_full, 4), gflops=round(gflops, 4),
        gate_params=max_params, gate_gflops=max_gflops,
        peer=row.name, peer_params_m=row.params_m, peer_gflops=row.gflops,
    )


def family_reports(imgsz=IMGSZ):
    from .spec import V11_MODEL_IDS

    return [measure(model_id, imgsz=imgsz) for model_id in V11_MODEL_IDS]


def format_table(reports):
    lines = [
        f"{'model':10s} {'params':>12s} {'GFLOPs':>8s} {'gate params':>12s} {'gate GF':>8s} "
        f"{'vs peer params':>15s} {'vs peer GF':>11s}  verdict",
    ]
    for r in reports:
        lines.append(
            f"{r.model_id:10s} {r.params:>12,d} {r.gflops:>8.3f} {r.gate_params:>12,d} "
            f"{r.gate_gflops:>8.3f} {-100 * r.params_under_peer:>+14.2f}% "
            f"{-100 * r.gflops_under_peer:>+10.2f}%  {'PASS' if r.passes else 'FAIL'} ({r.peer})"
        )
    return "\n".join(lines)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    reports = family_reports()
    if "--json" in argv:
        print(json.dumps([r.to_dict() for r in reports], indent=2))
    else:
        print(format_table(reports))
    return 0 if all(r.passes for r in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BudgetReport", "deploy_copy", "count_params", "full_gflops", "thop_gflops",
    "measure_module", "measure", "family_reports", "format_table", "main",
]
