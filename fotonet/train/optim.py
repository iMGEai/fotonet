"""Optimizers: AdamW setup and MuonSGD (first used by fotonet v1.1).

``MuonSGD`` is the fixed hybrid behind ``optimizer="musgd"`` for v1.1. The
shared ``fotonet.engine.optim.MuSGD`` stays untouched for v1. Differences:

- The orthogonalized update of each matrix is rescaled to a root-mean-square of
  ``rms`` (Moonlight's 0.2 * sqrt(max(rows, cols))), so one learning rate fits
  every layer shape and matches AdamW's update scale.
- Newton-Schulz runs once per *shape group*, batched over all parameters of
  that shape, instead of once per parameter in a Python loop.
- Non-matrix parameters (biases, norms) use plain momentum SGD.
"""
from __future__ import annotations

import math
from collections import defaultdict

import torch

_NS_COEFFS = (3.4445, -4.7750, 2.0315)


def newton_schulz(updates, steps=5, eps=1e-7):
    """Batched zero-power iteration on ``[G, rows, cols]`` matrices."""
    x = updates.float()
    transposed = x.shape[-2] > x.shape[-1]
    if transposed:
        x = x.transpose(-2, -1)
    x = x / x.norm(dim=(-2, -1), keepdim=True).clamp_min(eps)
    a, b, c = _NS_COEFFS
    for _ in range(int(steps)):
        gram = x @ x.transpose(-2, -1)
        x = a * x + (b * gram + c * gram @ gram) @ x
    if transposed:
        x = x.transpose(-2, -1)
    return x


class MuonSGD(torch.optim.Optimizer):
    """Muon for matrix-shaped weights, momentum SGD for the rest, decoupled weight decay."""

    def __init__(self, params, lr=1e-3, momentum=0.95, weight_decay=0.0, nesterov=True,
                 ns_steps=5, rms=0.2):
        if not math.isfinite(float(lr)) or lr <= 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=float(lr), momentum=float(momentum), weight_decay=float(weight_decay),
                        nesterov=bool(nesterov), ns_steps=int(ns_steps), rms=float(rms), use_muon=True)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, momentum, wd = group["lr"], group["momentum"], group["weight_decay"]
            muon_by_shape = defaultdict(list)
            for param in group["params"]:
                if param.grad is None:
                    continue
                if wd:
                    param.mul_(1.0 - lr * wd)
                state = self.state[param]
                buf = state.get("momentum_buffer")
                grad = param.grad.float()
                if buf is None:
                    buf = state["momentum_buffer"] = grad.clone()
                else:
                    buf.mul_(momentum).add_(grad)
                update = grad.add(buf, alpha=momentum) if group["nesterov"] else buf
                if group["use_muon"] and param.ndim >= 2:
                    matrix = update.reshape(param.shape[0], -1)
                    muon_by_shape[tuple(matrix.shape)].append((param, matrix))
                else:
                    param.add_(update.to(param.dtype), alpha=-lr)
            for (rows, cols), items in muon_by_shape.items():
                stacked = torch.stack([matrix for _, matrix in items])
                orth = newton_schulz(stacked, steps=group["ns_steps"])
                scale = group["rms"] * math.sqrt(max(rows, cols))
                for (param, _), update in zip(items, orth):
                    param.add_(update.reshape(param.shape).to(param.dtype), alpha=-lr * scale)
        return loss


def build_v11_optimizer(model, name="adamw", lr=1e-3, momentum=0.937, weight_decay=0.05, fused=False):
    """v1.1 optimizer: shared AdamW/SGD grouping, the fixed MuonSGD for ``musgd``."""
    from fotonet.engine.optim import build_optimizer, split_optimizer_params

    key = str(name or "adamw").lower().replace("-", "_")
    if key in {"musgd", "mu_sgd", "muon_sgd", "muon"}:
        g_bnw, g_w, g_b = split_optimizer_params(model)
        groups = [
            {"params": g_bnw, "weight_decay": 0.0, "use_muon": False},
            {"params": g_w, "weight_decay": float(weight_decay), "use_muon": True},
            {"params": g_b, "weight_decay": 0.0, "use_muon": False},
        ]
        return MuonSGD([g for g in groups if g["params"]], lr=lr, momentum=momentum)
    return build_optimizer(model, name=key, lr=lr, momentum=momentum, weight_decay=weight_decay, fused=fused)


__all__ = ["MuonSGD", "newton_schulz", "build_v11_optimizer"]
