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
    is_3d = (x.ndim == 3)
    for _ in range(int(steps)):
        gram = torch.bmm(x, x.transpose(-2, -1)) if is_3d else (x @ x.transpose(-2, -1))
        term = torch.baddbmm(gram, gram, gram, beta=b, alpha=c) if is_3d else (b * gram + c * (gram @ gram))
        x = torch.baddbmm(x, term, x, beta=a, alpha=1.0) if is_3d else (a * x + term @ x)
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
        self._buffers: dict[tuple, torch.Tensor] = {}

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, momentum, wd = group["lr"], group["momentum"], group["weight_decay"]
            ns_steps, rms = group["ns_steps"], group["rms"]
            use_muon = group["use_muon"]
            nesterov = group["nesterov"]

            params_with_grad = []
            grads = []
            states = []
            for p in group["params"]:
                if p.grad is not None:
                    params_with_grad.append(p)
                    grads.append(p.grad.float())
                    states.append(self.state[p])

            if not params_with_grad:
                continue

            if wd:
                torch._foreach_mul_(params_with_grad, 1.0 - lr * wd)

            bufs_to_update = []
            grads_to_update = []
            for p, g, st in zip(params_with_grad, grads, states):
                if "momentum_buffer" not in st:
                    st["momentum_buffer"] = g.clone()
                else:
                    bufs_to_update.append(st["momentum_buffer"])
                    grads_to_update.append(g)

            if bufs_to_update:
                torch._foreach_mul_(bufs_to_update, momentum)
                torch._foreach_add_(bufs_to_update, grads_to_update)

            bufs = [st["momentum_buffer"] for st in states]
            updates = torch._foreach_add(grads, bufs, alpha=momentum) if nesterov else bufs

            if not use_muon:
                torch._foreach_add_(params_with_grad, updates, alpha=-lr)
                continue

            buckets = defaultdict(list)
            sgd_params = []
            sgd_updates = []
            for param, update in zip(params_with_grad, updates):
                if param.ndim >= 2:
                    r = param.shape[0]
                    c = param.numel() // r
                    transposed = (r > c)
                    m, n = (c, r) if transposed else (r, c)
                    scale = rms * math.sqrt(max(r, c))
                    mat = update.reshape(r, c)
                    if transposed:
                        mat = mat.t()
                    buckets[m].append((param, mat, transposed, r, c, n, scale))
                else:
                    sgd_params.append(param)
                    sgd_updates.append(update)

            if sgd_params:
                torch._foreach_add_(sgd_params, sgd_updates, alpha=-lr)

            for m, items in buckets.items():
                max_n = max(item[5] for item in items)
                batch = len(items)
                buf_key = (batch, m, max_n, items[0][1].device)
                stacked = self._buffers.get(buf_key)
                if stacked is None or stacked.device != items[0][1].device:
                    stacked = self._buffers[buf_key] = torch.zeros(batch, m, max_n, device=items[0][1].device)
                else:
                    stacked.zero_()
                for i, item in enumerate(items):
                    stacked[i, :, :item[5]] = item[1]
                orth_stacked = newton_schulz(stacked, steps=ns_steps)
                for i, item in enumerate(items):
                    param, _, transposed, r, c, n, scale = item
                    orth = orth_stacked[i, :, :n]
                    if transposed:
                        orth = orth.t()
                    param.add_(orth.reshape(param.shape).to(param.dtype), alpha=-lr * scale)
        return loss


class MuAdamW(torch.optim.Optimizer):
    """Selective hybrid optimizer: batched Newton-Schulz Muon for hidden dense matrices, AdamW for the rest.

    Intended for CNN + Attention + Query-Decoder hybrids (``fotonet2n`` / Hi Bird):
    - ``use_muon=True`` groups run batched 5-step Newton-Schulz orthogonalized Nesterov updates
      (Moonlight RMS scaling ``0.2 * sqrt(max(rows, cols))``) with decoupled weight decay.
    - ``use_muon=False`` groups run exact decoupled AdamW (fp32 moments via ``_foreach`` ops) for
      stems, depthwise convolutions, attention QKV/proj, hypergraph incidence, query decoders,
      prediction heads, embeddings, norms, biases, and channel/residual gates.
    """

    def __init__(self, params, lr=1e-2, momentum=0.937, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=5e-4, nesterov=True, ns_steps=5, rms=0.2):
        if not math.isfinite(float(lr)) or lr <= 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(
            lr=float(lr),
            momentum=float(momentum),
            betas=(float(betas[0]), float(betas[1])),
            eps=float(eps),
            weight_decay=float(weight_decay),
            nesterov=bool(nesterov),
            ns_steps=int(ns_steps),
            rms=float(rms),
            use_muon=True,
            wd_mult=1.0,
        )
        super().__init__(params, defaults)
        self._buffers: dict[tuple, torch.Tensor] = {}

    @staticmethod
    def _is_matrix_param(param: torch.Tensor) -> bool:
        return param.ndim >= 2 and sum(1 for d in param.shape if d > 1) >= 2

    def _adamw_step(self, params, grads, states, *, lr, betas, eps, wd):
        if not params:
            return
        beta1, beta2 = float(betas[0]), float(betas[1])
        if wd:
            torch._foreach_mul_(params, 1.0 - lr * wd)

        by_step = defaultdict(list)
        for p, g, st in zip(params, grads, states):
            if "exp_avg" not in st:
                st["step"] = 0
                st["exp_avg"] = torch.zeros_like(g)
                st["exp_avg_sq"] = torch.zeros_like(g)
            st["step"] = int(st["step"]) + 1
            by_step[st["step"]].append((p, g, st["exp_avg"], st["exp_avg_sq"]))

        for step_num, items in by_step.items():
            ps = [x[0] for x in items]
            gs = [x[1] for x in items]
            ms = [x[2] for x in items]
            vs = [x[3] for x in items]

            torch._foreach_mul_(ms, beta1)
            torch._foreach_add_(ms, gs, alpha=1.0 - beta1)
            torch._foreach_mul_(vs, beta2)
            torch._foreach_addcmul_(vs, gs, gs, value=1.0 - beta2)

            bc1 = 1.0 - beta1 ** step_num
            bc2 = 1.0 - beta2 ** step_num
            step_size = lr / bc1
            inv_sqrt_bc2 = 1.0 / math.sqrt(bc2)

            denoms = torch._foreach_sqrt(vs)
            torch._foreach_mul_(denoms, inv_sqrt_bc2)
            torch._foreach_add_(denoms, eps)

            if all(p.dtype == torch.float32 for p in ps):
                torch._foreach_addcdiv_(ps, ms, denoms, value=-step_size)
            else:
                for p, m_t, d_t in zip(ps, ms, denoms):
                    p.add_((m_t / d_t).to(p.dtype), alpha=-step_size)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = float(group["lr"])
            wd = float(group["weight_decay"])
            use_muon = bool(group.get("use_muon", True))
            betas = group.get("betas", (0.9, 0.999))
            eps = float(group.get("eps", 1e-8))

            params_with_grad, grads, states = [], [], []
            for p in group["params"]:
                if p.grad is not None:
                    params_with_grad.append(p)
                    grads.append(p.grad.float())
                    states.append(self.state[p])
            if not params_with_grad:
                continue

            if not use_muon:
                self._adamw_step(params_with_grad, grads, states, lr=lr, betas=betas, eps=eps, wd=wd)
                continue

            # Separate true >=2D matrices for Muon from any accidental vector/scalar gate (fallback to AdamW)
            m_params, m_grads, m_states = [], [], []
            a_params, a_grads, a_states = [], [], []
            for p, g, st in zip(params_with_grad, grads, states):
                if self._is_matrix_param(p):
                    m_params.append(p)
                    m_grads.append(g)
                    m_states.append(st)
                else:
                    a_params.append(p)
                    a_grads.append(g)
                    a_states.append(st)

            if a_params:
                self._adamw_step(a_params, a_grads, a_states, lr=lr, betas=betas, eps=eps, wd=0.0)
            if not m_params:
                continue

            momentum = float(group["momentum"])
            nesterov = bool(group["nesterov"])
            ns_steps = int(group["ns_steps"])
            rms = float(group["rms"])

            if wd:
                torch._foreach_mul_(m_params, 1.0 - lr * wd)

            bufs_to_update, grads_to_update = [], []
            for g, st in zip(m_grads, m_states):
                if "momentum_buffer" not in st:
                    st["momentum_buffer"] = g.clone()
                else:
                    bufs_to_update.append(st["momentum_buffer"])
                    grads_to_update.append(g)
            if bufs_to_update:
                torch._foreach_mul_(bufs_to_update, momentum)
                torch._foreach_add_(bufs_to_update, grads_to_update)

            bufs = [st["momentum_buffer"] for st in m_states]
            updates = torch._foreach_add(m_grads, bufs, alpha=momentum) if nesterov else bufs

            buckets = defaultdict(list)
            for param, update in zip(m_params, updates):
                r = param.shape[0]
                c = param.numel() // r
                transposed = (r > c)
                m, n = (c, r) if transposed else (r, c)
                scale = rms * math.sqrt(max(r, c))
                mat = update.reshape(r, c)
                if transposed:
                    mat = mat.t()
                buckets[m].append((param, mat, transposed, r, c, n, scale))

            for m, items in buckets.items():
                max_n = max(item[5] for item in items)
                batch = len(items)
                buf_key = (batch, m, max_n, items[0][1].device)
                stacked = self._buffers.get(buf_key)
                if stacked is None or stacked.device != items[0][1].device:
                    stacked = self._buffers[buf_key] = torch.zeros(batch, m, max_n, device=items[0][1].device)
                else:
                    stacked.zero_()
                for i, item in enumerate(items):
                    stacked[i, :, :item[5]] = item[1]
                orth_stacked = newton_schulz(stacked, steps=ns_steps)
                for i, item in enumerate(items):
                    param, _, transposed, r, c, n, scale = item
                    orth = orth_stacked[i, :, :n]
                    if transposed:
                        orth = orth.t()
                    param.add_(orth.reshape(param.shape).to(param.dtype), alpha=-lr * scale)
        return loss


_MUADAMW_EXCLUDE_TOKENS = (
    "backbone.stem",
    ".attn.",
    "hyper_inc",
    "edge_proj",
    "cls_pred",
    "reg_pred",
    "aux_obj",
    "head.decoder",
    "head.query_up",
    "head.box_pe",
    "head.cdn_label_emb",
)


def _is_muadamw_excluded(name_mod: str, module: torch.nn.Module) -> bool:
    if isinstance(module, torch.nn.Embedding):
        return True
    if isinstance(module, torch.nn.Conv2d) and int(module.groups) > 1:
        return True
    return any(tok in name_mod for tok in _MUADAMW_EXCLUDE_TOKENS)


def build_v11_optimizer(model, name="adamw", lr=1e-3, momentum=0.937, weight_decay=0.05, fused=False, lr_mult=None):
    """v1.1 optimizer: shared AdamW/SGD grouping, MuonSGD for ``musgd``, selective MuAdamW for ``muadamw``."""
    from fotonet.engine.optim import build_optimizer, split_optimizer_params

    key = str(name or "adamw").lower().replace("-", "_")
    if key in {"muadamw", "muon_adamw", "mu_adamw"}:
        adamw_mult = 0.15
        if isinstance(lr_mult, dict):
            adamw_mult = float(lr_mult.get("adamw", lr_mult.get("default", 0.15)))
        elif isinstance(lr_mult, (int, float)):
            adamw_mult = float(lr_mult)

        g_muon, g_adamw_decay, g_adamw_nodecay = [], [], []
        seen = set()
        norm_types = (torch.nn.BatchNorm2d, torch.nn.GroupNorm, torch.nn.LayerNorm)

        for name_mod, module in model.named_modules():
            excluded = _is_muadamw_excluded(name_mod, module)
            is_norm = isinstance(module, norm_types)
            is_emb = isinstance(module, torch.nn.Embedding)
            is_hidden_matrix_mod = isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)) and not excluded
            for p_name, param in module.named_parameters(recurse=False):
                if not param.requires_grad or id(param) in seen:
                    continue
                seen.add(id(param))
                is_vec = param.ndim <= 1 or sum(1 for d in param.shape if d > 1) <= 1
                if p_name == "bias" or is_norm or is_emb or is_vec:
                    g_adamw_nodecay.append(param)
                elif is_hidden_matrix_mod and param.ndim in (2, 4):
                    g_muon.append(param)
                else:
                    g_adamw_decay.append(param)

        for param in model.parameters():
            if param.requires_grad and id(param) not in seen:
                seen.add(id(param))
                is_vec = param.ndim <= 1 or sum(1 for d in param.shape if d > 1) <= 1
                if is_vec:
                    g_adamw_nodecay.append(param)
                else:
                    g_adamw_decay.append(param)

        muon_wd = float(weight_decay)
        adamw_wd = min(muon_wd * 0.2, 1e-4) if muon_wd > 0.0 else 0.0
        adamw_wd_mult = (adamw_wd / muon_wd) if muon_wd > 0.0 else 0.0
        groups = [
            {"params": g_muon, "weight_decay": muon_wd, "wd_mult": 1.0, "use_muon": True, "lr_mult": 1.0},
            {"params": g_adamw_decay, "weight_decay": adamw_wd, "wd_mult": adamw_wd_mult,
             "use_muon": False, "lr_mult": adamw_mult, "betas": (0.9, 0.999)},
            {"params": g_adamw_nodecay, "weight_decay": 0.0, "wd_mult": 0.0,
             "use_muon": False, "lr_mult": adamw_mult, "betas": (0.9, 0.999)},
        ]
        return MuAdamW([g for g in groups if g["params"]], lr=lr, momentum=momentum, weight_decay=muon_wd)

    if key in {"musgd", "mu_sgd", "muon_sgd", "muon"}:
        cls_mult = 3.0
        if isinstance(lr_mult, dict):
            cls_mult = float(lr_mult.get("head.cls", lr_mult.get("cls", 3.0)))
        elif isinstance(lr_mult, (int, float)):
            cls_mult = float(lr_mult)

        g_bnw, g_w, g_b, g_cls = [], [], [], []
        seen = set()
        norm_types = (torch.nn.BatchNorm2d, torch.nn.GroupNorm, torch.nn.LayerNorm)

        for name_mod, module in model.named_modules():
            is_cls_head = "cls_pred" in name_mod or "cls_stem" in name_mod
            for p_name, param in module.named_parameters(recurse=False):
                if not param.requires_grad or id(param) in seen:
                    continue
                seen.add(id(param))
                if p_name == "bias":
                    g_b.append(param)
                elif isinstance(module, norm_types) or param.ndim <= 1:
                    g_bnw.append(param)
                elif is_cls_head and param.ndim >= 2:
                    g_cls.append(param)
                else:
                    g_w.append(param)

        for param in model.parameters():
            if param.requires_grad and id(param) not in seen:
                seen.add(id(param))
                if param.ndim <= 1:
                    g_b.append(param)
                else:
                    g_w.append(param)

        groups = [
            {"params": g_bnw, "weight_decay": 0.0, "use_muon": False, "lr_mult": 1.0},
            {"params": g_w, "weight_decay": float(weight_decay), "use_muon": True, "lr_mult": 1.0},
            {"params": g_b, "weight_decay": 0.0, "use_muon": False, "lr_mult": 1.0},
        ]
        if g_cls:
            groups.append({"params": g_cls, "weight_decay": float(weight_decay), "use_muon": True, "lr_mult": cls_mult})
        return MuonSGD([g for g in groups if g["params"]], lr=lr, momentum=momentum)
    return build_optimizer(model, name=key, lr=lr, momentum=momentum, weight_decay=weight_decay, fused=fused)


__all__ = ["MuAdamW", "MuonSGD", "newton_schulz", "build_v11_optimizer"]
