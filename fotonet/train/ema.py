"""Exponential moving average of weights and float buffers (foreach updates)."""

from __future__ import annotations

import math

import torch


class _EMA:
    """Exponential moving average of parameters and float buffers (foreach updates)."""

    def __init__(self, model, decay, tau=0.0):
        self.decay = float(decay)
        self.tau = float(tau)
        self.updates = 0
        self.shadow = {name: param.detach().clone().float() for name, param in model.named_parameters()}
        self._names = list(self.shadow)
        self.buf_names = [name for name, buf in model.named_buffers() if buf.is_floating_point()]
        self.shadow_buf = {name: model.get_buffer(name).detach().clone().float() for name in self.buf_names}
        self._shadow_list = [self.shadow[name] for name in self._names]
        self._shadow_buf_list = [self.shadow_buf[name] for name in self.buf_names]

    def effective_decay(self):
        if self.tau <= 0:
            return self.decay
        return self.decay * (1.0 - math.exp(-self.updates / self.tau))

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        weight = 1.0 - self.effective_decay()
        live = [param.detach() for param in model.parameters()]
        if all(p.dtype == torch.float32 for p in live):
            torch._foreach_lerp_(self._shadow_list, live, weight)
        else:
            torch._foreach_lerp_(self._shadow_list, [p.float() for p in live], weight)
        if self._shadow_buf_list:
            buffers = [model.get_buffer(name).detach().float() for name in self.buf_names]
            torch._foreach_lerp_(self._shadow_buf_list, buffers, weight)

    def state(self):
        return {
            "params": {name: self.shadow[name] for name in self._names},
            "buffers": dict(self.shadow_buf),
            "updates": self.updates,
        }

    def load_state_dict(self, state):
        self.updates = int(state.get("updates", 0) or 0)
        params = state.get("params", state)
        buffers = state.get("buffers", {})
        for name in self._names:
            if name in params:
                self.shadow[name].copy_(params[name].to(self.shadow[name].device, dtype=self.shadow[name].dtype))
        for name in self.buf_names:
            if name in buffers:
                self.shadow_buf[name].copy_(buffers[name].to(self.shadow_buf[name].device, dtype=self.shadow_buf[name].dtype))

    def deploy_state(self, model):
        out = {}
        for name, param in model.named_parameters():
            out[name] = self.shadow[name].to(dtype=param.dtype)
        for name, buf in model.named_buffers():
            if name in self.shadow_buf:
                out[name] = self.shadow_buf[name].to(dtype=buf.dtype)
            else:
                out[name] = buf.detach().clone()
        return out
