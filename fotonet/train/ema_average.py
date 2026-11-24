"""The mean of the last EMA snapshots of a run, taken every few epochs (first used by the fotonet v1.3 trainer)."""
from __future__ import annotations

import torch


class EmaAverager:
    """Running mean of EMA state dicts, held on the CPU and stored in the resume checkpoint.

    Floating-point tensors (weights and BatchNorm statistics) are averaged; integer buffers take the
    newest value.
    """

    def __init__(self):
        self.count = 0
        self.mean = {}

    @staticmethod
    def due(epoch, epochs, last, every):
        """Whether epoch ``epoch`` is one of the ``last`` snapshots, ``every`` epochs apart, ending at ``epochs``."""
        last, every = int(last), max(int(every), 1)
        if last <= 0:
            return False
        return (int(epochs) - int(epoch)) % every == 0 and int(epoch) > int(epochs) - last * every

    @torch.no_grad()
    def add(self, state):
        self.count += 1
        for name, value in state.items():
            value = value.detach().cpu()
            if not value.is_floating_point():
                self.mean[name] = value.clone()
            elif name not in self.mean:
                self.mean[name] = value.double()
            else:
                self.mean[name] += (value.double() - self.mean[name]) / self.count

    def result(self):
        """The averaged state as float32, ready for ``load_state_dict``."""
        return {name: value.float() if value.is_floating_point() else value for name, value in self.mean.items()}

    def state(self):
        return {"count": self.count, "mean": dict(self.mean)}

    def load_state_dict(self, state):
        self.count = int(state.get("count", 0) or 0)
        # A resume reads the checkpoint onto the training device; the running mean lives on the CPU.
        self.mean = {name: value.detach().cpu() for name, value in dict(state.get("mean") or {}).items()}


__all__ = ["EmaAverager"]
