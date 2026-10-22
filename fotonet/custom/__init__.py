"""Custom nodes: user-written parts for specialized work.

A custom node is an ordinary Python class or function registered with a
decorator. It declares its name and version (same rules as built-in parts, see
``fotonet.registry``), typed ports, parameters, what it needs and provides, and
whether it is export-safe, compile-safe, and deterministic.

Safety: a custom node is code with full access to the machine. Nothing loads
by default. ``enable("package")`` imports a node package on purpose;
``available()`` only lists packages that advertise the ``fotonet.nodes``
entry point, without importing them. A model or recipe that names a custom
part never imports anything: ``missing(refs)`` reports what is not installed.

    from fotonet.custom import module_node, Port, Param

    @module_node("myorg.tiny_attention", "1.0.0",
                 inputs={"x": Port.feature()}, outputs={"y": Port.feature(same_as="x")},
                 export_safe=True)
    class TinyAttention(nn.Module):
        heads = Param(int, default=2, choices=(1, 2, 4))

        def __init__(self, channels, heads=2):
            ...
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from enum import Enum

from fotonet import registry

ENTRY_POINT_GROUP = "fotonet.nodes"


@dataclass(frozen=True)
class Port:
    """A typed node input or output, checked when nodes are wired together."""

    kind: str
    channels: int | None = None
    stride: int | None = None
    same_as: str | None = None

    KINDS = ("feature", "features", "boxes", "logits", "targets", "loss", "images", "scalar")

    def __post_init__(self):
        if self.kind not in self.KINDS:
            raise ValueError(f"port kind must be one of {self.KINDS}, got {self.kind!r}.")

    @classmethod
    def feature(cls, channels=None, stride=None, same_as=None):
        return cls("feature", channels, stride, same_as)

    @classmethod
    def loss(cls):
        return cls("loss")


@dataclass(frozen=True)
class Param:
    """A node parameter: type, default, and allowed values or range (shown as a field in Studio)."""

    type: type
    default: object = None
    choices: tuple | None = None
    low: float | None = None
    high: float | None = None

    def check(self, name, value):
        if not isinstance(value, self.type):
            raise TypeError(f"{name} must be {self.type.__name__}, got {type(value).__name__}.")
        if self.choices is not None and value not in self.choices:
            raise ValueError(f"{name} must be one of {self.choices}, got {value!r}.")
        if self.low is not None and value < self.low or self.high is not None and value > self.high:
            raise ValueError(f"{name} must be within [{self.low}, {self.high}], got {value!r}.")
        return value


class Slot(str, Enum):
    """Where a trainer node plugs into the fixed training loop."""

    AUGMENT = "augment"
    ASSIGNER = "assigner"
    LOSS = "loss"
    LOSS_WEIGHT = "loss_weight"
    OPTIMIZER = "optimizer"
    SCHEDULE = "schedule"
    METRIC = "metric"
    CALLBACK = "callback"
    PROBE = "probe"


_SLOT_KIND = {
    Slot.AUGMENT: "augment", Slot.ASSIGNER: "assigner", Slot.LOSS: "loss", Slot.LOSS_WEIGHT: "loss",
    Slot.OPTIMIZER: "optimizer", Slot.SCHEDULE: "schedule", Slot.METRIC: "metric", Slot.CALLBACK: "callback",
    Slot.PROBE: "probe",
}


@dataclass(frozen=True)
class NodeSpec:
    name: str
    version: str
    kind: str
    inputs: dict = field(default_factory=dict)
    outputs: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)
    needs: tuple = ()
    provides: tuple = ()
    slot: str | None = None
    export_safe: bool = False
    compile_safe: bool = True
    deterministic: bool = True
    package: str | None = None


def _collect_params(obj):
    return {name: value for name, value in vars(obj).items() if isinstance(value, Param)} if isinstance(obj, type) else {}


def _register(obj, spec: NodeSpec):
    registry.register(spec.name, spec.version, spec.kind, obj, custom=True, package=spec.package)
    obj.part_name = spec.name
    obj.part_version = spec.version
    obj.node_spec = spec
    return obj


def module_node(name, version, *, inputs=None, outputs=None, kind="block", needs=(), provides=(),
                export_safe=False, compile_safe=True, deterministic=True):
    """Register a model-graph node (an ``nn.Module`` class): block, backbone, neck, or head."""
    if kind not in ("block", "backbone", "neck", "head"):
        raise ValueError(f"module nodes are blocks, backbones, necks, or heads, got {kind!r}.")

    def decorate(cls):
        spec = NodeSpec(name, version, kind, dict(inputs or {}), dict(outputs or {}), _collect_params(cls),
                        tuple(needs), tuple(provides), None, bool(export_safe), bool(compile_safe),
                        bool(deterministic), getattr(cls, "__module__", "").split(".")[0] or None)
        return _register(cls, spec)
    return decorate


def trainer_node(name, version, *, slot, needs=(), provides=(), compile_safe=True, deterministic=True):
    """Register a training-graph node (augmentation, assigner, loss term, metric, probe, ...)."""
    slot = Slot(slot)

    def decorate(obj):
        spec = NodeSpec(name, version, _SLOT_KIND[slot], {}, {}, _collect_params(obj), tuple(needs),
                        tuple(provides), slot.value, False, bool(compile_safe), bool(deterministic),
                        getattr(obj, "__module__", "").split(".")[0] or None)
        return _register(obj, spec)
    return decorate


_ENABLED: list[str] = []


def enable(package: str):
    """Import a node package on purpose. This runs its code."""
    module = importlib.import_module(package)
    if package not in _ENABLED:
        _ENABLED.append(package)
    return module


def enabled() -> list[str]:
    return list(_ENABLED)


def available() -> list[str]:
    """Node packages that advertise the ``fotonet.nodes`` entry point. Lists only; imports nothing."""
    from importlib.metadata import entry_points

    try:
        found = entry_points(group=ENTRY_POINT_GROUP)
    except TypeError:                       # Python < 3.10 API
        found = entry_points().get(ENTRY_POINT_GROUP, [])
    return sorted({ep.value.split(":")[0] for ep in found})


def missing(refs) -> list[str]:
    """Part references (``name@version``) that are not registered. Never imports anything."""
    registry.parts()
    out = []
    for ref in refs:
        try:
            registry.resolve(ref)
        except KeyError:
            out.append(str(ref))
    return out


def check(ref, *, channels=16, size=32, export=True):
    """Build a module node with its defaults, run forward and backward, and try ONNX export when export-safe.

    Returns a report dict; raises on the first failing check.
    """
    import io

    import torch

    info = registry.resolve(ref)
    spec = getattr(info.obj, "node_spec", None)
    if spec is None or info.kind not in ("block", "backbone", "neck", "head"):
        raise ValueError(f"{ref} is not a module node.")
    params = {name: p.default for name, p in spec.params.items()}
    for name, value in params.items():
        spec.params[name].check(name, value)
    module = info.obj(channels, **params)
    x = torch.randn(2, channels, size, size, requires_grad=True)
    y = module(x)
    if not isinstance(y, torch.Tensor):
        raise TypeError(f"{ref} returned {type(y).__name__}; module nodes return a tensor.")
    for out_name, port in spec.outputs.items():
        if port.same_as and tuple(y.shape) != tuple(x.shape):
            raise ValueError(f"{ref}.{out_name} must have the shape of {port.same_as}: {tuple(y.shape)} != {tuple(x.shape)}.")
    y.float().sum().backward()
    if x.grad is None or not torch.isfinite(x.grad).all():
        raise ValueError(f"{ref}: backward did not produce finite input gradients.")
    report = {"ref": info.ref, "output_shape": list(y.shape), "backward": True, "export": None}
    if export and spec.export_safe:
        buffer = io.BytesIO()
        torch.onnx.export(module.eval(), (x.detach(),), buffer, opset_version=17, dynamo=False)
        report["export"] = "onnx"
    golden = getattr(info.obj, "golden", None)
    if callable(golden):
        golden()
        report["golden"] = True
    return report


__all__ = [
    "ENTRY_POINT_GROUP", "NodeSpec", "Param", "Port", "Slot", "available", "check", "enable", "enabled",
    "missing", "module_node", "trainer_node",
]
