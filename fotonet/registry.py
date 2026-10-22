"""Versioned parts: one registry for model blocks, training pieces, and custom nodes.

Every reusable part has a name and a version, ``name@MAJOR.MINOR.PATCH``. The
version describes the math, not the code:

- PATCH: a fix that does not change outputs (a crash, an edge case, a message).
- MINOR: the same math, better code (faster, less memory), outputs within a
  tested tolerance. Every model that uses the part gets it.
- MAJOR: different math. New models only; a released model keeps its major.

A released part version never changes: ``tests/test_parts_golden.py`` pins its
outputs on seeded inputs. Built-in parts register themselves on import;
custom parts come from ``fotonet.custom`` and are marked ``custom``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

KINDS = ("block", "backbone", "neck", "head", "loss", "assigner", "optimizer", "schedule", "ema",
         "augment", "metric", "callback", "probe")


@dataclass(frozen=True)
class PartInfo:
    name: str
    version: str
    kind: str
    obj: object = field(compare=False, repr=False)
    custom: bool = False
    package: str | None = None

    @property
    def major(self) -> int:
        return int(self.version.split(".")[0])

    @property
    def ref(self) -> str:
        return f"{self.name}@{self.version}"


_PARTS: dict[str, dict[str, PartInfo]] = {}


def _check_version(version: str) -> str:
    pieces = str(version).split(".")
    if len(pieces) != 3 or not all(piece.isdigit() for piece in pieces):
        raise ValueError(f"part versions are MAJOR.MINOR.PATCH, got {version!r}.")
    return str(version)


def register(name: str, version: str, kind: str, obj, *, custom: bool = False, package: str | None = None) -> PartInfo:
    """Add one part version. Registering the same name and version twice is an error."""
    if kind not in KINDS:
        raise ValueError(f"unknown part kind {kind!r}; expected one of {', '.join(KINDS)}.")
    info = PartInfo(str(name), _check_version(version), kind, obj, custom=bool(custom), package=package)
    versions = _PARTS.setdefault(info.name, {})
    if info.version in versions and versions[info.version].obj is not obj:
        raise ValueError(f"{info.ref} is already registered.")
    versions[info.version] = info
    return info


def part(name: str, version: str, kind: str):
    """Decorator for built-in parts: registers the object and stamps ``part_name`` / ``part_version`` on it."""
    def decorate(obj):
        register(name, version, kind, obj)
        obj.part_name = name
        obj.part_version = version
        return obj
    return decorate


def _key(version: str):
    return tuple(int(piece) for piece in version.split("."))


def get(name: str, major: int | None = None) -> PartInfo:
    """The newest version of ``name`` (within ``major`` when given)."""
    versions = _PARTS.get(str(name))
    if not versions:
        raise KeyError(f"no part named {name!r}.")
    candidates = [info for info in versions.values() if major is None or info.major == int(major)]
    if not candidates:
        raise KeyError(f"{name!r} has no version with major {major}.")
    return max(candidates, key=lambda info: _key(info.version))


def resolve(ref: str) -> PartInfo:
    """``"nn.cspstar@1"`` (newest 1.x.y), ``"nn.cspstar@1.0.0"`` (exact), or ``"nn.cspstar"`` (newest)."""
    name, _, version = str(ref).partition("@")
    if not version:
        return get(name)
    if version.count(".") == 2:
        try:
            return _PARTS[name][version]
        except KeyError:
            raise KeyError(f"no part {ref!r}.") from None
    return get(name, int(version))


def parts(kind: str | None = None, *, custom: bool | None = None) -> list[PartInfo]:
    """Every registered part version, optionally filtered by kind or custom flag."""
    _load_builtins()
    out = [info for versions in _PARTS.values() for info in versions.values()
           if (kind is None or info.kind == kind) and (custom is None or info.custom == custom)]
    return sorted(out, key=lambda info: (info.name, _key(info.version)))


def lockfile(objects) -> dict[str, str]:
    """``{part name: exact version}`` for the parts used by ``objects`` (modules or callables).

    Stored in checkpoints so a result can be traced to the exact part versions.
    """
    import torch.nn as nn

    out = {}
    for obj in objects:
        items = obj.modules() if isinstance(obj, nn.Module) else [obj]
        for item in items:
            cls = item if isinstance(item, type) or callable(item) and not isinstance(item, nn.Module) else type(item)
            name = getattr(cls, "part_name", None)
            if name:
                out[name] = getattr(cls, "part_version")
    return dict(sorted(out.items()))


def _load_builtins():
    import fotonet.nn  # noqa: F401  (registers the built-in blocks)
    import fotonet.train  # noqa: F401  (registers the built-in training parts)


__all__ = ["KINDS", "PartInfo", "get", "lockfile", "part", "parts", "register", "resolve"]
