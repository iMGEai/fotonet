"""Versioned parts: one registry for model blocks, training pieces, and custom nodes.

Every reusable part has a name and a version, ``name@MAJOR.MINOR.PATCH``. The
version describes the math, not the code:

- PATCH: a fix that does not change outputs (a crash, an edge case, a message).
- MINOR: the same math, better code (faster, less memory), outputs within a
  tested tolerance. Every model that uses the part gets it.
- MAJOR: different math. New models only; a released model keeps its major.

From fotonet 1.2 on, a new built-in part carries the version of the fotonet
release it ships in, ``name@RELEASE`` (``assign.dual_o2o_o2m@1.2``). A release
version is a line of its own: it is reached by its exact ref only, never by a
major-only ref such as ``name@1``, which keeps meaning the newest
``1.MINOR.PATCH``. The parts that came out with v1.1 keep their ``1.0.0``.

A part that was renamed keeps its old ref as an alias (``alias()``), so run
files and checkpoints that name it still resolve.

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
    def release(self) -> bool:
        """True for a release version (``1.2``), False for ``MAJOR.MINOR.PATCH``."""
        return self.version.count(".") == 1

    @property
    def ref(self) -> str:
        return f"{self.name}@{self.version}"


_PARTS: dict[str, dict[str, PartInfo]] = {}
# old ref -> current ref, for renamed parts
_ALIASES: dict[str, str] = {}


def _check_version(version: str) -> str:
    pieces = str(version).split(".")
    ok = len(pieces) in (2, 3) and all(piece.isdigit() for piece in pieces)
    # Release versions started with fotonet 1.2; "1.0" or "1.1" would be a typo for a semver.
    if ok and len(pieces) == 2 and (int(pieces[0]), int(pieces[1])) < (1, 2):
        ok = False
    if not ok:
        raise ValueError(f"part versions are MAJOR.MINOR.PATCH or a fotonet release from 1.2 on, got {version!r}.")
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
    pieces = tuple(int(piece) for piece in version.split("."))
    return pieces + (0,) * (3 - len(pieces))


def alias(old_ref: str, new_ref: str) -> None:
    """Let ``old_ref`` (a renamed part's former ``name@version``) resolve to ``new_ref``."""
    old_ref, new_ref = str(old_ref), str(new_ref)
    if old_ref in _ALIASES and _ALIASES[old_ref] != new_ref:
        raise ValueError(f"{old_ref} is already an alias of {_ALIASES[old_ref]}.")
    name, _, version = old_ref.partition("@")
    if version in _PARTS.get(name, {}):
        raise ValueError(f"{old_ref} is a registered part; it cannot also be an alias.")
    _ALIASES[old_ref] = new_ref


def canonical_version(name: str, version: str) -> str:
    """The current version for ``name@version``: the alias target's, or ``version`` itself."""
    target = _ALIASES.get(f"{name}@{version}")
    return target.partition("@")[2] if target else str(version)


def get(name: str, major: int | None = None) -> PartInfo:
    """The newest version of ``name``; with ``major``, the newest ``major.MINOR.PATCH`` (never a release version)."""
    versions = _PARTS.get(str(name))
    if not versions:
        raise KeyError(f"no part named {name!r}.")
    candidates = [info for info in versions.values()
                  if major is None or (not info.release and info.major == int(major))]
    if not candidates:
        raise KeyError(f"{name!r} has no version with major {major}.")
    return max(candidates, key=lambda info: _key(info.version))


def resolve(ref: str) -> PartInfo:
    """``"nn.cspstar@1"`` (newest 1.x.y), ``"nn.cspstar@1.0.0"`` or ``"sched.lr@1.2"`` (exact), or
    ``"nn.cspstar"`` (newest). A renamed part's old ref resolves to its new one."""
    ref = _ALIASES.get(str(ref), str(ref))
    name, _, version = ref.partition("@")
    if not version:
        return get(name)
    if "." in version:
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


__all__ = ["KINDS", "PartInfo", "alias", "canonical_version", "get", "lockfile", "part", "parts", "register",
           "resolve"]
