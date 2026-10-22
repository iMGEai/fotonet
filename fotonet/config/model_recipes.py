"""Model recipes: a model is its layout plus the versioned parts it is built from.

Official recipes ship in the package under ``fotonet/config/models/<generation>/``.
Loading one only reads YAML; ``validate_model_recipe`` checks the schema, that
every part reference resolves in ``fotonet.registry``, the generation's layout
rules, and optionally the deploy budget.
"""
from __future__ import annotations

from pathlib import Path

import yaml

MODELS_DIR = Path(__file__).resolve().parent / "models"
GENERATIONS = {"v1.1": "v1_1"}
_TOP_KEYS = {"model_id", "generation", "graph_version", "layout", "parts"}
_V11_LAYOUT_KEYS = {
    "stem_channels", "p2_channels", "widths", "depths", "neck_channels", "neck_depths", "head_hidden",
    "attn_p5", "attn_p4", "head_levels",
}


def recipe_path(model_id: str, generation: str = "v1.1") -> Path:
    folder = GENERATIONS.get(str(generation))
    if folder is None:
        raise ValueError(f"unknown generation {generation!r}; expected one of {sorted(GENERATIONS)}.")
    return MODELS_DIR / folder / f"{str(model_id).strip().lower()}.yaml"


def load_model_recipe(ref, generation: str = "v1.1") -> dict:
    """Read a recipe by official model id or by path to a ``.yaml`` file."""
    path = Path(ref)
    if path.suffix.lower() not in (".yaml", ".yml"):
        path = recipe_path(ref, generation)
    if not path.is_file():
        raise FileNotFoundError(f"no model recipe at {path}.")
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a model recipe (expected a mapping).")
    return data


def iter_part_refs(parts) -> list[str]:
    refs = []
    for value in (parts or {}).values():
        refs.extend(value if isinstance(value, list) else [value])
    return [str(ref) for ref in refs]


def validate_model_recipe(recipe: dict, *, budget: bool = False) -> dict:
    """Check a recipe; returns it unchanged or raises ``ValueError`` naming every problem."""
    problems = []
    unknown = set(recipe) - _TOP_KEYS
    missing = _TOP_KEYS - set(recipe)
    if unknown:
        problems.append(f"unknown keys: {sorted(unknown)}")
    if missing:
        problems.append(f"missing keys: {sorted(missing)}")
    generation = recipe.get("generation")
    if generation not in GENERATIONS:
        problems.append(f"generation must be one of {sorted(GENERATIONS)}, got {generation!r}")
    from fotonet import registry

    registry.parts()                      # load the built-in parts
    for ref in iter_part_refs(recipe.get("parts")):
        try:
            registry.resolve(ref)
        except KeyError as exc:
            problems.append(str(exc).strip("'\""))
    layout = recipe.get("layout") or {}
    if generation == "v1.1" and not problems:
        if set(layout) != _V11_LAYOUT_KEYS:
            problems.append(f"v1.1 layout keys must be {sorted(_V11_LAYOUT_KEYS)}, got {sorted(layout)}")
        else:
            from fotonet.versions.v1_1.spec import GRAPH_VERSION, validate_layout

            if str(recipe.get("graph_version")) != GRAPH_VERSION:
                problems.append(f"v1.1 graph_version is {GRAPH_VERSION}, got {recipe.get('graph_version')!r}")
            try:
                validate_layout(**{key: tuple(v) if isinstance(v, list) else v for key, v in layout.items()},
                                nc=80, reg_max=1)
            except (TypeError, ValueError) as exc:
                problems.append(str(exc))
    if problems:
        raise ValueError(f"model recipe {recipe.get('model_id', '?')!r}: " + "; ".join(problems))
    if budget and generation == "v1.1":
        from fotonet.versions.v1_1.budget import measure

        report = measure(str(recipe["model_id"]))
        if not report.passes:
            raise ValueError(f"model recipe {recipe['model_id']!r} misses its budget gate: {report.to_dict()}")
    return recipe


__all__ = ["GENERATIONS", "MODELS_DIR", "iter_part_refs", "load_model_recipe", "recipe_path", "validate_model_recipe"]
