"""fotonet v2 family specifications (fotonet2n / Hi Bird).

Layout per tier lives in ``fotonet/config/models/v2/fotonet2n.yaml``.
Direct LTRB regression (reg_max is fixed at 1). The fingerprint is the built layout, not the family
name alone.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

V2_MODEL_IDS = ("fotonet2n",)
GRAPH_VERSION = "2.0-alpha"
LAYOUT_KEYS = ("stem_channels", "p2_channels", "stages", "junctions", "head_hidden", "neck_attention",
               "neck_down", "taps")
HYBRID_EXTRA_KEYS = ("stage_attn", "hyper_ace", "decoder_dim", "decoder_layers", "decoder_queries",
                     "decoder_context", "mlp_ratio")
HYBRID_DEFAULTS = {
    "stage_attn": None,
    "hyper_ace": None,
    "decoder_dim": None,
    "decoder_layers": None,
    "decoder_queries": None,
    "decoder_context": None,
    "mlp_ratio": None,
}
FINGERPRINT_KEYS = ("model_id", *LAYOUT_KEYS, *HYBRID_EXTRA_KEYS, "nc", "reg_max")
# Layout keys added after the first v2 draft, with the value that draft had built in. A stored
# config without them describes that draft, and a layout that keeps a first-draft value leaves the
# key out of the fingerprint, so first-draft checkpoints still rebuild their graph and still match.
FIRST_DRAFT = {"neck_attention": True, "neck_down": ("rep", "rep"), "taps": False, **HYBRID_DEFAULTS}
DOWN_KINDS = ("rep", "light")
DECODER_CONTEXT_MODES = ("edge_sample", "hyper_xattn")


def _row(row):
    out_width, block_width, blocks, ratio = row
    return int(out_width), int(block_width), int(blocks), float(ratio)


def normalize_layout(layout: dict) -> dict:
    """Layout values as ints, bools and tuples, whatever they were read as."""
    out = {
        "stem_channels": int(layout["stem_channels"]),
        "p2_channels": int(layout["p2_channels"]),
        "stages": tuple(_row(row) for row in layout["stages"]),
        "junctions": tuple(_row(row) for row in layout["junctions"]),
        "head_hidden": tuple(int(v) for v in layout["head_hidden"]),
        "neck_attention": bool(layout.get("neck_attention", FIRST_DRAFT["neck_attention"])),
        "neck_down": tuple(str(v) for v in layout.get("neck_down", FIRST_DRAFT["neck_down"])),
        "taps": bool(layout.get("taps", FIRST_DRAFT["taps"])),
    }
    if layout.get("stage_attn") is not None:
        out["stage_attn"] = tuple((int(r[0]), int(r[1])) for r in layout["stage_attn"])
    if layout.get("hyper_ace") is not None:
        out["hyper_ace"] = (int(layout["hyper_ace"][0]), int(layout["hyper_ace"][1]))
    if layout.get("decoder_dim") is not None:
        out["decoder_dim"] = int(layout["decoder_dim"])
    if layout.get("decoder_layers") is not None:
        out["decoder_layers"] = int(layout["decoder_layers"])
    if layout.get("decoder_queries") is not None:
        out["decoder_queries"] = int(layout["decoder_queries"])
    if layout.get("decoder_context") is not None:
        out["decoder_context"] = str(layout["decoder_context"]).strip().lower()
    if layout.get("mlp_ratio") is not None:
        out["mlp_ratio"] = float(layout["mlp_ratio"])
    return out


def validate_layout(*, stem_channels, p2_channels, stages, junctions, head_hidden, nc, reg_max,
                    neck_attention=True, neck_down=("rep", "rep"), taps=False,
                    stage_attn=None, hyper_ace=None, decoder_dim=None, decoder_layers=None,
                    decoder_queries=None, decoder_context=None, mlp_ratio=None):
    if len(stages) != 4 or len(junctions) != 4:
        raise ValueError("v2 needs four backbone stages and four neck junctions.")
    if len(head_hidden) != 2:
        raise ValueError("v2 head_hidden is (regression width, classification width).")
    for row in (*stages, *junctions):
        if len(row) != 4:
            raise ValueError("a v2 stage row is (output width, block width, blocks, middle ratio).")
        out_width, block_width, blocks, ratio = row
        if int(out_width) <= 0 or int(block_width) <= 0 or int(blocks) < 1 or not 0.0 < float(ratio) <= 4.0:
            raise ValueError(f"v2 stage row {tuple(row)} has a width, block count or ratio out of range.")
    if any(int(v) <= 0 for v in (stem_channels, p2_channels, nc, *head_hidden)):
        raise ValueError("v2 widths and class counts must be positive.")
    if any(int(row[0]) % 4 for row in (*stages[1:], *junctions)):
        raise ValueError("v2 pyramid widths must be divisible by 4.")
    if len(neck_down) != 2 or any(kind not in DOWN_KINDS for kind in neck_down):
        raise ValueError(f"v2 neck_down is two of {DOWN_KINDS} (the steps to P4 and to P5).")
    if taps:
        # P3, P4, P5: the backbone stage's width against the neck output it is added to.
        for level, (stage, junction) in enumerate(zip(stages[1:], junctions[1:]), start=3):
            if int(stage[0]) % int(junction[0]):
                raise ValueError(
                    f"v2 taps need the backbone width at P{level} ({stage[0]}) to be a multiple of the "
                    f"neck width there ({junction[0]}).")
    if stage_attn is not None:
        if len(stage_attn) != 4:
            raise ValueError("v2 stage_attn must list 4 (n_attn, area) pairs.")
        for n_a, area in stage_attn:
            if int(n_a) < 0 or int(area) < 1:
                raise ValueError(f"invalid stage_attn entry {(n_a, area)!r}.")
    if hyper_ace is not None:
        if len(hyper_ace) != 2 or int(hyper_ace[0]) <= 0 or int(hyper_ace[1]) <= 0:
            raise ValueError(f"v2 hyper_ace must be (hidden, num_edges) > 0, got {hyper_ace!r}.")
    if decoder_dim is not None and (int(decoder_dim) <= 0 or int(decoder_dim) % 4):
        raise ValueError(f"v2 decoder_dim must be positive and divisible by 4, got {decoder_dim!r}.")
    if decoder_layers is not None and int(decoder_layers) < 1:
        raise ValueError(f"v2 decoder_layers must be >= 1, got {decoder_layers!r}.")
    if decoder_queries is not None and int(decoder_queries) < 1:
        raise ValueError(f"v2 decoder_queries must be >= 1, got {decoder_queries!r}.")
    if decoder_context is not None and str(decoder_context) not in DECODER_CONTEXT_MODES:
        raise ValueError(f"v2 decoder_context must be one of {DECODER_CONTEXT_MODES}, got {decoder_context!r}.")
    if mlp_ratio is not None and float(mlp_ratio) <= 0.0:
        raise ValueError(f"v2 mlp_ratio must be > 0, got {mlp_ratio!r}.")
    if int(reg_max) != 1:
        raise ValueError("v2 uses direct LTRB regression; reg_max must be 1.")


@dataclass(frozen=True)
class V2Spec:
    """One v2 detector profile."""

    model_id: str
    stem_channels: int
    p2_channels: int
    stages: tuple
    junctions: tuple
    head_hidden: tuple
    neck_attention: bool = FIRST_DRAFT["neck_attention"]
    neck_down: tuple = FIRST_DRAFT["neck_down"]
    taps: bool = FIRST_DRAFT["taps"]
    stage_attn: tuple | None = None
    hyper_ace: tuple | None = None
    decoder_dim: int | None = None
    decoder_layers: int | None = None
    decoder_queries: int | None = None
    decoder_context: str | None = None
    mlp_ratio: float | None = None
    reg_max: int = 1
    nc: int = 80
    imgsz: int = 640

    def __post_init__(self) -> None:
        if self.model_id not in V2_MODEL_IDS:
            raise ValueError(f"Unknown v2 model id {self.model_id!r}; expected one of {V2_MODEL_IDS}.")
        validate_layout(stem_channels=self.stem_channels, p2_channels=self.p2_channels, stages=self.stages,
                        junctions=self.junctions, head_hidden=self.head_hidden, nc=self.nc, reg_max=self.reg_max,
                        neck_attention=self.neck_attention, neck_down=self.neck_down, taps=self.taps,
                        stage_attn=self.stage_attn, hyper_ace=self.hyper_ace, decoder_dim=self.decoder_dim,
                        decoder_layers=self.decoder_layers, decoder_queries=self.decoder_queries,
                        decoder_context=self.decoder_context, mlp_ratio=self.mlp_ratio)


def _load_family() -> dict[str, V2Spec]:
    """Read the tier layouts from ``fotonet/config/models/v2/*.yaml``."""
    from fotonet.config.model_recipes import load_model_recipe

    family = {}
    for model_id in V2_MODEL_IDS:
        recipe = load_model_recipe(model_id, generation="v2")
        if recipe.get("model_id") != model_id or str(recipe.get("graph_version")) != GRAPH_VERSION:
            raise ValueError(f"model recipe for {model_id} does not match v2 graph {GRAPH_VERSION}.")
        family[model_id] = V2Spec(model_id=model_id, **normalize_layout(dict(recipe["layout"])))
    return family


FAMILY: dict[str, V2Spec] = _load_family()


def get_spec(model_id: str) -> V2Spec:
    key = str(model_id).strip().lower()
    if key not in FAMILY:
        raise ValueError(f"Unknown v2 model id {model_id!r}; expected one of {sorted(FAMILY)}.")
    return FAMILY[key]


def _as_jsonable(value):
    if isinstance(value, (tuple, list)):
        return [_as_jsonable(item) for item in value]
    return value


def architecture_fingerprint(config: dict) -> str:
    """SHA256 of the layout that was actually built. Image size is not part of the identity."""
    payload = {"graph_version": GRAPH_VERSION}
    for key in FINGERPRINT_KEYS:
        if key in FIRST_DRAFT:
            value = _as_jsonable(config.get(key, FIRST_DRAFT[key]))
            if value != _as_jsonable(FIRST_DRAFT[key]):
                payload[key] = value
            continue
        if key not in config:
            raise KeyError(f"v2 fingerprint config is missing {key!r}.")
        payload[key] = _as_jsonable(config[key])
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(blob).hexdigest()


def fingerprint_matches(config: dict, stored: str) -> bool:
    if not stored or not isinstance(config, dict):
        return False
    try:
        return stored == architecture_fingerprint(config)
    except (KeyError, TypeError, ValueError):
        return False


__all__ = ["DOWN_KINDS", "FAMILY", "FINGERPRINT_KEYS", "FIRST_DRAFT", "GRAPH_VERSION", "LAYOUT_KEYS",
           "V2_MODEL_IDS", "V2Spec", "architecture_fingerprint", "fingerprint_matches", "get_spec",
           "normalize_layout", "validate_layout"]
