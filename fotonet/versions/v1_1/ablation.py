"""Subset ablation protocol for v1.1 training ideas.

Every idea is ranked the same way: train ``fotonetn`` with the
``fotonetn_ablation`` recipe on one fixed, seeded COCO subset, then compare
its ``mAP vs images seen`` curve with the baseline run.

Keep rule: a change stays only if it adds at least 0.3 mAP50-95 at the end, or
reaches the baseline's final mAP with at least 15% fewer images seen while
ending no more than 0.1 mAP below it.

Usage::

    python -m fotonet.versions.v1_1.ablation prepare data=coco.yaml out=runs/ablation/subset
    fotonet train model=fotonetn data=runs/ablation/subset/data.yaml \\
        recipe=fotonetn_ablation out_dir=runs/ablation/<name>
    python -m fotonet.versions.v1_1.ablation compare \\
        base=runs/ablation/base/fotonetn/metrics.jsonl cand=runs/ablation/<name>/fotonetn/metrics.jsonl
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys

import yaml

PROTOCOL = {
    "recipe": "fotonetn_ablation",
    "model": "fotonetn",
    "train_fraction": 0.20,
    "val_images": 1000,
    "seed": 0,
    "metric": "mAP50_95",
    "keep_gain": 0.003,         # +0.3 mAP; metrics files store fractions
    "keep_efficiency": 0.15,    # >= 15% fewer images to the baseline's final mAP
    "efficiency_tolerance": 0.001,  # ...while ending at most 0.1 mAP lower
}


def _digest(paths):
    return hashlib.sha256("\n".join(paths).encode("utf-8")).hexdigest()


def make_subset(data, out_dir, train_fraction=PROTOCOL["train_fraction"],
                val_images=PROTOCOL["val_images"], seed=PROTOCOL["seed"]):
    """Write a seeded train/val subset as image manifests plus a data YAML.

    Labels are found exactly as for the full dataset because the manifests
    list the original absolute image paths. Returns the new YAML path.
    """
    from fotonet.data.dataset import expand_image_sources
    from fotonet.utils.config import load_data_cfg

    fraction = float(train_fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"train_fraction must be in (0, 1], got {train_fraction!r}.")
    cfg = load_data_cfg(os.fspath(data))
    if not cfg.get("train") or not cfg.get("val"):
        raise ValueError("The data config needs both 'train' and 'val' sources for an ablation subset.")
    train_all = expand_image_sources(cfg["train"])
    val_all = expand_image_sources(cfg["val"])
    n_train = max(1, int(round(len(train_all) * fraction)))
    n_val = max(1, min(int(val_images), len(val_all)))
    train = sorted(random.Random(int(seed)).sample(train_all, n_train))
    val = sorted(random.Random(int(seed) + 1).sample(val_all, n_val))

    out_dir = os.path.abspath(os.fspath(out_dir))
    os.makedirs(out_dir, exist_ok=True)
    for name, paths in (("train.txt", train), ("val.txt", val)):
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as handle:
            handle.write("\n".join(paths) + "\n")
    names = cfg.get("names")
    data_cfg = {"path": out_dir, "train": "train.txt", "val": "val.txt"}
    if cfg.get("nc") is not None:
        data_cfg["nc"] = int(cfg["nc"])
    if names:
        data_cfg["names"] = [names[index] for index in range(len(names))]
    yaml_path = os.path.join(out_dir, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data_cfg, handle, sort_keys=False, allow_unicode=True)
    manifest = {
        "source": os.path.abspath(os.fspath(data)),
        "seed": int(seed),
        "train_fraction": fraction,
        "train_images": len(train),
        "val_images": len(val),
        "train_sha256": _digest(train),
        "val_sha256": _digest(val),
    }
    with open(os.path.join(out_dir, "subset.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return yaml_path


def train_command(data_yaml, run_dir, model=PROTOCOL["model"]):
    """The exact CLI line for one protocol run."""
    return (
        f"fotonet train model={model} data={data_yaml} "
        f"recipe={PROTOCOL['recipe']} out_dir={run_dir}"
    )


def read_curve(metrics_path, metric=PROTOCOL["metric"]):
    """``[(images_seen, value), ...]`` for every evaluated epoch of a run."""
    curve = []
    with open(os.fspath(metrics_path), "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if metric not in row:
                continue
            if "images_seen" not in row:
                raise ValueError(
                    f"{metrics_path}:{line_no} has no 'images_seen'; the run predates the "
                    "ablation protocol and cannot be compared on the images-seen axis."
                )
            curve.append((int(row["images_seen"]), float(row[metric])))
    if not curve:
        raise ValueError(f"{metrics_path} has no evaluated epochs with {metric!r}.")
    return sorted(curve)


def images_to_reach(curve, target):
    """Images seen when ``curve`` first reaches ``target`` (linear between evaluations).

    Before the first evaluation the curve is taken to start at (0, 0). Returns
    None when the target is never reached.
    """
    prev_images, prev_value = 0, 0.0
    for images, value in curve:
        if value >= target:
            if value == prev_value:
                return float(images)
            share = (target - prev_value) / (value - prev_value)
            return prev_images + max(0.0, min(1.0, share)) * (images - prev_images)
        prev_images, prev_value = images, value
    return None


def compare(baseline, candidate, metric=PROTOCOL["metric"], keep_gain=PROTOCOL["keep_gain"],
            keep_efficiency=PROTOCOL["keep_efficiency"],
            efficiency_tolerance=PROTOCOL["efficiency_tolerance"]):
    """Apply the keep rule to two runs' metrics files and return the verdict."""
    base = read_curve(baseline, metric) if not isinstance(baseline, list) else sorted(baseline)
    cand = read_curve(candidate, metric) if not isinstance(candidate, list) else sorted(candidate)
    base_final, cand_final = base[-1][1], cand[-1][1]
    gain = cand_final - base_final
    # A curve always reaches its own final value, at its last point at latest.
    base_images = images_to_reach(base, base_final) or float(base[-1][0])
    cand_images = images_to_reach(cand, base_final)
    saving = None if cand_images is None else 1.0 - cand_images / max(base_images, 1.0)
    by_gain = gain >= keep_gain
    by_efficiency = (
        saving is not None and saving >= keep_efficiency and gain >= -efficiency_tolerance
    )
    budget_mismatch = abs(cand[-1][0] - base[-1][0]) > 0.01 * max(base[-1][0], 1)
    if by_gain:
        reason = f"+{100 * gain:.2f} mAP at the end"
    elif by_efficiency:
        reason = f"baseline mAP reached with {100 * saving:.1f}% fewer images"
    else:
        reason = (
            f"{100 * gain:+.2f} mAP at the end"
            + ("" if saving is None else f", {100 * saving:+.1f}% images to the baseline mAP")
        )
    return {
        "metric": metric,
        "baseline_final": base_final,
        "candidate_final": cand_final,
        "gain": gain,
        "baseline_images_to_final": base_images,
        "candidate_images_to_baseline_final": cand_images,
        "images_saving": saving,
        "keep": bool(by_gain or by_efficiency),
        "reason": reason,
        "budget_mismatch": budget_mismatch,
    }


def _parse(argv):
    if not argv or argv[0] not in {"prepare", "compare"}:
        raise SystemExit(__doc__)
    options = {}
    for token in argv[1:]:
        if "=" not in token:
            raise SystemExit(f"Invalid argument {token!r}; use key=value.")
        key, value = token.split("=", 1)
        options[key.replace("-", "_")] = value
    return argv[0], options


def main(argv=None):
    command, options = _parse(list(sys.argv[1:] if argv is None else argv))
    if command == "prepare":
        if "data" not in options or "out" not in options:
            raise SystemExit("prepare needs data=<data.yaml> out=<folder>.")
        path = make_subset(
            options["data"], options["out"],
            train_fraction=float(options.get("fraction", PROTOCOL["train_fraction"])),
            val_images=int(options.get("val_images", PROTOCOL["val_images"])),
            seed=int(options.get("seed", PROTOCOL["seed"])),
        )
        print(f"subset={path}")
        print("run: " + train_command(path, "runs/ablation/<name>"))
        return 0
    if "base" not in options or "cand" not in options:
        raise SystemExit("compare needs base=<metrics.jsonl> cand=<metrics.jsonl>.")
    verdict = compare(options["base"], options["cand"], metric=options.get("metric", PROTOCOL["metric"]))
    print(json.dumps(verdict, indent=2))
    print(("KEEP: " if verdict["keep"] else "REJECT: ") + verdict["reason"])
    if verdict["budget_mismatch"]:
        print("warning: the two runs saw different numbers of images; check they used the same protocol.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PROTOCOL", "make_subset", "train_command", "read_curve", "images_to_reach", "compare", "main",
]
