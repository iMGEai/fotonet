"""Tiny starter datasets (``foto8``, ``foto128``) for first runs and smoke tests.

Built from COCO validation images with their YOLO labels, an attribution file
listing every image's license and source, and a data YAML. ``data="foto8"``
then works anywhere a data config is accepted.

``build_starter`` makes one locally from a COCO copy. ``resolve_starter``
finds a built copy in the cache, or downloads the published archive once
it exists (release asset ``v1.0.0/<name>.zip``; not published yet).
"""
from __future__ import annotations

import json
import os
import random
import shutil
import zipfile
from pathlib import Path

# (train images, val images, val reuses the train images)
STARTERS = {"foto8": (4, 4, False), "foto128": (128, 128, True)}


def _cache_root():
    return Path(os.environ.get("FOTONET_DATASETS_DIR", Path.home() / ".cache" / "fotonet" / "datasets")).expanduser()


def _attribution(annotations, files):
    """Per-image license lines from a COCO instances JSON, when available."""
    if not annotations or not Path(annotations).is_file():
        return ["COCO 2017 images: see https://cocodataset.org/#termsofuse (Flickr, Creative Commons licenses)."]
    with open(annotations, "r", encoding="utf-8") as handle:
        coco = json.load(handle)
    licenses = {lic["id"]: lic for lic in coco.get("licenses", [])}
    by_name = {img["file_name"]: img for img in coco.get("images", [])}
    lines = []
    for name in files:
        image = by_name.get(name, {})
        lic = licenses.get(image.get("license"), {})
        lines.append(f"- {name}: {lic.get('name', 'unknown license')} ({lic.get('url', '')}); "
                     f"source {image.get('flickr_url') or image.get('coco_url', 'unknown')}")
    return lines


def build_starter(data, name="foto8", out_dir=None, seed=0, annotations=None):
    """Build a starter dataset from the ``val`` split of a COCO-style data config."""
    from fotonet.data.dataset import DetectionDataset
    from fotonet.utils.config import load_data_cfg

    if name not in STARTERS:
        raise ValueError(f"Unknown starter {name!r}; expected one of {sorted(STARTERS)}.")
    n_train, n_val, shared = STARTERS[name]
    cfg = load_data_cfg(os.fspath(data))
    if not cfg.get("val"):
        raise ValueError("The source data config needs a 'val' split to draw starter images from.")
    source = DetectionDataset(cfg["val"], imgsz=640, cache_labels=True, num_classes=cfg.get("nc"))
    labelled = [i for i in range(len(source.img_files)) if len(source._get_label(i)["labels"])]
    count = n_train if shared else n_train + n_val
    picked = sorted(random.Random(int(seed)).sample(labelled, min(count, len(labelled))))
    train = picked[:n_train]
    splits = {"train": train, "val": train if shared else picked[n_train:n_train + n_val]}
    root = Path(out_dir) if out_dir else _cache_root() / name
    if root.exists():
        shutil.rmtree(root)
    copied = []
    for split, indices in splits.items():
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        for index in indices:
            image = Path(source.img_files[index])
            label = Path(source._get_label_path(str(image)))
            shutil.copy2(image, root / "images" / split / image.name)
            shutil.copy2(label, root / "labels" / split / label.name)
            copied.append(image.name)
    names = cfg.get("names") or {}
    data_yaml = root / "data.yaml"
    lines = [f"path: {root.as_posix()}", "train: images/train", "val: images/val", f"nc: {len(names) or cfg.get('nc')}"]
    if names:
        lines.append("names: " + json.dumps([names[i] for i in range(len(names))]))
    data_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "ATTRIBUTION.md").write_text(
        f"# {name}\n\nImages and labels from COCO 2017 val (https://cocodataset.org), "
        "annotations under CC BY 4.0. Per-image licenses:\n\n"
        + "\n".join(_attribution(annotations, sorted(set(copied)))) + "\n",
        encoding="utf-8",
    )
    return str(data_yaml)


def resolve_starter(name):
    """Path to a starter's data YAML: cached copy, else the published archive."""
    if name not in STARTERS:
        raise ValueError(f"Unknown starter {name!r}; expected one of {sorted(STARTERS)}.")
    root = _cache_root() / name
    data_yaml = root / "data.yaml"
    if data_yaml.is_file():
        return str(data_yaml)
    if os.environ.get("FOTONET_OFFLINE", "").strip().lower() not in {"1", "true", "yes", "on"}:
        from fotonet.models.release import RELEASE_HOST, _download

        archive = root.with_suffix(".zip")
        try:
            _download(f"{RELEASE_HOST}/v1.0.0/{name}.zip", archive)
            with zipfile.ZipFile(archive) as bundle:
                bundle.extractall(root)
            archive.unlink(missing_ok=True)
            if data_yaml.is_file():
                text = data_yaml.read_text(encoding="utf-8").splitlines()
                text = [f"path: {root.as_posix()}" if line.startswith("path:") else line for line in text]
                data_yaml.write_text("\n".join(text) + "\n", encoding="utf-8")
                return str(data_yaml)
        except Exception:
            pass
    raise FileNotFoundError(
        f"Starter dataset {name!r} is not available locally and could not be downloaded. Build it from a "
        f"COCO copy with: fotonet starter name={name} data=path/to/coco.yaml"
    )


def resolve_data(data):
    """Return ``data`` unchanged, or the YAML path when it names a starter dataset."""
    if isinstance(data, str) and data in STARTERS and not os.path.isfile(data):
        return resolve_starter(data)
    return data


__all__ = ["STARTERS", "build_starter", "resolve_starter", "resolve_data"]
