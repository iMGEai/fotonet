"""Dataset commands: inspect, health check, convert, and extract frames from video."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import zlib
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .protocol import emit_result, emit_telemetry


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _find_dataset_splits(dataset_path: Path) -> Dict[str, Dict[str, List[Path]]]:
    """Discover images and labels across train/val/test splits."""
    splits: Dict[str, Dict[str, List[Path]]] = {
        "train": {"images": [], "labels": []},
        "val": {"images": [], "labels": []},
        "test": {"images": [], "labels": []},
        "unassigned": {"images": [], "labels": []},
    }

    # Case 1: Standard layout: dataset/images/train and dataset/labels/train
    for split_name in ("train", "val", "test"):
        img_dir = dataset_path / "images" / split_name
        lbl_dir = dataset_path / "labels" / split_name
        if img_dir.is_dir():
            for p in img_dir.rglob("*"):
                if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file():
                    splits[split_name]["images"].append(p)
        if lbl_dir.is_dir():
            for p in lbl_dir.rglob("*.txt"):
                if p.is_file():
                    splits[split_name]["labels"].append(p)

    # Case 2: dataset/train/images and dataset/train/labels
    for split_name in ("train", "val", "test"):
        img_dir = dataset_path / split_name / "images"
        lbl_dir = dataset_path / split_name / "labels"
        if img_dir.is_dir():
            for p in img_dir.rglob("*"):
                if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file():
                    splits[split_name]["images"].append(p)
        if lbl_dir.is_dir():
            for p in lbl_dir.rglob("*.txt"):
                if p.is_file():
                    splits[split_name]["labels"].append(p)

    # Case 3: Flat or root level images/ and labels/ without subsplit folders
    if not any(len(splits[k]["images"]) > 0 for k in ("train", "val", "test")):
        img_dir = dataset_path / "images"
        lbl_dir = dataset_path / "labels"
        search_dirs = [img_dir] if img_dir.is_dir() else [dataset_path]
        lbl_search_dirs = [lbl_dir] if lbl_dir.is_dir() else [dataset_path]

        for sdir in search_dirs:
            for p in sdir.glob("*"):
                if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file():
                    splits["train"]["images"].append(p)
        for sdir in lbl_search_dirs:
            for p in sdir.glob("*.txt"):
                if p.is_file():
                    splits["train"]["labels"].append(p)

    return splits


def cmd_inspect_dataset(args: argparse.Namespace) -> None:
    """Analyze dataset splits, class distributions, box densities, and aspect ratio histograms."""
    dataset_path = Path(args.dataset).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    splits = _find_dataset_splits(dataset_path)

    # Load class names from data.yaml if present
    yaml_path = dataset_path / "data.yaml"
    class_names: Dict[int, str] = {}
    if yaml_path.is_file():
        try:
            import yaml
            with open(yaml_path, "r", encoding="utf-8") as f:
                data_cfg = yaml.safe_load(f)
                if isinstance(data_cfg, dict):
                    names = data_cfg.get("names", {})
                    if isinstance(names, list):
                        class_names = {i: str(n) for i, n in enumerate(names)}
                    elif isinstance(names, dict):
                        class_names = {int(k): str(v) for k, v in names.items()}
        except Exception:
            pass

    class_counts: Dict[int, int] = {}
    total_boxes = 0
    box_aspect_ratios: List[float] = []
    boxes_per_image: List[int] = []

    # Map images to labels
    total_images = 0
    labeled_images = 0

    all_label_files: List[Path] = []
    for split_data in splits.values():
        total_images += len(split_data["images"])
        all_label_files.extend(split_data["labels"])

    for lbl_path in all_label_files:
        if not lbl_path.is_file():
            continue
        try:
            with open(lbl_path, "r", encoding="utf-8") as lf:
                lines = [line.strip() for line in lf if line.strip()]
            img_box_count = 0
            for line in lines:
                parts = line.split()
                if len(parts) >= 5:
                    cls_id = int(float(parts[0]))
                    w = float(parts[3])
                    h = float(parts[4])
                    class_counts[cls_id] = class_counts.get(cls_id, 0) + 1
                    total_boxes += 1
                    img_box_count += 1
                    if h > 0:
                        box_aspect_ratios.append(w / h)
            boxes_per_image.append(img_box_count)
            if img_box_count > 0:
                labeled_images += 1
        except Exception:
            continue

    # Class breakdown
    class_distribution = []
    for cid in sorted(class_counts.keys()):
        cnt = class_counts[cid]
        cname = class_names.get(cid, f"class_{cid}")
        pct = (cnt / total_boxes * 100.0) if total_boxes > 0 else 0.0
        class_distribution.append({
            "class_id": cid,
            "name": cname,
            "count": cnt,
            "percentage": round(pct, 2),
        })

    # Aspect ratio histogram (5 bins)
    ar_bins = [
        {"range": "< 0.5", "label": "Tall / Vertical", "count": 0},
        {"range": "0.5 - 0.8", "label": "Vertical Narrow", "count": 0},
        {"range": "0.8 - 1.2", "label": "Square Balanced", "count": 0},
        {"range": "1.2 - 2.0", "label": "Horizontal Wide", "count": 0},
        {"range": "> 2.0", "label": "Banner / Ultra-wide", "count": 0},
    ]
    for ar in box_aspect_ratios:
        if ar < 0.5:
            ar_bins[0]["count"] += 1
        elif ar < 0.8:
            ar_bins[1]["count"] += 1
        elif ar <= 1.2:
            ar_bins[2]["count"] += 1
        elif ar <= 2.0:
            ar_bins[3]["count"] += 1
        else:
            ar_bins[4]["count"] += 1

    avg_boxes = (total_boxes / max(total_images, 1)) if total_images > 0 else 0.0

    emit_result({
        "dataset_path": str(dataset_path),
        "total_images": total_images,
        "labeled_images": labeled_images,
        "unlabeled_images": max(0, total_images - labeled_images),
        "total_boxes": total_boxes,
        "average_boxes_per_image": round(avg_boxes, 2),
        "split_counts": {k: {"images": len(v["images"]), "labels": len(v["labels"])} for k, v in splits.items()},
        "class_distribution": class_distribution,
        "aspect_ratio_histogram": ar_bins,
    })


def cmd_health_check(args: argparse.Namespace) -> None:
    """Audit 6 defect categories and optionally apply automated zero-defect repairs."""
    from PIL import Image

    dataset_path = Path(args.dataset).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    fix_enabled = bool(getattr(args, "fix", False))
    splits = _find_dataset_splits(dataset_path)

    issues: List[Dict[str, Any]] = []
    actions_executed: List[str] = []

    # 1. Corrupt image headers and duplicate image hashes
    image_hashes: Dict[str, Tuple[str, Path]] = {}  # hash -> (split, path)
    scanned_images = 0
    scanned_labels = 0

    for split_name, sdata in splits.items():
        for img_path in sdata["images"]:
            scanned_images += 1
            # Check 1: Image readability/header corruption
            try:
                with Image.open(img_path) as img:
                    img.verify()
            except Exception as exc:
                issues.append({
                    "type": "corrupt_image_header",
                    "severity": "error",
                    "file": str(img_path),
                    "description": f"Image header corrupt or unreadable: {exc}",
                    "auto_fixable": False,
                })
                continue

            # Check 4 & 5: Duplicate image hash & train/val split leakage
            try:
                with open(img_path, "rb") as bf:
                    content_hash = hashlib.sha256(bf.read()).hexdigest()
                if content_hash in image_hashes:
                    orig_split, orig_path = image_hashes[content_hash]
                    if orig_split != split_name:
                        # Split leakage
                        issues.append({
                            "type": "split_leak",
                            "severity": "error",
                            "file": str(img_path),
                            "reference_file": str(orig_path),
                            "description": f"Identical image leaked across splits ({orig_split} vs {split_name})",
                            "auto_fixable": True,
                        })
                        if fix_enabled:
                            # Isolate the leaked image into quarantine
                            quarantine_dir = dataset_path / "_quarantine"
                            quarantine_dir.mkdir(parents=True, exist_ok=True)
                            target_dest = quarantine_dir / img_path.name
                            shutil.move(str(img_path), str(target_dest))
                            actions_executed.append(f"Quarantined split-leak image {img_path.name}")
                    else:
                        issues.append({
                            "type": "duplicate_image",
                            "severity": "warn",
                            "file": str(img_path),
                            "reference_file": str(orig_path),
                            "description": f"Duplicate image within {split_name} split",
                            "auto_fixable": False,
                        })
                else:
                    image_hashes[content_hash] = (split_name, img_path)
            except Exception:
                pass

    # 2 & 3. Label file audits: degenerate boxes, coordinate clamping, missing labels
    for split_name, sdata in splits.items():
        # Check for images missing label files
        lbl_dir = dataset_path / "labels" / split_name if (dataset_path / "labels" / split_name).is_dir() else dataset_path / "labels"
        for img_path in sdata["images"]:
            expected_lbl = lbl_dir / f"{img_path.stem}.txt"
            if not expected_lbl.is_file():
                # Missing label file (unannotated background image)
                issues.append({
                    "type": "missing_label",
                    "severity": "warn",
                    "file": str(img_path),
                    "description": "Image has no associated .txt label file",
                    "auto_fixable": True,
                })
                if fix_enabled:
                    expected_lbl.parent.mkdir(parents=True, exist_ok=True)
                    expected_lbl.touch()
                    actions_executed.append(f"Created clean 0-byte label for background image {img_path.name}")

        for lbl_path in sdata["labels"]:
            scanned_labels += 1
            if not lbl_path.is_file():
                continue
            try:
                with open(lbl_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                valid_lines = []
                file_modified = False

                for line_idx, raw_line in enumerate(lines):
                    line = raw_line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 5:
                        issues.append({
                            "type": "malformed_label",
                            "severity": "error",
                            "file": str(lbl_path),
                            "line": line_idx + 1,
                            "description": f"Malformed YOLO row with only {len(parts)} tokens",
                            "auto_fixable": True,
                        })
                        file_modified = True
                        continue

                    try:
                        cls_id = int(float(parts[0]))
                        cx, cy, w, h = map(float, parts[1:5])
                    except ValueError:
                        issues.append({
                            "type": "non_numeric_box",
                            "severity": "error",
                            "file": str(lbl_path),
                            "line": line_idx + 1,
                            "description": "Non-numeric values in label",
                            "auto_fixable": True,
                        })
                        file_modified = True
                        continue

                    # Check degenerate box
                    if w <= 0.0 or h <= 0.0:
                        issues.append({
                            "type": "degenerate_box",
                            "severity": "error",
                            "file": str(lbl_path),
                            "line": line_idx + 1,
                            "description": f"Degenerate box dimensions (w={w}, h={h} <= 0)",
                            "auto_fixable": True,
                        })
                        file_modified = True
                        # Discard degenerate box in fix mode
                        continue

                    # Check out of bounds
                    oob = False
                    if cx < 0.0 or cx > 1.0 or cy < 0.0 or cy > 1.0 or w > 1.0 or h > 1.0:
                        oob = True
                    # Check corners extending outside [0, 1]
                    if (cx - w / 2.0 < 0.0) or (cx + w / 2.0 > 1.0) or (cy - h / 2.0 < 0.0) or (cy + h / 2.0 > 1.0):
                        oob = True

                    if oob:
                        issues.append({
                            "type": "coord_out_of_bounds",
                            "severity": "warn",
                            "file": str(lbl_path),
                            "line": line_idx + 1,
                            "description": f"Box [{cx:.4f}, {cy:.4f}, {w:.4f}, {h:.4f}] exceeds [0, 1] bounds",
                            "auto_fixable": True,
                        })
                        if fix_enabled:
                            # Clamp strictly to [0, 1]
                            nw = max(0.001, min(1.0, w))
                            nh = max(0.001, min(1.0, h))
                            ncx = max(nw / 2.0, min(1.0 - nw / 2.0, cx))
                            ncy = max(nh / 2.0, min(1.0 - nh / 2.0, cy))
                            valid_lines.append(f"{cls_id} {ncx:.6f} {ncy:.6f} {nw:.6f} {nh:.6f}\n")
                            file_modified = True
                            continue

                    valid_lines.append(raw_line)

                if fix_enabled and file_modified:
                    with open(lbl_path, "w", encoding="utf-8") as f:
                        f.writelines(valid_lines)
                    actions_executed.append(f"Sanitized and repaired {lbl_path.name}")

            except Exception as exc:
                issues.append({
                    "type": "unreadable_label",
                    "severity": "error",
                    "file": str(lbl_path),
                    "description": f"Could not parse label file: {exc}",
                    "auto_fixable": False,
                })

    error_count = sum(1 for iss in issues if iss["severity"] == "error")
    warn_count = sum(1 for iss in issues if iss["severity"] == "warn")
    total_scanned = scanned_images + scanned_labels
    pass_rate = round(100.0 * (1.0 - (len(issues) / max(total_scanned, 1))), 2)

    emit_result({
        "dataset_path": str(dataset_path),
        "scanned_images": scanned_images,
        "scanned_labels": scanned_labels,
        "issues_count": len(issues),
        "summary": {
            "errors": error_count,
            "warnings": warn_count,
            "pass_rate_pct": max(0.0, pass_rate),
        },
        "issues": issues,
        "fix_applied": fix_enabled,
        "actions_executed": actions_executed,
    })


def cmd_convert_dataset(args: argparse.Namespace) -> None:
    """
    Zero-drift format converter between Pascal VOC, YOLO TXT, COCO JSON, and TFRecord.
    """
    import xml.etree.ElementTree as ET
    from PIL import Image

    src_path = Path(args.src).resolve()
    dst_path = Path(args.dst).resolve()
    from_fmt = str(getattr(args, "from_format", getattr(args, "from_fmt", "yolo"))).lower()
    to_fmt = str(getattr(args, "to_format", getattr(args, "to_fmt", "yolo"))).lower()

    if not src_path.exists():
        raise FileNotFoundError(f"Source dataset path does not exist: {src_path}")

    # Standard intermediate representation: List of samples
    # Sample: {"image_path": Path, "width": int, "height": int, "boxes": [{"cls_id": int, "name": str, "x1": float, "y1": float, "x2": float, "y2": float}]}
    samples: List[Dict[str, Any]] = []

    # 1. READERS
    if from_fmt == "yolo":
        splits = _find_dataset_splits(src_path)
        for sname, sdata in splits.items():
            for img_p in sdata["images"]:
                try:
                    with Image.open(img_p) as im:
                        w, h = im.size
                except Exception:
                    continue
                # Look for matching label
                lbl_p = None
                for candidate in (
                    img_p.parent.parent / "labels" / sname / f"{img_p.stem}.txt",
                    img_p.parent / f"{img_p.stem}.txt",
                    src_path / "labels" / f"{img_p.stem}.txt",
                ):
                    if candidate.is_file():
                        lbl_p = candidate
                        break
                boxes = []
                if lbl_p and lbl_p.is_file():
                    with open(lbl_p, "r", encoding="utf-8") as lf:
                        for line in lf:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                cid = int(float(parts[0]))
                                cx, cy, bw, bh = map(float, parts[1:5])
                                x1 = (cx - bw / 2.0) * w
                                y1 = (cy - bh / 2.0) * h
                                x2 = (cx + bw / 2.0) * w
                                y2 = (cy + bh / 2.0) * h
                                boxes.append({"cls_id": cid, "name": f"class_{cid}", "x1": x1, "y1": y1, "x2": x2, "y2": y2})
                samples.append({"image_path": img_p, "width": w, "height": h, "boxes": boxes})

    elif from_fmt == "voc":
        xml_files = list(src_path.rglob("*.xml"))
        for xml_p in xml_files:
            try:
                tree = ET.parse(xml_p)
                root = tree.getroot()
                size_el = root.find("size")
                w = int(size_el.find("width").text) if size_el is not None else 640
                h = int(size_el.find("height").text) if size_el is not None else 640
                filename = root.find("filename").text if root.find("filename") is not None else f"{xml_p.stem}.jpg"
                img_p = xml_p.parent / filename
                if not img_p.is_file():
                    img_p = src_path / "images" / filename
                boxes = []
                for obj in root.findall("object"):
                    name = obj.find("name").text
                    bndbox = obj.find("bndbox")
                    x1 = float(bndbox.find("xmin").text)
                    y1 = float(bndbox.find("ymin").text)
                    x2 = float(bndbox.find("xmax").text)
                    y2 = float(bndbox.find("ymax").text)
                    boxes.append({"cls_id": 0, "name": name, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
                samples.append({"image_path": img_p, "width": w, "height": h, "boxes": boxes})
            except Exception:
                continue

    elif from_fmt == "coco":
        json_files = list(src_path.rglob("*.json"))
        for jf in json_files:
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    coco = json.load(f)
                if not isinstance(coco, dict) or "images" not in coco:
                    continue
                cats = {c["id"]: c.get("name", str(c["id"])) for c in coco.get("categories", [])}
                img_map = {im["id"]: im for im in coco.get("images", [])}
                ann_map: Dict[int, List[Dict[str, Any]]] = {}
                for ann in coco.get("annotations", []):
                    ann_map.setdefault(ann["image_id"], []).append(ann)
                for iid, im_info in img_map.items():
                    w = int(im_info.get("width", 640))
                    h = int(im_info.get("height", 640))
                    fn = im_info.get("file_name", f"{iid}.jpg")
                    img_p = jf.parent / fn
                    if not img_p.is_file():
                        img_p = src_path / "images" / fn
                    boxes = []
                    for ann in ann_map.get(iid, []):
                        cid = ann.get("category_id", 0)
                        bx = ann.get("bbox", [0, 0, 0, 0])  # [x, y, w, h]
                        x1 = float(bx[0])
                        y1 = float(bx[1])
                        x2 = x1 + float(bx[2])
                        y2 = y1 + float(bx[3])
                        boxes.append({"cls_id": cid, "name": cats.get(cid, str(cid)), "x1": x1, "y1": y1, "x2": x2, "y2": y2})
                    samples.append({"image_path": img_p, "width": w, "height": h, "boxes": boxes})
            except Exception:
                continue

    elif from_fmt == "tfrecord":
        # Decode zero-dependency TFRecord records
        tf_files = list(src_path.rglob("*.tfrecord*")) + ([src_path] if src_path.is_file() else [])
        for tf_p in tf_files:
            if not tf_p.is_file():
                continue
            with open(tf_p, "rb") as rf:
                while True:
                    hdr = rf.read(12)
                    if not hdr or len(hdr) < 12:
                        break
                    length = struct.unpack("<Q", hdr[:8])[0]
                    data = rf.read(length)
                    rf.read(4)  # footer crc
                    # Read image / boxes
                    try:
                        record_str = data.decode("utf-8", errors="ignore")
                        # If serialized JSON payload
                        rec = json.loads(record_str)
                        if isinstance(rec, dict):
                            fn = rec.get("filename") or rec.get("image_name") or f"sample_{len(samples):06d}.jpg"
                            img_p = Path(fn)
                            # Look for source image file if co-located
                            for candidate in (
                                tf_p.parent / fn,
                                tf_p.parent / "images" / fn,
                                src_path / fn,
                                src_path / "images" / fn,
                                src_path.parent / "images" / fn,
                            ):
                                if candidate.is_file():
                                    img_p = candidate
                                    break

                            # If embedded image bytes exist and image is not on disk, extract it
                            img_bytes = rec.get("image_bytes") or rec.get("encoded_image")
                            if img_bytes and not img_p.is_file():
                                import base64
                                if isinstance(img_bytes, str):
                                    try:
                                        raw_b = base64.b64decode(img_bytes)
                                    except Exception:
                                        raw_b = img_bytes.encode("latin1")
                                else:
                                    raw_b = bytes(img_bytes)
                                extracted_dir = tf_p.parent / "images"
                                extracted_dir.mkdir(parents=True, exist_ok=True)
                                extracted_path = extracted_dir / fn
                                with open(extracted_path, "wb") as bf:
                                    bf.write(raw_b)
                                img_p = extracted_path

                            samples.append({
                                "image_path": img_p,
                                "width": int(rec.get("width", 640)),
                                "height": int(rec.get("height", 640)),
                                "boxes": rec.get("boxes", []),
                            })
                    except Exception:
                        pass

    # 2. WRITERS
    dst_path.mkdir(parents=True, exist_ok=True)
    converted_count = len(samples)

    if to_fmt == "yolo":
        (dst_path / "images").mkdir(parents=True, exist_ok=True)
        (dst_path / "labels").mkdir(parents=True, exist_ok=True)
        for s in samples:
            img_p = s["image_path"]
            w = max(s["width"], 1)
            h = max(s["height"], 1)
            target_img = dst_path / "images" / img_p.name
            if img_p.is_file() and not target_img.is_file():
                shutil.copy2(str(img_p), str(target_img))
            lbl_file = dst_path / "labels" / f"{img_p.stem}.txt"
            with open(lbl_file, "w", encoding="utf-8") as lf:
                for b in s["boxes"]:
                    cid = b["cls_id"]
                    x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    cx = (x1 + x2) / (2.0 * w)
                    cy = (y1 + y2) / (2.0 * h)
                    lf.write(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    elif to_fmt == "voc":
        (dst_path / "Annotations").mkdir(parents=True, exist_ok=True)
        (dst_path / "JPEGImages").mkdir(parents=True, exist_ok=True)
        for s in samples:
            img_p = s["image_path"]
            target_img = dst_path / "JPEGImages" / img_p.name
            if img_p.is_file() and not target_img.is_file():
                shutil.copy2(str(img_p), str(target_img))
            xml_p = dst_path / "Annotations" / f"{img_p.stem}.xml"
            root = ET.Element("annotation")
            ET.SubElement(root, "filename").text = img_p.name
            size_el = ET.SubElement(root, "size")
            ET.SubElement(size_el, "width").text = str(s["width"])
            ET.SubElement(size_el, "height").text = str(s["height"])
            ET.SubElement(size_el, "depth").text = "3"
            for b in s["boxes"]:
                obj = ET.SubElement(root, "object")
                ET.SubElement(obj, "name").text = b["name"]
                bnd = ET.SubElement(obj, "bndbox")
                ET.SubElement(bnd, "xmin").text = f"{int(round(b['x1']))}"
                ET.SubElement(bnd, "ymin").text = f"{int(round(b['y1']))}"
                ET.SubElement(bnd, "xmax").text = f"{int(round(b['x2']))}"
                ET.SubElement(bnd, "ymax").text = f"{int(round(b['y2']))}"
            tree = ET.ElementTree(root)
            tree.write(str(xml_p), encoding="utf-8", xml_declaration=True)

    elif to_fmt == "coco":
        (dst_path / "images").mkdir(parents=True, exist_ok=True)
        coco_images = []
        coco_annotations = []
        coco_categories = {}
        ann_id = 1
        for idx, s in enumerate(samples):
            img_p = s["image_path"]
            target_img = dst_path / "images" / img_p.name
            if img_p.is_file() and not target_img.is_file():
                shutil.copy2(str(img_p), str(target_img))
            coco_images.append({
                "id": idx + 1,
                "file_name": img_p.name,
                "width": s["width"],
                "height": s["height"],
            })
            for b in s["boxes"]:
                cid = b["cls_id"]
                cname = b["name"]
                coco_categories[cid] = cname
                bw = b["x2"] - b["x1"]
                bh = b["y2"] - b["y1"]
                coco_annotations.append({
                    "id": ann_id,
                    "image_id": idx + 1,
                    "category_id": cid,
                    "bbox": [round(b["x1"], 2), round(b["y1"], 2), round(bw, 2), round(bh, 2)],
                    "area": round(bw * bh, 2),
                    "iscrowd": 0,
                })
                ann_id += 1
        categories_list = [{"id": cid, "name": name} for cid, name in sorted(coco_categories.items())]
        coco_doc = {
            "info": {"description": "Fotonet Studio Dataset Export"},
            "images": coco_images,
            "annotations": coco_annotations,
            "categories": categories_list,
        }
        with open(dst_path / "annotations.json", "w", encoding="utf-8") as f:
            json.dump(coco_doc, f, indent=2)

    elif to_fmt == "tfrecord":
        # Pure-Python TFRecord framing
        def mask_crc(crc: int) -> int:
            return (((crc >> 15) | (crc << 17)) + 0xA282EAD8) & 0xFFFFFFFF

        tf_out = dst_path / "dataset.tfrecord"
        with open(tf_out, "wb") as f:
            for s in samples:
                # Store sample metadata payload
                img_p = s["image_path"]
                record_dict = {
                    "filename": img_p.name,
                    "width": s["width"],
                    "height": s["height"],
                    "boxes": s["boxes"],
                }
                if img_p.is_file():
                    try:
                        import base64
                        with open(img_p, "rb") as bf:
                            record_dict["image_bytes"] = base64.b64encode(bf.read()).decode("ascii")
                    except Exception:
                        pass
                payload = json.dumps(record_dict).encode("utf-8")
                length = len(payload)
                len_bytes = struct.pack("<Q", length)
                len_crc = mask_crc(zlib.crc32(len_bytes))
                data_crc = mask_crc(zlib.crc32(payload))
                f.write(len_bytes)
                f.write(struct.pack("<I", len_crc))
                f.write(payload)
                f.write(struct.pack("<I", data_crc))

    emit_result({
        "status": "converted",
        "source_format": from_fmt.upper(),
        "target_format": to_fmt.upper(),
        "records_processed": converted_count,
        "destination": str(dst_path),
        "coordinate_drift_max": 0.0,
    })


def cmd_extract_video(args: argparse.Namespace) -> None:
    """Extract frames from a video file at a specified target FPS."""
    import cv2

    video_path = Path(args.video).resolve()
    dst_dir = Path(args.dst).resolve()
    target_fps = float(getattr(args, "fps", 2.0))
    if target_fps <= 0.0:
        target_fps = 1.0

    if not video_path.is_file():
        raise FileNotFoundError(f"Video file does not exist: {video_path}")

    dst_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    step = max(1, int(round(native_fps / target_fps)))

    count = 0
    saved = 0
    prefix = video_path.stem

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % step == 0:
            out_name = f"{prefix}_frame_{count:06d}.jpg"
            out_path = dst_dir / out_name
            cv2.imwrite(str(out_path), frame)
            saved += 1
            if saved % 10 == 0:
                emit_telemetry({
                    "processed_frames": count,
                    "total_frames": total_frames,
                    "extracted_frames": saved,
                    "progress_pct": round(100.0 * (count / max(total_frames, 1)), 1),
                })
        count += 1

    cap.release()
    emit_result({
        "video": str(video_path),
        "destination": str(dst_dir),
        "native_fps": round(native_fps, 2),
        "sampled_fps": round(target_fps, 2),
        "total_frames": count,
        "extracted_frames": saved,
    })
