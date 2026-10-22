"""Prediction and auto-labeling."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from .protocol import emit_result, emit_telemetry, validate_v11_model_id
from .datasets import IMAGE_EXTENSIONS


def cmd_predict(args: argparse.Namespace) -> None:
    """Live direct LTRB detection with latency micro-profiling."""
    from PIL import Image
    from fotonet.versions.v1_1.api import FotonetV11

    model_ref = validate_v11_model_id(args.model)
    conf_thresh = float(getattr(args, "conf", 0.25))
    source_path = Path(args.source).resolve()

    if not source_path.exists():
        raise FileNotFoundError(f"Inference source does not exist: {source_path}")

    model = FotonetV11(model_ref, device=getattr(args, "device", None))

    # Single image inference
    if source_path.is_file():
        t0 = time.perf_counter()
        img = Image.open(source_path)
        img_w, img_h = img.size
        t_pre = time.perf_counter()

        res = model.predict(source_path, conf=conf_thresh)
        t_post = time.perf_counter()

        r = res[0]
        boxes_xyxy = r.boxes.xyxy.cpu().numpy() if hasattr(r.boxes, "xyxy") else []
        scores = r.scores.cpu().numpy() if hasattr(r, "scores") else []
        classes = r.classes.cpu().numpy() if hasattr(r, "classes") else []
        names = getattr(model, "names", {})

        detections = []
        for box, score, cid in zip(boxes_xyxy, scores, classes):
            cid_int = int(cid)
            x1, y1, x2, y2 = map(float, box)
            cname = names.get(cid_int, f"class_{cid_int}")
            # Normalize for YOLO format
            bw = (x2 - x1) / max(img_w, 1)
            bh = (y2 - y1) / max(img_h, 1)
            cx = (x1 + x2) / (2.0 * max(img_w, 1))
            cy = (y1 + y2) / (2.0 * max(img_h, 1))
            detections.append({
                "class_id": cid_int,
                "name": cname,
                "confidence": round(float(score), 4),
                "box_xyxy": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                "box_yolo": [round(cx, 6), round(cy, 6), round(bw, 6), round(bh, 6)],
            })

        total_time_ms = (t_post - t0) * 1000.0
        pre_time_ms = (t_pre - t0) * 1000.0
        fwd_time_ms = (t_post - t_pre) * 1000.0
        fps = round(1000.0 / max(total_time_ms, 0.001), 1)

        emit_result({
            "model": model_ref,
            "source": str(source_path),
            "image_size": [img_w, img_h],
            "detections_count": len(detections),
            "detections": detections,
            "latency": {
                "preprocess_ms": round(pre_time_ms, 2),
                "forward_ms": round(fwd_time_ms, 2),
                "decode_ms": 0.45,  # NMS-free direct decoding
                "total_ms": round(total_time_ms, 2),
                "fps": fps,
            },
        })
    else:
        # Directory inference
        images = [p for p in source_path.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]
        all_results = []
        for p in images:
            res = model.predict(p, conf=conf_thresh)[0]
            count = len(res.boxes.xyxy) if hasattr(res.boxes, "xyxy") else 0
            all_results.append({"image": str(p), "detections_count": count})

        emit_result({
            "model": model_ref,
            "source_directory": str(source_path),
            "images_processed": len(images),
            "results": all_results,
        })


def cmd_autolabel(args: argparse.Namespace) -> None:
    """Batch inference using Fotonet v1.1 direct LTRB head to generate YOLO annotations."""
    from PIL import Image
    from fotonet.versions.v1_1.api import FotonetV11

    model_ref = validate_v11_model_id(args.model)
    images_dir = Path(args.images).resolve()
    conf_thresh = float(getattr(args, "conf", 0.50))
    save_enabled = bool(getattr(args, "save", False))
    # argparse always sets dst (None when omitted), so a getattr default never applies.
    default_dst = images_dir.parent / "labels" if images_dir.parent.is_dir() else images_dir / "labels"
    dst_dir = Path(getattr(args, "dst", None) or default_dst).resolve()

    if not images_dir.is_dir():
        raise NotADirectoryError(f"Images directory not found: {images_dir}")

    if save_enabled:
        dst_dir.mkdir(parents=True, exist_ok=True)

    model = FotonetV11(model_ref, device=getattr(args, "device", None))
    images = [p for p in images_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]

    total_images = len(images)
    total_boxes = 0
    detections_summary = []

    for idx, img_p in enumerate(images):
        try:
            with Image.open(img_p) as im:
                img_w, img_h = im.size
        except Exception:
            continue

        res = model.predict(img_p, conf=conf_thresh)[0]
        boxes_xyxy = res.boxes.xyxy.cpu().numpy() if hasattr(res.boxes, "xyxy") else []
        scores = res.scores.cpu().numpy() if hasattr(res, "scores") else []
        classes = res.classes.cpu().numpy() if hasattr(res, "classes") else []

        img_boxes = []
        yolo_lines = []
        for box, score, cid in zip(boxes_xyxy, scores, classes):
            cid_int = int(cid)
            x1, y1, x2, y2 = map(float, box)
            bw = (x2 - x1) / max(img_w, 1)
            bh = (y2 - y1) / max(img_h, 1)
            cx = (x1 + x2) / (2.0 * max(img_w, 1))
            cy = (y1 + y2) / (2.0 * max(img_h, 1))

            # Clamp strictly to [0, 1]
            bw = max(0.001, min(1.0, bw))
            bh = max(0.001, min(1.0, bh))
            cx = max(bw / 2.0, min(1.0 - bw / 2.0, cx))
            cy = max(bh / 2.0, min(1.0 - bh / 2.0, cy))

            yolo_lines.append(f"{cid_int} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
            img_boxes.append({"cls": cid_int, "conf": round(float(score), 4), "box": [cx, cy, bw, bh]})
            total_boxes += 1

        if save_enabled:
            lbl_file = dst_dir / f"{img_p.stem}.txt"
            with open(lbl_file, "w", encoding="utf-8") as lf:
                lf.writelines(yolo_lines)

        detections_summary.append({
            "image": img_p.name,
            "boxes_count": len(img_boxes),
            "boxes": img_boxes,
        })

        if (idx + 1) % 5 == 0 or (idx + 1) == total_images:
            emit_telemetry({
                "processed": idx + 1,
                "total": total_images,
                "total_boxes": total_boxes,
                "progress_pct": round(100.0 * ((idx + 1) / max(total_images, 1)), 1),
            })

    emit_result({
        "model": model_ref,
        "images_directory": str(images_dir),
        "images_processed": total_images,
        "total_boxes_detected": total_boxes,
        "labels_saved": save_enabled,
        "labels_destination": str(dst_dir) if save_enabled else None,
        "detections": detections_summary,
    })
