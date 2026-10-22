"""Augmentation preview and batch augmentation."""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .protocol import emit_result, emit_telemetry
from .datasets import _find_dataset_splits


def _albumentations_major(module) -> int:
    try:
        return int(str(module.__version__).split(".")[0])
    except (AttributeError, ValueError):
        return 1


def build_albumentations_filter(effect_dict: Dict[str, Any], img_shape: Tuple[int, int]) -> Optional[Any]:
    """Instantiate one of the 29 ForgeAugment filter effects."""
    import albumentations as A

    etype = str(effect_dict.get("type", "")).strip()
    p = float(effect_dict.get("probability", 1.0))
    params = effect_dict.get("params", {})
    h, w = img_shape
    # albumentations 2.x renamed or removed several arguments; passing the 1.x
    # names only warns, and the transform silently uses its defaults.
    albu2 = _albumentations_major(A) >= 2

    if etype == "BlurEffect":
        bl = int(params.get("blur_limit", 7))
        bl = bl if bl % 2 == 1 else bl + 1
        return A.Blur(blur_limit=bl, p=p)

    elif etype == "GaussianBlurEffect":
        bl = int(params.get("blur_limit", 7))
        bl = bl if bl % 2 == 1 else bl + 1
        smin = float(params.get("sigma_limit_min", 0.0))
        smax = float(params.get("sigma_limit_max", 0.0))
        sig = (smin, smax) if smax > 0.0 else 0
        return A.GaussianBlur(blur_limit=bl, sigma_limit=sig, p=p)

    elif etype == "MotionBlurEffect":
        bl = int(params.get("blur_limit", 7))
        bl = bl if bl % 2 == 1 else bl + 1
        return A.MotionBlur(blur_limit=bl, p=p)

    elif etype == "BrightnessContrastEffect":
        bl = float(params.get("brightness_limit", 0.2))
        cl = float(params.get("contrast_limit", 0.2))
        return A.RandomBrightnessContrast(brightness_limit=bl, contrast_limit=cl, p=p)

    elif etype == "BrightnessEffect":
        limit = float(params.get("limit", 0.2))
        return A.RandomBrightnessContrast(brightness_limit=limit, contrast_limit=0.0, p=p)

    elif etype == "ContrastEffect":
        limit = float(params.get("limit", 0.2))
        return A.RandomBrightnessContrast(brightness_limit=0.0, contrast_limit=limit, p=p)

    elif etype == "ExposureEffect":
        gmin = int(params.get("gamma_min", 80))
        gmax = int(params.get("gamma_max", 120))
        return A.RandomGamma(gamma_limit=(gmin, gmax), p=p)

    elif etype == "CLAHEEffect":
        cl = float(params.get("clip_limit", 4.0))
        tgs = int(params.get("tile_grid_size", 8))
        return A.CLAHE(clip_limit=cl, tile_grid_size=(tgs, tgs), p=p)

    elif etype == "ImageCompressionEffect":
        ql = int(params.get("quality_lower", 50))
        qu = int(params.get("quality_upper", 100))
        if albu2:
            return A.ImageCompression(quality_range=(ql, qu), p=p)
        return A.ImageCompression(quality_lower=ql, quality_upper=qu, p=p)

    elif etype == "GaussianNoiseEffect":
        vmin = float(params.get("var_limit_min", 10.0))
        vmax = float(params.get("var_limit_max", 50.0))
        if albu2:
            # 2.x takes the noise std as a fraction of 255; var_limit was a variance in pixel units.
            std = tuple(min(max(math.sqrt(max(v, 0.0)) / 255.0, 0.0), 1.0) for v in (vmin, vmax))
            return A.GaussNoise(std_range=std, p=p)
        return A.GaussNoise(var_limit=(vmin, vmax), p=p)

    elif etype == "ISONoiseEffect":
        csmin = float(params.get("color_shift_min", 0.01))
        csmax = float(params.get("color_shift_max", 0.05))
        imin = float(params.get("intensity_min", 0.1))
        imax = float(params.get("intensity_max", 0.5))
        return A.ISONoise(color_shift=(csmin, csmax), intensity=(imin, imax), p=p)

    elif etype == "RGBShiftEffect":
        rs = int(params.get("r_shift", 20))
        gs = int(params.get("g_shift", 20))
        bs = int(params.get("b_shift", 20))
        return A.RGBShift(r_shift_limit=rs, g_shift_limit=gs, b_shift_limit=bs, p=p)

    elif etype == "HueSaturationEffect":
        hs = int(params.get("hue_shift", 20))
        ss = int(params.get("sat_shift", 30))
        vs = int(params.get("val_shift", 20))
        return A.HueSaturationValue(hue_shift_limit=hs, sat_shift_limit=ss, val_shift_limit=vs, p=p)

    elif etype == "SharpenEffect":
        amin = float(params.get("alpha_min", 0.2))
        amax = float(params.get("alpha_max", 0.5))
        lmin = float(params.get("lightness_min", 0.5))
        lmax = float(params.get("lightness_max", 1.0))
        return A.Sharpen(alpha=(amin, amax), lightness=(lmin, lmax), p=p)

    elif etype == "UnsharpMaskEffect":
        bl = int(params.get("blur_limit", 7))
        bl = bl if bl % 2 == 1 else bl + 1
        alpha = float(params.get("alpha", 0.2))
        thresh = int(params.get("threshold", 10))
        return A.UnsharpMask(blur_limit=bl, alpha=alpha, threshold=thresh, p=p)

    elif etype == "RandomRainEffect":
        s_low = int(params.get("slant_lower", -10))
        s_up = int(params.get("slant_upper", 10))
        dl = int(params.get("drop_length", 20))
        dw = int(params.get("drop_width", 1))
        bv = int(params.get("blur_value", 3))
        bv = bv if bv % 2 == 1 else bv + 1
        bc = float(params.get("brightness_coefficient", 0.7))
        slant = {"slant_range": (s_low, s_up)} if albu2 else {"slant_lower": s_low, "slant_upper": s_up}
        return A.RandomRain(
            **slant, drop_length=dl,
            drop_width=dw, blur_value=bv, brightness_coefficient=bc, p=p,
        )

    elif etype == "RandomFogEffect":
        flow = float(params.get("fog_coef_lower", 0.3))
        fup = float(params.get("fog_coef_upper", 1.0))
        ac = float(params.get("alpha_coef", 0.08))
        if albu2:
            return A.RandomFog(fog_coef_range=(flow, fup), alpha_coef=ac, p=p)
        return A.RandomFog(fog_coef_lower=flow, fog_coef_upper=fup, alpha_coef=ac, p=p)

    elif etype == "RandomSunFlareEffect":
        rad = int(params.get("src_radius", min(400, max(50, w // 2))))
        return A.RandomSunFlare(src_radius=rad, p=p)

    elif etype == "HorizontalFlipEffect":
        return A.HorizontalFlip(p=p)

    elif etype == "VerticalFlipEffect":
        return A.VerticalFlip(p=p)

    elif etype == "RotateEffect":
        lim = int(params.get("limit", 15))
        return A.Rotate(limit=lim, p=p)

    elif etype == "SafeRotateEffect":
        return A.RandomRotate90(p=p)

    elif etype == "RandomCropEffect":
        smin = float(params.get("scale_min", 0.7))
        smax = float(params.get("scale_max", 0.9))
        return A.RandomResizedCrop(size=(h, w), scale=(smin, smax), ratio=(0.9, 1.1), p=p)

    elif etype == "CenterCropEffect":
        scale = float(params.get("scale", 0.8))
        crop_frac = -(1.0 - scale) / 2.0
        return A.CropAndPad(percent=crop_frac, keep_size=True, p=p)

    elif etype == "RandomResizedCropEffect":
        smin = float(params.get("scale_min", 0.5))
        smax = float(params.get("scale_max", 1.0))
        rmin = float(params.get("ratio_min", 0.75))
        rmax = float(params.get("ratio_max", 1.33))
        return A.RandomResizedCrop(size=(h, w), scale=(smin, smax), ratio=(rmin, rmax), p=p)

    elif etype == "PerspectiveEffect":
        sc = float(params.get("scale", 0.05))
        return A.Perspective(scale=sc, p=p)

    elif etype == "ElasticTransformEffect":
        alpha = float(params.get("alpha", 1.0))
        sigma = float(params.get("sigma", 50.0))
        if albu2:
            # The affine part (alpha_affine) was removed from ElasticTransform in 2.x.
            return A.ElasticTransform(alpha=alpha, sigma=sigma, p=p)
        alpha_affine = float(params.get("alpha_affine", 50.0))
        return A.ElasticTransform(alpha=alpha, sigma=sigma, alpha_affine=alpha_affine, p=p)

    elif etype == "GridDistortionEffect":
        ns = int(params.get("num_steps", 5))
        dl = float(params.get("distort_limit", 0.3))
        return A.GridDistortion(num_steps=ns, distort_limit=dl, p=p)

    elif etype == "OpticalDistortionEffect":
        dl = float(params.get("distort_limit", 0.5))
        if albu2:
            # shift_limit no longer exists in 2.x.
            return A.OpticalDistortion(distort_limit=dl, p=p)
        sl = float(params.get("shift_limit", 0.5))
        return A.OpticalDistortion(distort_limit=dl, shift_limit=sl, p=p)

    return None


def _clamp_yolo_boxes(boxes: List[List[float]]) -> List[List[float]]:
    """Strictly clamp bounding boxes within [0, 1] range to prevent Albumentations assertion errors."""
    sanitized = []
    for box in boxes:
        if len(box) < 4:
            continue
        cx, cy, w, h = box[:4]
        nw = max(0.001, min(1.0, float(w)))
        nh = max(0.001, min(1.0, float(h)))
        ncx = max(nw / 2.0, min(1.0 - nw / 2.0, float(cx)))
        ncy = max(nh / 2.0, min(1.0 - nh / 2.0, float(cy)))
        sanitized.append([ncx, ncy, nw, nh])
    return sanitized


def cmd_augment_preview(args: argparse.Namespace) -> None:
    """Run an Albumentations augmentation pipeline on a single image and return base64 and boxes."""
    import cv2
    import albumentations as A

    image_path = Path(args.image).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Parse input boxes: JSON string or file
    boxes_input = args.boxes
    boxes_raw = []
    if os.path.isfile(boxes_input):
        with open(boxes_input, "r", encoding="utf-8") as f:
            boxes_raw = json.load(f)
    elif isinstance(boxes_input, str):
        boxes_raw = json.loads(boxes_input)

    # Format of boxes: can be list of [cx, cy, w, h] or list of [cls, cx, cy, w, h]
    clean_boxes = []
    labels = []
    for item in boxes_raw:
        if len(item) == 5:
            labels.append(int(item[0]))
            clean_boxes.append(item[1:5])
        elif len(item) == 4:
            labels.append(0)
            clean_boxes.append(item)

    clean_boxes = _clamp_yolo_boxes(clean_boxes)

    # Parse pipeline: JSON string or file
    pipeline_raw = []
    pipeline_input = args.pipeline
    if os.path.isfile(pipeline_input):
        with open(pipeline_input, "r", encoding="utf-8") as f:
            pipeline_raw = json.load(f)
    elif isinstance(pipeline_input, str):
        pipeline_raw = json.loads(pipeline_input)

    effects_list = pipeline_raw.get("effects", pipeline_raw) if isinstance(pipeline_raw, dict) else pipeline_raw

    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    # Build Albumentations transform list
    transforms = []
    for eff in effects_list:
        if eff.get("enabled", True):
            t = build_albumentations_filter(eff, (h, w))
            if t is not None:
                transforms.append(t)

    if transforms:
        compose = A.Compose(
            transforms,
            bbox_params=A.BboxParams(format="yolo", label_fields=["labels"], min_visibility=0.1),
        )
        try:
            res = compose(image=img_rgb, bboxes=clean_boxes, labels=labels[: len(clean_boxes)])
            trans_img = res["image"]
            trans_boxes = _clamp_yolo_boxes(res["bboxes"])
            trans_labels = res["labels"]
        except Exception:
            trans_img = img_rgb
            trans_boxes = clean_boxes
            trans_labels = labels
    else:
        trans_img = img_rgb
        trans_boxes = clean_boxes
        trans_labels = labels

    # Encode to base64 JPEG
    trans_bgr = cv2.cvtColor(trans_img, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", trans_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError("Failed to encode augmented image to JPEG buffer.")

    b64_str = "data:image/jpeg;base64," + base64.b64encode(buf).decode("ascii")

    emit_result({
        "image_base64": b64_str,
        "width": trans_img.shape[1],
        "height": trans_img.shape[0],
        "boxes": trans_boxes,
        "labels": trans_labels,
    })


def cmd_augment_batch(args: argparse.Namespace) -> None:
    """Batch dataset synthesis producing augmented images and YOLO .txt labels."""
    import cv2
    import albumentations as A

    dataset_path = Path(args.dataset).resolve()
    multiplier = int(getattr(args, "multiplier", 3))
    # argparse always sets dst (None when omitted), so a getattr default never applies.
    out_dir = Path(getattr(args, "dst", None) or dataset_path / "augmented").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_images = out_dir / "images"
    out_labels = out_dir / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    # Parse pipeline
    pipeline_input = args.pipeline
    pipeline_raw = []
    if os.path.isfile(pipeline_input):
        with open(pipeline_input, "r", encoding="utf-8") as f:
            pipeline_raw = json.load(f)
    elif isinstance(pipeline_input, str):
        pipeline_raw = json.loads(pipeline_input)

    effects_list = pipeline_raw.get("effects", pipeline_raw) if isinstance(pipeline_raw, dict) else pipeline_raw

    splits = _find_dataset_splits(dataset_path)
    all_images = []
    for sdata in splits.values():
        all_images.extend(sdata["images"])

    total_images = len(all_images)
    generated_count = 0

    for idx, img_p in enumerate(all_images):
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        # Find matching label
        lbl_p = None
        for candidate in (
            img_p.parent.parent / "labels" / img_p.parent.name / f"{img_p.stem}.txt",
            img_p.parent / f"{img_p.stem}.txt",
            dataset_path / "labels" / f"{img_p.stem}.txt",
        ):
            if candidate.is_file():
                lbl_p = candidate
                break

        bboxes = []
        labels = []
        if lbl_p and lbl_p.is_file():
            with open(lbl_p, "r", encoding="utf-8") as lf:
                for line in lf:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        labels.append(int(float(parts[0])))
                        bboxes.append(list(map(float, parts[1:5])))

        bboxes = _clamp_yolo_boxes(bboxes)

        # Build pipeline
        transforms = []
        for eff in effects_list:
            if eff.get("enabled", True):
                t = build_albumentations_filter(eff, (h, w))
                if t is not None:
                    transforms.append(t)

        compose = A.Compose(
            transforms,
            bbox_params=A.BboxParams(format="yolo", label_fields=["labels"], min_visibility=0.1),
        ) if transforms else None

        for m_idx in range(multiplier):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            rand_suffix = random.randint(1000, 9999)
            aug_name = f"aug_{m_idx}_{ts}_{rand_suffix}_{img_p.stem}.jpg"
            aug_lbl_name = f"aug_{m_idx}_{ts}_{rand_suffix}_{img_p.stem}.txt"

            if compose is not None and bboxes:
                try:
                    res = compose(image=img_rgb, bboxes=bboxes, labels=labels[: len(bboxes)])
                    t_img = res["image"]
                    t_boxes = _clamp_yolo_boxes(res["bboxes"])
                    t_labels = res["labels"]
                except Exception:
                    t_img = img_rgb
                    t_boxes = bboxes
                    t_labels = labels
            else:
                t_img = img_rgb
                t_boxes = bboxes
                t_labels = labels

            # Save image
            cv2.imwrite(str(out_images / aug_name), cv2.cvtColor(t_img, cv2.COLOR_RGB2BGR))
            # Save label
            with open(out_labels / aug_lbl_name, "w", encoding="utf-8") as lf:
                for box, cid in zip(t_boxes, t_labels):
                    lf.write(f"{int(cid)} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n")

            generated_count += 1

        if (idx + 1) % 5 == 0 or (idx + 1) == total_images:
            emit_telemetry({
                "processed_images": idx + 1,
                "total_images": total_images,
                "generated_count": generated_count,
                "progress_pct": round(100.0 * ((idx + 1) / max(total_images, 1)), 1),
            })

    emit_result({
        "dataset": str(dataset_path),
        "destination": str(out_dir),
        "input_images": total_images,
        "multiplier": multiplier,
        "generated_images": generated_count,
        "generated_labels": generated_count,
    })
