"""
Fotonet Studio CLI Bridge (fotonet.cli.studio_bridge)
=====================================================
Line-delimited JSON IPC bridge connecting Fotonet Studio (desktop/fotonet-studio)
to the Fotonet v1.1 computer vision engine.

Protocol Contract:
- Telemetry events: {"type": "telemetry", "data": {...}}
- Completion events: {"type": "result", "status": "ok", "data": {...}}
- Error events:      {"type": "error", "message": "...", "traceback": "..."}

Strict Architectural Governance:
- Supports exclusively Fotonet v1.1 family (fotonetn, fotonets, fotonetm, fotonetl).
- Absolute ban on internal numeric architecture counters in user-facing surfaces.
- Zero dependencies on external YOLO packages (e.g. ultralytics).
- Bit-exact numerical parity verification (L1 error < 1e-5) on ONNX export.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import List, Optional

from .protocol import (  # noqa: F401
    emit_error, emit_json, emit_result, emit_telemetry, _REAL_STDOUT, V11_MODEL_IDS,
    validate_v11_model_id,
)
from .environment import cmd_env  # noqa: F401
from .datasets import (  # noqa: F401
    cmd_convert_dataset, cmd_extract_video, cmd_health_check, cmd_inspect_dataset, _find_dataset_splits,
    IMAGE_EXTENSIONS,
)
from .augment import (  # noqa: F401
    build_albumentations_filter, _clamp_yolo_boxes, cmd_augment_batch, cmd_augment_preview,
)
from .inference import cmd_autolabel, cmd_predict  # noqa: F401
from .training import cmd_evaluate, cmd_train  # noqa: F401
from .export import cmd_codegen, cmd_export  # noqa: F401


# ============================================================================
# 9. Main Dispatcher & Subcommand CLI Router
# ============================================================================

def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser for Fotonet Studio CLI Bridge."""
    parser = argparse.ArgumentParser(
        prog="python -m fotonet.cli.studio_bridge",
        description="Fotonet Studio IPC Bridge for Fotonet v1.1 engine",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. env
    subparsers.add_parser("env", help="Hardware, PyTorch, CUDA, and model query")

    # 2. inspect-dataset
    p_inspect = subparsers.add_parser("inspect-dataset", help="Dataset analysis & distributions")
    p_inspect.add_argument("--dataset", required=True, help="Path to dataset directory")

    # 3. health-check
    p_health = subparsers.add_parser("health-check", help="Audits 6 defect categories with auto-repair")
    p_health.add_argument("--dataset", required=True, help="Path to dataset directory")
    p_health.add_argument("--fix", action="store_true", help="Apply auto-repairs")

    # 4. convert-dataset
    p_conv = subparsers.add_parser("convert-dataset", help="Zero-drift converter (VOC, YOLO, COCO, TFRecord)")
    p_conv.add_argument("--src", required=True, help="Source dataset path")
    p_conv.add_argument("--from", dest="from_format", default="yolo", help="Source format (voc, yolo, coco, tfrecord)")
    p_conv.add_argument("--from-format", dest="from_format", help=argparse.SUPPRESS)
    p_conv.add_argument("--to", dest="to_format", default="yolo", help="Target format (voc, yolo, coco, tfrecord)")
    p_conv.add_argument("--to-format", dest="to_format", help=argparse.SUPPRESS)
    p_conv.add_argument("--dst", required=True, help="Destination directory")

    # 5. extract-video
    p_vid = subparsers.add_parser("extract-video", help="Video keyframe extraction via OpenCV")
    p_vid.add_argument("--video", required=True, help="Path to input video")
    p_vid.add_argument("--fps", type=float, default=2.0, help="Target FPS sampling rate")
    p_vid.add_argument("--dst", required=True, help="Destination directory for extracted frames")

    # 6. augment-preview
    p_aprev = subparsers.add_parser("augment-preview", help="ForgeAugment single-image preview with 29 filters")
    p_aprev.add_argument("--image", required=True, help="Path to input image")
    p_aprev.add_argument("--boxes", default="[]", help="Bounding boxes JSON string or file")
    p_aprev.add_argument("--pipeline", default="[]", help="Pipeline configuration JSON string or file")

    # 7. augment-batch
    p_abatch = subparsers.add_parser("augment-batch", help="Batch synthetic dataset generator")
    p_abatch.add_argument("--dataset", required=True, help="Path to source dataset")
    p_abatch.add_argument("--pipeline", default="[]", help="Pipeline configuration JSON string or file")
    p_abatch.add_argument("--multiplier", type=int, default=3, help="Variations per image")
    p_abatch.add_argument("--dst", help="Destination directory")

    # 8. autolabel
    p_auto = subparsers.add_parser("autolabel", help="Auto-labeling using Fotonet v1.1 direct LTRB head")
    p_auto.add_argument("--model", required=True, help="Fotonet v1.1 model ID or checkpoint path")
    p_auto.add_argument("--images", required=True, help="Directory containing images")
    p_auto.add_argument("--conf", type=float, default=0.50, help="Confidence threshold")
    p_auto.add_argument("--save", action="store_true", help="Save YOLO .txt labels")
    p_auto.add_argument("--dst", help="Destination labels directory")
    p_auto.add_argument("--device", help="Computation device (e.g. cuda, cpu)")

    # 9. train
    p_train = subparsers.add_parser("train", help="Launch supervised training supervisor with live telemetry")
    p_train.add_argument("--model", required=True, help="Model ID (fotonetn, fotonets, fotonetm, fotonetl)")
    p_train.add_argument("--data", required=True, help="Dataset configuration YAML path")
    p_train.add_argument("--epochs", type=int, default=120, help="Total training epochs")
    p_train.add_argument("--batch", type=int, default=16, help="Batch size")
    p_train.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    p_train.add_argument("--imgsz", type=int, default=640, help="Input resolution")
    p_train.add_argument("--device", default="auto", help="Compute device")
    p_train.add_argument("--optimizer", default="adamw", help="Optimizer (adamw, sgd)")
    p_train.add_argument("--out-dir", dest="out_dir", default="runs/train", help="Output directory")

    # 10. predict
    p_pred = subparsers.add_parser("predict", help="Live detection with latency profiling")
    p_pred.add_argument("--model", required=True, help="Fotonet v1.1 model ID or checkpoint path")
    p_pred.add_argument("--source", required=True, help="Image, directory, or video path")
    p_pred.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p_pred.add_argument("--iou", type=float, default=0.50,
                        help="Ignored: the model is NMS-free (kept so existing Studio calls still parse)")
    p_pred.add_argument("--device", help="Compute device")

    # 11. evaluate
    p_eval = subparsers.add_parser("evaluate", help="Full COCO benchmark evaluation (mAP, PR, Confusion)")
    p_eval.add_argument("--model", required=True, help="Fotonet v1.1 model ID or checkpoint path")
    p_eval.add_argument("--data", required=True, help="Dataset YAML path")
    p_eval.add_argument("--conf", type=float, default=0.25,
                        help="Operating confidence threshold for precision/recall (mAP keeps every detection)")
    p_eval.add_argument("--iou", type=float, default=0.50, help="Operating IoU threshold")
    p_eval.add_argument("--imgsz", type=int, default=640, help="Evaluation resolution")
    p_eval.add_argument("--device", help="Compute device")

    # 12. export
    p_exp = subparsers.add_parser("export", help="Multi-target export with L1 < 1e-5 numerical parity check")
    p_exp.add_argument("--model", required=True, help="Fotonet v1.1 model ID or checkpoint path")
    p_exp.add_argument("--format", default="onnx", help="Export target format (onnx, tensorrt, coreml, torchscript)")
    p_exp.add_argument("--precision", default="fp32", help="Precision (fp32, fp16, int8)")
    p_exp.add_argument("--verify", action="store_true", default=True, help="Verify numerical parity")
    p_exp.add_argument("--imgsz", type=int, default=640, help="Input resolution")
    p_exp.add_argument("--batch", type=int, default=1, help="Export batch size")
    p_exp.add_argument("--dynamic", action="store_true", help="Dynamic axes")
    p_exp.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    p_exp.add_argument("--out", help="Output artifact destination path")
    p_exp.add_argument("--device", help="Export compute device")

    # 13. codegen
    p_cg = subparsers.add_parser("codegen", help="Autonomous zero-framework production code generator")
    p_cg.add_argument("--target", required=True, help="Target language (python, cpp, node)")
    p_cg.add_argument("--model", required=True, help="Model path (.onnx or .engine)")
    p_cg.add_argument("--classes", help="Class names list JSON or file")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI Entrypoint executing subcommands and emitting line-delimited JSON."""
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        emit_error("Invalid CLI bridge arguments. Usage: python -m fotonet.cli.studio_bridge <command> [args]")
        return exc.code if isinstance(exc.code, int) else 1

    cmd_map = {
        "env": cmd_env,
        "inspect-dataset": cmd_inspect_dataset,
        "health-check": cmd_health_check,
        "convert-dataset": cmd_convert_dataset,
        "extract-video": cmd_extract_video,
        "augment-preview": cmd_augment_preview,
        "augment-batch": cmd_augment_batch,
        "autolabel": cmd_autolabel,
        "train": cmd_train,
        "predict": cmd_predict,
        "evaluate": cmd_evaluate,
        "export": cmd_export,
        "codegen": cmd_codegen,
    }

    handler = cmd_map.get(args.command)
    if not handler:
        emit_error(f"Unknown studio bridge command: {args.command}")
        return 1

    try:
        handler(args)
        return 0
    except Exception as exc:
        emit_error(str(exc), traceback_str=traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
