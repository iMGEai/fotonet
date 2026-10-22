"""Blank-slate measurements for the v1.1 family.

Run from the repo root:

    python -m fotonet.versions.v1_1.bench

Reports, on the current GPU, for fotonetn / fotonets / fotonetm / fotonetl:

- train parameters (O2M head included) and deploy parameters (O2M stripped)
- conv GMACs and GFLOPs (2 x MACs) at 640, batch 1, deploy graph
- model-forward latency, eval, batch 1 and batch 8
- end-to-end latency: uint8 BGR frames through letterbox, forward, and top-k
- compute-bound training it/s (matcher + loss + backward) at batch 1 and 8
- dataloader training it/s on randomly generated JPEG images

GFLOPs count Conv2d multiply-adds only. BatchNorm, SiLU, residual adds,
and the box decode are not included. SiLU and the decode are cheap next
to the dense 3x3 stack; the number is still a conv-only figure.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

MODELS = ("fotonetn", "fotonets", "fotonetm", "fotonetl")
IMGSZ = 640
NC = 80


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def count_conv_macs(model, images: torch.Tensor) -> int:
    """Conv2d multiply-adds for one forward. Linear layers are included."""
    total = [0]

    def conv_hook(module, inputs, output):
        x = inputs[0]
        n, _, h, w = output.shape
        cin = module.in_channels // module.groups
        total[0] += int(n * module.out_channels * cin * module.kernel_size[0] * module.kernel_size[1] * h * w)

    def linear_hook(module, inputs, output):
        total[0] += int(output.numel() * module.in_features)

    hooks = []
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))
    with torch.inference_mode():
        model(images)
    for hook in hooks:
        hook.remove()
    return total[0]


def _median_ms(times):
    ordered = sorted(times)
    return ordered[len(ordered) // 2]


def bench_forward(model, batch: int, warmup: int = 20, iters: int = 50) -> float:
    images = torch.randn(batch, 3, IMGSZ, IMGSZ, device="cuda")
    for _ in range(warmup):
        model(images)
    _sync()
    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        model(images)
        end.record()
        _sync()
        times.append(start.elapsed_time(end))
    return _median_ms(times)


def bench_e2e(model_id: str, batch: int, warmup: int = 8, iters: int = 20) -> float:
    """uint8 BGR frames, 480x640, through the public predict path."""
    from fotonet import Fotonet

    api = Fotonet(model_id, device="cuda")
    frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(batch)]
    for _ in range(warmup):
        api.predict_bgr(frames if batch > 1 else frames[0], imgsz=IMGSZ, conf=0.25)
    _sync()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        api.predict_bgr(frames if batch > 1 else frames[0], imgsz=IMGSZ, conf=0.25)
        _sync()
        times.append((time.perf_counter() - t0) * 1000.0)
    del api
    torch.cuda.empty_cache()
    return _median_ms(times)


def _one_train_step(model, opt, images, gt, labels, valid):
    from fotonet.train.losses import loss_fn
    from fotonet.train.assign import matcher_fn

    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(images, use_o2m=True)
    match = matcher_fn(
        out["pred_logits_o2o"].float(), out["pred_boxes_o2o"].float(),
        out["anchor_points"].float(), out["stride_tensor"].float(),
        gt, labels, valid, model.head.strides,
        float(images.shape[-2]), float(images.shape[-1]),
        out["pred_logits_o2m"].float(), out["pred_boxes_o2m"].float(),
        quality_mix=images.new_zeros(()), prior_mix=images.new_ones(()),
    )
    tgt_m, tgt_o, wgt_m, wgt_o, sb_m, sa_m, sg_m, rw_m, sb_o, sa_o, sg_o, rw_o = match
    total = loss_fn(
        out["pred_logits_o2o"].float(), out["pred_boxes_o2o"].float(),
        tgt_o, wgt_o, sb_o, sa_o, sg_o, rw_o, gt,
        out["pred_logits_o2m"].float(), out["pred_boxes_o2m"].float(),
        tgt_m, wgt_m, sb_m, sa_m, sg_m, rw_m,
    )[0]
    if not torch.isfinite(total):
        raise RuntimeError("non-finite training loss")
    total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    opt.step()
    return float(total.detach())


def bench_train_compute(model_id: str, batch: int, warmup: int = 5, steps: int = 20):
    from fotonet.versions.v1_1.graph import V11Detector

    try:
        model = V11Detector(model_id, nc=NC).cuda().train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
        images = torch.rand(batch, 3, IMGSZ, IMGSZ, device="cuda")
        gt = torch.zeros(batch, 4, 4, device="cuda")
        gt[:, 0] = torch.tensor([0.5, 0.5, 0.2, 0.2], device="cuda")
        gt[:, 1] = torch.tensor([0.3, 0.4, 0.1, 0.15], device="cuda")
        labels = torch.zeros(batch, 4, dtype=torch.long, device="cuda")
        labels[:, 1] = 1
        valid = torch.zeros(batch, 4, dtype=torch.bool, device="cuda")
        valid[:, :2] = True
        for _ in range(warmup):
            _one_train_step(model, opt, images, gt, labels, valid)
        _sync()
        t0 = time.perf_counter()
        last = 0.0
        for _ in range(steps):
            last = _one_train_step(model, opt, images, gt, labels, valid)
        _sync()
        elapsed = time.perf_counter() - t0
        return {"it_s": round(steps / elapsed, 3), "ms": round(1000.0 * elapsed / steps, 2), "loss": round(last, 3)}
    except torch.cuda.OutOfMemoryError:
        return {"it_s": None, "ms": None, "loss": None, "error": "out of memory"}
    finally:
        torch.cuda.empty_cache()


def bench_train_loader(model_id: str, batch: int, epochs: int = 2, n_images: int = 32):
    """Real trainer on random JPEGs. Epoch 2 is the number; epoch 1 includes cuDNN autotune."""
    from PIL import Image

    from fotonet.versions.v1_1.engine import train

    root = Path(tempfile.mkdtemp(prefix="f11bench_"))
    (root / "images").mkdir()
    (root / "labels").mkdir()
    rng = np.random.default_rng(0)
    for i in range(n_images):
        img = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
        Image.fromarray(img).save(root / "images" / f"{i:03d}.jpg", quality=90)
        (root / "labels" / f"{i:03d}.txt").write_text(
            "0 0.50 0.50 0.20 0.20\n1 0.30 0.40 0.10 0.15\n", encoding="utf-8"
        )
    (root / "data.yaml").write_text(
        f"path: {root.as_posix()}\ntrain: images\nval:\nnc: 2\nnames: ['a', 'b']\n",
        encoding="utf-8",
    )
    try:
        out = train(
            model_id=model_id,
            data=str(root / "data.yaml"),
            imgsz=IMGSZ,
            epochs=epochs,
            batch=batch,
            workers=0,
            max_images=n_images,
            val_period=epochs + 1,
            out_dir=str(root / "run"),
            gpu_augment=True,
            compile=False,
            close_mosaic=0,
            nbs=batch,
            warmup_epochs=0,
        )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {"error": "out of memory"}
    metrics = Path(out["metrics_path"]).read_text(encoding="utf-8").strip().splitlines()
    rows = [json.loads(line) for line in metrics if line.strip()]
    torch.cuda.empty_cache()
    return {
        "epochs": [{"epoch": r["epoch"], "it_s": r["it_s"], "loss": r["loss"], "seconds": r["epoch_seconds"]} for r in rows],
        "last_checkpoint": out["last_checkpoint"],
        "root": str(root),
    }


def _model_row(model_id: str) -> dict:
    from fotonet.versions.v1_1.graph import V11Detector

    train_model = V11Detector(model_id, nc=NC).cuda().train()
    train_params = count_params(train_model)
    deploy = V11Detector(model_id, nc=NC).cuda().eval()
    deploy.strip_o2m_for_inference()
    deploy_params = count_params(deploy)
    dummy = torch.randn(1, 3, IMGSZ, IMGSZ, device="cuda")
    macs = count_conv_macs(deploy, dummy)
    row = {
        "model": model_id,
        "train_params": train_params,
        "deploy_params": deploy_params,
        "gmacs_640": round(macs / 1e9, 3),
        "gflops_640": round(2.0 * macs / 1e9, 3),
        "forward_b1_ms": round(bench_forward(deploy, 1), 3),
    }
    try:
        row["forward_b8_ms"] = round(bench_forward(deploy, 8), 3)
    except torch.cuda.OutOfMemoryError:
        row["forward_b8_ms"] = None
        torch.cuda.empty_cache()
    del train_model, deploy
    torch.cuda.empty_cache()
    row["e2e_b1_ms"] = round(bench_e2e(model_id, 1), 3)
    try:
        row["e2e_b8_ms"] = round(bench_e2e(model_id, 8), 3)
    except torch.cuda.OutOfMemoryError:
        row["e2e_b8_ms"] = None
        torch.cuda.empty_cache()
    print(json.dumps(row), flush=True)
    return row


def main():
    if not torch.cuda.is_available():
        print("CUDA is required for these measurements.", file=sys.stderr)
        return 1
    torch.backends.cudnn.benchmark = True
    device_name = torch.cuda.get_device_name(0)
    print(f"device {device_name} torch {torch.__version__}", flush=True)
    rows = [_model_row(model_id) for model_id in MODELS]
    train_rows = []
    for model_id in MODELS:
        for batch in (1, 8):
            measured = bench_train_compute(model_id, batch)
            measured.update({"model": model_id, "batch": batch, "kind": "compute"})
            print(json.dumps(measured), flush=True)
            train_rows.append(measured)
            torch.cuda.empty_cache()
    loader_rows = []
    for model_id, batch in (("fotonetn", 8), ("fotonets", 8), ("fotonetm", 4), ("fotonetl", 2)):
        measured = bench_train_loader(model_id, batch)
        measured.update({"model": model_id, "batch": batch, "kind": "loader"})
        print(json.dumps({k: v for k, v in measured.items() if k != "root"}), flush=True)
        loader_rows.append(measured)
        torch.cuda.empty_cache()
    # Resume the nano run for one more epoch to prove the checkpoint is loadable.
    nano = next(row for row in loader_rows if row["model"] == "fotonetn" and "last_checkpoint" in row)
    from fotonet.versions.v1_1.engine import train
    resumed = train(
        model_id="fotonetn",
        data=str(Path(nano["root"]) / "data.yaml"),
        resume=nano["last_checkpoint"],
        imgsz=IMGSZ,
        epochs=3,
        batch=8,
        workers=0,
        max_images=32,
        val_period=99,
        out_dir=str(Path(nano["root"]) / "run"),
        gpu_augment=True,
        compile=False,
        close_mosaic=0,
        nbs=8,
        warmup_epochs=0,
    )
    resume_lines = Path(resumed["metrics_path"]).read_text(encoding="utf-8").strip().splitlines()
    resume_row = json.loads(resume_lines[-1])
    print(json.dumps({"kind": "resume", "epoch": resume_row["epoch"], "it_s": resume_row["it_s"], "loss": resume_row["loss"]}), flush=True)

    out_dir = Path("runs")
    out_dir.mkdir(exist_ok=True)
    payload = {
        "device": device_name,
        "torch": torch.__version__,
        "imgsz": IMGSZ,
        "models": rows,
        "train_compute": train_rows,
        "train_loader": [{k: v for k, v in row.items() if k != "root"} for row in loader_rows],
        "resume": {"epoch": resume_row["epoch"], "it_s": resume_row["it_s"], "loss": resume_row["loss"]},
    }
    dest = out_dir / "v11_bench.json"
    dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {dest}", flush=True)
    import shutil

    for row in loader_rows:                      # generated JPEG datasets and their runs
        if row.get("root"):
            shutil.rmtree(row["root"], ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
