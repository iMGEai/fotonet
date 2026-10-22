"""``env``: report the Python, torch, CUDA, and fotonet environment."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

from .protocol import emit_result, V11_MODEL_IDS


def cmd_env(args: argparse.Namespace) -> None:
    """Query system hardware, GPU, PyTorch, CUDA, and supported Fotonet v1.1 models."""
    import torch

    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else "CPU (Host Runtime)"
    vram_total_mb = 0
    vram_free_mb = 0
    if gpu_available:
        try:
            free_b, total_b = torch.cuda.mem_get_info(0)
            vram_total_mb = int(total_b // (1024 * 1024))
            vram_free_mb = int(free_b // (1024 * 1024))
        except Exception:
            vram_total_mb = 0
            vram_free_mb = 0

    gpu_temp = None
    if gpu_available:
        try:
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=1.0,
            )
            if res.returncode == 0 and res.stdout.strip().isdigit():
                gpu_temp = int(res.stdout.strip())
        except Exception:
            pass

    cuda_version = torch.version.cuda if gpu_available else None
    torch_version = torch.__version__

    v11_profiles = {
        "fotonetn": {"params": "2.06M", "default_imgsz": 640, "flops": "4.8G", "target": "Micro/Edge Embedded"},
        "fotonets": {"params": "4.10M", "default_imgsz": 640, "flops": "9.6G", "target": "Real-time Edge Robotics"},
        "fotonetm": {"params": "9.80M", "default_imgsz": 640, "flops": "22.4G", "target": "Balanced Production"},
        "fotonetl": {"params": "18.4M", "default_imgsz": 640, "flops": "41.2G", "target": "High-accuracy Server"},
    }

    emit_result({
        "status": "online",
        "platform": sys.platform,
        "python_version": sys.version.split()[0],
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "vram_total_mb": vram_total_mb,
        "vram_free_mb": vram_free_mb,
        "gpu_temperature_c": gpu_temp,
        "cuda_version": cuda_version,
        "torch_version": torch_version,
        "models": list(V11_MODEL_IDS),
        "profiles": v11_profiles,
    })
