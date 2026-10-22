# Google Colab & Cloud Training Guide

Train **fotonet** models (`fotonetn`, `fotonets`, `fotonetm`, `fotonetl`) on Google Colab (free-tier Tesla T4, V100, A100), cloud Linux VMs, macOS (Apple Silicon MPS), and Windows workstations.

---

## 3-Cell Quickstart in Google Colab

### Cell 1: Environment & Hardware Setup
```bash
# Verify GPU availability (Tesla T4, A100, L4)
!nvidia-smi

# Install fotonet in developer mode
!git clone https://github.com/iMGEai/fotonet.git
%cd fotonet
!pip install -e .
```

### Cell 2: Dataset Preparation
You can use any standard YOLO/COCO dataset YAML or directory structure:
```bash
# Example: Download COCO8 or COCO128 mini dataset for testing
import urllib.request
import zipfile
import os

url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8.zip"
urllib.request.urlretrieve(url, "coco8.zip")
with zipfile.ZipFile("coco8.zip", "r") as zip_ref:
    zip_ref.extractall("datasets")
```

Create or verify `coco8.yaml`:
```yaml
path: datasets/coco8
train: images/train
val: images/val
nc: 80
```

### Cell 3: Start Training

#### Option A: Unified Python API (Recommended)
```python
from fotonet import Fotonet

# 1. Initialize modern architecture (fotonetn, fotonets, fotonetm, fotonetl)
model = Fotonet("fotonetn")

# 2. Train with automatic platform and hardware adaptation
results = model.train(
    data="coco8.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device="auto",        # Automatically selects CUDA (fp16 on T4, bf16 on Ampere+)
    output_style="auto",  # Automatically formats as a clean Markdown table in Colab
)

# 3. Predict on an image using the trained weights
results = model.predict("datasets/coco8/images/val/000000000036.jpg", conf=0.25)
results[0].save("output.jpg")
```

#### Option B: Unified CLI
```bash
fotonet train model=fotonetn data=coco8.yaml epochs=100 imgsz=640 batch=16
```

---

## Automatic Colab Hardening & Resilience

fotonet includes built-in engine safeguards that automatically activate when running inside Google Colab:

1. **Tesla T4 Precision Auto-Detection**:
   - Google Colab free-tier GPUs (Tesla T4, Turing architecture) lack native `bfloat16` Tensor Core execution units.
   - The engine automatically detects T4 and transparently switches to `torch.float16` AMP paired with dynamic `GradScaler`, preventing CUDA crashes and maximizing hardware throughput.
   - On Ampere+ GPUs (A100, H100, RTX 30/40), native `torch.bfloat16` AMP is activated automatically.

2. **Multiprocessing Starvation Protection**:
   - Google Colab provides 2 vCPUs and a constrained shared memory (`/dev/shm`) limit.
   - Worker processes are automatically clamped to `min(workers, os.cpu_count())` with pin-memory tuning to prevent thread deadlocks and OOM errors.

3. **Colab-Friendly Telemetry**:
   - In standard Jupyter / Colab notebooks, frequent carriage returns (`\r`) can scramble cell outputs.
   - In non-TTY / Colab sessions, fotonet defaults to a clean Markdown table:
   ```text
   | Epoch | Stage  | Imgsz | Total Loss | Box Loss | Cls Loss | Aux Loss | mAP@50 | mAP@50:95 | Time    |
   |-------|--------|-------|------------|----------|----------|----------|--------|-----------|---------|
   | 1     | warmup | 640   | 5.8420     | 0.4921   | 1.8420   | 0.9412   | 0.1240 | 0.0812    | 01m 12s |
   | 2     | warmup | 640   | 4.9310     | 0.3842   | 1.4520   | 0.7812   | 0.2310 | 0.1540 *  | 01m 10s |
   ```
   - Each row is emitted cleanly and can be copied directly into GitHub issues, pull requests, or research notes.

4. **Resilient Compilation**:
   - `compile=False` by default for maximum compatibility across cloud environments without requiring Triton or C++ build tools.
   - When `compile=True` is requested, failures automatically fall back to eager execution with a helpful log instead of aborting the run.
