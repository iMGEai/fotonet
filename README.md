# fotonet


![Turkey](https://img.shields.io/badge/Made%20in-Turkey-E30A17) ![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)

`fotonet` is an open-source, NMS-free object-detection library for Python covering local inference, video tracking, training, validation, and model export.

## Models

| Model | Generation | Parameters | Compute (`640 × 640`) | COCO mAP50-95 | Status |
| --- | --- | --- | --- | --- | --- |
| `fotonet2n` | v2 (Hi Bird) | 2.69M | 6.41 GFLOPs | 37.5 | Available |
| `fotonetn` | v1.1 (UltraDense) | 2.85M | 6.95 GFLOPs | 32.1 | Available |
| `fotonets` | v1.1 (UltraDense) | 11.49M | 27.43 GFLOPs | — | Architecture in package |
| `fotonetm` | v1.1 (UltraDense) | 22.23M | 63.38 GFLOPs | — | Architecture in package |
| `fotonetl` | v1.1 (UltraDense) | 70.32M | 161.51 GFLOPs | — | Architecture in package |
| `fotonete` | v1 | 1.70M | 5.20 GFLOPs | 17.1 | Available |

`fotonet2n` (Hi Bird, fotonet 2.0) reaches **37.5 mAP50-95** (55.6 mAP50, 40.0 mAP75) on COCO val2017 from a 300-epoch from-scratch single-GPU run in a 5.98 MB Apache-2.0 checkpoint with zero deployed `BatchNorm2d` layers and fixed `[B, 300, nc+4]` NMS-free outputs. `fotonetn` stays available at 32.1 mAP50-95 (`fotonetn-ex.pt` keeps the first 29.9 weights). See [published weights](docs/model-zoo.md#published-weights) and [trainer 2.0](docs/training.md#trainer-20).

## Install

```bash
pip install fotonet
```

For a development checkout:

```bash
git clone https://github.com/iMGEai/fotonet.git
cd fotonet
pip install -e ".[dev]"
```

## Quick start

Load the released checkpoint and run inference from Python:

```python
from fotonet import Fotonet

model = Fotonet("fotonet2n")
results = model.predict("image.jpg", conf=0.25, imgsz=640)

for detection in results[0].boxes:
    print(detection.cls, detection.conf, detection.xyxy)  # pixel corners (x1, y1, x2, y2)
```

For a BGR frame from OpenCV:

```python
frame = cv2.imread("image.jpg")
results = model.predict_bgr(frame, conf=0.25, imgsz=640)
```

## Train

Training uses a YAML dataset configuration and writes checkpoints to a local run directory:

```bash
fotonet train \
  model=fotonet2n \
  data=path/to/data.yaml \
  epochs=100 \
  batch=16 \
  imgsz=640 \
  run_dir=runs/fotonet2n
```

Resume an interrupted run:

```bash
fotonet train \
  model=fotonet2n \
  data=path/to/data.yaml \
  resume=runs/fotonet2n/fotonet2n_last.pt
```

Weights and training outputs are not stored in the Git repository.

## Export

```python
from fotonet import Fotonet

model = Fotonet("fotonet2n")
output = model.export(
    format="onnx",
    path="exports/fotonet2n.onnx",
    imgsz=640,
)
print(output["artifact"])
```

See the [export guide](docs/export.md) for ONNX, TensorRT, and TorchScript options.

## Documentation

- [Documentation portal](docs/index.md)
- [Installation](docs/installation.md)
- [Quick start](docs/quickstart.md)
- [Inference and results](docs/inference.md)
- [Training and resume](docs/training.md)
- [Google Colab & cloud training](docs/colab_training.md)
- [Models and runtime](docs/model-zoo.md)
- [Model configuration](docs/model-config.md)
- [Export](docs/export.md)
- [Transform API](docs/transform-api.md)
- [Security](docs/security.md)
- [Contributing](docs/contributing.md)

## License

Apache License 2.0. See [LICENSE](LICENSE).
