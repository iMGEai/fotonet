# fotonet

`fotonet` is an open-source, NMS-free object-detection library for Python covering local inference, video tracking, training, validation, and model export.

## Models

| Model | Generation | Parameters | Compute (`640 × 640`) | Status |
| --- | --- | --- | --- | --- |
| `fotonetn` | v1.1 (UltraDense) | 2.85M | 6.95 GFLOPs | Available |
| `fotonets` | v1.1 (UltraDense) | 11.49M | 27.43 GFLOPs | Architecture in package, weights coming soon |
| `fotonetm` | v1.1 (UltraDense) | 22.23M | 63.38 GFLOPs | Architecture in package, weights coming soon |
| `fotonetl` | v1.1 (UltraDense) | 70.32M | 161.51 GFLOPs | Architecture in package, weights coming soon |
| `fotonete` | v1 | 1.70M | 5.20 GFLOPs | Available |

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

model = Fotonet("fotonetn")
results = model.predict("image.jpg", conf=0.25, imgsz=640)

for detection in results[0].boxes:
    print(detection.cls, detection.conf, detection.xyxy)
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
  model=fotonetn \
  data=path/to/data.yaml \
  epochs=100 \
  batch=16 \
  imgsz=640 \
  run_dir=runs/fotonetn
```

Resume an interrupted run:

```bash
fotonet train \
  model=fotonetn \
  data=path/to/data.yaml \
  resume=runs/fotonetn/fotonet_last.pt
```

Weights and training outputs are not stored in the Git repository.

## Export

```python
from fotonet import Fotonet

model = Fotonet("fotonetn")
output = model.export(
    format="onnx",
    path="exports/fotonetn.onnx",
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