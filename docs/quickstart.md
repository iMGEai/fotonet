# Quick Start

`Fotonet("fotonetn")` loads `fotonetn.pt` from the `v1.1` GitHub release, and
`Fotonet("fotonete")` loads `fotonete.pt` from the `v1` release. Each checkpoint
is downloaded once and cached locally; neither is bundled in the Python package.

Calling `train()` on either model fine-tunes from those weights. A different
class count keeps the backbone and neck and starts only the class predictors fresh.

## Construct a Model

```python
from fotonet import Fotonet

model = Fotonet("path/to/checkpoint.pt")
```

Use `Fotonet()` when you intentionally need an untrained architecture for
training or custom checkpoint loading.

To use a local checkpoint explicitly, pass its path:

```python
from fotonet import Fotonet

model = Fotonet("path/to/checkpoint.pt")
```

Set `FOTONETE_MODEL_PATH` to override the release checkpoint, or set
`FOTONETE_CACHE_DIR` to choose the download cache directory.

## Run Inference With a Checkpoint

```python
from fotonet import Fotonet

model = Fotonet("my_checkpoint.pt")
results = model.predict("image.jpg", conf=0.25)
result = results[0]
print(result)
```

## Inspect Results

```python
for box in result.boxes:
    print(box.cls, box.conf, box.xywh, box.xyxy)
```

## CLI inference

```bash
fotonet predict model=fotonete source=image.jpg conf=0.25 save=true
```

## Train a YOLO dataset

```bash
python -m fotonet.cli.main train model=fotonete data=data.yaml epochs=100 imgsz=640 batch=16 run_dir=runs/fotonete
```

Add `dry_run=true` to resolve the graph, checkpoint, paths, and training arguments
without constructing a trainer or starting training. Add `resume=<checkpoint>` to continue
the same run from `runs/fotonete/fotonet_last.pt`.
