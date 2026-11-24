# Training and resume

fotonet supports one production training policy: ordinary uniform sampling.
The deploy graph uses one-to-one predictions; the one-to-many branch is
training-only and can be stripped from inference checkpoints.

## Dataset YAML

```yaml
path: /path/to/dataset
train: images/train
val: images/val
nc: 2
names:
  0: cat
  1: dog
```

Labels use normalized YOLO rows:

```text
class_id x_center y_center width height
```

The default `annotation_policy="fix"` clamps edge-crossing boxes and drops
malformed/duplicate rows. Use `annotation_policy="error"` for a strict audit.
Public validation requires an independent `val` source and never silently
uses training data.

### COCO annotation files

A split can also name a COCO annotation file directly. No conversion to text
labels is needed:

```yaml
path: /path/to/coco
train:
  format: coco
  images: train2017
  annotations: annotations/instances_train2017.json
val:
  format: coco
  images: val2017
  annotations: annotations/instances_val2017.json
nc: 80
```

Categories become classes in the order of their ids. Crowd annotations
(`iscrowd: 1`) are handled the way COCO intends:

- Validation scores against the annotation file with the official COCO
  protocol, so a detection inside a crowd region is neither a hit nor a miss.
- Training with trainer `2.0` keeps crowd regions out of the loss: an anchor
  inside one is neither object nor background for that region's class. Turn
  this off with `crowd_ignore=false`. Older trainers leave crowd annotations
  out, as a text conversion does.

A text conversion of COCO drops crowd regions, so a model trained and scored
on one sees crowds of people as background in training and is scored against
missing boxes in validation.

## CLI launcher

The CLI uses `key=value` arguments and validates them before starting a long run.

```bash
python -m fotonet.cli.main train \
  model=fotonete \
  data=path/to/data.yaml \
  epochs=300 \
  batch=16 \
  imgsz=640 \
  run_dir=runs/fotonete \
  dry_run=true
```

Inspect its resolved graph/checkpoint identity and arguments without starting
training:

```bash
fotonet train model=fotonete data=path/to/data.yaml dry_run=true
```

## Trainer 2.0

fotonet 2.0 introduces trainer `2.0` (`fotonet2n_scratch`) for the v2 family
(`fotonet2n`, Hi Bird). The published `fotonet2n` weights (37.5 mAP50-95 on
COCO val2017) were trained from scratch in 300 epochs with it:

- task-aligned assignment (`assign.tal@2.0`) with stride-8 small-object support
  and native COCO `iscrowd` masking
- hybrid Stage-0 + Stage-1 GeoSA Top-300 deep supervision with static-shape
  64-query contrastive denoising (`loss.tal@2.0`)
- selective `MuAdamW` (`optim.muadamw@2.0`): Muon on hidden dense matrices,
  AdamW at 0.15x LR on stem, depthwise, attention, decoder, and head layers
- 300-epoch WSD-Cosine schedule with a 30-epoch clean tail (mosaic, GPU mixup,
  and GPU copy-paste closed at epoch 271) and EMA checkpoint averaging

Train `fotonet2n` from scratch with it:

```bash
fotonet train \
  model=fotonet2n \
  pretrained=False \
  data=path/to/coco.yaml \
  recipe=fotonet2n_scratch \
  out_dir=runs/fotonet2n
```

`pretrained=False` matters: without it, `model=fotonet2n` starts from the
published `v2` weights and fine-tunes them.

## Trainer 1.2

fotonet 1.2 adds a second trainer for the v1.1 family. The model is the same;
only the training changes. The published `fotonetn` weights were trained with
it. What it changes:

- a stronger loss on the deployed head, so it keeps fewer duplicate boxes
- a warmup, stable and decay learning-rate schedule
- the image size steps up during training (512, then 576, then 640 px)
- a mosaic that zooms and shifts the whole canvas
- guard rails during training: duplicates, precision and recall on part of
  the validation set, and a forecast of the final mAP

Train `fotonetn` from scratch with it:

```bash
fotonet train \
  model=fotonetn \
  pretrained=False \
  data=path/to/data.yaml \
  recipe=fotonetn_scratch_v12 \
  out_dir=runs/fotonetn
```

`pretrained=False` matters: without it, `model=fotonetn` starts from the
published weights and fine-tunes them.

A recipe picks its trainer with the `trainer` key (`"1.1"`, `"1.2"`, `"1.3"`, or `"2.0"`); recipes
that name none use 1.1, the trainer behind the first `fotonetn` weights. All
stay available.

## Resume after interruption

```bash
fotonet train \
  model=fotonete \
  data=path/to/data.yaml \
  epochs=300 \
  batch=16 \
  run_dir=runs/fotonete \
  resume=runs/fotonete/fotonet_last.pt
```

Resume restores model, EMA, optimizer, scheduler, AMP scaler, epoch/global
step, and RNG state. It rejects a different model, class count/order, training
protocol, or incomplete checkpoint.

## Python API

```python
from fotonet import Fotonet

model = Fotonet("fotonete", nc=2)
summary = model.train(
    data="data.yaml",
    epochs=300,
    imgsz=640,
    batch=16,
    save_dir="runs/fotonete",
)
```

Resume through Python:

```python
model = Fotonet("fotonete")
summary = model.train(
    data="data.yaml",
    epochs=300,
    weights="runs/fotonete/fotonet_last.pt",
    resume=True,
    save_dir="runs/fotonete",
)
```

`resume=True` continues complete state. `pretrained=True` starts a fresh
optimizer/scheduler from a supported weight. They are mutually exclusive.

Checkpoint deserialization is always tensor-only. There is no unsafe pickle
switch and no metadata-free reconstruction.

## Validation semantics

`val_conf=0.0` retains finite detections for score-ranked COCO evaluation.
`operating_conf` and `operating_iou` report a separate deployment operating
point. Sanitized/noncanonical annotations are labeled as such and must not be
reported as official COCO metrics.
