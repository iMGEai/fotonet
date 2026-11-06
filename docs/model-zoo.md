# Models and runtime

`fotonet` includes the v1.1 UltraDense family (`fotonetn`, `fotonets`, `fotonetm`, `fotonetl`) and the v1 `fotonete` detector.

## Model facts

| Model ID | Generation | Input size | Classes | Deployed parameters | Compute (`640 × 640`) | Status |
| --- | --- | --- | --- | --- | --- | --- |
| `fotonetn` | v1.1 (UltraDense) | `640 × 640` | `80` COCO | `2.85M` | `6.95 GFLOPs` | Released |
| `fotonets` | v1.1 (UltraDense) | `640 × 640` | `80` COCO | `11.49M` | `27.43 GFLOPs` | Architecture in package, weights coming soon |
| `fotonetm` | v1.1 (UltraDense) | `640 × 640` | `80` COCO | `22.23M` | `63.38 GFLOPs` | Architecture in package, weights coming soon |
| `fotonetl` | v1.1 (UltraDense) | `640 × 640` | `80` COCO | `70.32M` | `161.51 GFLOPs` | Architecture in package, weights coming soon |
| `fotonete` | v1 | `640 × 640` | `80` COCO | `1.70M` | `5.20 GFLOPs` | Released |

See [model configuration](model-config.md) for the model settings and checkpoint
compatibility rules.

## Published weights

`Fotonet("fotonetn")` downloads the newest published `fotonetn` weights once and
keeps them in the local cache.

All `fotonetn` weights live in the `v1.1` GitHub release:

| Asset | Weights | COCO val2017 mAP50-95 | mAP50 |
| --- | --- | --- | --- |
| `fotonetn.pt` (current) | `fotonetn`, trained with [trainer 1.2](training.md#trainer-12) | 32.1 | 47.5 |
| `fotonetn-ex.pt` | the first `fotonetn` | 29.9 | 42.9 |

Measured on all 5,000 val2017 images at 640 px, up to 300 boxes per image,
scored with pycocotools. Both are the same model with the same size and speed;
only the training differs.

fotonet checks the cached file against the release at most once a day and
downloads it again when the release has newer weights. To keep using the first
weights, download `fotonetn-ex.pt` from the release and load it by path:

```python
model = Fotonet("fotonetn-ex.pt")
```

With `FOTONET_OFFLINE=1`, fotonet uses the cached file and makes no network
call.

## Runtime measurements

The website benchmark section publishes the current measurements together with
the hardware and software used for the run.

For application code, start with the [quickstart](quickstart.md) and
[inference guide](inference.md). For deployment artifacts, see the
[export guide](export.md).