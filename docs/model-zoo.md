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

## Runtime measurements

The website benchmark section publishes the current measurements together with
the hardware and software used for the run.

For application code, start with the [quickstart](quickstart.md) and
[inference guide](inference.md). For deployment artifacts, see the
[export guide](export.md).