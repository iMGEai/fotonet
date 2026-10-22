# Model configuration

fotonet ships fixed model definitions for the v1.1 UltraDense family (`fotonetn`, `fotonets`, `fotonetm`, `fotonetl`) and the v1 `fotonete` detector.

```python
from fotonet import Fotonet

model = Fotonet("fotonetn")          # 80 COCO classes
custom = Fotonet("fotonetn", nc=2)   # custom class count for training
```

## Architecture verification

Each model size defines fixed backbone/neck channel widths, feature strides (`8`, `16`, `32`), and distributional regression width (`reg_max`). Checkpoint loading verifies the model identity and class count before loading tensor weights.
