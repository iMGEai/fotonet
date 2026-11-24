"""``batch: auto``: the largest training batch the GPU's free memory holds."""
from __future__ import annotations

import torch

# Forward and backward are measured; the matcher, the loss and the GPU loader come on top.
MEMORY_FRACTION = 0.6
SMALLEST = 8


def fit_batch(base_bytes, bytes_per_image, budget_bytes, nbs):
    """The largest of 8, 16, 32, ... (at most ``nbs``) whose memory ``base + slope * batch`` fits the budget."""
    best = SMALLEST
    batch = SMALLEST
    while batch <= int(nbs):
        if base_bytes + bytes_per_image * batch <= budget_bytes:
            best = batch
        batch *= 2
    return best


def auto_batch(model, imgsz, device, amp_dtype, nbs, channels_last=False, default=16):
    """Measure the training graph at batch 4 and 8 and pick a batch from the fitted memory line.

    BatchNorm statistics are put back afterwards, so the measurement leaves the model as it was.
    On anything but CUDA the ``default`` is returned.
    """
    if device.type != "cuda":
        return int(default)
    saved = {name: buf.clone() for name, buf in model.named_buffers()}
    peaks = []
    torch.cuda.empty_cache()
    before = torch.cuda.memory_allocated(device)
    free, _total = torch.cuda.mem_get_info(device)
    for batch in (4, 8):
        torch.cuda.reset_peak_memory_stats(device)
        images = torch.rand(batch, 3, int(imgsz), int(imgsz), device=device)
        if channels_last:
            images = images.contiguous(memory_format=torch.channels_last)
        with torch.autocast(device.type, dtype=amp_dtype):
            out = model(images, use_o2m=True)
        sum(value.float().mean() for value in out.values() if torch.is_tensor(value) and value.requires_grad).backward()
        model.zero_grad(set_to_none=True)
        peaks.append(torch.cuda.max_memory_allocated(device))
        del images, out
    with torch.no_grad():
        for name, buf in model.named_buffers():
            buf.copy_(saved[name])
    slope = max((peaks[1] - peaks[0]) / 4.0, 1.0)
    base = peaks[0] - 4.0 * slope
    torch.cuda.empty_cache()
    return fit_batch(base, slope, before + free * MEMORY_FRACTION, nbs)


__all__ = ["auto_batch", "fit_batch"]
