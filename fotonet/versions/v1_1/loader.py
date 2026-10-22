"""Moved to ``fotonet.data.gpu_loader``; this module keeps the old import path."""
from fotonet.data.gpu_loader import (  # noqa: F401
    _JPEG_MAGIC, _ORIENTATION_TAG, image_header, _resized_shape, Part, GeometryMixin,
    collate_geometry, _decode_cpu, GpuMosaicAssembler, _END, DecodePrefetcher,
)
