"""The v1.1 trainer lives in ``engine``. This module forwards to it."""
from .engine import train, train_cli

__all__ = ["train", "train_cli"]
