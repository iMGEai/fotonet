"""Validation result object returned by ``Fotonet.val``.

A ``dict`` subclass, so code written against the historical dict keeps
working (``metrics["mAP50_95"]``), with attribute access on top.
"""
from __future__ import annotations


class Metrics(dict):
    """COCO detection metrics with named accessors."""

    names = None

    @property
    def map(self):
        """mAP50-95."""
        return self.get("mAP50_95")

    @property
    def map50(self):
        return self.get("mAP50")

    @property
    def map75(self):
        return self.get("mAP75")

    @property
    def map_small(self):
        return self.get("mAP_small")

    @property
    def map_medium(self):
        return self.get("mAP_medium")

    @property
    def map_large(self):
        return self.get("mAP_large")

    @property
    def precision(self):
        """Micro precision at the operating confidence and IoU."""
        return self.get("precision_at_conf", self.get("precision"))

    @property
    def recall(self):
        return self.get("recall_at_conf", self.get("recall"))

    @property
    def per_class(self):
        """Per-class metrics keyed by class name (or id when names are unknown)."""
        raw = self.get("per_class") or {}
        names = self.names or {}
        out = {}
        for key, value in raw.items():
            try:
                class_id = int(key)
            except (TypeError, ValueError):
                out[key] = value
                continue
            out[names.get(class_id, class_id)] = value
        return out

    def summary(self):
        rows = [("mAP50-95", self.map), ("mAP50", self.map50), ("mAP75", self.map75),
                ("precision", self.precision), ("recall", self.recall)]
        return "  ".join(f"{name} {value:.4f}" for name, value in rows if isinstance(value, (int, float)))

    def __repr__(self):
        return f"Metrics({self.summary()})"


__all__ = ["Metrics"]
