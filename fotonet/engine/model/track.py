"""``Fotonet`` multi-object tracking over a video or image stream."""


class TrackMixin:
    """Multi-object tracking over a video or image stream."""

    def track(self, source, *, persist=False, tracker="iou", tracker_iou=0.3, max_age=30, **kwargs):
        """Track ordered frames.

        ``tracker='bytetrack'``: Kalman motion model with two-stage
        (confident, then low-confidence) association. ``tracker='iou'``: the
        dependency-free same-class IoU tracker. ``persist=True`` keeps track
        state across calls (frame-by-frame use).
        """
        kind = str(tracker).lower()
        if kind not in {"iou", "bytetrack"}:
            raise ValueError("tracker must be 'bytetrack' or 'iou'.")
        current = getattr(self, "_tracker", None)
        if kind == "bytetrack":
            from fotonet.engine.bytetrack import ByteTracker

            if not persist or not isinstance(current, ByteTracker):
                current = ByteTracker(track_buffer=max_age)
                if persist:
                    self._tracker = current
            if "conf" not in kwargs:
                kwargs["conf"] = current.low   # ByteTrack needs the low-confidence detections too
        else:
            from fotonet.engine.tracker import IoUTracker

            if (not persist or not isinstance(current, IoUTracker) or current.iou_threshold != float(tracker_iou)
                    or current.max_age != int(max_age)):
                current = IoUTracker(iou_threshold=tracker_iou, max_age=max_age)
                if persist:
                    self._tracker = current

        stream = bool(kwargs.get("stream", False))
        predictions = self.predict(source, **kwargs)
        if stream:
            return (current.update(result) for result in predictions)
        return [current.update(result) for result in predictions]
