"""ByteTrack multi-object tracking for ``Fotonet.track(tracker="bytetrack")``.

Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection
Box" (ECCV 2022). Constant-velocity Kalman filter over (cx, cy, aspect, height),
then two association passes: confident detections against all tracks, and the
low-confidence remainder against tracks still unmatched, so occluded objects
keep their identity instead of being dropped. Lost tracks are kept for
``track_buffer`` frames. Association is class-agnostic, as in the paper; each
output detection keeps its own class.

Pure NumPy plus SciPy's Hungarian solver (already a fotonet dependency).
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from fotonet.engine.results import Results



class KalmanFilter:
    """Constant-velocity Kalman filter on (cx, cy, a, h, vx, vy, va, vh)."""

    def __init__(self):
        self.motion = np.eye(8)
        self.motion[:4, 4:] = np.eye(4)
        self.update_mat = np.eye(4, 8)
        self.std_position = 1.0 / 20
        self.std_velocity = 1.0 / 160

    def initiate(self, measurement):
        mean = np.r_[measurement, np.zeros(4)]
        h = measurement[3]
        std = [2 * self.std_position * h, 2 * self.std_position * h, 1e-2, 2 * self.std_position * h,
               10 * self.std_velocity * h, 10 * self.std_velocity * h, 1e-5, 10 * self.std_velocity * h]
        return mean, np.diag(np.square(std))

    def predict(self, mean, covariance):
        h = mean[3]
        std = [self.std_position * h, self.std_position * h, 1e-2, self.std_position * h,
               self.std_velocity * h, self.std_velocity * h, 1e-5, self.std_velocity * h]
        mean = self.motion @ mean
        covariance = self.motion @ covariance @ self.motion.T + np.diag(np.square(std))
        return mean, covariance

    def update(self, mean, covariance, measurement):
        h = mean[3]
        std = [self.std_position * h, self.std_position * h, 1e-1, self.std_position * h]
        projected_mean = self.update_mat @ mean
        projected_cov = self.update_mat @ covariance @ self.update_mat.T + np.diag(np.square(std))
        gain = np.linalg.solve(projected_cov, (covariance @ self.update_mat.T).T).T
        mean = mean + gain @ (measurement - projected_mean)
        covariance = covariance - gain @ projected_cov @ gain.T
        return mean, covariance


def _xyxy_to_xyah(box):
    w, h = box[2] - box[0], box[3] - box[1]
    return np.array([box[0] + w / 2, box[1] + h / 2, w / max(h, 1e-9), h], dtype=np.float64)


def _xyah_to_xyxy(state):
    cx, cy, a, h = state[:4]
    w = a * h
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float64)


def _iou_matrix(a, b):
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def _assign(cost, threshold):
    """Hungarian matching keeping pairs with cost <= threshold."""
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    rows, cols = linear_sum_assignment(cost)
    matches = [(r, c) for r, c in zip(rows, cols) if cost[r, c] <= threshold]
    matched_r = {r for r, _ in matches}
    matched_c = {c for _, c in matches}
    return (
        matches,
        [r for r in range(cost.shape[0]) if r not in matched_r],
        [c for c in range(cost.shape[1]) if c not in matched_c],
    )


class _Track:
    def __init__(self, box, score, cls, kalman):
        self.kalman = kalman
        self.mean, self.covariance = kalman.initiate(_xyxy_to_xyah(box))
        self.score = float(score)
        self.cls = int(cls)
        self.track_id = 0
        self.activated = False
        self.lost_frames = 0

    def predict(self):
        mean = self.mean.copy()
        if self.lost_frames:
            mean[7] = 0.0
        self.mean, self.covariance = self.kalman.predict(mean, self.covariance)

    def update(self, box, score, cls):
        self.mean, self.covariance = self.kalman.update(self.mean, self.covariance, _xyxy_to_xyah(box))
        self.score, self.cls = float(score), int(cls)
        self.lost_frames = 0

    @property
    def box(self):
        return _xyah_to_xyxy(self.mean)


class ByteTracker:
    """Track ``Results`` frame by frame; returns Results restricted to tracked boxes with ids."""

    def __init__(self, track_high_thresh=0.5, track_low_thresh=0.1, new_track_thresh=0.6,
                 track_buffer=30, match_thresh=0.8):
        self.high = float(track_high_thresh)
        self.low = float(track_low_thresh)
        self.new = float(new_track_thresh)
        self.buffer = int(track_buffer)
        self.match = float(match_thresh)
        self.kalman = KalmanFilter()
        self.reset()

    def reset(self):
        self.tracked, self.lost = [], []
        self.frame = 0
        self._next_id = 1

    def update(self, result: Results) -> Results:
        self.frame += 1
        width, height = result.image_size
        host_boxes, host_scores, host_classes, _ = result.boxes._host()
        boxes = host_boxes.astype(np.float64)
        scores = host_scores.astype(np.float64)
        classes = host_classes.astype(np.int64)
        xyxy = np.stack((
            (boxes[:, 0] - boxes[:, 2] / 2) * width, (boxes[:, 1] - boxes[:, 3] / 2) * height,
            (boxes[:, 0] + boxes[:, 2] / 2) * width, (boxes[:, 1] + boxes[:, 3] / 2) * height,
        ), axis=1) if len(boxes) else np.zeros((0, 4))
        high = np.flatnonzero(scores >= self.high)
        low = np.flatnonzero((scores >= self.low) & (scores < self.high))

        pool = self.tracked + self.lost
        for track in pool:
            track.predict()
        output = {}

        # Pass 1: confident detections against every track (score-fused IoU cost).
        track_boxes = np.array([t.box for t in pool]).reshape(-1, 4)
        cost = 1.0 - _iou_matrix(track_boxes, xyxy[high]) * scores[high][None, :]
        matches, unmatched_tracks, unmatched_high = _assign(cost, self.match)
        for t, d in matches:
            det = high[d]
            pool[t].update(xyxy[det], scores[det], classes[det])
            pool[t].activated = True
            output[det] = pool[t]
        remaining = [pool[t] for t in unmatched_tracks if pool[t] in self.tracked]
        lost_unmatched = [pool[t] for t in unmatched_tracks if pool[t] in self.lost]

        # Pass 2: low-confidence detections against tracks the first pass left over.
        rem_boxes = np.array([t.box for t in remaining]).reshape(-1, 4)
        cost = 1.0 - _iou_matrix(rem_boxes, xyxy[low])
        matches, still_unmatched, _ = _assign(cost, 0.5)
        for t, d in matches:
            det = low[d]
            remaining[t].update(xyxy[det], scores[det], classes[det])
            remaining[t].activated = True
            output[det] = remaining[t]

        # New tracks from confident detections nobody claimed.
        new_tracks = []
        for d in unmatched_high:
            det = high[d]
            if scores[det] < self.new:
                continue
            track = _Track(xyxy[det], scores[det], classes[det], self.kalman)
            track.track_id = self._next_id
            self._next_id += 1
            track.activated = self.frame == 1
            new_tracks.append(track)
            if track.activated:
                output[det] = track

        for track in [remaining[t] for t in still_unmatched] + lost_unmatched:
            track.lost_frames += 1
        alive = [t for t in self.tracked + self.lost + new_tracks if t.lost_frames <= self.buffer]
        self.tracked = [t for t in alive if t.lost_frames == 0]
        # A new track (after the first frame) is reported from its second match on.
        self.lost = [t for t in alive if t.lost_frames > 0]

        keep = sorted(output)
        ids = torch.tensor([output[i].track_id for i in keep], dtype=torch.long)
        index = torch.as_tensor(keep, dtype=torch.long)
        device = result.boxes_tensor.device
        return Results(
            result.orig_img,
            result.boxes_tensor[index.to(device)],
            result.scores[index.to(result.scores.device)],
            result.classes[index.to(result.classes.device)],
            result.names,
            ids.to(device),
            result.orig_shape,
        )


__all__ = ["ByteTracker", "KalmanFilter"]
