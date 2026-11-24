"""Task-aligned label assignment (first used by the fotonet v2 trainer).

One rule serves both heads. For every ground-truth box, the anchors whose centre lies inside it are
ranked by ``score^alpha * IoU^beta`` (TOOD's alignment metric) and the best ``topk`` become its
positives. An anchor claimed by several boxes keeps the one it overlaps most. With ``keep`` set, each
box then keeps only its best ``keep`` anchors, which turns the same rule into the one-to-one match
of the deployed head (``topk=7, keep=1``).

``small`` and ``grow`` widen the centre test for tiny boxes: a side shorter than ``small`` pixels
counts as ``grow`` pixels. Without it a box narrower than a stride can hold no anchor centre at all
and trains nothing.

Positive targets are soft. Each box's metric is rescaled so that its best anchor's target equals
that box's best IoU, so the class score learns to rank by localization quality.

The work is done on the (box, anchor) pairs that pass the centre test, a few percent of all pairs:

- ``candidate_pairs`` runs the centre test. It reads only the ground truth and the anchor grid, so
  one call serves every head of a training step.
- ``assign_pairs`` ranks those pairs and returns the positives with their soft targets. With
  ``fixed=True`` it never waits for the device and its output size depends only on the shapes.
- ``task_aligned_assign`` is the same assignment with dense ``[batch, anchors]`` outputs.

Equal metrics are ranked by anchor index, lowest first.
"""
from __future__ import annotations

import torch

# Bits kept for the box index next to an IoU when one sortable integer holds both.
_BOX_BITS = 20
_BOX_MAX = (1 << _BOX_BITS) - 1
_FLOAT_BITS = 0x7FFFFFFF


def _pair_iou(a, b, eps=1e-7):
    """IoU of broadcastable xyxy boxes."""
    width = (torch.minimum(a[..., 2], b[..., 2]) - torch.maximum(a[..., 0], b[..., 0])).clamp(min=0.0)
    height = (torch.minimum(a[..., 3], b[..., 3]) - torch.maximum(a[..., 1], b[..., 1])).clamp(min=0.0)
    inter = width * height
    area_a = (a[..., 2] - a[..., 0]).clamp(min=0.0) * (a[..., 3] - a[..., 1]).clamp(min=0.0)
    area_b = (b[..., 2] - b[..., 0]).clamp(min=0.0) * (b[..., 3] - b[..., 1]).clamp(min=0.0)
    return inter / (area_a + area_b - inter + eps)


def _ordered_bits(values):
    """Non-negative float32 values as integers that sort the same way."""
    return values.float().contiguous().view(torch.int32).to(torch.int64) & _FLOAT_BITS


@torch.no_grad()
def candidate_pairs(anchors, gt_boxes, gt_mask, *, small=0.0, grow=0.0, eps=1e-9):
    """Every (image, box, anchor) whose anchor centre lies inside the box.

    anchors    [N, 2]    anchor centres in pixels
    gt_boxes   [B, M, 4] ground-truth boxes, xyxy in pixels
    gt_mask    [B, M]    True for real boxes (the rest is padding)

    Returns three index vectors ``(image, box, anchor)``, ordered by image, then box, then anchor.
    """
    image, box = gt_mask.bool().nonzero(as_tuple=True)
    real = gt_boxes[image, box]                                                           # [G, 4]
    if float(small) > 0.0:
        # Tiny boxes are widened for this test only.
        centre = (real[:, :2] + real[:, 2:]) * 0.5
        size = real[:, 2:] - real[:, :2]
        size = torch.where(size < float(small), size.new_tensor(float(grow)), size)
        real = torch.cat((centre - size * 0.5, centre + size * 0.5), -1)
    ax, ay = anchors[:, 0][None], anchors[:, 1][None]
    inside = ((ax > real[:, 0:1] + eps) & (ax < real[:, 2:3] - eps)
              & (ay > real[:, 1:2] + eps) & (ay < real[:, 3:4] - eps))                    # [G, N]
    row, anchor = inside.nonzero(as_tuple=True)
    return image[row], box[row], anchor


def max_positives(shape, topk, keep=None):
    """The most positives ``assign_pairs`` can return for ``shape = (batch, boxes per image, anchors)``."""
    batch, n_gt, n_anchors = (int(v) for v in shape)
    k = min(int(topk), n_anchors)
    per_box = int(keep) if keep is not None and int(keep) < k else k
    return batch * n_gt * per_box


@torch.no_grad()
def assign_pairs(pair_scores, pair_iou, pairs, shape, *, topk=10, keep=None, alpha=0.5, beta=6.0, eps=1e-9,
                 fixed=False):
    """Pick the positives among candidate pairs.

    pair_scores  [P] class probability of each pair's anchor, at its box's class
    pair_iou     [P] IoU of each pair's predicted box with its ground-truth box
    pairs        the ``(image, box, anchor)`` vectors of ``candidate_pairs``
    shape        ``(batch, boxes per image, anchors)``

    Returns ``(chosen, quality)``: the positions of the positive pairs in the pair vectors, and each
    one's soft target. An anchor is positive for at most one box.

    With ``fixed``, the result always has ``max_positives(shape, topk, keep)`` rows: the positives
    first, then padding rows with quality 0 that point at candidate pairs spread over the batch.
    Nothing waits for the device then, and the shapes do not depend on the batch's content. Without
    it, exactly the positives are returned. There must be at least one pair (callers check
    ``pairs[0].numel()``).
    """
    image, box, anchor = pairs
    batch, n_gt, n_anchors = (int(v) for v in shape)
    count = int(image.numel())
    metric = pair_scores.to(pair_iou.dtype).pow(float(alpha)) * pair_iou.pow(float(beta))

    # Pairs in the order "box by box, best metric first". The pairs arrive ordered by anchor, and
    # the sort is stable, so equal metrics stay in anchor order.
    group = image * n_gt + box
    order = torch.sort((group << 32) | (_FLOAT_BITS - _ordered_bits(metric)), stable=True).indices
    group, metric, iou = group[order], metric[order], pair_iou[order]
    box_s = box[order]
    spot = image[order] * n_anchors + anchor[order]
    sizes = torch.zeros(batch * n_gt, dtype=torch.int64, device=group.device).index_add_(0, group, torch.ones_like(group))
    first = sizes.cumsum(0) - sizes                              # where each box's pairs start
    position = torch.arange(count, device=group.device)

    # The best ``topk`` anchors of each box.
    k = min(int(topk), n_anchors)
    pos = position - first[group] < k

    # For O2O (keep is not None and keep < k), isolate the primary candidates per box FIRST before conflict resolution
    if keep is not None and int(keep) < k:
        before = pos.long().cumsum(0) - pos.long()
        primary_pos = pos & ((before - before[first[group]]) < int(keep))

        claims = torch.zeros(batch * n_anchors, dtype=torch.int64, device=group.device).index_add_(0, spot, primary_pos.long())
        vote = torch.where(primary_pos, (_ordered_bits(iou) << _BOX_BITS) | (_BOX_MAX - box_s), torch.zeros_like(box_s))
        best = torch.zeros(batch * n_anchors, dtype=torch.int64, device=group.device)
        best = best.scatter_reduce_(0, spot, vote, "amax", include_self=True)[spot]
        winner = torch.where((best >> _BOX_BITS) > 0, _BOX_MAX - (best & _BOX_MAX), torch.zeros_like(best))
        pos = primary_pos & ((claims[spot] <= 1) | (box_s == winner))

        # Fallback pass: give losers their second best candidate if vacant (no host .any() sync)
        loser_boxes = torch.zeros(batch * n_gt, dtype=torch.int64, device=group.device).index_add_(0, group, pos.long()) == 0
        claimed_anchors = torch.zeros(batch * n_anchors, dtype=torch.int64, device=group.device).index_add_(0, spot, pos.long()) > 0
        fallback_pos = (position - first[group] == 1) & loser_boxes[group] & (~claimed_anchors[spot])
        fb_claims = torch.zeros(batch * n_anchors, dtype=torch.int64, device=group.device).index_add_(0, spot, fallback_pos.long())
        fb_vote = torch.where(fallback_pos, (_ordered_bits(iou) << _BOX_BITS) | (_BOX_MAX - box_s), torch.zeros_like(box_s))
        fb_best = torch.zeros(batch * n_anchors, dtype=torch.int64, device=group.device)
        fb_best = fb_best.scatter_reduce_(0, spot, fb_vote, "amax", include_self=True)[spot]
        fb_winner = torch.where((fb_best >> _BOX_BITS) > 0, _BOX_MAX - (fb_best & _BOX_MAX), torch.zeros_like(fb_best))
        fallback_pos = fallback_pos & ((fb_claims[spot] <= 1) | (box_s == fb_winner))
        pos = pos | fallback_pos
    else:
        # Standard O2M multi-box conflict resolution
        claims = torch.zeros(batch * n_anchors, dtype=torch.int64, device=group.device).index_add_(0, spot, pos.long())
        vote = torch.where(pos, (_ordered_bits(iou) << _BOX_BITS) | (_BOX_MAX - box_s), torch.zeros_like(box_s))
        best = torch.zeros(batch * n_anchors, dtype=torch.int64, device=group.device)
        best = best.scatter_reduce_(0, spot, vote, "amax", include_self=True)[spot]
        winner = torch.where((best >> _BOX_BITS) > 0, _BOX_MAX - (best & _BOX_MAX), torch.zeros_like(best))
        pos = pos & ((claims[spot] <= 1) | (box_s == winner))

    # Soft targets: the box's best positive gets that box's best IoU.
    weight = pos.to(metric.dtype)
    zero = metric.new_zeros(batch * n_gt)
    best_metric = zero.scatter_reduce(0, group, metric * weight, "amax", include_self=True)
    best_iou = zero.scatter_reduce(0, group, iou * weight, "amax", include_self=True)
    quality = metric * best_iou[group] / (best_metric[group] + eps)
    if not fixed:
        chosen = pos.nonzero(as_tuple=True)[0]
        return order[chosen], quality[chosen]
    rows = max_positives(shape, topk, keep)
    chosen = torch.nonzero_static(pos, size=rows, fill_value=-1)[:, 0]
    real = chosen >= 0
    # Padding rows point at different pairs: their zero gradients then do not all land on one
    # address in the backward pass.
    spread = torch.arange(rows, device=chosen.device) % count
    return torch.where(real, order[chosen.clamp(min=0)], spread), torch.where(real, quality[chosen.clamp(min=0)], 0.0)


@torch.no_grad()
def task_aligned_assign(scores, boxes, anchors, gt_boxes, gt_labels, gt_mask, *, topk=10, keep=None,
                        alpha=0.5, beta=6.0, small=0.0, grow=0.0, eps=1e-9):
    """Assign anchors to ground truth.

    scores     [B, N, C] class probabilities (after sigmoid)
    boxes      [B, N, 4] predicted boxes, xyxy in pixels
    anchors    [N, 2]    anchor centres in pixels
    gt_boxes   [B, M, 4] ground-truth boxes, xyxy in pixels
    gt_labels  [B, M]    class ids
    gt_mask    [B, M]    True for real boxes (the rest is padding)

    Returns ``(target_scores [B, N, C], target_boxes [B, N, 4], fg [B, N] bool, gt_index [B, N])``.
    """
    batch, n_anchors, n_classes = scores.shape
    n_gt = int(gt_boxes.shape[1])
    target_scores = torch.zeros_like(scores)
    fg = torch.zeros(batch, n_anchors, dtype=torch.bool, device=scores.device)
    gt_index = torch.zeros(batch, n_anchors, dtype=torch.long, device=scores.device)
    if n_gt == 0:
        return target_scores, torch.zeros_like(boxes), fg, gt_index
    labels = gt_labels.long().clamp(0, n_classes - 1)
    pairs = candidate_pairs(anchors, gt_boxes, gt_mask, small=small, grow=grow, eps=eps)
    image, box, anchor = pairs
    chosen, quality = assign_pairs(
        scores[image, anchor, labels[image, box]], _pair_iou(gt_boxes[image, box], boxes[image, anchor]),
        pairs, (batch, n_gt, n_anchors), topk=topk, keep=keep, alpha=alpha, beta=beta, eps=eps,
    )
    image, box, anchor = image[chosen], box[chosen], anchor[chosen]
    target_scores[image, anchor, labels[image, box]] = quality.to(scores.dtype)
    fg[image, anchor] = True
    gt_index[image, anchor] = box
    target_boxes = gt_boxes.gather(1, gt_index[..., None].expand(-1, -1, 4))
    return target_scores, target_boxes, fg, gt_index


__all__ = ["_ordered_bits", "_pair_iou", "assign_pairs", "candidate_pairs", "max_positives", "task_aligned_assign"]
