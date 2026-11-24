"""fotonet2n (Hi Bird) training criterion: DEIM MAL + GeoSA GPU 1-to-1 + CDN + Self-Distillation.

Key components (zero extra deployed inference cost):
1. DEIM Matchability-Aware Loss (MAL) on both dense heads and GeoSA decoder layers
   (sparse+dense memory-efficient formulation with zero [B, 8400, 80] target alloc).
2. GT-size-normalized box L1 loss (equal relative edge gradient for small and large objects) + CIoU.
3. Stride-8-restricted ``tal_small`` (widens boxes < 16 px strictly on P3 stride-8 anchors).
4. GT-identity O2M->O2O class + box self-distillation + fixed-shape GPU NMS duplicate trap penalty.
5. Zero-host-sync vectorized 3-pass GPU conflict-free 1-to-1 matching for Layer 1 & Layer 2 GeoSA
   queries (deep supervision with MAL + CIoU + scale-normalized L1).
6. Static-shape training-only Contrastive Denoising (CDN, 32 pos + 32 hard-neg queries per image)
   executed through ``head_ref.forward_cdn(...)`` with block-diagonal attention isolation.
"""
from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F

from .assign import xywh_to_xyxy_t
from .losses import bbox_ciou
from .tal import _ordered_bits, _pair_iou, assign_pairs, max_positives


class Plan(NamedTuple):
    """Ground-truth plan prepared once per batch for both heads."""
    anchors: torch.Tensor        # [N, 2] anchor centres in pixels
    scale: torch.Tensor          # [4] width, height, width, height
    gt_boxes: torch.Tensor       # [B, M, 4] xyxy in pixels, real boxes first in each row, padding zero
    labels: torch.Tensor         # [B, M] class ids
    pairs: tuple                 # (image, box, anchor) of every anchor centre inside a box
    crowd: tuple | None          # (counted, anchors_in), see HybridTalCriterion._crowd


def fixed_gpu_nms_duplicate_penalty(logits_o2o, logits_o2m, boxes_o2m, scale, spot_o, quality_o,
                                    iou_thresh=0.5, w_dup=0.5):
    """Fixed-shape [B, 300, 300] NMS duplicate trap penalty running entirely on GPU."""
    batch, n_anchors, nc = logits_o2m.shape
    K = min(300, n_anchors)
    with torch.no_grad():
        raw_scores_m, classes_m = logits_o2m.max(dim=-1)
        _, topk_anchors = raw_scores_m.topk(K, dim=-1)
        topk_scores = raw_scores_m.gather(1, topk_anchors).float().sigmoid()
        topk_classes = classes_m.gather(1, topk_anchors)

        topk_boxes = boxes_o2m.gather(1, topk_anchors.unsqueeze(-1).expand(-1, -1, 4))
        topk_xyxy = xywh_to_xyxy_t(topk_boxes.float()) * scale

        b1 = topk_xyxy.unsqueeze(2)
        b2 = topk_xyxy.unsqueeze(1)
        iou_matrix = _pair_iou(b1, b2)

        upper = torch.triu(torch.ones(K, K, device=logits_o2m.device, dtype=torch.bool), diagonal=1)
        same_class = (topk_classes.unsqueeze(2) == topk_classes.unsqueeze(1))
        suppressed = (iou_matrix > float(iou_thresh)) & same_class & upper.unsqueeze(0) & (topk_scores.unsqueeze(2) > 0.05)
        is_duplicate = suppressed.any(dim=1)

        o2o_pos_count = torch.zeros(batch * n_anchors, dtype=torch.int32, device=spot_o.device)
        if spot_o.numel() > 0:
            o2o_pos_count.index_add_(0, spot_o, (quality_o > 0).to(torch.int32))

        b_idx = torch.arange(batch, device=topk_anchors.device).unsqueeze(1).expand(-1, K)
        topk_spots = b_idx * n_anchors + topk_anchors
        is_pos_in_o2o = (o2o_pos_count > 0)[topk_spots]

        trap_mask = is_duplicate & (~is_pos_in_o2o)

    o2o_topk_logits = logits_o2o.gather(1, topk_anchors.unsqueeze(-1).expand(-1, -1, nc))
    trap_logits = o2o_topk_logits.gather(-1, topk_classes.unsqueeze(-1)).squeeze(-1)

    bce = F.binary_cross_entropy_with_logits(trap_logits, torch.zeros_like(trap_logits), reduction="none")
    penalty = (bce * trap_mask).sum() / trap_mask.sum().clamp(min=1.0)
    return penalty * float(w_dup)


@torch.no_grad()
def candidate_pairs_stride8(anchors, stride_per_anchor, gt_boxes, gt_mask, *, small=16.0, grow=16.0, eps=1e-9):
    """Anchor-in-box centre test where ``tal_small`` widening applies strictly to stride-8 P3 anchors."""
    image, box = gt_mask.bool().nonzero(as_tuple=True)
    real = gt_boxes[image, box]  # [G, 4]
    ax, ay = anchors[:, 0][None], anchors[:, 1][None]
    inside_orig = (
        (ax > real[:, 0:1] + eps) & (ax < real[:, 2:3] - eps)
        & (ay > real[:, 1:2] + eps) & (ay < real[:, 3:4] - eps)
    )
    if float(small) > 0.0:
        centre = (real[:, :2] + real[:, 2:]) * 0.5
        size = real[:, 2:] - real[:, :2]
        grown = torch.where(size < float(small), size.new_tensor(float(grow)), size)
        widened = torch.cat((centre - grown * 0.5, centre + grown * 0.5), -1)
        inside_wide = (
            (ax > widened[:, 0:1] + eps) & (ax < widened[:, 2:3] - eps)
            & (ay > widened[:, 1:2] + eps) & (ay < widened[:, 3:4] - eps)
        )
        is_p3 = (stride_per_anchor <= 8.5)[None, :]
        inside = inside_orig | (inside_wide & is_p3)
    else:
        inside = inside_orig
    row, anchor = inside.nonzero(as_tuple=True)
    return image[row], box[row], anchor


def mal_head_loss(logits, boxes, obj_logits, spot, entry, target, quality, kept, obj_target, counted, obj_counted,
                  scale, w_cls, w_box, w_l1, w_obj, mal_gamma=0.0):
    """Per-head loss: quality-aligned full-BCE (default mal_gamma=0.0) or negative-modulated MAL + CIoU + GT-norm L1."""
    gamma = float(mal_gamma)
    q_f = quality.float()
    pos_mask = (q_f > 0.0).to(torch.float32)
    norm = q_f.sum().clamp(min=1.0)

    flat = logits.float().reshape(-1)
    at_pos = flat[entry]

    if gamma <= 0.0:
        if counted is None:
            cls = (F.softplus(flat).sum() - (at_pos * q_f).sum()) / norm
        else:
            bg = (F.softplus(flat) * counted).sum()
            cls = (bg - (at_pos * q_f).sum() + (F.softplus(at_pos) * kept.float()).sum()) / norm
    else:
        neg_w = flat.detach().sigmoid().pow(gamma)
        at_pos_neg_w = neg_w[entry]
        if counted is None:
            bg = (neg_w * F.softplus(flat)).sum()
            pos_delta = (pos_mask * ((1.0 - at_pos_neg_w) * F.softplus(at_pos) - at_pos * q_f)).sum()
            cls = (bg + pos_delta) / norm
        else:
            bg = (neg_w * F.softplus(flat) * counted).sum()
            pos_sub = (pos_mask * (1.0 - kept.float()) * at_pos_neg_w * F.softplus(at_pos)).sum()
            pos_add = (pos_mask * (F.softplus(at_pos) - at_pos * q_f)).sum()
            cls = (bg - pos_sub + pos_add) / norm

    pred = xywh_to_xyxy_t(boxes.reshape(-1, 4)[spot].float()) * scale
    box = ((1.0 - bbox_ciou(pred, target)) * q_f).sum() / norm

    # GT-size-normalized L1 (equal relative edge sensitivity across small and large objects)
    gt_w = (target[:, 2] - target[:, 0]).clamp(min=8.0)
    gt_h = (target[:, 3] - target[:, 1]).clamp(min=8.0)
    gt_norm = torch.stack((gt_w, gt_h, gt_w, gt_h), dim=-1)
    gap = torch.cat((target[:, :2] - pred[:, :2], pred[:, 2:] - target[:, 2:]), dim=-1) / gt_norm
    l1 = (gap.abs().clamp(max=4.0).mean(-1) * q_f).sum() / norm

    total = w_cls * cls + w_box * box + w_l1 * l1
    obj = total.new_zeros(())
    if obj_logits is not None and w_obj > 0.0:
        obj_l = F.binary_cross_entropy_with_logits(obj_logits.float(), obj_target, reduction="none")
        if obj_counted is not None:
            obj_l = obj_l * obj_counted
        obj = obj_l.sum() / norm
        total = total + w_obj * obj
    return total, cls.detach(), box.detach(), obj.detach()


def mal_head_loss_o2o(logits, boxes, obj_logits, spot, entry, target, quality, kept, obj_target, counted, obj_counted,
                      scale, w_cls, w_box, w_l1, w_obj, mal_gamma=0.0):
    return mal_head_loss(logits, boxes, obj_logits, spot, entry, target, quality, kept, obj_target, counted, obj_counted,
                         scale, w_cls, w_box, w_l1, w_obj, mal_gamma=mal_gamma)


def mal_head_loss_o2m(logits, boxes, obj_logits, spot, entry, target, quality, kept, obj_target, counted, obj_counted,
                      scale, w_cls, w_box, w_l1, w_obj, mal_gamma=0.0):
    return mal_head_loss(logits, boxes, obj_logits, spot, entry, target, quality, kept, obj_target, counted, obj_counted,
                         scale, w_cls, w_box, w_l1, w_obj, mal_gamma=mal_gamma)


def gt_identity_distillation_v2(logits_o2o, boxes_o2o, logits_o2m, boxes_o2m, scale,
                                assign_o, assign_m, batch, n_gt, nc,
                                w_distill=0.5, w_distill_box=2.0):
    """GT-matched O2M->O2O self-distillation with fixed .sum(-1) class BCE (no 80x attenuation)."""
    spot_o, image_o, box_o, _, quality_o = assign_o
    spot_m, image_m, box_m, _, quality_m = assign_m

    pos_o = quality_o > 0
    with torch.no_grad():
        g_id_m = image_m * n_gt + box_m
        q_m = quality_m.float()
        q_bits = _ordered_bits(q_m)
        best_m_code = torch.zeros(batch * n_gt, dtype=torch.int64, device=spot_m.device)
        best_m_code = best_m_code.scatter_reduce(0, g_id_m, (q_bits << 32) | spot_m, "amax", include_self=True)
        has_teacher = (best_m_code >> 32) > 0
        teacher_spot = best_m_code & 0xFFFFFFFF

        g_id_o = image_o * n_gt + box_o
        valid_match = pos_o & has_teacher[g_id_o]

    s_o = spot_o
    s_m = teacher_spot[g_id_o]
    w_match = torch.where(valid_match, quality_o.float(), quality_o.new_zeros((), dtype=torch.float32))
    norm = w_match.sum().clamp(min=1.0)

    # Fixed .sum(-1) across classes (scaled by 0.25 to balance CIoU box distillation)
    teacher_prob = logits_o2m.reshape(-1, nc)[s_m].float().sigmoid().detach()
    student_logit = logits_o2o.reshape(-1, nc)[s_o].float()
    cls_distill = F.binary_cross_entropy_with_logits(student_logit, teacher_prob, reduction="none").sum(-1)
    loss_cls = (cls_distill * w_match).sum() / norm

    teacher_box = xywh_to_xyxy_t(boxes_o2m.reshape(-1, 4)[s_m].float().detach()) * scale
    student_box = xywh_to_xyxy_t(boxes_o2o.reshape(-1, 4)[s_o].float()) * scale
    box_distill = 1.0 - bbox_ciou(student_box, teacher_box)
    loss_box = (box_distill * w_match).sum() / norm

    return float(w_distill) * (0.25 * loss_cls + float(w_distill_box) * loss_box)


def _gpu_query_1to1_loss(q_logits, q_boxes, gt_boxes_px, gt_labels, gt_mask, scale,
                         *, w_cls, w_box, w_l1, alpha=1.0, beta=6.0, mal_gamma=0.0):
    """Zero-host-sync vectorized 3-pass GPU conflict-free 1-to-1 matching + quality-BCE/CIoU/scale-L1 loss."""
    b, k, nc = q_logits.shape
    m = gt_boxes_px.shape[1]
    logits_f = q_logits.float()
    boxes_px = xywh_to_xyxy_t(q_boxes.float()) * scale  # [B, K, 4]
    gamma = float(mal_gamma)

    if m == 0:
        if gamma <= 0.0:
            cls_loss = F.softplus(logits_f).sum()
        else:
            neg_w = logits_f.detach().sigmoid().pow(gamma)
            cls_loss = (neg_w * F.softplus(logits_f)).sum()
        zero = cls_loss.new_zeros(())
        return w_cls * cls_loss, cls_loss.detach(), zero

    with torch.no_grad():
        probs = logits_f.sigmoid()                                                  # [B, K, C]
        gt_lab_exp = gt_labels.unsqueeze(1).expand(-1, k, -1)                       # [B, K, M]
        cls_p = probs.gather(2, gt_lab_exp)                                         # [B, K, M]
        iou_mat = _pair_iou(boxes_px.unsqueeze(2), gt_boxes_px.unsqueeze(1))        # [B, K, M]
        score = cls_p.pow(float(alpha)) * iou_mat.pow(float(beta)) + 1e-4 * iou_mat

        valid_gt = gt_mask.bool()                                                   # [B, M]
        q_taken = torch.zeros(b, k, dtype=torch.bool, device=q_logits.device)
        g_taken = torch.zeros(b, m, dtype=torch.bool, device=q_logits.device)
        matched_gt = torch.zeros(b, k, dtype=torch.long, device=q_logits.device)
        matched_iou = torch.zeros(b, k, dtype=torch.float32, device=q_logits.device)
        q_idx = torch.arange(k, device=q_logits.device)[None, :, None]
        g_idx = torch.arange(m, device=q_logits.device)[None, None, :]

        # 3-pass vectorized conflict-free 1-to-1 assignment on GPU (zero host sync)
        for _ in range(3):
            avail_mask = (~q_taken[:, :, None]) & (~g_taken[:, None, :]) & valid_gt[:, None, :]
            avail = torch.where(avail_mask, score, score.new_full((), -1.0))
            best_q_val, best_q = avail.max(dim=1)                                   # [B, M]
            claimed = (q_idx == best_q[:, None, :]) & (best_q_val[:, None, :] > 0.0)
            claim_score = torch.where(claimed, avail, avail.new_full((), -1.0))
            win_val, win_g = claim_score.max(dim=2)                                 # [B, K]
            new_match = win_val > 0.0
            q_taken = q_taken | new_match
            matched_gt = torch.where(new_match, win_g, matched_gt)
            win_iou = iou_mat.gather(2, win_g.unsqueeze(-1)).squeeze(-1)
            matched_iou = torch.where(new_match, win_iou, matched_iou)
            # Mark matched GTs as taken
            matched_g_mask = ((g_idx == win_g[:, :, None]) & new_match[:, :, None]).any(dim=1)
            g_taken = g_taken | matched_g_mask

        pos_mask = q_taken.to(torch.float32)                                        # [B, K]
        target_cls = gt_labels.gather(1, matched_gt)                                # [B, K]
        target_boxes = gt_boxes_px.gather(1, matched_gt.unsqueeze(-1).expand(-1, -1, 4))  # [B, K, 4]
        quality = matched_iou * pos_mask

    norm = quality.sum().clamp(min=1.0)
    at_pos = logits_f.gather(2, target_cls.unsqueeze(-1)).squeeze(-1)               # [B, K]
    if gamma <= 0.0:
        # Quality-aligned full-BCE: full 1.0x suppression on all ~285 unmatched duplicate queries
        cls_loss = (F.softplus(logits_f).sum() - (at_pos * quality).sum()) / norm
    else:
        neg_w = logits_f.detach().sigmoid().pow(gamma)
        bg_loss = (neg_w * F.softplus(logits_f)).sum()
        at_pos_neg_w = neg_w.gather(2, target_cls.unsqueeze(-1)).squeeze(-1)
        pos_delta = (pos_mask * ((1.0 - at_pos_neg_w) * F.softplus(at_pos) - at_pos * quality)).sum()
        cls_loss = (bg_loss + pos_delta) / norm

    # Box loss (CIoU + GT-size-normalized L1) weighted by quality (floor 0.1 on matched queries)
    box_w = torch.where(q_taken, quality.clamp(min=0.1), quality.new_zeros(()))
    box_norm = box_w.sum().clamp(min=1.0)
    pred_flat = boxes_px.reshape(-1, 4)
    tgt_flat = target_boxes.reshape(-1, 4)
    ciou_l = ((1.0 - bbox_ciou(pred_flat, tgt_flat)).reshape(b, k) * box_w).sum() / box_norm

    gt_w = (target_boxes[..., 2] - target_boxes[..., 0]).clamp(min=8.0)
    gt_h = (target_boxes[..., 3] - target_boxes[..., 1]).clamp(min=8.0)
    gt_norm = torch.stack((gt_w, gt_h, gt_w, gt_h), dim=-1)
    gap = torch.cat((target_boxes[..., :2] - boxes_px[..., :2], boxes_px[..., 2:] - target_boxes[..., 2:]), dim=-1) / gt_norm
    l1_l = (gap.abs().clamp(max=4.0).mean(-1) * box_w).sum() / box_norm

    total = w_cls * cls_loss + w_box * ciou_l + w_l1 * l1_l
    return total, cls_loss.detach(), ciou_l.detach()


def _cdn_layer_loss(l_logits, l_boxes, g_px, g_lab, valid_f, scale, w_cls, w_box, w_l1, mal_gamma=0.0):
    """Compilable fixed-shape per-layer CDN loss (32 pos + 32 hard-neg)."""
    b, g = g_lab.shape
    gamma = float(mal_gamma)
    lf = l_logits.float()
    pos_l = lf[:, :g, :]                                                        # [B, G, C]
    neg_l = lf[:, g:, :]                                                        # [B, G, C]
    pos_b_px = xywh_to_xyxy_t(l_boxes[:, :g, :].float()) * scale                # [B, G, 4]

    with torch.no_grad():
        iou_pos = _pair_iou(pos_b_px, g_px) * valid_f                           # [B, G]
    norm = valid_f.sum().clamp(min=1.0)

    at_pos = pos_l.gather(2, g_lab.unsqueeze(-1)).squeeze(-1)
    if gamma <= 0.0:
        cls_pos = ((F.softplus(pos_l) * valid_f.unsqueeze(-1)).sum() - (at_pos * iou_pos).sum()) / norm
        cls_neg = (F.softplus(neg_l) * valid_f.unsqueeze(-1)).sum() / norm
    else:
        pos_neg_w = pos_l.detach().sigmoid().pow(gamma)
        bg_pos = (pos_neg_w * F.softplus(pos_l) * valid_f.unsqueeze(-1)).sum()
        at_pos_w = pos_neg_w.gather(2, g_lab.unsqueeze(-1)).squeeze(-1)
        cls_pos = (bg_pos + (valid_f * ((1.0 - at_pos_w) * F.softplus(at_pos) - at_pos * iou_pos)).sum()) / norm
        neg_w = neg_l.detach().sigmoid().pow(gamma)
        cls_neg = (neg_w * F.softplus(neg_l) * valid_f.unsqueeze(-1)).sum() / norm

    # Box CIoU + scale-normalized L1 on valid positive CDN queries
    ciou_l = ((1.0 - bbox_ciou(pos_b_px.reshape(-1, 4), g_px.reshape(-1, 4))).reshape(b, g) * valid_f).sum() / norm
    gt_w = (g_px[..., 2] - g_px[..., 0]).clamp(min=8.0)
    gt_h = (g_px[..., 3] - g_px[..., 1]).clamp(min=8.0)
    gt_norm = torch.stack((gt_w, gt_h, gt_w, gt_h), dim=-1)
    gap = torch.cat((g_px[..., :2] - pos_b_px[..., :2], pos_b_px[..., 2:] - g_px[..., 2:]), dim=-1) / gt_norm
    l1_l = (gap.abs().clamp(max=4.0).mean(-1) * valid_f).sum() / norm

    return w_cls * (cls_pos + cls_neg) + w_box * ciou_l + w_l1 * l1_l


def _cdn_loss(head_ref, hyper_map, hyper_tokens, gt_xywh, gt_boxes_px, gt_labels, gt_mask, scale,
              *, w_cls, w_box, w_l1, mal_gamma=0.0, n_groups=32, forward_fn=None, layer_loss_fn=None):
    """Static-shape training-only Contrastive Denoising (32 pos + 32 hard-neg queries per image)."""
    if head_ref is None or getattr(head_ref, "cdn_label_emb", None) is None:
        zero = gt_xywh.new_zeros(())
        return zero
    b, m, _ = gt_xywh.shape
    device = gt_xywh.device
    nc = int(head_ref.nc)
    g = int(n_groups)

    with torch.no_grad():
        # Pad or slice GT to fixed size g=32 (prepare already sorted real boxes to the front)
        if m >= g:
            g_xywh = gt_xywh[:, :g].float()
            g_px = gt_boxes_px[:, :g].float()
            g_lab = gt_labels[:, :g].long()
            g_valid = gt_mask[:, :g].bool()
        else:
            pad = g - m
            g_xywh = F.pad(gt_xywh.float(), (0, 0, 0, pad))
            g_px = F.pad(gt_boxes_px.float(), (0, 0, 0, pad))
            g_lab = F.pad(gt_labels.long(), (0, pad))
            g_valid = F.pad(gt_mask.bool(), (0, pad))

        # Positive noisy queries: +-15% centre shift, [0.85, 1.15] scale jitter, 10% label flip
        pos_shift = (torch.rand(b, g, 2, device=device) * 2.0 - 1.0) * 0.15
        pos_scale = 0.85 + torch.rand(b, g, 2, device=device) * 0.30
        pos_cxcy = (g_xywh[..., :2] + pos_shift * g_xywh[..., 2:]).clamp(0.0, 1.0)
        pos_wh = (g_xywh[..., 2:] * pos_scale).clamp(1e-3, 1.0)
        pos_boxes = torch.cat((pos_cxcy, pos_wh), dim=-1)

        flip = torch.rand(b, g, device=device) < 0.10
        rand_cls = torch.randint(0, nc, (b, g), device=device)
        pos_in_lab = torch.where(flip, rand_cls, g_lab)
        pos_in_lab = torch.where(g_valid, pos_in_lab, torch.full_like(pos_in_lab, nc))

        # Hard-negative queries: shifted 0.55..1.0 * wh away (near-miss / duplicate traps) -> background target
        sign = torch.where(torch.rand(b, g, 2, device=device) < 0.5, -1.0, 1.0)
        neg_shift = sign * (0.55 + torch.rand(b, g, 2, device=device) * 0.45)
        neg_cxcy = (g_xywh[..., :2] + neg_shift * g_xywh[..., 2:]).clamp(0.0, 1.0)
        neg_wh = (g_xywh[..., 2:] * (0.8 + torch.rand(b, g, 2, device=device) * 0.4)).clamp(1e-3, 1.0)
        neg_boxes = torch.cat((neg_cxcy, neg_wh), dim=-1)
        neg_in_lab = torch.where(g_valid, g_lab, torch.full_like(g_lab, nc))

        dn_boxes = torch.cat((pos_boxes, neg_boxes), dim=1)                         # [B, 2G, 4]
        dn_labels = torch.cat((pos_in_lab, neg_in_lab), dim=1)                      # [B, 2G]

        # Block-diagonal contrastive mask: pos i and neg i attend only to each other (no cross-GT leakage)
        idx = torch.arange(g, device=device)
        pair_id = torch.cat((idx, idx), dim=0)                                      # [2G]
        same_group = pair_id[:, None] == pair_id[None, :]                           # [2G, 2G]
        attn_mask = torch.where(same_group, 0.0, -1e4).to(hyper_map.dtype)[None, None, :, :]

    fwd = head_ref.forward_cdn if forward_fn is None else forward_fn
    dec_logits, dec_boxes = fwd(dn_boxes, dn_labels, hyper_map, hyper_tokens, attn_mask)
    if not dec_logits:
        return gt_xywh.new_zeros(())

    valid_f = g_valid.to(torch.float32)                                             # [B, G]
    total_cdn = gt_xywh.new_zeros(())
    n_layers = float(len(dec_logits))
    l_fn = _cdn_layer_loss if layer_loss_fn is None else layer_loss_fn

    for l_logits, l_boxes in zip(dec_logits, dec_boxes):
        total_cdn = total_cdn + l_fn(l_logits, l_boxes, g_px, g_lab, valid_f, scale, w_cls, w_box, w_l1, mal_gamma) / n_layers

    return total_cdn


class HybridTalCriterion:
    """Criterion for fotonet2n (Hi Bird): MAL + GeoSA GPU 1-to-1 + CDN + Self-Distillation."""

    def __init__(self, cfg, strides, head_ref=None):
        self.cfg = cfg
        self.strides = tuple(float(s) for s in strides)
        self.head_ref = head_ref
        self.head_loss = mal_head_loss
        self.head_loss_o2o = mal_head_loss_o2o
        self.head_loss_o2m = mal_head_loss_o2m
        self.distill_loss = gt_identity_distillation_v2
        self.dup_loss = fixed_gpu_nms_duplicate_penalty
        self.query_loss = _gpu_query_1to1_loss
        self.cdn_forward = None
        self.cdn_layer_loss = _cdn_layer_loss
        self._stride_cache: dict[tuple, torch.Tensor] = {}

    def _stride_vector(self, anchors: torch.Tensor, height: float, width: float) -> torch.Tensor:
        key = (int(anchors.shape[0]), int(height), int(width), str(anchors.device))
        vec = self._stride_cache.get(key)
        if vec is not None:
            return vec
        parts = []
        for s in self.strides:
            ny, nx = int(round(height / s)), int(round(width / s))
            parts.append(torch.full((ny * nx,), s, device=anchors.device, dtype=torch.float32))
        cat = torch.cat(parts, dim=0)
        if cat.shape[0] != anchors.shape[0]:
            cat = torch.full((anchors.shape[0],), self.strides[0], device=anchors.device, dtype=torch.float32)
        self._stride_cache[key] = cat
        return cat

    @staticmethod
    def _crowd(ignore, anchors, scale, nc):
        if ignore is None:
            return None
        boxes, classes, valid = ignore
        batch, n_anchors = boxes.shape[0], anchors.shape[0]
        counted = torch.ones(batch * n_anchors * int(nc), dtype=torch.bool, device=anchors.device)
        anchors_in = torch.zeros(batch * n_anchors, dtype=torch.bool, device=anchors.device)
        image, column = valid.nonzero(as_tuple=True)
        if image.numel() > 0:
            xyxy = xywh_to_xyxy_t(boxes[image, column].float()) * scale
            ax, ay = anchors[:, 0][None], anchors[:, 1][None]
            inside = (ax >= xyxy[:, 0:1]) & (ax <= xyxy[:, 2:3]) & (ay >= xyxy[:, 1:2]) & (ay <= xyxy[:, 3:4])
            row, anchor = inside.nonzero(as_tuple=True)
            spot = image[row] * n_anchors + anchor
            counted[spot * int(nc) + classes[image, column].long().clamp(0, int(nc) - 1)[row]] = False
            anchors_in[spot] = True
        return counted, anchors_in.view(batch, n_anchors)

    @torch.no_grad()
    def prepare(self, gt_xywh, labels, mask, height, width, anchors, nc, ignore=None):
        mask = mask.bool()
        if mask.shape[1] > 0:
            order = mask.to(torch.uint8).argsort(dim=1, descending=True, stable=True)
            gt_xywh = gt_xywh.gather(1, order[..., None].expand(-1, -1, 4))
            labels = labels.gather(1, order)
            mask = mask.gather(1, order)
        anchors = anchors.float()
        scale = anchors.new_tensor([float(width), float(height), float(width), float(height)])
        gt_boxes = xywh_to_xyxy_t(gt_xywh.float()) * scale * mask[..., None]
        labels = labels.long().clamp(0, int(nc) - 1)
        small = 16.0 if bool(self.cfg.get("tal_small", True)) else 0.0
        grow = 16.0
        s_vec = self._stride_vector(anchors, height, width)
        pairs = candidate_pairs_stride8(anchors, s_vec, gt_boxes, mask, small=small, grow=grow)
        plan = Plan(anchors, scale, gt_boxes, labels, pairs, self._crowd(ignore, anchors, scale, nc))
        # Stash sorted normalized gt_xywh & mask for GeoSA & CDN without breaking Plan namedtuple
        self._last_gt = (gt_xywh, mask)
        return plan

    def _head(self, out, name, plan, *, topk, keep):
        cfg = self.cfg
        logits, boxes = out[f"pred_logits_{name}"], out[f"pred_boxes_{name}"]
        batch, n_anchors, nc = logits.shape
        obj_logits = None
        w_obj = float(cfg.get("w_obj", 0.0))
        if name == "o2m" and w_obj > 0.0 and out.get("pred_obj_o2m") is not None:
            obj_logits = out["pred_obj_o2m"]
        gt_boxes, labels, crowd = plan.gt_boxes, plan.labels, plan.crowd
        shape = (batch, gt_boxes.shape[1], n_anchors)
        with torch.no_grad():
            image, box, anchor = plan.pairs
            if image.numel() == 0:
                rows = max_positives(shape, topk, keep)
                image = box = anchor = image.new_zeros(rows)
                quality = gt_boxes.new_zeros(rows)
            else:
                chosen, quality = assign_pairs(
                    logits[image, anchor, labels[image, box]].float().sigmoid(),
                    _pair_iou(gt_boxes[image, box], xywh_to_xyxy_t(boxes[image, anchor].float()) * plan.scale),
                    plan.pairs, shape, topk=topk, keep=keep, alpha=float(cfg["alpha"]), beta=float(cfg["beta"]),
                    fixed=True,
                )
                image, box, anchor = image[chosen], box[chosen], anchor[chosen]
            spot = image * n_anchors + anchor
            entry = spot * nc + labels[image, box] if labels.numel() else spot * nc
            target = gt_boxes[image, box] if gt_boxes.numel() else gt_boxes.new_zeros(spot.numel(), 4)
            counted = obj_counted = obj_target = None
            kept = torch.zeros_like(quality)
            if crowd is not None:
                counted = crowd[0]
                kept = (~counted[entry] & (quality > 0.0)).to(quality.dtype)
            if obj_logits is not None:
                obj_target = torch.zeros(batch * n_anchors, dtype=quality.dtype, device=quality.device)
                obj_target = obj_target.index_add_(0, spot, quality).view(batch, n_anchors)
                if crowd is not None:
                    obj_counted = ~(crowd[1] & (obj_target <= 0.0))
        fn = self.head_loss if self.head_loss is not mal_head_loss else (
            self.head_loss_o2o if name == "o2o" else self.head_loss_o2m
        )
        total, cls_val, box_val, obj_val = fn(
            logits, boxes, obj_logits, spot, entry, target, quality, kept, obj_target, counted, obj_counted,
            plan.scale, float(cfg["w_cls"]), float(cfg["w_box"]), float(cfg["w_l1"]), w_obj,
            float(cfg.get("mal_gamma", 0.0)),
        )
        return (total, cls_val, box_val, obj_val), (spot, image, box, anchor, quality)

    def __call__(self, out, gt_xywh, labels, mask, height, width, progress, ignore=None, plan=None):
        cfg = self.cfg
        if plan is None:
            anchors = out["anchor_points"].float() * out["stride_tensor"].float()
            plan = self.prepare(gt_xywh, labels, mask, height, width, anchors, out["pred_logits_o2o"].shape[-1],
                                ignore=ignore)
        gt_xywh_sorted, gt_mask_sorted = getattr(self, "_last_gt", (gt_xywh, mask))
        zero = plan.anchors.new_zeros(())
        w_cls = float(cfg["w_cls"])
        w_box = float(cfg["w_box"])
        w_l1 = float(cfg["w_l1"])
        mal_gamma = float(cfg.get("mal_gamma", 0.0))

        # 1. Stage-0 Dense O2O Proposal Supervision
        (loss_s0, cls_s0, box_s0, _), assign_o = self._head(
            out, "o2o", plan, topk=int(cfg["o2o_candidates"]), keep=int(cfg["o2o_topk"])
        )
        loss_o = loss_s0
        cls_report, box_report = cls_s0, box_s0

        # 2. Stage-1 GeoSA Decoder Deep Supervision (Layer 1 + Layer 2)
        dec_logits = out.get("dec_logits")
        dec_boxes = out.get("dec_boxes")
        w_dec = float(cfg.get("w_dec", 1.0))
        if dec_logits and dec_boxes and w_dec > 0.0:
            dec_total = zero
            n_dec = len(dec_logits)
            # Layer weights emphasize final Layer 2 slightly (0.4, 0.6 for 2 layers)
            layer_weights = [1.0 / n_dec] * n_dec if n_dec != 2 else [0.4, 0.6]
            last_cls, last_box = cls_s0, box_s0
            for lw, dl, db in zip(layer_weights, dec_logits, dec_boxes):
                l_tot, l_cls, l_box = self.query_loss(
                    dl, db, plan.gt_boxes, plan.labels, gt_mask_sorted, plan.scale,
                    w_cls=w_cls, w_box=w_box, w_l1=w_l1,
                    alpha=float(cfg["alpha"]), beta=float(cfg["beta"]), mal_gamma=mal_gamma,
                )
                dec_total = dec_total + lw * l_tot
                last_cls, last_box = l_cls, l_box
            loss_o = 0.5 * loss_s0 + w_dec * dec_total
            cls_report, box_report = last_cls, last_box

        # 3. Training-Only Contrastive Denoising (CDN)
        w_cdn = float(cfg.get("w_cdn", 0.5))
        if w_cdn > 0.0 and self.head_ref is not None and out.get("hyper_map") is not None:
            cdn_l = _cdn_loss(
                self.head_ref, out["hyper_map"], out["hyper_tokens"],
                gt_xywh_sorted, plan.gt_boxes, plan.labels, gt_mask_sorted, plan.scale,
                w_cls=w_cls, w_box=w_box, w_l1=w_l1, mal_gamma=mal_gamma,
                forward_fn=self.cdn_forward, layer_loss_fn=self.cdn_layer_loss,
            )
            loss_o = loss_o + w_cdn * cdn_l

        # 4. Auxiliary Dense O2M-C CNN Branch + Corrected Self-Distillation + GPU NMS Duplicate Trap
        cons = zero
        if out.get("pred_logits_o2m") is not None:
            (loss_m, cls_m, box_m, obj), assign_m = self._head(out, "o2m", plan, topk=int(cfg["topk"]), keep=None)
            share = min(max(float(progress), 0.0), 1.0)
            p_start, p_end = float(cfg["prog_o2m_start"]), float(cfg["prog_o2m_end"])
            if "prog_o2m_floor" in cfg and p_start > 0.0:
                p_end = max(p_end, min(float(cfg["prog_o2m_floor"]), p_start))
            w_m = p_start + (p_end - p_start) * share
            # Never zero out O2O early: active from 0.6 -> 1.0 across the run
            w_o_min = float(cfg.get("prog_o2o_start", 0.6))
            w_o = w_o_min + (float(cfg["prog_total"]) - w_o_min) * share

            o2o_d = float(cfg.get("o2o_distill") or 0.0)
            raw_distill = cfg.get("w_distill")
            raw_dup = cfg.get("w_dup")
            w_distill = float(raw_distill) if (raw_distill is not None and float(raw_distill) > 0.0) else o2o_d
            w_dup = float(raw_dup) if (raw_dup is not None and float(raw_dup) > 0.0) else o2o_d
            if w_distill > 0.0 or w_dup > 0.0:
                loss_distill = zero
                loss_dup = zero
                if w_distill > 0.0:
                    loss_distill = self.distill_loss(
                        out["pred_logits_o2o"], out["pred_boxes_o2o"],
                        out["pred_logits_o2m"], out["pred_boxes_o2m"],
                        plan.scale, assign_o, assign_m,
                        out["pred_logits_o2o"].shape[0], plan.gt_boxes.shape[1], out["pred_logits_o2o"].shape[-1],
                        w_distill=w_distill, w_distill_box=float(cfg.get("w_distill_box", 2.0)),
                    )
                if w_dup > 0.0:
                    loss_dup = self.dup_loss(
                        out["pred_logits_o2o"], out["pred_logits_o2m"], out["pred_boxes_o2m"],
                        plan.scale, assign_o[0], assign_o[4],
                        iou_thresh=float(cfg.get("dup_iou_thresh", 0.5)), w_dup=w_dup,
                    )
                cons = loss_distill + loss_dup
                loss_o = loss_o + cons
        else:
            loss_m, cls_m, box_m, obj = zero, zero, zero, zero
            w_m, w_o = 0.0, float(cfg["prog_total"])

        total = (w_m * loss_m + w_o * loss_o) * float(cfg["loss_scale"])
        return total, (cls_report, box_report, cls_m, box_m, cons.detach(), obj)


__all__ = [
    "HybridTalCriterion",
    "candidate_pairs_stride8",
    "gt_identity_distillation_v2",
    "mal_head_loss",
    "mal_head_loss_o2m",
    "mal_head_loss_o2o",
]
