"""Per-step machinery: NaN-safe gradient accumulation and the torch.compile guard."""
from __future__ import annotations

import torch

from fotonet.utils.console import log_event

from fotonet.train.distill import decode_labels
from fotonet.train.losses import o2o_branch_loss
from fotonet.train.assign import o2o_group_targets
from .data import _m_bucket


def _bank_grads(params, bank, banked, finite):
    """Move one micro-batch's gradients into a flat bank; a non-finite batch adds zeros.

    ``banked`` is ``(seen, valid)``: which parameters received a gradient in
    the window so far, and how many micro-batches were finite. ``finite`` is
    the loss check. Returns ``(bank, banked, ok)``; no host sync.
    """
    grads = [p.grad for p in params]
    seen_now = [g is not None for g in grads]
    seen = seen_now if banked is None else [a or b for a, b in zip(banked[0], seen_now)]
    valid = banked[1] if banked is not None else torch.zeros((), device=finite.device)
    if not any(seen_now):
        return bank, (seen, valid + finite.float()), finite
    flat = torch.cat([
        (g if g is not None else torch.zeros_like(p)).reshape(-1).float() for g, p in zip(grads, params)
    ])
    ok = finite & torch.isfinite(flat).all()
    flat = torch.where(ok, flat, 0.0)
    for p in params:
        p.grad = None
    bank = flat if bank is None else bank.add_(flat)
    return bank, (seen, valid + ok.float()), ok


def _restore_grads(params, bank, seen, scale=None):
    """Put the banked window gradients back on the parameters that received any.

    ``scale`` (a device scalar) turns the sum over the finite micro-batches
    into their mean when some were dropped.
    """
    if bank is None or seen is None:
        return
    if scale is not None:
        bank.mul_(scale)
    targets, sources = [], []
    offset = 0
    for p, got in zip(params, seen):
        n = p.numel()
        if got:
            # empty_like keeps the parameter's memory format (channels_last):
            # fused AdamW wants grads laid out like their parameters.
            p.grad = torch.empty_like(p)
            targets.append(p.grad)
            sources.append(bank[offset:offset + n].view(p.shape))
        offset += n
    if hasattr(torch, "_foreach_copy_"):
        torch._foreach_copy_(targets, sources)
    else:
        for target, source in zip(targets, sources):
            target.copy_(source)


def _shape_key(args, kwargs):
    """Shapes and dtypes of the tensor arguments: what makes torch.compile recompile."""
    def key(value):
        if isinstance(value, torch.Tensor):
            return (tuple(value.shape), value.dtype)
        if isinstance(value, (bool, int, float, str)):
            return value          # compile specializes on Python scalars too (use_o2m)
        if isinstance(value, (list, tuple)):
            return tuple(key(item) for item in value)
        return None

    return tuple(key(arg) for arg in args), tuple(sorted((name, key(value)) for name, value in kwargs.items()))


class _EagerFallback:
    """Call the compiled function; if compiling it fails, use eager for the run.

    ``torch.compile(dynamic=False)`` is lazy and compiles again for every new
    input shape, so the first call at each shape (the first step, and every
    image-size change) is guarded. Errors at a shape that already ran are real
    errors and propagate.
    """

    def __init__(self, compiled, eager, name):
        self.compiled = compiled
        self.eager = eager
        self.name = name
        self.proven = set()
        self.failed = False

    def __call__(self, *args, **kwargs):
        if self.failed:
            return self.eager(*args, **kwargs)
        key = _shape_key(args, kwargs)
        if key in self.proven:
            return self.compiled(*args, **kwargs)
        try:
            out = self.compiled(*args, **kwargs)
        except Exception as exc:
            log_event(f"torch.compile failed for the {self.name} ({type(exc).__name__}: {exc}); using eager.",
                      level="warn")
            self.failed = True
            self.compiled = self.eager
            return self.eager(*args, **kwargs)
        self.proven.add(key)
        return out


class TrainStep:
    """One training micro-step, plus the optimizer step every ``accum_steps``.

    Built once per run from everything that does not change between steps.
    ``__call__`` prepares the batch, runs forward, matching, and the loss,
    backpropagates (dropping non-finite micro-batches without a host sync),
    and steps the optimizer and EMA at the end of each accumulation window.
    """

    def __init__(self, *, cfg, model, compiled, match, loss, opt, ema, scaler, augmenter, dev, mmax,
                 use_compile, use_amp, amp_dtype, channels_last, decode_targets, teacher, single_cls,
                 groups_on, accum_steps, prof):
        self.cfg = cfg
        self.model = model
        self.compiled = compiled
        self.match = match
        self.loss = loss
        self.opt = opt
        self.ema = ema
        self.scaler = scaler
        self.augmenter = augmenter
        self.dev = dev
        self.mmax = mmax
        self.use_compile = use_compile
        self.use_amp = use_amp
        self.amp_dtype = amp_dtype
        self.channels_last = channels_last
        self.decode_targets = decode_targets
        self.teacher = teacher
        self.single_cls = single_cls
        self.groups_on = groups_on
        self.accum_steps = accum_steps
        self.prof = prof
        self.grad_params = [p for p in model.parameters() if p.requires_grad]
        self.bank, self.banked = None, None
        self.gt_overflow = 0

    def __call__(self, imgs, targets, *, epoch, step_i, n_steps, lr, quality_mix, prior_mix, use_o2m):
        """Run one micro-step. Returns ``(loss row, dropped-batch count, batch size)`` as device tensors."""
        images, gtmap, lab, mask, gt_weight = self._batch(imgs, targets)
        self.prof.mark("augment")
        for group in self.opt.param_groups:
            group["lr"] = lr
        total, parts, finite = self._losses(images, gtmap, lab, mask, gt_weight, use_o2m, quality_mix, prior_mix)
        # The last window of an epoch can be short (n_steps not a multiple of
        # accum_steps); average over the batches it really has.
        window_start = (step_i // self.accum_steps) * self.accum_steps
        window = min(self.accum_steps, n_steps - window_start)
        dropped = self._backward(total, finite, window)
        self.prof.mark("backward")
        if (step_i + 1) % self.accum_steps == 0 or (step_i + 1) == n_steps:
            self._optimizer_step(total, epoch, step_i, window)
        self.prof.mark("optimizer")
        row = torch.stack([total.detach(), *parts])
        row = torch.where(finite, row, torch.zeros_like(row))   # keep NaN out of the epoch means
        return row, dropped, images.shape[0]

    def _batch(self, imgs, targets):
        """Device images and padded ``[B, M]`` targets, sliced to the batch's GT count."""
        dev, mmax = self.dev, self.mmax
        if self.augmenter is not None:
            images, boxes, labels, valid = self.augmenter(imgs.to(dev, non_blocking=True), targets)
            self.gt_overflow += sum(1 for target in targets if int(target["boxes"].shape[0]) > mmax)
        else:
            images = imgs.to(dev, non_blocking=True)
            images = images.float().div_(255.0) if images.dtype == torch.uint8 else images.float()
            b = images.shape[0]
            boxes = torch.zeros(b, mmax, 4, device=dev)
            labels = torch.zeros(b, mmax, dtype=torch.long, device=dev)
            valid = torch.zeros(b, mmax, dtype=torch.bool, device=dev)
            for index, target in enumerate(targets):
                count = min(int(target["boxes"].shape[0]), mmax)
                if int(target["boxes"].shape[0]) > mmax:
                    self.gt_overflow += 1
                if count:
                    boxes[index, :count] = target["boxes"][:count].to(dev)
                    labels[index, :count] = target["labels"][:count].to(dev)
                    valid[index, :count] = True
        max_gt = max((int(target["boxes"].shape[0]) for target in targets), default=0)
        m_used = mmax if self.use_compile else _m_bucket(max_gt, mmax)
        gtmap = boxes[:, :m_used]
        lab = labels[:, :m_used]
        mask = valid[:, :m_used]
        gt_weight = None
        if self.decode_targets:
            lab, weights = decode_labels(lab)
            if self.teacher is not None:
                gt_weight = weights
            if self.single_cls:
                lab = torch.zeros_like(lab)
        if self.channels_last:
            images = images.contiguous(memory_format=torch.channels_last)
        return images, gtmap, lab, mask, gt_weight

    def _losses(self, images, gtmap, lab, mask, gt_weight, use_o2m, qm, prior):
        """Forward, matching, and loss. Returns ``(masked total, loss parts, finite flag)``."""
        cfg = self.cfg
        if self.use_amp:
            with torch.autocast(self.dev.type, dtype=self.amp_dtype):
                out = self.compiled(images, use_o2m=use_o2m)
        else:
            out = self.compiled(images, use_o2m=use_o2m)
        self.prof.mark("forward")
        anchors = out["anchor_points"].float()
        strides = out["stride_tensor"].float()
        o2m_logits = out.get("pred_logits_o2m")
        o2m_boxes = out.get("pred_boxes_o2m")
        o2m_obj = out.get("pred_obj_o2m")
        logits_o = out["pred_logits_o2o"].float()
        boxes_o = out["pred_boxes_o2o"].float()
        matched = self.match(
            logits_o, boxes_o, anchors, strides, gtmap, lab, mask, self.model.head.strides,
            float(images.shape[-2]), float(images.shape[-1]),
            None if o2m_logits is None else o2m_logits.float(),
            None if o2m_boxes is None else o2m_boxes.float(),
            topk=int(cfg["topk"]), o2o_topk=int(cfg["o2o_topk"]),
            alpha=float(cfg["alpha"]), beta=float(cfg["beta"]),
            o2o_beta=float(cfg["o2o_beta"]), radius=float(cfg["radius"]),
            o2o_center_prior=float(cfg["o2o_center_prior"]),
            slack_cells=float(cfg["slack_cells"]),
            quality_mix=qm, prior_mix=prior,
            o2o_metric=str(cfg["o2o_metric"]), candidates=int(cfg["matcher_candidates"]),
            gt_weight=gt_weight, return_state=self.groups_on,
        )
        (tgt_m, tgt_o, wgt_m, wgt_o, sb_m, sa_m, sg_m, rw_m, sb_o, sa_o, sg_o, rw_o) = matched[:12]
        values = self.loss(
            logits_o, boxes_o, tgt_o, wgt_o, sb_o, sa_o, sg_o, rw_o, gtmap,
            None if o2m_logits is None else o2m_logits.float(),
            None if o2m_boxes is None else o2m_boxes.float(),
            tgt_m, wgt_m, sb_m, sa_m, sg_m, rw_m,
            w_box=float(cfg["w_box"]), w_o2o=float(cfg["w_o2o"]),
            w_cons=float(cfg["w_cons"]), iou_type=str(cfg["iou_type"]),
            preds_obj_o2m=None if o2m_obj is None else o2m_obj.float(),
            w_obj=float(cfg["w_obj"]), w_runner_up=float(cfg["w_runner_up"]),
            o2o_neg_alpha=float(cfg["o2o_neg_alpha"]), o2o_neg_gamma=float(cfg["o2o_neg_gamma"]),
        )
        total, l_cls, l_box, l_cls_m, l_box_m, l_cons, l_obj = values
        if self.groups_on and "pred_logits_o2o_groups" in out:
            state = matched[12]
            for g_logits, g_boxes in zip(out["pred_logits_o2o_groups"], out["pred_boxes_o2o_groups"]):
                g_logits, g_boxes = g_logits.float(), g_boxes.float()
                g_targets = o2o_group_targets(
                    state, g_logits, g_boxes, o2o_topk=int(cfg["o2o_topk"]),
                    o2o_metric=str(cfg["o2o_metric"]), alpha=float(cfg["alpha"]),
                    beta=float(cfg["beta"]), o2o_beta=float(cfg["o2o_beta"]),
                    o2o_center_prior=float(cfg["o2o_center_prior"]), quality_mix=qm,
                    gt_weight=gt_weight,
                )
                g_loss, _, _ = o2o_branch_loss(
                    g_logits, g_boxes, *g_targets, gtmap, w_box=float(cfg["w_box"]),
                    iou_type=str(cfg["iou_type"]),
                )
                total = total + float(cfg["w_group"]) * g_loss
        self.prof.mark("assign_loss")
        finite = torch.isfinite(total)
        total = torch.where(finite, total, torch.zeros_like(total))
        return total, (l_cls, l_box, l_cls_m, l_box_m, l_cons, l_obj), finite

    def _backward(self, total, finite, window):
        """Backpropagate one micro-batch; returns 1.0 (device) if it was dropped as non-finite."""
        scaled = total / window
        if self.scaler is not None:
            dropped = (~finite).float()
            self.scaler.scale(scaled).backward()
            return dropped
        scaled.backward()
        # Zeroing a non-finite loss is not enough: its backward still
        # multiplies NaN into the gradients. Bank each micro-batch's
        # gradients and drop the ones that are not finite, on the
        # device (no host sync), so one bad batch does not discard the
        # whole accumulation window.
        self.bank, self.banked, ok = _bank_grads(self.grad_params, self.bank, self.banked, finite)
        return (~ok).float()

    def _optimizer_step(self, total, epoch, step_i, window):
        """Clip, step, and update the EMA at the end of an accumulation window."""
        model, opt, scaler = self.model, self.opt, self.scaler
        any_valid = None
        if scaler is not None:
            scaler.unscale_(opt)
        elif self.banked is not None:
            any_valid = self.banked[1] > 0
            _restore_grads(self.grad_params, self.bank, self.banked[0], window / self.banked[1].clamp(min=1.0))
            self.bank, self.banked = None, None
        clip = float(self.cfg["grad_clip"])
        trainable = [p for p in model.parameters() if p.grad is not None]
        if clip > 0:
            gnorm = torch.nn.utils.clip_grad_norm_(trainable, clip)
        else:
            gnorm = torch.norm(torch.stack([p.grad.detach().norm(2) for p in trainable]), 2) if trainable else total.new_zeros(())
        step_ok = torch.isfinite(torch.as_tensor(gnorm))
        if any_valid is not None:
            step_ok = step_ok & any_valid     # every micro-batch was dropped: no step
        if bool(step_ok):
            if scaler is not None:
                scaler.step(opt)
            else:
                opt.step()
            self.ema.update(model)
        else:
            bad = [n for n, p in model.named_parameters()
                   if p.grad is not None and not torch.isfinite(p.grad).all()]
            log_event(
                f"epoch {epoch} step {step_i}: non-finite gradient in {len(bad)} tensor(s) "
                f"({', '.join(bad[:3])}{', ...' if len(bad) > 3 else ''}), step skipped; "
                f"gradient norm {float(gnorm)}",
                level="warn",
            )
        if scaler is not None:
            scaler.update()
        opt.zero_grad(set_to_none=True)
