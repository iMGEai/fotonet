"""Detection heads: deployed one-to-one head plus training-only one-to-many head, objectness, and O2O groups."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv


FEATURE_STRIDES = (8, 16, 32)


def _clip_norm_xywh(boxes: torch.Tensor) -> torch.Tensor:
    """Clip a normalized cxcywh box by clipping its xyxy corners to the image."""
    cx, cy, w, h = boxes.unbind(-1)
    x1 = (cx - w * 0.5).clamp(0.0, 1.0)
    y1 = (cy - h * 0.5).clamp(0.0, 1.0)
    x2 = (cx + w * 0.5).clamp(0.0, 1.0)
    y2 = (cy + h * 0.5).clamp(0.0, 1.0)
    return torch.stack((
        (x1 + x2) * 0.5,
        (y1 + y2) * 0.5,
        (x2 - x1).clamp(min=0.0),
        (y2 - y1).clamp(min=0.0),
    ), -1)


class TaskHead(nn.Module):
    """Per-level regression and classification towers with their predictors."""

    def __init__(self, in_channels, reg_hidden, cls_hidden, nc):
        super().__init__()
        self.reg_stem = nn.ModuleList(
            nn.Sequential(Conv(c, reg_hidden, 3), Conv(reg_hidden, reg_hidden, 3)) for c in in_channels
        )
        self.reg_pred = nn.ModuleList(nn.Conv2d(reg_hidden, 4, 1) for _ in in_channels)
        self.cls_stem = nn.ModuleList(
            nn.Sequential(Conv(c, cls_hidden, 3), Conv(cls_hidden, cls_hidden, 1)) for c in in_channels
        )
        self.cls_pred = nn.ModuleList(nn.Conv2d(cls_hidden, nc, 1) for _ in in_channels)

    def towers(self, i, feat):
        return self.reg_stem[i](feat), self.cls_stem[i](feat)


class PredictorPair(nn.Module):
    """Extra class and box predictors on shared towers (training-only O2O groups)."""

    def __init__(self, levels, reg_hidden, cls_hidden, nc):
        super().__init__()
        self.reg_pred = nn.ModuleList(nn.Conv2d(reg_hidden, 4, 1) for _ in range(levels))
        self.cls_pred = nn.ModuleList(nn.Conv2d(cls_hidden, nc, 1) for _ in range(levels))


def _round8(value):
    return max(8, int(math.ceil(float(value) / 8.0)) * 8)


class DetectHead(nn.Module):
    """Deployed O2O head plus training-only O2M head, objectness, and O2O groups."""

    def __init__(self, in_channels, nc=80, hidden=(48, 96), strides=FEATURE_STRIDES, imgsz=640,
                 o2m_mult=1.5, o2o_groups=1, aux_objectness=True):
        super().__init__()
        self.nc = int(nc)
        self.reg_max = 1
        self.imgsz = int(imgsz)
        self.strides = tuple(int(s) for s in strides)
        if len(self.strides) != len(in_channels):
            raise ValueError("Number of head strides must match number of feature maps.")
        reg_hidden, cls_hidden = (int(v) for v in hidden)
        self.detach_o2o = True
        self.o2o = TaskHead(in_channels, reg_hidden, cls_hidden, self.nc)
        m_reg, m_cls = _round8(reg_hidden * o2m_mult), _round8(cls_hidden * o2m_mult)
        self.o2m = TaskHead(in_channels, m_reg, m_cls, self.nc)
        self.aux_obj = (
            nn.ModuleList(nn.Conv2d(m_cls, 1, 1) for _ in in_channels) if aux_objectness else None
        )
        self.o2o_groups = nn.ModuleList(
            PredictorPair(len(in_channels), reg_hidden, cls_hidden, self.nc)
            for _ in range(max(int(o2o_groups), 1) - 1)
        )
        self.init_biases()

    @property
    def has_o2m(self) -> bool:
        return self.o2m is not None

    def strip_o2m_for_inference(self):
        self.o2m = None
        self.aux_obj = None
        self.o2o_groups = nn.ModuleList()
        self._anchor_cache = None
        return self

    def init_biases(self):
        cls_preds = list(self.o2o.cls_pred)
        reg_preds = list(self.o2o.reg_pred)
        if self.o2m is not None:
            cls_preds += list(self.o2m.cls_pred)
            reg_preds += list(self.o2m.reg_pred)
        for group in self.o2o_groups:
            cls_preds += list(group.cls_pred)
            reg_preds += list(group.reg_pred)
        if self.aux_obj is not None:
            cls_preds += list(self.aux_obj)
        # Predictor 1x1 convs start near zero (std 0.01, as in RetinaNet): fan-out
        # Kaiming on a 1- or 4-channel output gives weights ~1.4 and initial
        # logits in the hundreds, which saturates the losses from step one.
        for module in cls_preds + reg_preds:
            nn.init.normal_(module.weight, std=0.01)
        for module in cls_preds:
            nn.init.constant_(module.bias, -4.595)   # initial foreground confidence ~1%
        for module in reg_preds:
            nn.init.constant_(module.bias, 1.85)     # softplus(1.85) ~ 2 grid cells

    @staticmethod
    def _grid(nx, ny, device, dtype):
        # Built in float32 and cast: ONNX Range has no float16 form, so an FP16
        # export would otherwise be invalid.
        yv, xv = torch.meshgrid(
            torch.arange(ny, device=device, dtype=torch.float32) + 0.5,
            torch.arange(nx, device=device, dtype=torch.float32) + 0.5,
            indexing="ij",
        )
        return torch.stack((xv, yv), 2).view(1, -1, 2).to(dtype)

    @staticmethod
    def _dist2bbox(ltrb, anchor, stride, imgsz):
        """LTRB distances in cell units to normalized cxcywh. imgsz is (width, height) or one side."""
        lt, rb = ltrb.chunk(2, -1)
        x1y1 = (anchor - lt) * stride
        x2y2 = (anchor + rb) * stride
        c_xy = (x1y1 + x2y2) * 0.5
        wh = (x2y2 - x1y1).clamp(min=0.0)
        if isinstance(imgsz, (tuple, list)):
            width, height = float(imgsz[0]), float(imgsz[1])
            scale = ltrb.new_tensor([width, height, width, height])
        elif torch.is_tensor(imgsz):
            size = imgsz.to(device=ltrb.device, dtype=ltrb.dtype).reshape(-1)
            # [w, h, w, h] by concatenating slices: CoreML cannot stack 0-d tensors.
            width, height = size[:1], size[-1:]
            scale = torch.cat((width, height, width, height))
        else:
            side = ltrb.new_tensor(float(imgsz))
            scale = torch.stack((side, side, side, side))
        return torch.cat((c_xy, wh), -1) / scale

    def _decode(self, cls_raw, reg_raw, grid, stride, imgsz):
        b = cls_raw.shape[0]
        logits = cls_raw.reshape(b, self.nc, -1).permute(0, 2, 1)
        dist = F.softplus(reg_raw.reshape(b, 4, -1)).permute(0, 2, 1)
        return logits, self._dist2bbox(dist, grid, stride, imgsz)

    def _eval_anchors(self, feats, imgsz):
        f0 = feats[0]
        w, h = (float(imgsz[0]), float(imgsz[1])) if isinstance(imgsz, (tuple, list)) else (float(imgsz), float(imgsz))
        key = (tuple((int(f.shape[-2]), int(f.shape[-1])) for f in feats), f0.device, f0.dtype, w, h)
        cache = getattr(self, "_anchor_cache", None)
        if cache is None or cache[0] != key:
            grids = [self._grid(f.shape[-1], f.shape[-2], f0.device, f0.dtype) for f in feats]
            strides = [
                torch.full((1, f.shape[-2] * f.shape[-1], 1), float(s), device=f0.device, dtype=f0.dtype)
                for f, s in zip(feats, self.strides)
            ]
            cache = (key, torch.cat(grids, dim=1), torch.cat(strides, dim=1), f0.new_tensor([w, h, w, h]))
            self._anchor_cache = cache
        return cache[1], cache[2], cache[3]

    def forward(self, feats, imgsz=None, use_o2m: bool = True):
        if imgsz is None:
            imgsz = float(self.imgsz)
        if (
            not self.training
            and not torch.onnx.is_in_onnx_export()
            and not torch.jit.is_tracing()
            and not torch.is_tensor(imgsz)
        ):
            anchor, stride, scale = self._eval_anchors(feats, imgsz)
            cls_o2o, reg_o2o = [], []
            for i, feat in enumerate(feats):
                reg_h, cls_h = self.o2o.towers(i, feat)
                cls_o2o.append(self.o2o.cls_pred[i](cls_h).flatten(2))
                reg_o2o.append(self.o2o.reg_pred[i](reg_h).flatten(2))
            logits = torch.cat(cls_o2o, dim=2).permute(0, 2, 1)
            lt, rb = F.softplus(torch.cat(reg_o2o, dim=2)).permute(0, 2, 1).chunk(2, -1)
            x1y1, x2y2 = (anchor - lt) * stride, (anchor + rb) * stride
            boxes = torch.cat(((x1y1 + x2y2) * 0.5, (x2y2 - x1y1).clamp(min=0.0)), -1) / scale
            return torch.cat((logits.clamp(-25.0, 25.0), boxes), dim=-1)
        train = self.training
        run_o2m = bool(train and use_o2m and self.o2m is not None)
        cls_o2o, box_o2o, anchors, stride_rows = [], [], [], []
        cls_o2m, box_o2m, obj_o2m = [], [], []
        groups = [([], []) for _ in self.o2o_groups] if train else []
        for i, feat in enumerate(feats):
            stride = float(self.strides[i])
            grid = self._grid(feat.shape[-1], feat.shape[-2], feat.device, feat.dtype)
            source = feat.detach() if (train and self.detach_o2o) else feat
            reg_h, cls_h = self.o2o.towers(i, source)
            logits, boxes = self._decode(self.o2o.cls_pred[i](cls_h), self.o2o.reg_pred[i](reg_h), grid, stride, imgsz)
            cls_o2o.append(logits)
            box_o2o.append(boxes)
            if not train:
                continue
            anchors.append(grid.squeeze(0))
            stride_rows.append(torch.full(
                (feat.shape[-2] * feat.shape[-1], 1), stride, device=feat.device, dtype=feat.dtype,
            ))
            for (g_cls, g_box), group in zip(groups, self.o2o_groups):
                g_logits, g_boxes = self._decode(group.cls_pred[i](cls_h), group.reg_pred[i](reg_h), grid, stride, imgsz)
                g_cls.append(g_logits)
                g_box.append(g_boxes)
            if run_o2m:
                reg_m, cls_m = self.o2m.towers(i, feat)
                logits_m, boxes_m = self._decode(self.o2m.cls_pred[i](cls_m), self.o2m.reg_pred[i](reg_m), grid, stride, imgsz)
                cls_o2m.append(logits_m)
                box_o2m.append(boxes_m)
                if self.aux_obj is not None:
                    obj_o2m.append(self.aux_obj[i](cls_m).flatten(1))
        pred_logits = torch.cat(cls_o2o, dim=1)
        pred_boxes = torch.cat(box_o2o, dim=1)
        if not train:
            return torch.cat((pred_logits.clamp(-25.0, 25.0), pred_boxes), dim=-1)
        result = {
            "pred_logits_o2o": pred_logits,
            "pred_boxes_o2o": pred_boxes,
            "anchor_points": torch.cat(anchors, dim=0),
            "stride_tensor": torch.cat(stride_rows, dim=0),
        }
        if groups:
            result["pred_logits_o2o_groups"] = [torch.cat(c, dim=1) for c, _ in groups]
            result["pred_boxes_o2o_groups"] = [torch.cat(b, dim=1) for _, b in groups]
        if run_o2m:
            result["pred_logits_o2m"] = torch.cat(cls_o2m, dim=1)
            result["pred_boxes_o2m"] = torch.cat(box_o2m, dim=1)
            if obj_o2m:
                result["pred_obj_o2m"] = torch.cat(obj_o2m, dim=1)
        return result
