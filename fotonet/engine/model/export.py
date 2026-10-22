"""``Fotonet`` checkpoint saving and model export."""
import os
import torch


class ExportMixin:
    """Checkpoint saving and model export."""

    @staticmethod
    def _state_dict_for_save(model, half=False, strip_o2m=False):
        state = model.state_dict()
        out = {}
        for k, v in state.items():
            if strip_o2m and (
                k.startswith("head.cls_o2m.")
                or k.startswith("head.reg_o2m.")
                or k.startswith("head.quality_o2m.")
            ):
                continue
            if torch.is_tensor(v):
                v = v.detach().cpu()
                out[k] = v.half() if half and v.dtype.is_floating_point else v
            else:
                out[k] = v
        return out

    def save(self, model_path, inference_only=False, half=False):
        model_path = os.fspath(model_path)
        parent = os.path.dirname(model_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        if getattr(self, "is_v1_1", False):
            # v1.1 keeps its own self-identifying format. ``half`` stores the weights as
            # FP16; the loader casts them back to the model dtype on the way in.
            if self.runtime_format is not None:
                raise RuntimeError("Exported runtime artifacts cannot be saved as checkpoints.")
            from fotonet.versions.v1_1.api import save_payload

            source_model = self.model
            if not inference_only and self._training_model_before_deploy is not None:
                source_model = self._training_model_before_deploy
            if not inference_only and bool(getattr(source_model, "_fused", False)):
                raise RuntimeError(
                    "Refusing to save a fused graph as a trainable checkpoint. "
                    "Use inference_only=True or reload an unfused model first."
                )
            save_payload(
                source_model, model_path, epoch=getattr(self, "loaded_epoch", None),
                inference_only=bool(inference_only), names=self.names, half=bool(half),
            )
            print(f"[INFO] Saved to '{model_path}'.")
            return
        # ``prepare_for_inference`` intentionally keeps an unfused graph for
        # future training. A normal save must serialize that graph, not the
        # disposable fused deployment copy, or a reloaded checkpoint would
        # silently lose BatchNorm training behavior.
        source_model = self.model
        if not inference_only and self._training_model_before_deploy is not None:
            source_model = self._training_model_before_deploy
        raw = source_model._orig_mod if hasattr(source_model, "_orig_mod") else source_model
        if not inference_only and bool(getattr(raw, "_fused", False)):
            raise RuntimeError(
                "Refusing to save a fused graph as a trainable checkpoint. "
                "Use inference_only=True or reload/restore an unfused training model first."
            )
        strip_o2m = bool(inference_only)
        checkpoint = {
            "model": self._state_dict_for_save(raw, half=half, strip_o2m=strip_o2m),
            "checkpoint_format": self.CHECKPOINT_FORMAT,
            "architecture_schema": self.architecture_schema,
            "architecture_fingerprint": self.architecture_fingerprint,
            "training_protocol": None if inference_only else self.TRAINING_PROTOCOL,
            "model_id": self.model_id,
            "nc": self.nc,
            "names": self.names,
            "inference_only": bool(inference_only),
            "stripped_o2m": strip_o2m,
            "has_o2m": not strip_o2m,
            "fused": bool(getattr(raw, "_fused", False)),
            "model_config": getattr(raw, "model_config", self._model_cfg),
            "raw_output": "[B,N,nc+4] class_logits+normalized_xywh",
        }
        torch.save(checkpoint, model_path)
        print(f"[INFO] Saved to '{model_path}'.")

    def export(self, path=None, format="onnx", imgsz=640, batch=1, dynamic=False,
               half=False, simplify=True, int8=False, opset=17, device=None, verify=True, **kwargs):
        """Export an inference graph and return ``{'artifact', 'metadata'}``.

        Returning metadata makes preprocessing and coordinate contracts visible
        to deployment callers instead of burying them beside the file.
        """
        if getattr(self, "is_v1_1", False):
            from fotonet.versions.v1_1.api import export_v11
            return export_v11(
                self, path=path, format=format, imgsz=imgsz, batch=batch, dynamic=dynamic,
                half=half, simplify=simplify, opset=opset, device=device, verify=verify,
                int8=int8, **kwargs,
            )
        from fotonet.export import export_model
        return export_model(
            self,
            path=path,
            format=format,
            imgsz=imgsz,
            batch=batch,
            dynamic=dynamic,
            half=half,
            simplify=simplify,
            int8=int8,
            opset=opset,
            device=device,
            verify=verify,
            **kwargs,
        )
