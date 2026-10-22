"""``Fotonet`` model summary and export benchmarks."""
import os
import torch



def _onnx_cuda_available():
    try:
        import onnxruntime
    except ImportError:
        return False
    return "CUDAExecutionProvider" in onnxruntime.get_available_providers()


class InfoMixin:
    """Model summary and export benchmarks."""

    def info(self, imgsz=640, verbose=True):
        """Deployed parameters and GFLOPs (fused, training-only heads removed).

        For v1.1 models this also reports the YOLO27 cost gate of the tier.
        """
        if self.runtime_format is not None:
            raise RuntimeError("info() needs the PyTorch model, not an exported runtime artifact.")
        from fotonet.versions.v1_1.budget import deploy_copy, measure, measure_module

        raw = self._training_model_before_deploy or self.model
        if getattr(self, "is_v1_1", False):
            report = measure(raw, model_id=self.model_id, imgsz=imgsz)
            out = report.to_dict()
        else:
            params, g_thop, g_full = measure_module(deploy_copy(raw), imgsz)
            gflops = max(g_full, g_thop) if g_thop is not None else g_full
            out = {"model_id": self.model_id, "params": params, "gflops": round(gflops, 4),
                   "gflops_thop": None if g_thop is None else round(g_thop, 4), "gflops_full": round(g_full, 4)}
        if verbose:
            line = f"{out['model_id']}: {out['params']:,} params, {out['gflops']:.3f} GFLOPs @ {imgsz}"
            if "gate_params" in out:
                line += (f" | {out['peer']} gate {out['gate_params']:,} / {out['gate_gflops']:.3f}: "
                         f"{'PASS' if out['passes'] else 'FAIL'}")
            print(line)
        return out

    def benchmark(self, formats=("pytorch", "torchscript", "onnx"), imgsz=640, batch=1, device=None,
                  half=False, runs=50, warmup=10, verbose=True, out_dir=None):
        """Export to each format and time a forward pass of the deployed graph.

        Returns one dict per format: ``format``, ``ok``, ``ms`` (median),
        ``fps``, ``artifact``, ``error``. TensorRT engines are timed with
        ``trtexec`` when it is installed. Exported files are kept in
        ``out_dir`` when given; otherwise they go to a temporary folder that is
        removed afterwards, and ``artifact`` is None.
        """
        import shutil
        import statistics
        import subprocess
        import tempfile
        import time

        device = torch.device(device or self.device)
        h, w = self._normalize_imgsz(imgsz)
        rows = []
        keep = out_dir is not None
        if keep:
            os.makedirs(out_dir, exist_ok=True)
        workdir = os.fspath(out_dir) if keep else tempfile.mkdtemp(prefix="fotonet_bench_")

        def timed(module, use_half, run_device=device):
            x = torch.rand(int(batch), 3, h, w, device=run_device)
            if use_half:
                x = x.half()
            with torch.inference_mode():
                for _ in range(int(warmup)):
                    module(x)
                times = []
                for _ in range(int(runs)):
                    if run_device.type == "cuda":
                        torch.cuda.synchronize(run_device)
                    start = time.perf_counter()
                    module(x)
                    if run_device.type == "cuda":
                        torch.cuda.synchronize(run_device)
                    times.append((time.perf_counter() - start) * 1000.0)
            return statistics.median(times)

        for fmt in formats:
            fmt = str(fmt).lower()
            row = {"format": fmt, "ok": False, "ms": None, "fps": None, "artifact": None, "error": None}
            try:
                if fmt == "pytorch":
                    from fotonet.versions.v1_1.budget import deploy_copy

                    module = deploy_copy(self._training_model_before_deploy or self.model).to(device)
                    use_half = bool(half and device.type == "cuda")
                    if use_half:
                        module = module.half()
                    row["ms"] = timed(module, use_half)
                else:
                    suffix = {"onnx": ".onnx", "torchscript": ".torchscript", "engine": ".engine",
                              "tensorrt": ".engine"}.get(fmt)
                    if suffix is None:
                        raise ValueError(f"benchmark does not time format {fmt!r}")
                    path = os.path.join(workdir, f"{self.model_id}{suffix}")
                    out = self.export(path=path, format=fmt, imgsz=(h, w), batch=batch,
                                      half=bool(half and device.type == "cuda"), device=device)
                    row["artifact"] = out["artifact"]
                    if suffix == ".engine":
                        trtexec = shutil.which("trtexec")
                        if trtexec is None:
                            raise RuntimeError("timing a TensorRT engine needs trtexec")
                        text = subprocess.run(
                            [trtexec, f"--loadEngine={out['artifact']}", "--useCudaGraph", "--noDataTransfers",
                             "--warmUp=500", "--duration=3"], capture_output=True, text=True, timeout=300,
                        ).stdout
                        line = next(l for l in text.splitlines() if "GPU Compute Time" in l)
                        row["ms"] = float(line.split("median = ")[1].split(" ms")[0])
                    else:
                        run_device = device
                        if fmt == "onnx" and device.type == "cuda" and not _onnx_cuda_available():
                            # A CPU-only onnxruntime cannot run on the GPU; time it on the CPU and say so.
                            run_device = torch.device("cpu")
                        row["device"] = str(run_device)
                        runtime = type(self)(out["artifact"], device=str(run_device))
                        row["ms"] = timed(runtime.model, bool(half and run_device.type == "cuda"), run_device)
                row["ok"] = True
                row["fps"] = round(1000.0 * int(batch) / row["ms"], 1)
                row["ms"] = round(row["ms"], 3)
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
            if verbose:
                status = f"{row['ms']} ms, {row['fps']} img/s" if row["ok"] else f"failed ({row['error']})"
                where = f" on {row['device']}" if row.get("device") and row["device"] != str(device) else ""
                print(f"{fmt:12s} {status}{where}")
        if not keep:
            # Exports are tens of MB each; do not leave them behind in the temp folder.
            # Release runtimes first: Windows will not delete a file that is still open.
            import gc

            gc.collect()
            shutil.rmtree(workdir, ignore_errors=True)
            for row in rows:
                row["artifact"] = None
        return rows
