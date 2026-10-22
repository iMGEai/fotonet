"""The training loop."""
from __future__ import annotations

import gc
import json
import os
import time

import torch
from torch.utils.data import DataLoader

from fotonet.data.augmentations.hyp import build_augment_hyp
from fotonet.data.dataset import build_detection_dataset
from fotonet.engine.runtime import configure_data_worker
from fotonet.registry import lockfile
from fotonet.utils.config import default_class_names
from fotonet.utils.console import (
    AgyProgressBar, format_eta, log_event, print_completion_card, print_epoch_summary, print_train_card,
)

from ..api import V11_FORMAT, load_checkpoint, load_weights, save_payload
from fotonet.train.distill import TeacherCache
from ..graph import build_from_config
from fotonet.data.augmentations.gpu import GpuAugmenter
from fotonet.data.gpu_loader import DecodePrefetcher, GpuMosaicAssembler, collate_geometry
from fotonet.train.losses import loss_fn
from fotonet.train.assign import matcher_fn, o2o_group_targets
from fotonet.train.optim import build_v11_optimizer
from fotonet.train.profiler import StepProfiler, format_summary

from .options import (
    _check_options, _recipe_overrides, _require_square, _resolve_compile, DEFAULTS, select_device,
)
from .schedule import _epoch_ramp, _lr_at, _o2m_enabled, _Schedule, _should_validate
from .data import (
    _close_loader, _collate, _epoch_data_plan, _load_data_cfg, _sample_fraction,
    _SingleClassView, _TrainDataset,
)
from .state import _EMA, _freeze, _resolve_training_model, _run_dir, capture_rng_state, restore_rng_state
from .callbacks import _callbacks, _fire
from .step import _EagerFallback, TrainStep
from .live import LIVE_KEYS as LIVE_KEYS_SET, KnobSchedule, LiveTuner, SCHEDULE_KEYS


def _train(model, model_id, data, recipe, imgsz, resume, **kwargs):
    _check_options(kwargs)
    callbacks = _callbacks(kwargs.pop("callbacks", None))
    cfg = dict(DEFAULTS)
    recipe_cfg = _recipe_overrides(recipe)
    cfg.update({key: value for key, value in recipe_cfg.items() if key not in ("augment_hyp", "knob_schedule")})
    cfg.update({key: value for key, value in kwargs.items() if key in DEFAULTS and value is not None})
    knob_steps = kwargs.get("knob_schedule")
    if knob_steps is None:
        knob_steps = recipe_cfg.get("knob_schedule")
    side = _require_square(imgsz if imgsz else cfg["imgsz"])
    cfg["imgsz"] = side
    if float(cfg.get("w_dfl") or 0.0) != 0.0:
        log_event("w_dfl is ignored: v1.1 direct regression has no distribution-focal term.", level="warn")
    aug_hyp = build_augment_hyp({
        **dict(recipe_cfg.get("augment_hyp") or {}),
        **dict(kwargs.get("augment_hyp") or {}),
    })
    aug_hyp["close_mosaic"] = int(cfg["close_mosaic"])
    # Checked now, so a bad entry fails before any data loads rather than at its epoch.
    schedule = KnobSchedule(knob_steps, cfg, log_event) if knob_steps else None

    dev = select_device(cfg.get("device"))
    if dev.type == "cuda":
        # The host only does bookkeeping here. Small CPU tensor ops every step
        # otherwise wake the intra-op pool, whose idle threads spin: three
        # spinning threads cost ~2.8 cores on a 4-core box.
        torch.set_num_threads(1)
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        if torch.cuda.is_bf16_supported():
            amp_dtype, scaler, precision = torch.bfloat16, None, "cuda bf16"
        else:
            amp_dtype = torch.float16
            scaler = torch.amp.GradScaler("cuda")
            precision = "cuda fp16"
    else:
        amp_dtype, scaler, precision = torch.float32, None, f"{dev.type} fp32"
    use_amp = dev.type == "cuda"
    gpu_aug = bool(cfg["gpu_augment"]) and dev.type == "cuda"
    if bool(cfg["gpu_augment"]) and not gpu_aug:
        log_event("GPU augmentation needs CUDA; using the CPU augmentation path.", level="warn")
    gpu_decode = gpu_aug and bool(cfg["gpu_decode"])
    if gpu_decode and (aug_hyp.get("mixup", 0.0) > 0.0 or aug_hyp.get("copy_paste", 0.0) > 0.0 or cfg["predecode"]):
        log_event("GPU decode does not do mixup, copy-paste, or pre-decode; using the CPU loader.", level="warn")
        gpu_decode = False
    channels_last = bool(cfg["channels_last"]) and dev.type == "cuda"
    jpeg_decoder = str(cfg["jpeg_decoder"]).lower()
    if jpeg_decoder not in ("auto", "workers", "nvjpeg"):
        raise ValueError(f"jpeg_decoder must be 'auto', 'workers', or 'nvjpeg', got {cfg['jpeg_decoder']!r}.")
    if jpeg_decoder == "auto":
        jpeg_decoder = "workers" if int(cfg["workers"]) >= 2 else "nvjpeg"

    train_target, val_target, nc_data, data_names = _load_data_cfg(data)
    single_cls = bool(cfg["single_cls"])
    nc_final = 1 if single_cls else int(cfg["nc"] or nc_data)
    if single_cls:
        names = {0: "object"}
    else:
        names = (
            dict(data_names)
            if data_names and len(data_names) == nc_final
            else default_class_names(nc_final)
        )
    if float(cfg["fraction"]) < 1.0:
        train_target = _sample_fraction(train_target, cfg["fraction"])
        log_event(f"fraction: training on {len(train_target)} images")

    head_options = {
        "o2m_mult": float(cfg["o2m_mult"]),
        "o2o_groups": int(cfg["o2o_groups"]),
        "aux_objectness": float(cfg["w_obj"]) > 0.0,
    }
    model, transfer = _resolve_training_model(model, model_id, nc_final, side, head_options)
    if transfer is not None:
        log_event(
            f"model rebuilt: {len(transfer['transferred'])} tensors transferred, "
            f"{len(transfer['reinitialized'])} initialized fresh (nc={nc_final})"
        )
    model.head.detach_o2o = bool(cfg["o2o_detach"])
    if model.head.detach_o2o and int(cfg["o2m_every_n_steps"] or 1) > 1:
        log_event("o2o_detach: the O2M head is the backbone's only teacher, so it runs every step.")
        cfg["o2m_every_n_steps"] = 1
    model = model.to(dev)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    model.class_names = dict(names)
    model.train()
    frozen = _freeze(model, cfg["freeze"])
    if frozen:
        log_event(f"freeze: {len(frozen)} parameter tensors frozen")

    raw_resume = resume
    resuming = bool(resume) and str(resume).strip().lower() not in {"false", "0", "no"}
    quality_set = any(kwargs.get(key) is not None for key in ("quality_start_epoch", "quality_warmup_epochs"))
    if not resuming and getattr(model, "trained_weights", False) and not quality_set:
        # The quality ramp exists for random-init boxes. A fine-tune or finish()
        # restarts the epoch count, so the ramp would switch quality targets off
        # and matching back to the center prior on an already trained model.
        fresh_heads = bool(transfer and transfer.get("reinitialized"))
        if fresh_heads:
            # Trained features, freshly initialized predictors (new classes, or
            # an inference-only file without the O2M head): a short ramp.
            cfg["quality_start_epoch"], cfg["quality_warmup_epochs"] = 1, 3
            prior_epochs = 2
            log_event("starting from trained weights with new heads: quality targets ramp in over 3 epochs")
        else:
            cfg["quality_start_epoch"], cfg["quality_warmup_epochs"] = 1, 0
            prior_epochs = 0
            log_event("starting from trained weights: IoU-quality targets on from the first epoch")
        if kwargs.get("center_prior_epochs") is None:
            cfg["center_prior_epochs"] = prior_epochs
    out_dir = _run_dir(cfg, model.model_id, resuming)
    os.makedirs(out_dir, exist_ok=True)
    metrics_path = os.path.join(out_dir, "metrics.jsonl")
    last_path = os.path.join(out_dir, f"{model.model_id}_last.pt")
    best_path = os.path.join(out_dir, f"{model.model_id}_best.pt")

    teacher = None
    if cfg["teacher_cache"]:
        teacher = TeacherCache(
            cfg["teacher_cache"], conf=float(cfg["teacher_conf"]),
            weight=float(cfg["teacher_weight"]), nc=None if single_cls else nc_final,
        )
        log_event(f"teacher pseudo-labels: {len(teacher)} images, conf >= {teacher.conf}, weight {teacher.weight}")
    max_images = int(cfg["max_images"] or 0)

    def make_dataset(image_dir, image_size, hyp, augment, labels=None):
        # ``labels``: an already loaded label cache to share instead of reading
        # every label file again (image-size and mosaic-close rebuilds).
        dataset = _TrainDataset(
            image_dir, imgsz=int(image_size), augment=augment, cache_labels=labels is None,
            cache_to_ram=False, augment_hyp=hyp, num_classes=nc_data if single_cls else nc_final,
            gpu_post=bool(augment and gpu_aug), gpu_decode=bool(augment and gpu_decode),
            teacher=teacher, verbose=False,
        )
        dataset.jpeg_decoder = jpeg_decoder
        dataset.mosaic_buffer = max(int(cfg["mosaic_buffer"] or 0), 0)
        if labels is not None:
            dataset.cache = labels
        if max_images and max_images < len(dataset.img_files):
            dataset.img_files = dataset.img_files[:max_images]
        dataset.set_total_epochs(int(cfg["epochs"]))
        return dataset

    ds = make_dataset(train_target, side, aug_hyp, True)
    if bool(cfg["predecode"]):
        from fotonet.data.predecode import ensure_cache
        root = cfg["predecode_dir"] or os.path.join(str(cfg["out_dir"]), "_predecode")
        cache, stats = ensure_cache(
            ds.img_files, root, max_side=int(cfg["predecode_max_side"]),
            cache_code=f"m{int(cfg['predecode_max_side'])}",
        )
        ds._predecode = cache
        log_event(
            f"predecode: {stats['images']} images, {stats['bytes'] / 1e6:.1f} MB, "
            f"{'rebuilt' if stats['rebuilt'] else 'cache hit'}"
        )
    n_train = len(ds.img_files)
    collate = collate_geometry if gpu_decode else _collate

    def make_loader(dataset):
        # Read at every build, so a live change of workers takes effect at the next rebuild.
        workers = min(int(cfg["workers"]), max(os.cpu_count() or 1, 1))
        loader_kwargs = dict(
            batch_size=int(cfg["batch"]), shuffle=True, num_workers=workers,
            pin_memory=(dev.type == "cuda" and not gpu_decode), drop_last=True, collate_fn=collate,
        )
        if workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 2
            # One OpenCV thread per worker: the workers are the parallelism.
            loader_kwargs["worker_init_fn"] = configure_data_worker
        return DataLoader(dataset, **loader_kwargs)

    loader = make_loader(ds)
    if len(loader) < 1:
        raise SystemExit("Training loader is empty. Lower batch or raise max_images.")
    steps_per_epoch = len(loader)
    accum_steps = max(round(float(cfg["nbs"]) / int(cfg["batch"])), 1)
    mmax = int(cfg["mmax"])

    opt = build_v11_optimizer(
        model, name=str(cfg["optimizer"]), lr=float(cfg["lr0"]),
        momentum=float(cfg["momentum"]), weight_decay=float(cfg["weight_decay"]),
        fused=dev.type == "cuda",
    )
    ema = _EMA(model, cfg["ema_decay"], tau=float(cfg["ema_tau"]))
    augmenter = None
    if gpu_aug:
        try:
            augmenter = GpuAugmenter(aug_hyp, dev, side, mmax=mmax, compile_core=bool(cfg["compile_augment"]))
        except Exception as exc:
            log_event(f"GPU augmentation unavailable ({exc}); using CPU augmentation.", level="warn")
            gpu_aug = gpu_decode = False
            ds.gpu_post = ds.gpu_decode = False
            loader = make_loader(ds)
    assembler = GpuMosaicAssembler(dev) if gpu_decode else None

    start_epoch, best_map, best_epoch = 1, -1.0, 0
    if resuming:
        if isinstance(resume, bool) or str(resume).strip().lower() in {"true", "1", "yes"}:
            rpath = last_path if os.path.isfile(last_path) else os.path.join(out_dir, "last.pt")
        else:
            rpath = os.fspath(resume)
        if not os.path.isfile(rpath):
            raise SystemExit(f"Resume checkpoint not found: {rpath}")
        ckpt = load_checkpoint(rpath, map_location=dev)
        if ckpt.get("format") != V11_FORMAT or "optimizer" not in ckpt or "ema" not in ckpt:
            raise SystemExit(f"{rpath} is not a v1.1 training checkpoint (need model, optimizer, and ema).")
        load_weights(model, ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        ema.load_state_dict(ckpt["ema"])
        if scaler is not None and ckpt.get("scaler"):
            scaler.load_state_dict(ckpt["scaler"])
        restore_rng_state(ckpt.get("rng"), augmenter)
        start_epoch = int(ckpt["epoch"]) + 1
        best_map = float(ckpt.get("best_map", -1.0))
        best_epoch = int(ckpt.get("best_epoch", ckpt["epoch"]) or 0)
        if ckpt.get("epochs") and float(cfg["time"] or 0.0) > 0.0:
            # A time-budgeted run keeps the epoch count it was fitted to.
            cfg["epochs"] = int(ckpt["epochs"])
            cfg["time"] = 0.0
        if start_epoch > int(cfg["epochs"]):
            raise SystemExit(f"Resume epoch {start_epoch} is past epochs={cfg['epochs']}.")
        log_event(f"Resumed {rpath} at epoch {start_epoch} (best {best_map:.4f}).")

    compiled = model
    match = matcher_fn
    loss = loss_fn
    use_compile = _resolve_compile(cfg["compile"], dev, steps_per_epoch * int(cfg["epochs"]))
    if use_compile:
        try:
            torch._dynamo.config.recompile_limit = 16
            compiled = _EagerFallback(torch.compile(model, dynamic=False), model, "model")
            match = _EagerFallback(torch.compile(matcher_fn, dynamic=False), matcher_fn, "matcher")
            loss = _EagerFallback(torch.compile(loss_fn, dynamic=False), loss_fn, "loss")
            log_event("torch.compile enabled for the model, matcher, and loss (first steps compile).")
        except Exception as exc:
            log_event(f"torch.compile skipped ({exc}).", level="warn")
            compiled, match, loss = model, matcher_fn, loss_fn
            use_compile = False

    val_loader = None
    val_model = None
    n_val = 0
    if val_target:
        val_ds = build_detection_dataset(
            val_target, imgsz=side, num_classes=nc_data if single_cls else nc_final, augment=False,
            cache_labels=True, verbose=False,
        )
        n_val = len(val_ds.img_files)
        if single_cls:
            val_ds = _SingleClassView(val_ds)
        val_workers = min(int(cfg["val_workers"]), max(os.cpu_count() or 1, 1))
        # Persistent: validation runs every epoch, and starting worker
        # processes again each time costs seconds on Windows.
        val_loader = DataLoader(
            val_ds, batch_size=int(cfg["val_batch"]), shuffle=False,
            num_workers=val_workers, pin_memory=(dev.type == "cuda"), collate_fn=_collate,
            worker_init_fn=configure_data_worker if val_workers > 0 else None,
            persistent_workers=val_workers > 0,
        )
        val_model = build_from_config(model.model_id, model.model_config, nc=nc_final, imgsz=side).to(dev).eval()
        if channels_last:
            val_model = val_model.to(memory_format=torch.channels_last)

    epochs = int(cfg["epochs"])
    plan = _Schedule(cfg, epochs, steps_per_epoch)
    lr0, lrf = float(cfg["lr0"]), float(cfg["lrf"])
    gstep = (start_epoch - 1) * steps_per_epoch
    print_train_card(
        model.model_id, sum(p.numel() for p in model.parameters()), precision,
        n_train, n_val, epochs, side, int(cfg["batch"]), accum_steps,
        str(cfg["optimizer"]), lr0, plan.kind, use_amp,
        "full" if val_loader is not None else "none",
        resume_path=str(raw_resume) if resuming else None,
        recipe_name=str(recipe) if recipe else None,
    )
    if gpu_decode:
        where = "in the workers" if jpeg_decoder == "workers" else "on the GPU"
        log_event(f"data: JPEG decode {where}; mosaic, affine, HSV, and flips on the GPU")
    elif gpu_aug:
        log_event("augmentation: mosaic and mixup on CPU, affine/HSV/flip/BGR on GPU")
    writer = None
    if bool(cfg["tensorboard"]):
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(os.path.join(out_dir, "tensorboard"))
        except Exception as exc:
            log_event(f"TensorBoard unavailable ({exc}).", level="warn")
    _fire(callbacks, "on_train_start", {"out_dir": out_dir, "model": model, "config": dict(cfg)})

    t_run = time.perf_counter()
    stopped_early = False
    time_budget_applied = not float(cfg["time"] or 0.0) > 0.0
    opt.zero_grad(set_to_none=True)

    def rebuild(image_size, mosaic):
        hyp = dict(aug_hyp)
        hyp["mosaic"] = float(mosaic)
        dataset = make_dataset(train_target, image_size, hyp, True, labels=ds.cache)
        if getattr(ds, "_predecode", None) is not None:
            dataset._predecode = ds._predecode
        return dataset, make_loader(dataset)

    prof = StepProfiler(dev, steps=int(cfg["profile_steps"] or 0))
    profile_written = False
    mosaic_closed = False
    decode_targets = teacher is not None or single_cls
    groups_on = int(cfg["o2o_groups"]) > 1
    lr_now = lr0
    # Training parts used by this run, recorded in every checkpoint next to the model's own.
    train_parts = lockfile([loss_fn, matcher_fn, o2o_group_targets, _EMA, _lr_at, _epoch_ramp])
    step = TrainStep(
        cfg=cfg, model=model, compiled=compiled, match=match, loss=loss, opt=opt, ema=ema, scaler=scaler,
        augmenter=augmenter, dev=dev, mmax=mmax, use_compile=use_compile, use_amp=use_amp,
        amp_dtype=amp_dtype, channels_last=channels_last, decode_targets=decode_targets, teacher=teacher,
        single_cls=single_cls, groups_on=groups_on, accum_steps=accum_steps, prof=prof,
    )
    tuner = LiveTuner(out_dir, cfg, aug_hyp, log_event) if bool(cfg["live_tuning"]) else None
    if (tuner is not None or schedule is not None) and use_compile:
        # Live and scheduled changes reach the compiled step as new Python constants; allow more recompiles than the default.
        import importlib
        dynamo_config = importlib.import_module("torch._dynamo").config
        for name in ("recompile_limit", "cache_size_limit"):
            if hasattr(dynamo_config, name):
                setattr(dynamo_config, name, max(int(getattr(dynamo_config, name)), 64))
    epoch = start_epoch
    while epoch <= epochs:
        applied = tuner.poll(epoch) if tuner is not None else {}
        scheduled = schedule.apply(epoch, skip=tuner.pinned if tuner is not None else ()) if schedule is not None else {}
        live_rebuild = False
        if applied or scheduled:
            keys = set(applied) | set(scheduled)
            lr0, lrf = float(cfg["lr0"]), float(cfg["lrf"])
            if keys & SCHEDULE_KEYS:
                plan = _Schedule(cfg, epochs, steps_per_epoch)
                aug_hyp["close_mosaic"] = int(cfg["close_mosaic"])
            if "weight_decay" in keys:
                for group in opt.param_groups:
                    if float(group.get("weight_decay", 0.0)) > 0.0:
                        group["weight_decay"] = float(cfg["weight_decay"])
            if "momentum" in keys:
                for group in opt.param_groups:
                    if "betas" in group:
                        group["betas"] = (float(cfg["momentum"]), group["betas"][1])
                    elif "momentum" in group:
                        group["momentum"] = float(cfg["momentum"])
            if keys & {"ema_decay", "ema_tau"}:
                ema.decay, ema.tau = float(cfg["ema_decay"]), float(cfg["ema_tau"])
            if "o2o_detach" in keys:
                model.head.detach_o2o = bool(cfg["o2o_detach"])
            aug_keys = keys - LIVE_KEYS_SET
            if augmenter is not None and aug_keys:
                augmenter.hyp.update({key: aug_hyp[key] for key in aug_keys})
            live_rebuild = bool(aug_keys) or "workers" in keys
        want = plan.imgsz_for(epoch) if plan.imgsz_for else side
        was_closed = mosaic_closed
        must_rebuild, mosaic_closed, mosaic_p = _epoch_data_plan(
            epoch, ds.imgsz, want, mosaic_closed, plan.close_from, aug_hyp.get("mosaic", 0.0),
        )
        if must_rebuild or live_rebuild:
            # Stop the old loader's persistent workers before the new ones
            # start, and give cached blocks sized for the old image size back:
            # on Windows a full GPU spills into system RAM instead of failing,
            # so the allocator would never free them by itself.
            _close_loader(loader)
            batches = loader = None
            ds, loader = rebuild(want, mosaic_p)
            if dev.type == "cuda":
                gc.collect()
                torch.cuda.empty_cache()
            model.set_imgsz(int(ds.imgsz))
            if augmenter is not None:
                augmenter.imgsz = int(want)
            note = [f"imgsz {want}"]
            if mosaic_closed and not was_closed:
                note.append("mosaic closed")
            log_event(f"epoch {epoch}: {', '.join(note)}")
        ds.set_epoch(epoch - 1)
        quality = _epoch_ramp(epoch - 1, cfg["quality_start_epoch"], cfg["quality_warmup_epochs"])
        qm = torch.tensor(quality, device=dev)
        prior_epochs = int(cfg["center_prior_epochs"] or 0)
        if prior_epochs > 0:
            prior = torch.tensor(1.0 - _epoch_ramp(epoch - 1, 1, prior_epochs), device=dev)
        else:
            prior = torch.tensor(1.0 - quality, device=dev)
        bar = AgyProgressBar(total=len(loader), desc="train", epoch=epoch, total_epochs=epochs, unit="it", window_size=50)
        bar.set_phase(plan.phase(epoch, gstep, cfg["warmup_epochs"]))
        sums = None
        skipped = torch.zeros((), device=dev)
        step_i = 0
        t_ep = time.perf_counter()
        # Without a terminal the bar prints only at epoch end; log a heartbeat
        # with throughput at every 10% of the epoch instead.
        tenth_marks = {max(1, round(len(loader) * k / 10)) for k in range(1, 10)}
        opt.zero_grad(set_to_none=True)
        # GPU decode runs one or two batches ahead on a side stream.
        batches = DecodePrefetcher(loader, assembler, int(ds.imgsz)) if gpu_decode else loader
        for imgs, targets in batches:
            prof.begin_step()
            use_o2m = _o2m_enabled(epoch, step_i, cfg)
            lr_now = plan.lr(gstep, lr0, lrf) * float(cfg["lr_scale"])
            row, dropped, batch_images = step(
                imgs, targets, epoch=epoch, step_i=step_i, n_steps=len(loader), lr=lr_now,
                quality_mix=qm, prior_mix=prior, use_o2m=use_o2m,
            )
            skipped = skipped + dropped
            sums = row if sums is None else sums + row
            step_i += 1
            gstep += 1
            bar.update(1)
            if step_i % 50 == 0:
                bar.set_postfix(loss=f"{float(row[0]):.3f}", cls=f"{float(row[1]):.3f}", box=f"{float(row[2]):.3f}", lr=f"{lr_now:.1e}")
            bar.set_run(gstep, plan.total_steps, (plan.total_steps - gstep) / max(step_i / max(time.perf_counter() - t_ep, 1e-6), 1e-6), time.perf_counter() - t_run)
            if not bar.is_tty and step_i in tenth_marks:
                elapsed = max(time.perf_counter() - t_ep, 1e-6)
                log_event(
                    f"epoch {epoch}: {round(100 * step_i / len(loader))}% ({step_i}/{len(loader)} steps), "
                    f"{step_i / elapsed:.2f} it/s, {step_i * int(cfg['batch']) / elapsed:.0f} img/s"
                )
            prof.end_step(batch_images)
        bar.close()
        if prof.finished and not profile_written:
            summary = prof.write(os.path.join(out_dir, "profile.json"))
            log_event(format_summary(summary))
            profile_written = True
        wall = time.perf_counter() - t_ep
        mean = (sums / max(step_i, 1)).tolist() if sums is not None else [0.0] * 7
        skipped_steps = int(skipped.item())
        if skipped_steps:
            log_event(f"epoch {epoch}: {skipped_steps} non-finite loss step(s) contributed nothing", level="warn")
        entry = {
            "epoch": epoch, "loss": round(mean[0], 4), "cls_loss": round(mean[1], 4),
            "box_loss": round(mean[2], 4), "cls_m_loss": round(mean[3], 4),
            "box_m_loss": round(mean[4], 4), "cons_loss": round(mean[5], 4),
            "obj_loss": round(mean[6], 4),
            "quality_mix": round(quality, 4), "lr": round(lr_now, 8),
            "train_imgsz": int(ds.imgsz), "epoch_seconds": round(wall, 3),
            "it_s": round(step_i / max(wall, 1e-9), 3),
            # Learning-efficiency axis for ablations: images the optimizer has
            # seen so far, resume included (drop_last keeps every batch full).
            "images_seen": int(gstep) * int(cfg["batch"]),
            "elapsed_seconds": round(time.perf_counter() - t_run, 3),
            "skipped_steps": skipped_steps,
        }
        if applied:
            entry["live_changes"] = {key: new for key, (_old, new) in applied.items()}
        if scheduled:
            entry["scheduled_changes"] = {key: new for key, (_old, new) in scheduled.items()}
        is_best = False
        val_metrics = None
        if val_loader is not None and _should_validate(epoch, epochs, cfg["val_period"], plan.val_final):
            from fotonet.engine.validation import evaluate_detection_model
            val_model.load_state_dict(ema.deploy_state(model), strict=True)
            val_metrics = evaluate_detection_model(
                val_model, val_loader, dev, nc_final, class_names=names, progress=False,
                amp=bool(cfg["val_amp"]) and dev.type == "cuda",
                coco_backend="fast" if bool(cfg["fast_val"]) else "pycocotools",
            )
            score = float(val_metrics.get(cfg["best_metric"], 0.0))
            entry[cfg["best_metric"]] = round(score, 4)
            for key in ("mAP50", "mAP75"):
                if key != cfg["best_metric"] and val_metrics.get(key) is not None:
                    entry[key] = round(float(val_metrics[key]), 4)
            if score > best_map:
                best_map, best_epoch = score, epoch
                is_best = True
            _fire(callbacks, "on_val_end", {"epoch": epoch, "metrics": val_metrics})
        with open(metrics_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
        if writer is not None:
            for key, value in entry.items():
                if isinstance(value, (int, float)) and key != "epoch":
                    writer.add_scalar(key, value, epoch)
        resume_state = {"rng": capture_rng_state(augmenter), "best_epoch": best_epoch, "epochs": epochs,
                        "parts": train_parts}
        if scaler is not None:
            resume_state["scaler"] = scaler.state_dict()
        save_payload(
            model, last_path, epoch=epoch, ema=ema.state(), optimizer=opt.state_dict(),
            best_map=best_map, inference_only=False, names=names, extra=resume_state,
        )
        if is_best:
            save_payload(
                model, best_path, epoch=epoch, best_map=best_map, inference_only=False,
                state=ema.deploy_state(model), names=names,
                extra={"metric": cfg["best_metric"], "metric_value": best_map, "weights_source": "ema",
                       "parts": train_parts},
            )
        print_epoch_summary(
            epoch, epochs, wall,
            {"loss": entry["loss"], "cls_loss": entry["cls_loss"], "box_loss": entry["box_loss"], "lr": entry["lr"]},
            val_metrics=val_metrics, is_best=is_best, val_mode="full", regression_name="box",
            run_progress=f"{gstep}/{plan.total_steps} steps",
            run_eta=format_eta((plan.total_steps - gstep) / max(step_i / max(wall, 1e-9), 1e-6)),
        )
        _fire(callbacks, "on_epoch_end", dict(entry))

        if not time_budget_applied:
            time_budget_applied = True
            epoch_cost = time.perf_counter() - t_ep
            remaining = float(cfg["time"]) * 3600.0 - (time.perf_counter() - t_run)
            fitted = max(epoch, epoch + int(remaining // max(epoch_cost, 1e-6)))
            if fitted != epochs:
                epochs = fitted
                plan = _Schedule(cfg, epochs, steps_per_epoch)
                ds.set_total_epochs(epochs)
                log_event(f"time budget {cfg['time']} h: training {epochs} epochs ({epoch_cost:.0f} s each)")
        patience = int(cfg["patience"] or 0)
        if patience and best_epoch and epoch - best_epoch >= patience:
            log_event(f"early stop: no {cfg['best_metric']} improvement for {patience} epochs (best at epoch {best_epoch})")
            stopped_early = True
            break
        epoch += 1

    if prof.enabled and not profile_written and prof.summary()["steps"]:
        # The run ended before profile_steps were collected; keep what there is.
        summary = prof.write(os.path.join(out_dir, "profile.json"))
        log_event(format_summary(summary) + f" (partial: {summary['steps']} steps)")
    if step.gt_overflow:
        log_event(f"{step.gt_overflow} images had more than mmax={mmax} boxes; extras were dropped.")
    if assembler is not None and assembler.cpu_fallbacks:
        log_event(f"{assembler.cpu_fallbacks} JPEGs were decoded on the CPU (unsupported by nvJPEG).")
    if writer is not None:
        writer.close()
    _close_loader(loader)
    _close_loader(val_loader)
    model.eval()
    completed = min(epoch, epochs) - start_epoch + 1
    print_completion_card({
        "training_wall_time_sec": time.perf_counter() - t_run,
        "epochs_completed": completed,
        "epochs_requested": epochs,
        "best_score": best_map,
        "best_metric": cfg["best_metric"],
        "best_checkpoint": best_path if val_loader is not None else None,
        "last_checkpoint": last_path,
        "training_run_id": out_dir,
    })
    result = {
        "best_checkpoint": best_path if val_loader is not None and os.path.isfile(best_path) else None,
        "last_checkpoint": last_path,
        "best_metric": best_map,
        "best_epoch": best_epoch,
        "metrics_path": metrics_path,
        "names": dict(names),
        "out_dir": out_dir,
        "epochs": epochs,
        "stopped_early": stopped_early,
    }
    _fire(callbacks, "on_train_end", dict(result))
    return result
