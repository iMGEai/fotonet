"""fotonet command line interface.

Examples:
  fotonet train model=fotonete data=datasets/coco/coco.yaml epochs=250 batch=16 imgsz=640
  fotonet train model=fotonetn data=foto8 epochs=30
  fotonet predict model=fotonete source=image.jpg conf=0.25 save=true classes=person,car
  fotonet track model=fotonetn source=street.mp4 tracker=bytetrack
  fotonet val model=fotonete data=datasets/coco/coco.yaml imgsz=640
  fotonet export model=fotonete format=onnx path=fotonet.onnx
  fotonet info model=fotonetn
  fotonet benchmark model=fotonetn formats=pytorch,onnx
  fotonet audit data=datasets/coco/coco.yaml split=train
  fotonet finish run=runs/train/fotonetn data=data.yaml decay_epochs=10
  fotonet distill model=teacher.pt data=data.yaml out=teacher_cache.pt
  fotonet starter name=foto8 data=datasets/coco/coco.yaml
  fotonet parts kind=block
  fotonet nodes package=my_nodes
"""

import ast
import os
import sys
from importlib.metadata import PackageNotFoundError, version


TASKS = {"train", "predict", "track", "val", "export", "info", "benchmark", "audit", "finish",
         "distill", "starter", "parts", "nodes"}
# Tasks that do not load a model first.
_MODEL_FREE = {"audit", "finish", "starter", "parts", "nodes"}


def _coerce(value):
    raw = str(value).strip()
    low = raw.lower()
    if low in {"true", "false"}:
        return low == "true"
    if low in {"none", "null"}:
        return None
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return value


def parse_args(argv):
    args = {}
    task = None
    remaining = list(argv)

    if remaining and "=" not in remaining[0]:
        task = remaining.pop(0).lower()

    for token in remaining:
        if "=" not in token:
            raise SystemExit(f"Invalid argument '{token}'. Use key=value.")
        key, value = token.split("=", 1)
        args[key.replace("-", "_")] = _coerce(value)

    task = str(task or "predict").lower()
    if task not in TASKS:
        raise SystemExit(f"Invalid task. Use one of: {', '.join(sorted(TASKS))}.")
    return task, args


def _pop_bool(args, key, default=False):
    value = args.pop(key, default)
    return bool(value)


def _parse_imgsz(value):
    """Accept the public scalar or (height, width) image-size form."""
    if isinstance(value, int):
        return value
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return int(value[0]), int(value[1])
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit("imgsz must be an integer or an (height,width) pair, e.g. imgsz=(640,960).") from exc


def _print_help():
    print(__doc__.strip())


def _version_text():
    try:
        from fotonet import __version__ as package_version
    except ImportError:
        try:
            package_version = version("fotonet")
        except PackageNotFoundError:
            package_version = "unknown"
    return f"fotonet {package_version}"


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in {"-h", "--help", "help"}:
        _print_help()
        return
    if argv[0] in {"-V", "--version", "version"}:
        print(_version_text())
        return
    from fotonet import Fotonet

    task, args = parse_args(argv)
    if task in _MODEL_FREE:
        return _model_free_task(task, args)
    model_name = args.pop("model", None)
    if not model_name:
        raise SystemExit("Every task requires model=fotonete or model=path/to/checkpoint.pt.")
    explicit_imgsz = "imgsz" in args
    imgsz = _parse_imgsz(args.pop("imgsz", 640))

    # One loader for every model reference. ``Fotonet`` recognizes v1.1 ids
    # and v1.1 checkpoints itself, reading each file once with the safe
    # tensor-only unpickler.
    from fotonet.versions import is_v11_model

    constructor = {}
    if is_v11_model(model_name) and "pretrained" in args:
        # For a v1.1 id this chooses published weights vs. an untrained build.
        constructor["pretrained"] = bool(args.pop("pretrained"))
    if task in ("train", "info", "benchmark"):
        # Training starts from an untrained build, and size and speed do not
        # depend on the weights, so "predictions are meaningless" does not apply.
        import warnings

        from fotonet.engine.model import UntrainedModelWarning

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UntrainedModelWarning)
            model = Fotonet(model_name, **constructor)
    else:
        model = Fotonet(model_name, **constructor)

    if task == "train" and getattr(model, "is_v1_1", False):
        data = args.pop("data", None)
        if not data:
            raise SystemExit("Training requires data=path/to/data.yaml.")
        recipe = args.pop("recipe", None)
        if explicit_imgsz:
            args["imgsz"] = imgsz
        print(f"fotonet train | model={model_name} data={data}" + (f" recipe={recipe}" if recipe else ""))
        if recipe is None:
            result = model.train(data=data, **args)
        else:
            result = model.train_from_recipe(data=data, recipe=recipe, **args)
        if isinstance(result, dict):
            if result.get("best_checkpoint"):
                print(f"best_checkpoint={result['best_checkpoint']}")
            print(f"last_checkpoint={result['last_checkpoint']}")
        return

    if task == "train":
        data = args.pop("data", None)
        if not data:
            raise SystemExit("Training requires data=path/to/data.yaml.")
        recipe = args.pop("recipe", None)
        epochs = int(args.pop("epochs", 100)) if recipe is None else None
        save_path = args.pop("save", "fotonet_trained.pt")
        if recipe is None:
            print(f"fotonet train | model={model_name} data={data} epochs={epochs} imgsz={imgsz}")
            result = model.train(data=data, epochs=epochs, imgsz=imgsz, **args)
        else:
            print(f"fotonet train | model={model_name} data={data} recipe={recipe}")
            result = model.train_from_recipe(data=data, recipe=recipe, imgsz=imgsz, **args)
        if save_path:
            best_path = result.get("best_checkpoint") if isinstance(result, dict) else None
            if best_path:
                print(f"best_checkpoint={best_path}")
            else:
                model.save(str(save_path), inference_only=True, half=True)
                print(f"saved={save_path}")
        return

    if task in {"predict", "track"}:
        source = args.pop("source", None)
        if source is None or source == "":
            raise SystemExit(f"{task} requires source=image.jpg|folder|video|webcam-index.")
        conf = float(args.pop("conf", 0.25))
        save = _pop_bool(args, "save", True)
        show = _pop_bool(args, "show", False)
        if isinstance(args.get("classes"), str):
            args["classes"] = [item.strip() for item in args["classes"].split(",") if item.strip()]
        args.setdefault("project", "runs")
        args.setdefault("name", task)
        runner = model.track if task == "track" else model.predict
        # Streamed, so a long video never accumulates every result in memory.
        results = runner(source, imgsz=imgsz, conf=conf, stream=True, save=save, **args)
        count = 0
        for result in results:
            if show:
                result.show()
            count += 1
        print(f"fotonet {task} | model={model_name} source={source} images={count}")
        if save:
            print(f"saved_dir={getattr(model, 'last_save_dir', None)}")
        return

    if task == "info":
        model.info(imgsz=imgsz)
        return

    if task == "benchmark":
        formats = args.pop("formats", "pytorch,torchscript,onnx")
        if isinstance(formats, str):
            formats = [item.strip() for item in formats.split(",") if item.strip()]
        model.benchmark(formats=formats, imgsz=imgsz, **args)
        return

    if task == "distill":
        from fotonet.train.distill import build_cache

        data = args.pop("data", None)
        if not data:
            raise SystemExit("distill requires data=path/to/data.yaml.")
        out = args.pop("out", "teacher_cache.pt")
        print(f"teacher_cache={build_cache(model, data, out, imgsz=imgsz, **args)}")
        return

    if task == "val":
        data = args.pop("data", None)
        if not data:
            raise SystemExit("Validation requires data=path/to/data.yaml.")
        metrics = model.val(data=data, imgsz=imgsz, **args)
        print("fotonet val")
        for key, value in metrics.items():
            if key.endswith("per_class"):
                continue
            print(f"{key}={value}")
        return

    if task == "export":
        fmt = str(args.pop("format", "onnx")).lower()
        path = args.pop("path", f"fotonet_export.{fmt}")
        out = model.export(path=path, format=fmt, imgsz=imgsz, **args)
        print(f"exported={out['artifact']}")
        print(f"metadata={out['metadata']}")


def _model_free_task(task, args):
    if task == "audit":
        import json

        from fotonet.data.audit import audit_dataset

        data = args.pop("data", None)
        if not data:
            raise SystemExit("audit requires data=path/to/data.yaml.")
        print(json.dumps(audit_dataset(data, **args), indent=2, default=str))
        return
    if task == "finish":
        from fotonet.versions.v1_1.engine import finish

        run = args.pop("run", None)
        if not run:
            raise SystemExit("finish requires run=<run folder or *_last.pt>.")
        result = finish(run, decay_epochs=int(args.pop("decay_epochs", 10)), **args)
        print(f"last_checkpoint={result['last_checkpoint']}")
        if result.get("best_checkpoint"):
            print(f"best_checkpoint={result['best_checkpoint']}")
        return
    if task == "starter":
        from fotonet.data.starter import build_starter

        data = args.pop("data", None)
        if not data:
            raise SystemExit("starter requires data=path/to/coco.yaml (the source of the images).")
        print(f"data={build_starter(data, **args)}")
        return
    if task == "parts":
        from fotonet import registry

        kind = args.pop("kind", None)
        for info in registry.parts(kind):
            print(f"{info.ref:<32} {info.kind:<10} {'custom ' + str(info.package) if info.custom else 'built-in'}")
        return
    if task == "nodes":
        from fotonet import custom, registry

        package = args.pop("package", None)
        if not package:
            for name in custom.available():
                print(f"{name}  (installed, not enabled; enable with: fotonet nodes package={name})")
            return
        # Explicit: this imports and runs the package's code.
        custom.enable(str(package))
        failed = 0
        for info in registry.parts(custom=True):
            if info.package != str(package).split(".")[0] or info.kind not in ("block", "backbone", "neck", "head"):
                continue
            try:
                report = custom.check(info.ref)
                print(f"ok    {info.ref}  output {report['output_shape']}  export {report['export'] or '-'}")
            except Exception as exc:
                failed += 1
                print(f"FAIL  {info.ref}  {type(exc).__name__}: {exc}")
        if failed:
            raise SystemExit(f"{failed} node(s) failed their checks.")
        return


if __name__ == "__main__":
    main()
