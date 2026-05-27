import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_ROOT = SCRIPT_DIR.parent
DATASET_PROCESS_ROOT = SKILL_ROOT.parent
AUTO_ANNOTATION_SCRIPT = DATASET_PROCESS_ROOT / "data-auto-annotation" / "scripts" / "sam3-predict.py"
GENERATION_SCRIPT = DATASET_PROCESS_ROOT / "image-dataset-generation" / "scripts" / "run_generation.py"
LOG_DIR: Path | None = None

os.environ.setdefault("PYTHONIOENCODING", "utf-8")


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def _write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _run(cmd: List[str], dry_run: bool = False) -> None:
    print("[pipeline] " + " ".join(cmd))
    if dry_run:
        return
    completed = subprocess.run(cmd, capture_output=True, check=False)
    stdout_text = _decode_bytes(completed.stdout)
    stderr_text = _decode_bytes(completed.stderr)
    _write_command_logs(cmd, stdout_text, stderr_text)
    if stdout_text:
        _safe_print(stdout_text, stream=sys.stdout)
    if stderr_text:
        _safe_print(stderr_text, stream=sys.stderr)
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, cmd, output=completed.stdout, stderr=completed.stderr)


def _training_command(args: argparse.Namespace, training_input: Path) -> List[str]:
    training_script = str(SCRIPT_DIR / "run_yolo_training.py")
    conda_env_name = str(getattr(args, "conda_env_name", "") or "").strip()
    if not conda_env_name:
        return [sys.executable, training_script, "--input", str(training_input)]

    current_env = str(os.environ.get("CONDA_DEFAULT_ENV", "") or "").strip()
    if current_env == conda_env_name:
        return [sys.executable, training_script, "--input", str(training_input)]

    conda_exe = str(os.environ.get("CONDA_EXE", "") or "").strip() or "conda"
    return [
        conda_exe,
        "run",
        "--no-capture-output",
        "-n",
        conda_env_name,
        "python",
        training_script,
        "--input",
        str(training_input),
    ]


def _decode_bytes(value: bytes | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    for encoding in ("utf-8", "gb18030", "gbk"):
        try:
            return value.decode(encoding)
        except UnicodeDecodeError:
            continue
    return value.decode("utf-8", errors="replace")


def _write_command_logs(cmd: List[str], stdout_text: str, stderr_text: str) -> None:
    if LOG_DIR is None:
        return
    stem = _command_log_stem(cmd)
    if not stem:
        return
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    _append_text(LOG_DIR / f"{stem}-stdout.txt", stdout_text)
    _append_text(LOG_DIR / f"{stem}-stderr.txt", stderr_text)
    if stem == "yolo-training":
        _write_clean_yolo_log(stdout_text)


def _append_text(path: Path, text: str) -> None:
    prefix = "\n\n" if path.exists() and path.stat().st_size > 0 else ""
    path.open("a", encoding="utf-8").write(prefix + (text or ""))


def _safe_print(text: str, *, stream: object) -> None:
    try:
        print(text, end="" if text.endswith("\n") else "\n", file=stream)
    except UnicodeEncodeError:
        encoding = getattr(stream, "encoding", None) or "utf-8"
        safe = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
        print(safe, end="" if safe.endswith("\n") else "\n", file=stream)


def _write_clean_yolo_log(stdout_text: str) -> None:
    if LOG_DIR is None:
        return
    clean = _clean_terminal_text(stdout_text)
    selected: list[str] = []
    for line in clean.splitlines():
        stripped = " ".join(line.strip().split())
        stripped = re.sub(r":\s+\d+%.*$", "", stripped)
        if not stripped:
            continue
        lower = stripped.lower()
        if (
            "epoch" in lower and "box_loss" in lower
            or re.match(r"^\d+/\d+\s+", stripped)
            or stripped.startswith("all ")
            or stripped.startswith("person ")
            or stripped.startswith("cigarette ")
            or stripped.startswith("Starting training")
            or stripped.endswith("epochs completed in")
            or "epochs completed in" in stripped
            or stripped.startswith("Results saved to")
        ):
            selected.append(stripped)
    (LOG_DIR / "yolo-training-epochs.txt").write_text("\n".join(selected).strip() + ("\n" if selected else ""), encoding="utf-8")


def _clean_terminal_text(text: str) -> str:
    text = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", text or "")
    text = text.replace("\r", "\n")
    return text


def _command_log_stem(cmd: List[str]) -> str:
    joined = " ".join(str(item) for item in cmd).replace("\\", "/")
    if "sam3-predict.py" in joined:
        return "data-auto-annotation"
    if "run_generation.py" in joined:
        return "image-dataset-generation"
    if "run_yolo_training.py" in joined:
        return "yolo-training"
    return ""


def _initialize_log_dir(project_dir: Path) -> None:
    global LOG_DIR
    LOG_DIR = (project_dir.parent / "logs") if project_dir.name == "training_run" else (project_dir / "logs")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    for name in (
        "data-auto-annotation-stdout.txt",
        "data-auto-annotation-stderr.txt",
        "image-dataset-generation-stdout.txt",
        "image-dataset-generation-stderr.txt",
        "yolo-training-stdout.txt",
        "yolo-training-stderr.txt",
    ):
        (LOG_DIR / name).touch(exist_ok=True)


def _inspect(dataset_root: Path, work_dir: Path, dry_run: bool) -> Dict:
    output = work_dir / "dataset_inspection.json"
    _run([sys.executable, str(SCRIPT_DIR / "inspect_dataset_input.py"), "--dataset-root", str(dataset_root), "--output", str(output)], dry_run)
    if dry_run:
        return {"status": "dry_run", "format": "unknown"}
    result = _load_json(output)
    if result.get("status") != "ok":
        raise ValueError(f"Dataset inspection failed: {result}")
    return result


def _prepare_real_coco(inspection: Dict, work_dir: Path, task_labels: List[str], dry_run: bool) -> Path:
    fmt = inspection.get("format")
    if fmt == "coco":
        return Path(str(inspection["coco_json"])).resolve()

    real_coco = work_dir / "real_coco.json"
    if fmt == "yolo":
        class_names = ",".join(task_labels)
        classes_file = inspection.get("classes_file")
        class_arg = str(classes_file or class_names)
        _run(
            [
                sys.executable,
                str(SCRIPT_DIR / "convert_yolo_to_coco.py"),
                "--images-dir",
                str(inspection["images_dir"]),
                "--labels-dir",
                str(inspection["labels_dir"]),
                "--output",
                str(real_coco),
                "--class-names",
                class_arg,
            ],
            dry_run,
        )
        return real_coco

    if fmt == "unlabeled":
        if not task_labels:
            raise ValueError("Unlabeled datasets require --labels, for example: --labels person bottle")
        _run(
            [
                sys.executable,
                str(AUTO_ANNOTATION_SCRIPT),
                "--input-dir",
                str(inspection["images_dir"]),
                "--text-prompts",
                *task_labels,
                "--source",
                "real",
                "--output",
                str(real_coco),
            ],
            dry_run,
        )
        return real_coco

    raise ValueError(f"Unsupported dataset format: {fmt}. Convert labels to COCO or YOLO first.")


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def _plan_synthetic(
    real_coco: Path,
    task: str,
    generation_prompt: str,
    split_train: float,
    work_dir: Path,
    max_synthetic: int,
    planner_llm: Dict[str, Any],
    dry_run: bool,
) -> Path:
    plan_path = work_dir / "synthetic_plan.json"
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "plan_synthetic_augmentation.py"),
        "--coco",
        str(real_coco),
        "--task",
        task,
        "--split-train",
        str(split_train),
        "--max-synthetic",
        str(max_synthetic),
        "--generation-prompt",
        generation_prompt,
        "--planner",
        "llm",
        "--output",
        str(plan_path),
    ]
    if planner_llm:
        planner_llm_path = work_dir / "planner_llm_config.json"
        _write_json(planner_llm_path, planner_llm)
        cmd.extend(["--llm-config", str(planner_llm_path)])
    _run(cmd, dry_run)
    return plan_path


def _read_recommended_synthetic_count(plan_path: Path) -> int:
    plan = _load_json(plan_path)
    return int(plan.get("recommended_synthetic_count", 0) or 0)


def _generate_images(plan_path: Path, ref_image: Path, output_dir: Path, dry_run: bool) -> Path:
    plan = _load_json(plan_path)
    synthetic_images_root = output_dir / "synthetic_images"
    synthetic_images_root.mkdir(parents=True, exist_ok=True)

    for idx, item in enumerate(plan.get("generation_plan", []), start=1):
        count = int(item.get("count", 0))
        prompt = str(item.get("prompt", "")).strip()
        if count <= 0 or not prompt:
            continue
        scene_dir = synthetic_images_root / f"scene_{idx:02d}"
        scene_dir.mkdir(parents=True, exist_ok=True)
        for n in range(count):
            input_json = output_dir / "generation_inputs" / f"scene_{idx:02d}_{n + 1:05d}.json"
            _write_json(
                input_json,
                {
                    "task": {"input_image": str(ref_image), "prompt": prompt, "output_dir": str(scene_dir)},
                },
            )
            _run([sys.executable, str(GENERATION_SCRIPT), "--input", str(input_json)], dry_run)
    return synthetic_images_root


def _annotate_synthetic(synthetic_images_root: Path, labels: List[str], output_dir: Path, dry_run: bool) -> Path:
    if not labels:
        raise ValueError("Synthetic auto-annotation requires --labels")
    synthetic_coco = output_dir / "synthetic_coco.json"
    _run(
        [
            sys.executable,
            str(AUTO_ANNOTATION_SCRIPT),
            "--input-dir",
            str(synthetic_images_root),
            "--text-prompts",
            *labels,
            "--source",
            "synthetic",
            "--is-synthetic",
            "--output",
            str(synthetic_coco),
        ],
        dry_run,
    )
    return synthetic_coco


def _merge(real_coco: Path, synthetic_coco: Path, real_root: Path, synthetic_root: Path, output_dir: Path, dry_run: bool) -> Path:
    merged_coco = output_dir / "merged_coco.json"
    merged_images = output_dir / "merged_images"
    _run(
        [
            sys.executable,
            str(SCRIPT_DIR / "merge_real_synthetic_coco.py"),
            "--real-coco",
            str(real_coco),
            "--synthetic-coco",
            str(synthetic_coco),
            "--output",
            str(merged_coco),
            "--real-root",
            str(real_root),
            "--synthetic-root",
            str(synthetic_root),
            "--output-image-root",
            str(merged_images),
        ],
        dry_run,
    )
    return merged_coco


def _write_training_input(args: argparse.Namespace, coco_json: Path, dataset_root: Path, output_dir: Path, synthetic_enabled: bool) -> Path:
    input_json = output_dir / "training_input.json"
    payload = {
        "runtime": {"conda_env_name": args.conda_env_name, "enforce_conda_env": args.enforce_conda_env},
        "dataset": {
            "root_dir": str(dataset_root),
            "coco_json": str(coco_json),
            "split": {"train": args.split_train, "val": args.split_val, "test": args.split_test},
            "class_names": args.labels,
            "copy_images": True,
            "synthetic_policy": {
                "enabled": synthetic_enabled,
                "source_field": "source",
                "synthetic_values": ["synthetic", "generated", "gen"],
                "is_synthetic_fields": ["is_synthetic", "synthetic"],
                "val_real_only": True,
                "test_real_only": True,
                "synthetic_to_train_only": True,
            },
        },
        "training": {
            "task": args.training_task,
            "model": args.model,
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": args.device,
            "workers": args.workers,
            "patience": args.patience,
        },
        "output": {"project_dir": str(args.project_dir), "run_name": args.run_name},
    }
    _write_json(input_json, payload)
    return input_json


def _load_input_json(value: str) -> Dict[str, Any]:
    raw = (value or "").strip()
    if not raw:
        return {}
    candidate = Path(raw)
    if candidate.exists() and candidate.is_file():
        return _load_json(candidate)
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("--input-json must be a JSON object or a JSON file path")
    return payload


def _attachment_payload(attachment: Any) -> Dict[str, Any]:
    if isinstance(attachment, dict):
        return dict(attachment)
    return {}


def _attachment_path(item: Dict[str, Any]) -> str:
    return str(item.get("path") or item.get("local_path") or item.get("file_path") or "").strip()


def _first_attachment_dir(attachments: List[Any]) -> str:
    for raw in attachments:
        item = _attachment_payload(raw)
        path = _attachment_path(item)
        if path and Path(path).exists() and Path(path).is_dir():
            return path
    return ""


def _first_attachment_image(attachments: List[Any]) -> str:
    for raw in attachments:
        item = _attachment_payload(raw)
        path = _attachment_path(item)
        mime_type = str(item.get("mime_type") or item.get("content_type") or "").lower()
        name = str(item.get("name") or item.get("filename") or path).lower()
        if path and (mime_type.startswith("image/") or name.endswith(IMAGE_EXTENSIONS) or path.lower().endswith(IMAGE_EXTENSIONS)):
            return path
    return ""


def _coalesce(*values: Any, default: Any = None) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return default


def _bool_value(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _list_value(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    return [item.strip() for item in text.replace("，", ",").replace("、", ",").split(",") if item.strip()]


def _apply_input_json(args: argparse.Namespace) -> argparse.Namespace:
    payload = _load_input_json(args.input_json)
    if not payload:
        return args

    spec = payload.get("spec") if isinstance(payload.get("spec"), dict) else payload
    ctx = spec.get("workflow_context") if isinstance(spec.get("workflow_context"), dict) else {}
    training = spec.get("training") if isinstance(spec.get("training"), dict) else {}
    split = spec.get("split") if isinstance(spec.get("split"), dict) else {}
    dataset = spec.get("dataset") if isinstance(spec.get("dataset"), dict) else {}
    runtime = spec.get("runtime") if isinstance(spec.get("runtime"), dict) else {}
    output = spec.get("output") if isinstance(spec.get("output"), dict) else {}
    attachments = spec.get("attachments") if isinstance(spec.get("attachments"), list) else []

    args.dataset_root = _coalesce(args.dataset_root, dataset.get("root_dir"), ctx.get("dataset_root"), spec.get("dataset_root"), _first_attachment_dir(attachments), default="")
    args.coco_json = _coalesce(args.coco_json, dataset.get("coco_json"), ctx.get("coco_json"), spec.get("coco_json"), default="")
    args.ref_image = _coalesce(args.ref_image, ctx.get("ref_image"), spec.get("ref_image"), spec.get("reference_image"), _first_attachment_image(attachments), default=None)
    args.task = _coalesce(args.task, spec.get("task"), spec.get("task_description"), ctx.get("task"), default="")
    args.generation_prompt = _coalesce(args.generation_prompt, spec.get("generation_prompt"), spec.get("prompt"), ctx.get("generation_prompt"), default="")
    args.labels = args.labels or _list_value(_coalesce(spec.get("labels"), spec.get("class_names"), dataset.get("class_names"), ctx.get("labels"), default=[]))
    args.work_dir = _coalesce(args.work_dir, ctx.get("work_dir"), spec.get("work_dir"), output.get("work_dir"), default="")
    args.project_dir = _coalesce(args.project_dir, output.get("project_dir"), ctx.get("project_dir"), spec.get("project_dir"), default="")
    args.run_name = _coalesce(output.get("run_name"), ctx.get("run_name"), spec.get("run_name"), args.run_name, default="synthetic_train")

    args.skip_generation = args.skip_generation or _bool_value(spec.get("skip_generation"), False)
    args.skip_training = args.skip_training or _bool_value(spec.get("skip_training"), False)
    args.dry_run = args.dry_run or _bool_value(spec.get("dry_run"), False)

    args.timeout = int(_coalesce(spec.get("timeout"), args.timeout, default=args.timeout))
    args.max_synthetic = int(_coalesce(spec.get("max_synthetic"), args.max_synthetic, default=args.max_synthetic))
    args.planner_llm = spec.get("planner_llm") if isinstance(spec.get("planner_llm"), dict) else {}
    args.split_train = float(_coalesce(split.get("train"), dataset.get("split_train"), args.split_train, default=args.split_train))
    args.split_val = float(_coalesce(split.get("val"), dataset.get("split_val"), args.split_val, default=args.split_val))
    args.split_test = float(_coalesce(split.get("test"), dataset.get("split_test"), args.split_test, default=args.split_test))
    args.conda_env_name = _coalesce(runtime.get("conda_env_name"), spec.get("conda_env_name"), args.conda_env_name, default=args.conda_env_name)
    args.enforce_conda_env = args.enforce_conda_env or _bool_value(runtime.get("enforce_conda_env"), False)
    args.training_task = _coalesce(training.get("task"), args.training_task, default=args.training_task)
    args.model = _coalesce(training.get("model"), spec.get("model"), args.model, default=args.model)
    args.epochs = int(_coalesce(training.get("epochs"), spec.get("epochs"), args.epochs, default=args.epochs))
    args.imgsz = int(_coalesce(training.get("imgsz"), spec.get("imgsz"), args.imgsz, default=args.imgsz))
    args.batch = int(_coalesce(training.get("batch"), spec.get("batch"), args.batch, default=args.batch))
    args.device = _coalesce(training.get("device"), spec.get("device"), args.device, default=args.device)
    args.workers = int(_coalesce(training.get("workers"), spec.get("workers"), args.workers, default=args.workers))
    args.patience = int(_coalesce(training.get("patience"), spec.get("patience"), args.patience, default=args.patience))
    return args


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect dataset, optionally generate synthetic data, merge, and train YOLO.")
    parser.add_argument("--input-json", default="", help="Chat/spec JSON string or JSON file path. CLI args override empty fields.")
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--coco-json", default="", help="Optional existing COCO JSON. If provided, skip dataset label conversion/annotation.")
    parser.add_argument("--task", default="", help="Task description, e.g. bottle detection, helmet detection, smoking detection")
    parser.add_argument("--generation-prompt", default="", help="User-provided positive prompt for synthetic image generation")
    parser.add_argument("--labels", nargs="*", default=[], help="Detection labels/classes")
    parser.add_argument("--work-dir", default="")
    parser.add_argument("--project-dir", default="")
    parser.add_argument("--run-name", default="synthetic_train")
    parser.add_argument("--ref-image", default=None, help="Reference image for image generation. Required when planner recommends synthetic images.")
    parser.add_argument("--skip-generation", action="store_true", help="Only prepare real COCO and training input")
    parser.add_argument("--skip-training", action="store_true", help="Do not run training; only write training_input.json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-synthetic", type=int, default=2000)
    parser.set_defaults(planner_llm={})
    parser.add_argument("--split-train", type=float, default=0.7)
    parser.add_argument("--split-val", type=float, default=0.2)
    parser.add_argument("--split-test", type=float, default=0.1)
    parser.add_argument("--conda-env-name", default="yolo")
    parser.add_argument("--enforce-conda-env", action="store_true", default=False)
    parser.add_argument("--training-task", default="detect", choices=["detect", "segment"])
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=8)
    args = parser.parse_args()
    args = _apply_input_json(args)

    if not args.dataset_root:
        raise ValueError("dataset_root is required. Provide it in chat/spec or with --dataset-root.")
    if not args.task:
        raise ValueError("task is required. Provide it in chat/spec or with --task.")
    if not args.work_dir:
        raise ValueError("work_dir is required. Provide it in chat/spec or with --work-dir.")
    if not args.project_dir:
        raise ValueError("project_dir is required. Provide it in chat/spec or with --project-dir.")

    split_sum = args.split_train + args.split_val + args.split_test
    if abs(split_sum - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {split_sum}")

    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    _initialize_log_dir(Path(args.project_dir).resolve())
    dataset_root = Path(args.dataset_root).resolve()

    inspection = _inspect(dataset_root, work_dir, args.dry_run)
    if args.dry_run:
        print("[pipeline] dry-run completed after command planning.")
        return

    if args.coco_json:
        real_coco = Path(args.coco_json).resolve()
        if not real_coco.exists():
            raise FileNotFoundError(f"Provided coco_json does not exist: {real_coco}")
    else:
        real_coco = _prepare_real_coco(inspection, work_dir, args.labels, args.dry_run)
    training_coco = real_coco
    training_root = Path(inspection["images_dir"]).resolve()
    synthetic_policy_enabled = False

    if not args.skip_generation:
        plan_path = _plan_synthetic(
            real_coco,
            args.task,
            args.generation_prompt,
            args.split_train,
            work_dir,
            args.max_synthetic,
            args.planner_llm,
            args.dry_run,
        )
        recommended_count = _read_recommended_synthetic_count(plan_path)
        print(f"[pipeline] recommended_synthetic_count={recommended_count}")
        if recommended_count > 0:
            if not args.ref_image:
                raise ValueError("recommended_synthetic_count > 0, so --ref-image is required unless --skip-generation is set")
            synthetic_root = _generate_images(plan_path, Path(args.ref_image).resolve(), work_dir, args.dry_run)
            synthetic_coco = _annotate_synthetic(synthetic_root, args.labels, work_dir, args.dry_run)
            training_coco = _merge(real_coco, synthetic_coco, Path(inspection["images_dir"]).resolve(), synthetic_root, work_dir, args.dry_run)
            training_root = work_dir / "merged_images"
            synthetic_policy_enabled = True
        else:
            print("[pipeline] Synthetic generation skipped because the planner recommended 0 images.")

    training_input = _write_training_input(args, training_coco, training_root, work_dir, synthetic_policy_enabled)
    print(
        json.dumps(
            {"training_input": str(training_input), "coco_json": str(training_coco), "dataset_root": str(training_root)},
            ensure_ascii=False,
            indent=2,
        )
    )

    if not args.skip_training:
        _run(_training_command(args, training_input), args.dry_run)


if __name__ == "__main__":
    main()
