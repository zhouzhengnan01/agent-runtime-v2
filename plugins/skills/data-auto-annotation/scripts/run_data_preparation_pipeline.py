import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from urllib import request

from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_ROOT = SCRIPT_DIR.parent
DATASET_PROCESS_ROOT = SKILL_ROOT.parent
AUTO_ANNOTATION_SCRIPT = SCRIPT_DIR / "sam3-predict.py"
GENERATION_SCRIPT = DATASET_PROCESS_ROOT / "image-dataset-generation" / "scripts" / "run_composite.py"
PRODUCE_GENERATION_SCRIPT = DATASET_PROCESS_ROOT / "image-dataset-produce" / "scripts" / "run_generation.py"
LOG_DIR: Path | None = None
DEFAULT_ANNOTATION_PROVIDER = os.getenv("ANNOTATION_PROVIDER", "locate_sam3")

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
SAM3_MAX_PROMPTS_PER_LABEL = 1
SAM3_PROMPT_MAX_WORDS = 8
SAM3_INSTRUCTION_KEYWORDS = {
    "sam3",
    "label",
    "labels",
    "class",
    "classes",
    "category",
    "categories",
    "bbox",
    "bounding",
    "box",
    "annotation",
    "annotations",
    "annotate",
    "mapping",
    "mapped",
    "filter",
    "rule",
    "condition",
    "criteria",
    "标注",
    "边界框",
    "映射",
    "筛选",
    "过滤",
    "类别",
    "业务",
    "规则",
    "条件",
    "检测",
}


def _safe_print(message: str, *, file: Any | None = None, end: str = "\n", flush: bool = True) -> None:
    """Best-effort progress output; broken stdout/stderr must not stop data prep."""
    try:
        print(message, end=end, file=file or sys.stdout, flush=flush)
    except OSError:
        return


def _emit_synthetic_generation_progress(status: str, **data: Any) -> None:
    payload = {"status": status, **data}
    _safe_print("[data-prep-event] synthetic_generation " + json.dumps(payload, ensure_ascii=False, default=str))


def _synthetic_generation_terminal_status(payload: Dict[str, Any]) -> str:
    error = "\n".join(
        str(payload.get(key) or "")
        for key in ("synthetic_generation_error", "synthetic_generation_primary_error")
    ).lower()
    if any(marker in error for marker in ("timeout", "timed out", "readtimeout", "connecttimeout", "deadline exceeded", "gateway timeout", "超时")):
        return "timeout"
    return "failed"


def _count_generated_images(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(1 for item in path.rglob("*") if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS)


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def _write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _coco_category_names(coco_path: Path) -> list[str]:
    if not coco_path.is_file():
        return []
    try:
        payload = _load_json(coco_path)
    except Exception:
        return []
    names: list[str] = []
    seen: set[str] = set()
    categories = payload.get("categories") if isinstance(payload, dict) else []
    if not isinstance(categories, list):
        return []
    for category in sorted(
        [item for item in categories if isinstance(item, dict)],
        key=lambda item: int(item.get("id", 0) or 0),
    ):
        name = str(category.get("name") or "").strip()
        if name and name not in seen:
            names.append(name)
            seen.add(name)
    return names


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


def _append_text(path: Path, text: str) -> None:
    prefix = "\n\n" if path.exists() and path.stat().st_size > 0 else ""
    path.open("a", encoding="utf-8").write(prefix + (text or ""))


def _command_log_stem(cmd: List[str]) -> str:
    joined = " ".join(str(item) for item in cmd).replace("\\", "/")
    if "sam3-predict.py" in joined:
        return "data-auto-annotation"
    if "image-dataset-produce" in joined:
        return "image-dataset-produce"
    if "image-dataset-generation" in joined or "run_composite.py" in joined:
        return "image-dataset-generation"
    if "plan_synthetic_augmentation.py" in joined:
        return "synthetic-planner"
    if "prepare_yolo_dataset.py" in joined:
        return "dataset-preparation"
    return ""


def _write_command_logs(cmd: List[str], stdout_text: str, stderr_text: str) -> None:
    if LOG_DIR is None:
        return
    stem = _command_log_stem(cmd)
    if not stem:
        return
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    _append_text(LOG_DIR / f"{stem}-stdout.txt", stdout_text)
    _append_text(LOG_DIR / f"{stem}-stderr.txt", stderr_text)


def _run(cmd: List[str], dry_run: bool = False, *, retries: int = 1, retry_sleep: float = 5.0) -> str:
    _safe_print("[data-prep] " + " ".join(cmd))
    if dry_run:
        return ""
    attempts = max(1, int(retries))
    last_returncode = 1
    last_stdout = b""
    last_stderr = b""
    for attempt in range(1, attempts + 1):
        returncode, stdout_bytes, stderr_bytes = _run_streaming_once(cmd)
        last_returncode = returncode
        last_stdout = stdout_bytes
        last_stderr = stderr_bytes
        stdout_text = _decode_bytes(stdout_bytes)
        stderr_text = _decode_bytes(stderr_bytes)
        _write_command_logs(cmd, stdout_text, stderr_text)
        if returncode == 0:
            return stdout_text
        combined = f"{stdout_text}\n{stderr_text}"
        if attempt >= attempts or not _looks_transient_subprocess_error(combined):
            raise subprocess.CalledProcessError(returncode, cmd, output=stdout_bytes, stderr=stderr_bytes)
        message = f"[data-prep] transient command failure; retrying {attempt + 1}/{attempts} after {retry_sleep}s"
        _safe_print(message)
        _write_command_logs(cmd, message + "\n", "")
        time.sleep(max(0.0, float(retry_sleep)))
    raise subprocess.CalledProcessError(last_returncode, cmd, output=last_stdout, stderr=last_stderr)


def _run_streaming_once(cmd: List[str]) -> tuple[int, bytes, bytes]:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    stdout_chunks: list[bytes] = []
    stderr_chunks: list[bytes] = []

    def pump(stream: Any, chunks: list[bytes], target: Any) -> None:
        try:
            while True:
                chunk = stream.readline()
                if not chunk:
                    break
                chunks.append(chunk)
                text = _decode_bytes(chunk)
                _safe_print(text, end="" if text.endswith("\n") else "\n", file=target)
        finally:
            stream.close()

    threads = [
        threading.Thread(target=pump, args=(proc.stdout, stdout_chunks, sys.stdout), daemon=True),
        threading.Thread(target=pump, args=(proc.stderr, stderr_chunks, sys.stderr), daemon=True),
    ]
    for thread in threads:
        thread.start()
    returncode = proc.wait()
    for thread in threads:
        thread.join()
    return returncode, b"".join(stdout_chunks), b"".join(stderr_chunks)


def _looks_transient_subprocess_error(text: str) -> bool:
    lowered = (text or "").lower()
    markers = (
        "connectionreseterror",
        "connection reset",
        "connection aborted",
        "remote host",
        "远程主机",
        "timeout",
        "timed out",
        "temporarily unavailable",
        "protocolerror",
        "ssl",
    )
    return any(marker in lowered for marker in markers)


def _initialize_log_dir(output_dir: Path) -> None:
    global LOG_DIR
    LOG_DIR = output_dir.parent / "logs" if output_dir.name == "pipeline_work" else output_dir / "logs"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    for name in (
        "data-auto-annotation-stdout.txt",
        "data-auto-annotation-stderr.txt",
        "image-dataset-generation-stdout.txt",
        "image-dataset-generation-stderr.txt",
        "image-dataset-produce-stdout.txt",
        "image-dataset-produce-stderr.txt",
        "synthetic-planner-stdout.txt",
        "synthetic-planner-stderr.txt",
        "dataset-preparation-stdout.txt",
        "dataset-preparation-stderr.txt",
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


def _decode_image_error(image_path: Path) -> str | None:
    try:
        with Image.open(image_path) as image:
            image.load()
        return None
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def _collect_images_for_validation(images_dir: Path) -> list[Path]:
    if not images_dir.is_dir():
        return []
    return sorted(path for path in images_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def _filter_coco_for_valid_images(coco_path: Path, original_images_dir: Path, valid_images: list[Path], output_path: Path) -> Path:
    coco = _load_json(coco_path)
    valid_rel = {path.relative_to(original_images_dir).as_posix() for path in valid_images}
    valid_names = {path.name for path in valid_images}

    kept_images: list[dict[str, Any]] = []
    kept_ids: set[int] = set()
    for image in coco.get("images", []):
        if not isinstance(image, dict):
            continue
        file_name = str(image.get("file_name") or "").replace("\\", "/").strip()
        if not file_name:
            continue
        if file_name in valid_rel or Path(file_name).name in valid_names:
            kept_images.append(image)
            try:
                kept_ids.add(int(image.get("id")))
            except (TypeError, ValueError):
                continue

    kept_annotations: list[dict[str, Any]] = []
    for annotation in coco.get("annotations", []):
        if not isinstance(annotation, dict):
            continue
        try:
            image_id = int(annotation.get("image_id"))
        except (TypeError, ValueError):
            continue
        if image_id in kept_ids:
            kept_annotations.append(annotation)

    filtered = dict(coco)
    filtered["images"] = kept_images
    filtered["annotations"] = kept_annotations
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output_path, filtered)
    return output_path


def _sanitize_inspection_images(inspection: Dict, work_dir: Path, dry_run: bool) -> Dict:
    images_dir_raw = inspection.get("images_dir")
    if dry_run or not images_dir_raw:
        return inspection

    images_dir = Path(str(images_dir_raw)).resolve()
    image_paths = _collect_images_for_validation(images_dir)
    if not image_paths:
        return inspection

    valid_images: list[Path] = []
    invalid_images: list[dict[str, str]] = []
    for image_path in image_paths:
        error = _decode_image_error(image_path)
        if error is None:
            valid_images.append(image_path)
        else:
            invalid_images.append(
                {
                    "path": str(image_path),
                    "relative_path": image_path.relative_to(images_dir).as_posix(),
                    "error": error,
                }
            )

    sanitized = dict(inspection)
    sanitized["original_image_count"] = len(image_paths)
    sanitized["valid_image_count"] = len(valid_images)
    sanitized["invalid_image_count"] = len(invalid_images)
    sanitized["invalid_images_report"] = None
    if not invalid_images:
        return sanitized

    report_path = work_dir / "invalid_uploaded_images.json"
    _write_json(
        report_path,
        {
            "status": "warning",
            "message": "Invalid images were excluded from annotation and training.",
            "images_dir": str(images_dir),
            "total": len(image_paths),
            "valid": len(valid_images),
            "invalid": len(invalid_images),
            "invalid_images": invalid_images,
        },
    )
    sanitized["invalid_images_report"] = str(report_path)
    sanitized["invalid_images"] = invalid_images
    if not valid_images:
        raise ValueError(f"All uploaded images are invalid. See report: {report_path}")

    clean_images_dir = work_dir / "valid_uploaded_images"
    if clean_images_dir.exists():
        shutil.rmtree(clean_images_dir)
    for image_path in valid_images:
        relative = image_path.relative_to(images_dir)
        target = clean_images_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, target)

    sanitized["original_images_dir"] = str(images_dir)
    sanitized["images_dir"] = str(clean_images_dir)
    sanitized["image_count"] = len(valid_images)
    if str(sanitized.get("format") or "").lower() == "coco" and sanitized.get("coco_json"):
        sanitized["original_coco_json"] = sanitized["coco_json"]
        sanitized["coco_json"] = str(
            _filter_coco_for_valid_images(
                Path(str(sanitized["original_coco_json"])).resolve(),
                images_dir,
                valid_images,
                work_dir / "filtered_valid_images.coco.json",
            )
        )

    _safe_print(
        "[data-prep] excluded invalid images before annotation/training: "
        f"invalid={len(invalid_images)}, valid={len(valid_images)}, report={report_path}"
    )
    return sanitized


def _prepare_real_coco(
    inspection: Dict,
    work_dir: Path,
    task_labels: List[str],
    annotation_prompts: List[str],
    prompt_label_map: Dict[str, str],
    intent_items: List[Dict[str, Any]],
    annotation_provider: str,
    dry_run: bool,
) -> Path:
    fmt = inspection.get("format")
    images_dir = Path(str(inspection.get("images_dir") or "")).resolve()
    if fmt == "coco":
        coco_path = Path(str(inspection["coco_json"])).resolve()
        if not dry_run:
            _write_per_image_coco_sidecars(coco_path, images_dir)
        return coco_path

    real_coco = work_dir / "real_coco.json"
    if fmt == "yolo":
        class_names = ",".join(task_labels)
        classes_file = inspection.get("classes_file")
        class_arg = str(classes_file or class_names)
        _run([
            sys.executable,
            str(SCRIPT_DIR / "convert_yolo_to_coco.py"),
            "--images-dir",
            str(images_dir),
            "--labels-dir",
            str(inspection["labels_dir"]),
            "--output",
            str(real_coco),
            "--class-names",
            class_arg,
        ], dry_run)
        if not dry_run:
            _write_per_image_coco_sidecars(real_coco, images_dir)
        return real_coco

    if fmt == "unlabeled":
        if not task_labels:
            raise ValueError("Unlabeled datasets require labels, for example: labels=person,bottle")
        prompts = _annotation_prompts_for_sam3(task_labels, annotation_prompts, prompt_label_map, intent_items)
        try:
            cmd = [
                sys.executable,
                str(AUTO_ANNOTATION_SCRIPT),
                "--annotation-provider",
                annotation_provider,
                "--input-dir",
                str(images_dir),
                "--text-prompts",
                *prompts,
                "--class-names",
                *task_labels,
                "--prompt-label-map-json",
                json.dumps(_prompt_label_map_for_sam3(task_labels, prompts, prompt_label_map), ensure_ascii=False),
                "--source",
                "real",
                "--per-image-output-dir",
                str(images_dir),
                "--per-image-base-dir",
                str(images_dir),
                "--output",
                str(real_coco),
            ]
            _run(cmd, dry_run)
        except subprocess.CalledProcessError as exc:
            stdout_tail = _decode_bytes(exc.output)[-2000:]
            stderr_tail = _decode_bytes(exc.stderr)[-2000:]
            raise RuntimeError(
                "Real dataset auto-annotation failed while preparing COCO. "
                "Check the SAM3 service, uploaded dataset images, and labels.\n"
                f"annotation_provider={annotation_provider}\n"
                f"labels={task_labels}\n"
                f"annotation_prompts={prompts}\n"
                f"images_dir={images_dir}\n"
                f"stdout_tail:\n{stdout_tail}\n"
                f"stderr_tail:\n{stderr_tail}"
            ) from exc
        return real_coco

    raise ValueError(f"Unsupported dataset format: {fmt}. Convert labels to COCO or YOLO first.")


def _write_per_image_coco_sidecars(coco_path: Path, images_dir: Path) -> None:
    if not coco_path.is_file() or not images_dir.is_dir():
        return
    coco = _load_json(coco_path)
    categories = [dict(item) for item in coco.get("categories", []) if isinstance(item, dict)]
    images_by_id: dict[int, dict[str, Any]] = {}
    for image in coco.get("images", []):
        if not isinstance(image, dict):
            continue
        try:
            image_id = int(image.get("id"))
        except (TypeError, ValueError):
            continue
        images_by_id[image_id] = image
    annotations_by_image_id: dict[int, list[dict[str, Any]]] = {image_id: [] for image_id in images_by_id}
    for annotation in coco.get("annotations", []):
        if not isinstance(annotation, dict):
            continue
        try:
            image_id = int(annotation.get("image_id"))
        except (TypeError, ValueError):
            continue
        if image_id in annotations_by_image_id:
            annotations_by_image_id[image_id].append(annotation)
    for image_id, image in images_by_id.items():
        file_name = str(image.get("file_name") or "").replace("\\", "/").strip()
        if not file_name:
            continue
        image_path = _resolve_image_path_for_sidecar(images_dir, file_name)
        if image_path is None:
            continue
        _write_single_image_coco_sidecar(
            image_path,
            images_dir,
            image,
            annotations_by_image_id.get(image_id, []),
            categories,
        )


def _resolve_image_path_for_sidecar(images_dir: Path, file_name: str) -> Path | None:
    candidate = (images_dir / file_name).resolve()
    try:
        candidate.relative_to(images_dir.resolve())
    except ValueError:
        return None
    if candidate.is_file():
        return candidate
    fallback = images_dir / Path(file_name).name
    return fallback.resolve() if fallback.is_file() else None


def _write_single_image_coco_sidecar(
    image_path: Path,
    images_dir: Path,
    image: dict[str, Any],
    annotations: list[dict[str, Any]],
    categories: list[dict[str, Any]],
) -> Path:
    relative_image = image_path.resolve().relative_to(images_dir.resolve())
    sidecar = images_dir / relative_image.parent / f"{image_path.stem}_coco.json"
    local_image = dict(image)
    original_image_id = int(local_image.get("id", 1) or 1)
    local_image["id"] = 1
    local_annotations: list[dict[str, Any]] = []
    for annotation_id, annotation in enumerate(annotations, start=1):
        local_annotation = dict(annotation)
        local_annotation["id"] = annotation_id
        local_annotation["image_id"] = 1
        local_annotations.append(local_annotation)
    _write_json(
        sidecar,
        {
            "images": [local_image],
            "annotations": local_annotations,
            "categories": categories,
            "licenses": [],
            "info": {
                "description": "per-image real dataset COCO annotation",
                "original_image_id": original_image_id,
            },
        },
    )
    _safe_print(f"[data-prep] per-image real coco written: {sidecar}")
    return sidecar


def _plan_synthetic(
    real_coco: Path,
    task: str,
    generation_prompt: str,
    split_train: float,
    work_dir: Path,
    max_synthetic: int,
    planner_llm: Dict[str, Any],
    synthetic_count_button: bool,
    fixed_synthetic_count: int,
    dry_run: bool,
) -> Path:
    plan_path = work_dir / "synthetic_plan.json"
    planner = "llm" if synthetic_count_button else "fixed"
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
        planner,
        "--fixed-synthetic-count",
        str(max(0, fixed_synthetic_count)),
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


def _extract_generated_image_path(stdout_text: str, scene_dir: Path, before_files: set[str]) -> Path | None:
    for pattern in (r"result_image_path:\s*(.+)", r"Successfully saved:\s*(.+)", r"saved:\s*(.+)"):
        match = re.search(pattern, stdout_text or "")
        if match:
            candidate = Path(match.group(1).strip().strip("\"'"))
            if candidate.exists() and candidate.is_file():
                return candidate.resolve()
    after_files = {
        str(path.resolve())
        for path in scene_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    new_files = sorted(after_files - before_files)
    return Path(new_files[-1]).resolve() if new_files else None


def _timestamp_suffix() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _move_to_unique_synthetic_name(image_path: Path, scene_dir: Path, scene_index: int, item_index: int) -> Path:
    target = scene_dir / f"scene_{scene_index:02d}_{item_index:05d}_{_timestamp_suffix()}{image_path.suffix.lower() or '.png'}"
    counter = 1
    while target.exists():
        target = scene_dir / f"scene_{scene_index:02d}_{item_index:05d}_{_timestamp_suffix()}_{counter}{image_path.suffix.lower() or '.png'}"
        counter += 1
    if image_path.resolve() == target.resolve():
        return target
    image_path.replace(target)
    return target.resolve()


def _annotation_prompts_for_sam3(
    labels: List[str],
    annotation_prompts: List[str] | None,
    prompt_label_map: Dict[str, str] | None = None,
    intent_items: List[Dict[str, Any]] | None = None,
) -> List[str]:
    prompts = _dedupe_text([str(item).strip() for item in (annotation_prompts or []) if str(item).strip()])
    prompt_map = {str(prompt).strip(): str(label).strip() for prompt, label in (prompt_label_map or {}).items() if str(prompt).strip() and str(label).strip()}
    class_names = _dedupe_text(labels)
    validated_prompts = [
        prompt
        for prompt, label in prompt_map.items()
        if label in class_names
    ]
    if validated_prompts:
        return _dedupe_text(validated_prompts)
    selected: List[str] = []
    for label in class_names:
        selected.extend(_best_annotation_prompts_for_label(label, labels, prompts, prompt_map, intent_items or []))
    return _dedupe_text(selected or prompts or labels)


def _best_annotation_prompts_for_label(
    label: str,
    labels: List[str],
    prompts: List[str],
    prompt_label_map: Dict[str, str],
    intent_items: List[Dict[str, Any]],
    limit: int = SAM3_MAX_PROMPTS_PER_LABEL,
) -> List[str]:
    class_key = str(label or "").strip().lower()
    if not class_key:
        return []
    related_items = _intent_items_for_label(label, intent_items)
    entity_prompts = _entity_interaction_prompts_for_label(label, related_items)
    if entity_prompts:
        return entity_prompts[: max(1, limit)]
    candidates = _dedupe_text([
        *_visual_prompts_from_intent_items(related_items),
        *[prompt for prompt, mapped_label in prompt_label_map.items() if mapped_label == label],
        *prompts,
        *_generated_visual_prompts_for_label(label, related_items),
        _label_to_sam3_prompt(label),
        label,
    ])
    ranked = sorted(
        (
            (_sam3_prompt_score(prompt, label, labels, prompt_label_map, related_items), index, prompt)
            for index, prompt in enumerate(candidates)
            if str(prompt or "").strip()
            and not any(_prompt_is_forbidden_for_intent(prompt, item) for item in related_items)
        ),
        key=lambda item: (item[0], item[1]),
    )
    high_quality = [prompt for score, _, prompt in ranked if score < 80]
    selected = high_quality[: max(1, limit)] if high_quality else [prompt for score, _, prompt in ranked if score < 9000][: max(1, limit)]
    if selected:
        return selected
    fallback = _label_to_sam3_prompt(label) or label
    return [fallback] if fallback else []


def _best_annotation_prompt_for_label(
    label: str,
    labels: List[str],
    prompts: List[str],
    prompt_label_map: Dict[str, str],
) -> str:
    selected = _best_annotation_prompts_for_label(label, labels, prompts, prompt_label_map, [])
    return selected[0] if selected else ""


def _sam3_prompt_score(
    prompt: str,
    label: str,
    labels: List[str],
    prompt_label_map: Dict[str, str],
    intent_items: List[Dict[str, Any]] | None = None,
) -> int:
    prompt_text = str(prompt or "").strip()
    label_text = str(label or "").strip()
    if not _is_valid_sam3_visual_prompt(prompt_text):
        return 10000
    prompt_key = prompt_text.lower()
    label_key = label_text.lower()
    canonical_label_prompt = _label_to_sam3_prompt(label_text).lower()
    mapped_key = str(prompt_label_map.get(prompt_text, "") or "").strip().lower()
    prompt_tokens = _sam3_prompt_tokens(prompt_text)
    label_tokens = _sam3_prompt_tokens(_label_to_sam3_prompt(label_text))
    intent_items = intent_items or []
    behavior_tokens = _intent_behavior_tokens(intent_items, label_text)
    target_tokens = _intent_target_tokens(intent_items)

    constrained_items = [item for item in intent_items if _is_constrained_intent_item(item)]
    if constrained_items:
        if any(_prompt_is_forbidden_for_intent(prompt_text, item) for item in constrained_items):
            return 10000
        primary_prompts = {
            str(item.get("primary_sam3_prompt") or "").strip().lower()
            for item in constrained_items
            if str(item.get("primary_sam3_prompt") or "").strip()
        }
        if prompt_key in primary_prompts and (len(labels) == 1 or mapped_key == label_key):
            return 0
        if mapped_key == label_key:
            return 2
        if prompt_key == canonical_label_prompt:
            return 30
        if _looks_related_prompt(prompt_text, label_text):
            return 40
        return 100

    # person_fall/person_running 这类业务行为标签必须优先使用带行为约束的自然语言 prompt。
    # 直接用 person 会把画面里的所有人都标成该行为，只有没有更好候选时才兜底使用。
    if label_key.startswith("person_"):
        if mapped_key == label_key and _looks_person_behavior_sam3_prompt(prompt_text, label_text, behavior_tokens):
            return 0
        if _looks_person_behavior_sam3_prompt(prompt_text, label_text, behavior_tokens):
            return 5
        if prompt_key == canonical_label_prompt:
            return 10
        if target_tokens and prompt_tokens.intersection(target_tokens):
            return 15
        if mapped_key == label_key:
            return 20
        if prompt_key == "person":
            return 80

    # 训练类别本身是 person/cigarette/face 这类实体时，优先使用同名实体 prompt，
    # 避免误选 person smoking 等行为短语。
    if not label_key.startswith("person_") and (prompt_key == label_key or prompt_key == canonical_label_prompt):
        return 0
    if not label_key.startswith("person_") and mapped_key == label_key and (prompt_key == label_key or prompt_key == canonical_label_prompt):
        return 0
    if mapped_key == label_key and _looks_entity_like_sam3_prompt(prompt_text, labels):
        return 2
    if _looks_entity_like_sam3_prompt(prompt_text, labels) and prompt_tokens.intersection(label_tokens):
        return 4
    if mapped_key == label_key:
        return 20
    if len(labels) == 1:
        return 30
    if _looks_related_prompt(prompt_text, label_text):
        return 40
    return 100


def _sam3_prompt_tokens(text: str) -> set[str]:
    return {item for item in re.split(r"[^a-z0-9]+", str(text or "").lower()) if item}


def _label_lookup_key(label: str) -> str:
    text = str(label or "").strip().lower()
    text = re.sub(r"[^a-z0-9_]+", "_", text).strip("_")
    return text


def _is_valid_sam3_visual_prompt(prompt: str) -> bool:
    text = str(prompt or "").strip()
    if not text:
        return False
    lower = text.lower()
    if re.search(r"[\u4e00-\u9fff]", text):
        return False
    if "_" in text:
        return False
    if any(keyword in lower for keyword in SAM3_INSTRUCTION_KEYWORDS):
        return False
    tokens = _sam3_prompt_tokens(text)
    if not tokens:
        return False
    if len(tokens) > SAM3_PROMPT_MAX_WORDS:
        return False
    if len(text) > 80:
        return False
    return True


def _intent_items_for_label(label: str, intent_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    label_key = str(label or "").strip().lower()
    result: List[Dict[str, Any]] = []
    for item in intent_items or []:
        if not isinstance(item, dict):
            continue
        candidates = [str(item.get("label") or ""), str(item.get("train_label") or "")]
        if isinstance(item.get("training_labels"), list):
            candidates.extend(str(value) for value in item.get("training_labels", []) if str(value).strip())
        if isinstance(item.get("business_labels"), list):
            candidates.extend(str(value) for value in item.get("business_labels", []) if str(value).strip())
        if label_key in {value.strip().lower() for value in candidates if value.strip()}:
            result.append(item)
    return result


def _visual_prompts_from_intent_items(items: List[Dict[str, Any]]) -> List[str]:
    values: List[str] = []
    for item in items:
        if _is_constrained_intent_item(item):
            primary = str(item.get("primary_sam3_prompt") or "").strip()
            if primary:
                values.append(primary)
            keys = ("sam3_prompts",)
        elif _is_entity_interaction_item(item):
            keys = ("sam3_prompts", "training_labels")
        else:
            keys = ("sam3_prompts", "visual_states", "annotation_prompts")
        for key in keys:
            raw = item.get(key)
            if isinstance(raw, list):
                values.extend(str(value).strip() for value in raw if str(value).strip())
    return _dedupe_text([
        value
        for value in values
        if _is_valid_sam3_visual_prompt(value)
        and not any(_prompt_is_forbidden_for_intent(value, item) for item in items)
    ])


def _entity_interaction_prompts_for_label(label: str, items: List[Dict[str, Any]]) -> List[str]:
    label_key = _label_lookup_key(label)
    prompts: List[str] = []
    for item in items:
        if not _is_entity_interaction_item(item):
            continue
        for key in ("training_labels", "sam3_prompts", "annotation_prompts"):
            raw = item.get(key)
            if not isinstance(raw, list):
                continue
            for value in raw:
                prompt = str(value or "").strip()
                if not _is_valid_sam3_visual_prompt(prompt):
                    continue
                if _label_lookup_key(prompt) == label_key or _label_lookup_key(_label_to_sam3_prompt(prompt)) == label_key:
                    prompts.append(_label_to_sam3_prompt(prompt))
    return _dedupe_text(prompts)


def _is_entity_interaction_item(item: Dict[str, Any]) -> bool:
    task_type = str(item.get("type") or item.get("task_type") or "").strip().lower()
    if _is_person_attribute_item(item):
        return False
    if task_type in {"entity_interaction", "object_interaction", "person_object_interaction"}:
        return True
    return False


def _is_person_attribute_item(item: Dict[str, Any]) -> bool:
    task_type = str(item.get("type") or item.get("task_type") or "").strip().lower()
    return task_type in {"person_attribute_detection", "person_attribute", "appearance_detection", "wearing_detection"}


def _is_constrained_intent_item(item: Dict[str, Any]) -> bool:
    if _is_entity_interaction_item(item):
        return False
    strategy = str(item.get("annotation_strategy") or "").strip().lower()
    if strategy in {"constrained_target", "requires_review", "candidate_and_verify"}:
        return True
    task_type = str(item.get("type") or item.get("task_type") or "").strip().lower()
    if task_type in {
        "person_attribute_detection",
        "person_attribute",
        "appearance_detection",
        "wearing_detection",
        "behavior_detection",
        "behavior",
        "behaviour",
        "action",
        "activity",
        "state_detection",
        "state",
        "status",
        "anomaly_detection",
    }:
        return True
    return any(bool(_phrase_values([item.get(field)])) for field in (
        "required_attributes",
        "required_actions",
        "required_states",
        "required_relations",
    ))


def _prompt_is_forbidden_for_intent(prompt: str, item: Dict[str, Any]) -> bool:
    if not _is_constrained_intent_item(item):
        return False
    prompt_key = str(prompt or "").strip().lower()
    if not prompt_key:
        return True
    primary = str(item.get("primary_sam3_prompt") or "").strip().lower()
    if primary and prompt_key == primary:
        return False
    forbidden = {
        str(value).strip().lower()
        for value in _phrase_values([item.get("forbidden_direct_prompts")])
        if str(value).strip()
    }
    forbidden.update(
        str(value).strip().lower()
        for value in _phrase_values([item.get("base_entity"), item.get("bbox_target"), item.get("subject")])
        if str(value).strip()
    )
    return prompt_key in forbidden


def _generated_visual_prompts_for_label(label: str, items: List[Dict[str, Any]]) -> List[str]:
    prompts: List[str] = []
    label_prompt = _label_to_sam3_prompt(label)
    if label_prompt:
        prompts.append(label_prompt)
    for item in items:
        subject = _first_valid_phrase([item.get("subject"), "person" if str(label).lower().startswith("person_") else ""])
        behaviors = _phrase_values([item.get("behavior"), item.get("action"), item.get("state"), item.get("activity")])
        targets = _phrase_values([item.get("target_objects"), item.get("objects"), item.get("tools")])
        for behavior in behaviors:
            if subject:
                prompts.append(f"{subject} {behavior}")
                if len(_sam3_prompt_tokens(behavior)) == 1:
                    prompts.append(f"{behavior} {subject}")
        for target in targets:
            if subject:
                prompts.append(f"{subject} with {target}")
                prompts.append(f"{subject} holding {target}")
                prompts.append(f"{subject} using {target}")
            prompts.append(f"{target} player")
            prompts.append(target)
    return _dedupe_text([prompt for prompt in prompts if _is_valid_sam3_visual_prompt(prompt)])


def _phrase_values(values: List[Any]) -> List[str]:
    result: List[str] = []
    for value in values:
        if isinstance(value, list):
            result.extend(str(item).strip() for item in value if str(item).strip())
        else:
            text = str(value or "").strip()
            if text:
                result.append(text)
    return _dedupe_text([value for value in result if _is_valid_sam3_visual_prompt(value)])


def _first_valid_phrase(values: List[Any]) -> str:
    phrases = _phrase_values(values)
    return phrases[0] if phrases else ""


def _intent_behavior_tokens(items: List[Dict[str, Any]], label: str) -> set[str]:
    tokens = _expanded_person_behavior_tokens(label)
    for item in items:
        for value in _phrase_values([item.get("behavior"), item.get("action"), item.get("state"), item.get("activity")]):
            tokens.update(_expanded_person_behavior_tokens(value))
            tokens.update(_sam3_prompt_tokens(value))
        for value in _phrase_values([item.get("visual_states"), item.get("sam3_prompts")]):
            prompt_tokens = _sam3_prompt_tokens(value)
            if prompt_tokens.intersection({"person", "people", "human", "man", "woman"}):
                tokens.update(prompt_tokens - {"person", "people", "human", "man", "woman"})
    return tokens


def _intent_target_tokens(items: List[Dict[str, Any]]) -> set[str]:
    tokens: set[str] = set()
    for item in items:
        for value in _phrase_values([item.get("target_objects"), item.get("objects"), item.get("tools"), item.get("observable_entities")]):
            tokens.update(_sam3_prompt_tokens(value))
    return tokens


def _looks_person_behavior_sam3_prompt(prompt: str, label: str, behavior_tokens: set[str] | None = None) -> bool:
    prompt_tokens = _sam3_prompt_tokens(prompt)
    if not prompt_tokens.intersection({"person", "people", "human", "man", "woman"}):
        return False
    behavior_tokens = behavior_tokens or _expanded_person_behavior_tokens(label)
    return bool(behavior_tokens and prompt_tokens.intersection(behavior_tokens))


def _expanded_person_behavior_tokens(label: str) -> set[str]:
    text = re.sub(r"^person[_\\s-]*", "", str(label or "").strip().lower())
    base_tokens = {item for item in re.split(r"[^a-z0-9]+", text) if item}
    expanded = set(base_tokens)
    synonyms = {
        "fall": {"fall", "falls", "falling", "fallen", "down", "lying"},
        "run": {"run", "runs", "running"},
        "running": {"run", "runs", "running"},
        "fight": {"fight", "fights", "fighting"},
        "sleep": {"sleep", "sleeps", "sleeping", "asleep"},
        "climb": {"climb", "climbs", "climbing"},
        "smoke": {"smoke", "smokes", "smoking", "cigarette"},
        "smoking": {"smoke", "smokes", "smoking", "cigarette"},
        "phone": {"phone", "mobile", "cellphone", "calling"},
        "use": {"use", "using"},
        "play": {"play", "playing"},
        "kick": {"kick", "kicking"},
    }
    for token in list(base_tokens):
        expanded.update(synonyms.get(token, set()))
        expanded.update(_generic_behavior_token_forms(token))
    return expanded


def _generic_behavior_token_forms(token: str) -> set[str]:
    text = str(token or "").strip().lower()
    if not text:
        return set()
    forms = {text}
    if text.endswith("ing") and len(text) > 4:
        forms.add(text[:-3])
        if len(text) > 5 and text[-4] == text[-5]:
            forms.add(text[:-4])
    elif text.endswith("ed") and len(text) > 3:
        forms.add(text[:-2])
    elif text.endswith("s") and len(text) > 3:
        forms.add(text[:-1])
    else:
        forms.add(text + "s")
        forms.add(text + "ed")
        if text.endswith("e"):
            forms.add(text[:-1] + "ing")
        elif len(text) >= 3 and text[-1] not in "aeiou" and text[-2] in "aeiou":
            forms.add(text + text[-1] + "ing")
        else:
            forms.add(text + "ing")
    return forms


def _looks_entity_like_sam3_prompt(prompt: str, labels: List[str]) -> bool:
    text = str(prompt or "").strip().lower()
    if not text:
        return False
    tokens = _sam3_prompt_tokens(text)
    if not tokens:
        return False
    behavior_tokens = {
        "smoking",
        "falling",
        "fallen",
        "fight",
        "fighting",
        "running",
        "playing",
        "using",
        "holding",
        "sleeping",
        "climbing",
        "riding",
        "kick",
        "kicking",
    }
    if tokens.intersection(behavior_tokens):
        return False
    normalized_labels = {_label_to_sam3_prompt(label).lower() for label in labels}
    return text in normalized_labels or len(tokens) <= 2


def _prompt_label_map_for_sam3(labels: List[str], prompts: List[str], prompt_label_map: Dict[str, str] | None) -> Dict[str, str]:
    class_names = _dedupe_text(labels)
    result: Dict[str, str] = {}
    for prompt, label in (prompt_label_map or {}).items():
        prompt_text = str(prompt or "").strip()
        label_text = str(label or "").strip()
        if prompt_text and label_text in class_names:
            result[prompt_text] = label_text
    for prompt in prompts:
        mapped = _best_sam3_prompt_mapped_label(prompt, class_names)
        if mapped:
            result.setdefault(prompt, mapped)
    for label in class_names:
        fallback_prompt = _label_to_sam3_prompt(label) or label
        result.setdefault(fallback_prompt, label)
    return result


def _best_sam3_prompt_mapped_label(prompt: str, labels: List[str]) -> str:
    prompt_tokens = _sam3_prompt_tokens(prompt)
    if not prompt_tokens:
        return ""
    best_label = ""
    best_score = 0
    for label in labels:
        label_prompt = _label_to_sam3_prompt(label)
        label_tokens = _sam3_prompt_tokens(label_prompt)
        behavior_tokens = _expanded_person_behavior_tokens(label)
        score = len(prompt_tokens.intersection(label_tokens)) + len(prompt_tokens.intersection(behavior_tokens))
        if score > best_score:
            best_label = label
            best_score = score
    return best_label


def _label_to_sam3_prompt(label: str) -> str:
    text = str(label or "").strip().lower()
    text = re.sub(r"[^a-z0-9_]+", "_", text).strip("_")
    if not text:
        return ""
    if text.startswith("person_"):
        return "person " + text.removeprefix("person_").replace("_", " ")
    return text.replace("_", " ")


def _looks_related_prompt(prompt: str, label: str) -> bool:
    prompt_tokens = {item for item in re.split(r"[^a-z0-9]+", str(prompt).lower()) if item}
    label_tokens = {item for item in re.split(r"[^a-z0-9]+", _label_to_sam3_prompt(label).lower()) if item}
    return bool(prompt_tokens and label_tokens and prompt_tokens.intersection(label_tokens))


def _dedupe_text(values: List[str]) -> List[str]:
    result: List[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def _annotate_one_synthetic(
    image_path: Path,
    labels: List[str],
    annotation_prompts: List[str],
    prompt_label_map: Dict[str, str],
    intent_items: List[Dict[str, Any]],
    annotation_provider: str,
    output_json: Path,
    dry_run: bool,
) -> Path:
    if not labels:
        raise ValueError("Synthetic auto-annotation requires labels")
    prompts = _annotation_prompts_for_sam3(labels, annotation_prompts, prompt_label_map, intent_items)
    _run([
        sys.executable,
        str(AUTO_ANNOTATION_SCRIPT),
        "--annotation-provider",
        annotation_provider,
        "--input-json",
        json.dumps({"image_path": str(image_path)}, ensure_ascii=False),
        "--text-prompts",
        *prompts,
        "--class-names",
        *labels,
        "--prompt-label-map-json",
        json.dumps(_prompt_label_map_for_sam3(labels, prompts, prompt_label_map), ensure_ascii=False),
        "--source",
        "synthetic",
        "--is-synthetic",
        "--output",
        str(output_json),
    ], dry_run)
    if not dry_run:
        _safe_print(f"[data-prep] synthetic coco written: {output_json}")
    return output_json


def _combine_coco_files(coco_files: List[Path], output_path: Path) -> Path:
    if not coco_files:
        raise ValueError("No synthetic partial COCO files were produced")
    merged = {"images": [], "annotations": [], "categories": [], "licenses": [], "info": {"description": "streaming synthetic SAM3 auto-annotation export"}}
    category_name_to_id: dict[str, int] = {}
    next_image_id = 1
    next_annotation_id = 1
    for coco_path in coco_files:
        coco = _load_json(coco_path)
        local_category_map: dict[int, int] = {}
        for category in coco.get("categories", []):
            name = str(category.get("name") or "").strip()
            if not name:
                continue
            if name not in category_name_to_id:
                category_name_to_id[name] = len(category_name_to_id) + 1
                new_category = dict(category)
                new_category["id"] = category_name_to_id[name]
                merged["categories"].append(new_category)
            local_category_map[int(category["id"])] = category_name_to_id[name]

        image_id_map: dict[int, int] = {}
        for image in coco.get("images", []):
            old_id = int(image["id"])
            image_id_map[old_id] = next_image_id
            new_image = dict(image)
            new_image["id"] = next_image_id
            new_image["source"] = "synthetic"
            new_image["is_synthetic"] = True
            merged["images"].append(new_image)
            next_image_id += 1

        for ann in coco.get("annotations", []):
            old_image_id = int(ann["image_id"])
            old_category_id = int(ann["category_id"])
            if old_image_id not in image_id_map or old_category_id not in local_category_map:
                continue
            new_ann = dict(ann)
            new_ann["id"] = next_annotation_id
            new_ann["image_id"] = image_id_map[old_image_id]
            new_ann["category_id"] = local_category_map[old_category_id]
            merged["annotations"].append(new_ann)
            next_annotation_id += 1
    _write_json(output_path, merged)
    return output_path


def _generate_and_annotate_synthetic(
    plan_path: Path,
    image1: Path,
    image2: Path,
    labels: List[str],
    annotation_prompts: List[str],
    prompt_label_map: Dict[str, str],
    intent_items: List[Dict[str, Any]],
    annotation_provider: str,
    output_dir: Path,
    dry_run: bool,
) -> tuple[Path, Path]:
    plan = _load_json(plan_path)
    synthetic_images_root = output_dir / "synthetic_images"
    synthetic_images_root.mkdir(parents=True, exist_ok=True)
    annotation_root = output_dir / "synthetic_annotations"
    annotation_root.mkdir(parents=True, exist_ok=True)
    partial_coco_files: List[Path] = []
    for idx, item in enumerate(plan.get("generation_plan", []), start=1):
        count = int(item.get("count", 0))
        prompt = str(item.get("prompt", "")).strip()
        if count <= 0 or not prompt:
            continue
        scene_dir = synthetic_images_root
        scene_dir.mkdir(parents=True, exist_ok=True)
        for n in range(count):
            input_json = output_dir / "generation_inputs" / f"scene_{idx:02d}_{n + 1:05d}.json"
            _write_json(input_json, {"task": {"image1": str(image1), "image2": str(image2), "prompt": prompt, "output_dir": str(scene_dir)}})
            before_files = {
                str(path.resolve())
                for path in scene_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            }
            stdout_text = _run([sys.executable, str(GENERATION_SCRIPT), "--input", str(input_json)], dry_run, retries=3, retry_sleep=8.0)
            if dry_run:
                continue
            generated_image = _extract_generated_image_path(stdout_text, scene_dir, before_files)
            if generated_image is None:
                raise RuntimeError(f"Composition completed but no composite image was found for {input_json}")
            generated_image = _move_to_unique_synthetic_name(generated_image, scene_dir, idx, n + 1)
            _safe_print(f"[data-prep] annotating composite image immediately: {generated_image}")
            partial_coco = annotation_root / f"{generated_image.stem}_coco.json"
            _annotate_one_synthetic(generated_image, labels, annotation_prompts, prompt_label_map, intent_items, annotation_provider, partial_coco, dry_run)
            partial_coco_files.append(partial_coco)
    synthetic_coco = output_dir / "synthetic_coco.json"
    if not dry_run:
        _combine_coco_files(partial_coco_files, synthetic_coco)
    return synthetic_images_root, synthetic_coco


def _collect_image1_reference_images(image1: Path) -> List[Path]:
    if image1.is_file() and image1.suffix.lower() in IMAGE_EXTENSIONS:
        return [image1.resolve()]
    if not image1.is_dir():
        return []
    return sorted(
        (path.resolve() for path in image1.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: str(path).casefold(),
    )


def _fallback_total_synthetic_count(plan: Dict[str, Any], reference_count: int) -> int:
    try:
        recommended = int(plan.get("recommended_synthetic_count", 0) or 0)
    except (TypeError, ValueError):
        recommended = 0
    planned_total = 0
    for item in plan.get("generation_plan", []):
        if not isinstance(item, dict):
            continue
        try:
            planned_total += max(0, int(item.get("count", 0) or 0))
        except (TypeError, ValueError):
            continue
    return max(reference_count, recommended, planned_total)


def _produce_synthetic_count(
    plan: Dict[str, Any],
    reference_count: int,
    produce_count_button: bool,
    fixed_produce_synthetic_count: int,
    max_synthetic: int,
) -> int:
    if reference_count <= 0:
        return 0
    if produce_count_button:
        total = max(1, _fallback_total_synthetic_count(plan, reference_count))
    else:
        total = max(0, int(fixed_produce_synthetic_count))
    if max_synthetic >= 0:
        total = min(total, max_synthetic)
    return max(0, total)


def _fallback_prompt_sequence(
    plan: Dict[str, Any],
    labels: List[str],
    total_count: int,
    planner_llm: Dict[str, Any] | None = None,
) -> tuple[List[str], str]:
    prompts = _llm_image_dataset_produce_prompts(plan, labels, total_count, planner_llm or {})
    if prompts:
        return prompts, "llm"
    return _heuristic_image_dataset_produce_prompts(plan, labels, total_count), "heuristic_reference_anchored"


def _llm_image_dataset_produce_prompts(
    plan: Dict[str, Any],
    labels: List[str],
    total_count: int,
    planner_llm: Dict[str, Any],
) -> List[str]:
    base_url = str(planner_llm.get("base_url") or "").rstrip("/")
    model = str(planner_llm.get("model") or "").strip()
    if not base_url or not model:
        return []
    count = max(1, min(12, int(total_count or 1)))
    class_names = labels or [str(item).strip() for item in plan.get("class_names", []) if str(item).strip()]
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a senior computer vision synthetic-data prompt planner. "
                    "Create prompts for the image-dataset-produce skill, which uses one image1 reference image as a visual reference. "
                    "The prompt must be suitable for image-to-image generation with flux2. "
                    "You cannot see the reference image, so do not invent a specific unrelated scene, location, identity, or background. "
                    "Every prompt must explicitly tell flux2 to use the provided input/reference image as the primary visual reference, "
                    "preserve its scene category, camera angle, background layout, lighting direction, and visual style, "
                    "and create a realistic variant for object detection rather than an identical copy. "
                    "Do not mention image2.zip, compositing, pasted objects, cutouts, extraction, or overlaying one image onto another. "
                    "Return JSON only: {\"prompts\":[\"...\"]}."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task_description": plan.get("task_description", ""),
                        "labels": class_names,
                        "image_size_summary": plan.get("image_size_summary", {}),
                        "difficulty_signals": plan.get("difficulty_signals", []),
                        "prompt_count": count,
                        "requirements": [
                            "Each prompt must include the phrase provided input image or reference image.",
                            "Preserve the reference image scene type, background layout, camera angle, composition, lighting, and visual style.",
                            "Do not invent a new unrelated environment such as a city street, garden, rooftop bar, living room, or parking garage unless the reference image already shows it.",
                            "Generate a visibly new but scene-consistent variant with small natural changes in target pose, placement, visibility, and local lighting.",
                            "Ensure target classes are clear and annotatable with bounding boxes.",
                            "Avoid exact duplication of the reference image.",
                            "Synthetic images are for training only.",
                        ],
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        "temperature": float(planner_llm.get("temperature", 0.4)),
        "max_tokens": int(planner_llm.get("max_tokens", 1024)),
    }
    headers = {"Content-Type": "application/json"}
    api_key = str(planner_llm.get("api_key") or "")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = request.Request(f"{base_url}/chat/completions", data=data, headers=headers, method="POST")
        timeout = int(planner_llm.get("timeout", 120) or 120)
        with request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        parsed = json.loads(raw)
        choices = parsed.get("choices") or []
        content = str((choices[0].get("message") or {}).get("content") or "").strip() if choices else ""
        prompt_payload = _parse_json_object_from_text(content)
        prompts = prompt_payload.get("prompts") if isinstance(prompt_payload, dict) else []
        return _validated_image_dataset_produce_prompts(prompts, plan, labels, total_count)
    except Exception as exc:
        _safe_print(f"[data-prep] image-dataset-produce prompt planner unavailable; using local prompt template: {exc}", file=sys.stderr)
        return []


def _parse_json_object_from_text(content: str) -> Dict[str, Any]:
    cleaned = (content or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    if not cleaned.startswith("{"):
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            return {}
        cleaned = match.group(0)
    parsed = json.loads(cleaned)
    return parsed if isinstance(parsed, dict) else {}


def _validated_image_dataset_produce_prompts(
    raw_prompts: Any,
    plan: Dict[str, Any],
    labels: List[str],
    total_count: int,
) -> List[str]:
    if not isinstance(raw_prompts, list):
        return []
    prompts: List[str] = []
    for item in raw_prompts:
        prompt = str(item or "").strip()
        if not prompt or _looks_like_composite_prompt(prompt) or not _is_reference_anchored_prompt(prompt):
            continue
        prompts.append(_ensure_image_dataset_produce_constraints(prompt, plan, labels))
    if not prompts:
        return []
    count = max(1, int(total_count or len(prompts)))
    while len(prompts) < min(count, 12):
        prompts.append(prompts[len(prompts) % len(prompts)])
    return prompts


def _is_reference_anchored_prompt(prompt: str) -> bool:
    lowered = prompt.lower()
    markers = (
        "input image",
        "reference image",
        "provided image",
        "source image",
        "given image",
        "参考图",
        "输入图",
        "原图",
    )
    return any(marker in lowered for marker in markers)


def _looks_like_composite_prompt(prompt: str) -> bool:
    lowered = prompt.lower()
    markers = (
        "image2.zip",
        "image2",
        "composite",
        "compositing",
        "paste",
        "pasted",
        "cutout",
        "extract",
        "overlay",
        "合成",
        "贴到",
        "抠图",
    )
    return any(marker in lowered for marker in markers)


def _ensure_image_dataset_produce_constraints(prompt: str, plan: Dict[str, Any], labels: List[str]) -> str:
    label_text = ", ".join(labels or [str(item).strip() for item in plan.get("class_names", []) if str(item).strip()]) or "target objects"
    return (
        "Use the provided input image as the primary visual reference. "
        "Preserve its scene category, background layout, camera angle, composition, lighting direction, and visual style. "
        "Do not replace the scene with an unrelated location or invented environment. "
        f"{prompt.strip()} "
        f"Target classes for object detection: {label_text}. "
        "Generate a realistic variant of the same referenced scene, not an identical copy. "
        "Keep target objects clearly visible and annotatable while allowing small natural changes in pose, scale, occlusion, and object placement. "
        "No text, watermark, logo, collage, pasted object, or obvious image editing artifact."
    )


def _heuristic_image_dataset_produce_prompts(plan: Dict[str, Any], labels: List[str], total_count: int) -> List[str]:
    class_names = labels or [str(item).strip() for item in plan.get("class_names", []) if str(item).strip()]
    label_text = ", ".join(class_names) if class_names else "target objects"
    task = str(plan.get("task_description") or "object detection training").strip()
    base = (
        f"Create a realistic variant of the provided input image for this YOLO object detection task: {task}. "
        f"The image must contain clearly annotatable target classes: {label_text}. "
        "Use the input image as the primary reference for scene type, camera angle, composition, background layout, resolution, and lighting; "
        "do not invent a different location, and do not duplicate the input image exactly. "
    )
    variations = [
        "preserve the original scene and camera framing while slightly changing target pose, hand position, and object visibility",
        "preserve the original background and lighting while adding mild occlusion and natural scale variation for the target classes",
        "preserve the original environment while changing only small details such as posture, target placement, and local shadows",
        "preserve the original camera angle and composition while making target objects clearer and easier to annotate",
        "preserve the original scene type while introducing realistic clutter near the target without hiding it",
        "preserve the original visual style while creating a harder but still labelable sample with slight blur or lighting variation",
    ]
    prompts = [
        _ensure_image_dataset_produce_constraints(base + f"Scene variation: {variation}.", plan, class_names)
        for variation in variations
    ]
    count = max(1, int(total_count or 1))
    while len(prompts) < min(count, 12):
        prompts.extend(prompts)
    return prompts[: max(1, min(count, len(prompts)))]


def _generate_and_annotate_synthetic_with_produce(
    plan_path: Path,
    image1: Path,
    labels: List[str],
    annotation_prompts: List[str],
    prompt_label_map: Dict[str, str],
    intent_items: List[Dict[str, Any]],
    annotation_provider: str,
    output_dir: Path,
    dry_run: bool,
    planner_llm: Dict[str, Any] | None = None,
    produce_count_button: bool = True,
    fixed_produce_synthetic_count: int = 5,
    max_synthetic: int = 2000,
    random_seed: int | None = None,
) -> tuple[Path, Path]:
    if not PRODUCE_GENERATION_SCRIPT.is_file():
        raise FileNotFoundError(f"image-dataset-produce generation script not found: {PRODUCE_GENERATION_SCRIPT}")
    plan = _load_json(plan_path)
    reference_images = _collect_image1_reference_images(image1)
    if not reference_images:
        raise ValueError(f"No image1 reference images found for image-dataset-produce fallback: {image1}")

    synthetic_images_root = output_dir / "synthetic_images"
    synthetic_images_root.mkdir(parents=True, exist_ok=True)
    annotation_root = output_dir / "synthetic_annotations"
    annotation_root.mkdir(parents=True, exist_ok=True)
    partial_coco_files: List[Path] = []
    produce_total = _produce_synthetic_count(
        plan,
        len(reference_images),
        produce_count_button,
        fixed_produce_synthetic_count,
        max_synthetic,
    )
    if produce_total <= 0:
        raise ValueError("image-dataset-produce fallback requested 0 synthetic images")
    prompts, prompt_source = _fallback_prompt_sequence(plan, labels, produce_total, planner_llm)
    _write_json(
        output_dir / "generation_inputs" / "produce_prompts.json",
        {
            "prompt_source": prompt_source,
            "prompt_count": produce_total,
            "reference_image_root": str(image1),
            "labels": labels,
            "prompts": prompts,
        },
    )
    rng = random.Random(random_seed)

    for generation_index in range(1, produce_total + 1):
        reference_image = rng.choice(reference_images)
        prompt = prompts[(generation_index - 1) % len(prompts)]
        input_json = output_dir / "generation_inputs" / f"produce_{generation_index:05d}.json"
        _write_json(input_json, {"task": {"input_image": str(reference_image), "prompt": prompt, "output_dir": str(synthetic_images_root)}})
        before_files = {
            str(path.resolve())
            for path in synthetic_images_root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        }
        stdout_text = _run([sys.executable, str(PRODUCE_GENERATION_SCRIPT), "--input", str(input_json)], dry_run, retries=3, retry_sleep=8.0)
        if dry_run:
            continue
        generated_image = _extract_generated_image_path(stdout_text, synthetic_images_root, before_files)
        if generated_image is None:
            raise RuntimeError(f"image-dataset-produce completed but no generated image was found for {input_json}")
        generated_image = _move_to_unique_synthetic_name(generated_image, synthetic_images_root, generation_index, 1)
        _safe_print(f"[data-prep] annotating image-dataset-produce image immediately: {generated_image}")
        partial_coco = annotation_root / f"{generated_image.stem}_coco.json"
        _annotate_one_synthetic(generated_image, labels, annotation_prompts, prompt_label_map, intent_items, annotation_provider, partial_coco, dry_run)
        partial_coco_files.append(partial_coco)

    synthetic_coco = output_dir / "synthetic_coco.json"
    if not dry_run:
        _combine_coco_files(partial_coco_files, synthetic_coco)
    return synthetic_images_root, synthetic_coco


def _exception_text(exc: Exception) -> str:
    parts = [str(exc)]
    if isinstance(exc, subprocess.CalledProcessError):
        parts.append(_decode_bytes(exc.output))
        parts.append(_decode_bytes(exc.stderr))
        parts.extend(str(item) for item in getattr(exc, "cmd", []) or [])
    return "\n".join(part for part in parts if part)


def _looks_like_generation_api_unavailable(exc: Exception) -> bool:
    text = _exception_text(exc).lower()
    markers = (
        "quota",
        "credit",
        "credits",
        "balance",
        "insufficient",
        "billing",
        "payment",
        "rate limit",
        "rate_limit",
        "too many requests",
        "429",
        "402",
        "401",
        "403",
        "api-key is blocked",
        "blocked",
        "tokencloud",
        "wan2.7",
        "额度",
        "配额",
        "余额",
        "欠费",
        "限额",
        "频率",
        "账户",
        "账号",
        "不可用",
    )
    if any(marker in text for marker in markers):
        return True
    return isinstance(exc, subprocess.CalledProcessError) and "run_composite.py" in text


def _synthetic_generation_failure_payload(exc: Exception) -> Dict[str, Any]:
    stdout_tail = ""
    stderr_tail = ""
    if isinstance(exc, subprocess.CalledProcessError):
        stdout_tail = _decode_bytes(exc.output)[-2000:]
        stderr_tail = _decode_bytes(exc.stderr)[-2000:]
    detail = stderr_tail or stdout_tail or str(exc)
    return {
        "synthetic_generation_status": "failed",
        "synthetic_generation_error": detail[-2000:],
        "synthetic_generation_error_type": exc.__class__.__name__,
        "synthetic_generation_fallback": "real_dataset_only",
        "synthetic_generation_stdout_tail": stdout_tail,
        "synthetic_generation_stderr_tail": stderr_tail,
    }


def _merge(real_coco: Path, synthetic_coco: Path, real_root: Path, synthetic_root: Path, output_dir: Path, dry_run: bool) -> Path:
    merged_coco = output_dir / "merged_coco.json"
    merged_images = output_dir / "merged_images"
    _run([
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
    ], dry_run)
    return merged_coco


def _export_combined_dataset(coco_path: Path, image_root: Path, output_dir: Path, dry_run: bool) -> Path:
    combined_dir = output_dir / "combined_dataset"
    if dry_run:
        return combined_dir
    combined_images = combined_dir / "images"
    combined_images.mkdir(parents=True, exist_ok=True)
    for image_path in image_root.rglob("*"):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        target = combined_images / image_path.name
        counter = 1
        while target.exists():
            target = combined_images / f"{image_path.stem}_{counter}{image_path.suffix.lower()}"
            counter += 1
        target.write_bytes(image_path.read_bytes())
    _write_json(combined_dir / "annotations.coco.json", _load_json(coco_path))
    return combined_dir


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


def _normalize_yolo_task(value: Any) -> str:
    text = str(value or "detect").strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"segment", "seg", "segmentation", "instance_segmentation"}:
        return "segment"
    return "detect"


def _normalize_annotation_provider(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    if text in {"locate", "locateanything", "locate_anything", "locate_sam3"}:
        return "locate_sam3"
    return "sam3"


def _list_value(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    return [item.strip() for item in re.split(r"[,，、\s]+", text) if item.strip()]


def _dict_value(value: Any) -> Dict[str, str]:
    if isinstance(value, dict):
        return {str(key).strip(): str(item).strip() for key, item in value.items() if str(key).strip() and str(item).strip()}
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return _dict_value(parsed)
    return {}


def _intent_items_value(value: Any) -> List[Dict[str, Any]]:
    if isinstance(value, list):
        return [dict(item) for item in value if isinstance(item, dict)]
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return _intent_items_value(parsed)
    return []


def _apply_input_json(args: argparse.Namespace) -> argparse.Namespace:
    payload = _load_input_json(args.input_json)
    if not payload:
        return args
    spec = payload.get("spec") if isinstance(payload.get("spec"), dict) else payload
    ctx = spec.get("workflow_context") if isinstance(spec.get("workflow_context"), dict) else {}
    split = spec.get("split") if isinstance(spec.get("split"), dict) else {}
    dataset = spec.get("dataset") if isinstance(spec.get("dataset"), dict) else {}
    output = spec.get("output") if isinstance(spec.get("output"), dict) else {}

    args.dataset_root = _coalesce(args.dataset_root, dataset.get("root_dir"), ctx.get("dataset_root"), spec.get("dataset_root"), default="")
    args.coco_json = _coalesce(args.coco_json, dataset.get("coco_json"), ctx.get("coco_json"), spec.get("coco_json"), default="")
    args.ref_image = _coalesce(args.ref_image, ctx.get("ref_image"), spec.get("ref_image"), spec.get("reference_image"), default=None)
    args.image1 = _coalesce(args.image1, spec.get("image1"), spec.get("input_image1"), ctx.get("image1"), args.ref_image, default=None)
    args.image2 = _coalesce(args.image2, spec.get("image2"), spec.get("input_image2"), ctx.get("image2"), default=None)
    args.task = _coalesce(args.task, spec.get("task"), spec.get("task_description"), ctx.get("task"), default="")
    args.generation_prompt = _coalesce(args.generation_prompt, spec.get("generation_prompt"), spec.get("prompt"), ctx.get("generation_prompt"), default="")
    args.labels = args.labels or _list_value(_coalesce(spec.get("labels"), spec.get("class_names"), dataset.get("class_names"), ctx.get("labels"), default=[]))
    args.annotation_prompts = args.annotation_prompts or _list_value(_coalesce(spec.get("annotation_prompts"), ctx.get("annotation_prompts"), default=[]))
    args.annotation_prompt_map = _dict_value(_coalesce(spec.get("annotation_prompt_map"), spec.get("prompt_label_map"), ctx.get("annotation_prompt_map"), ctx.get("prompt_label_map"), args.annotation_prompt_map, default={}))
    args.annotation_provider = _normalize_annotation_provider(
        _coalesce(
            args.annotation_provider,
            spec.get("annotation_provider"),
            ctx.get("annotation_provider"),
            DEFAULT_ANNOTATION_PROVIDER,
            default=DEFAULT_ANNOTATION_PROVIDER,
        )
    )
    args.intent_items = args.intent_items or _intent_items_value(_coalesce(spec.get("intent_items"), ctx.get("intent_items"), default=[]))
    args.work_dir = _coalesce(args.work_dir, ctx.get("work_dir"), spec.get("work_dir"), output.get("work_dir"), default="")
    args.output_dir = _coalesce(args.output_dir, ctx.get("output_dir"), spec.get("output_dir"), output.get("output_dir"), default="")
    args.skip_generation = args.skip_generation or _bool_value(spec.get("skip_generation"), False)
    args.generation_skip_reason = str(
        _coalesce(
            args.generation_skip_reason,
            spec.get("generation_skip_reason"),
            ctx.get("generation_skip_reason"),
            default="",
        )
        or ""
    ).strip()
    args.reuse_synthetic_data = args.reuse_synthetic_data or _bool_value(spec.get("reuse_synthetic_data"), False)
    args.dry_run = args.dry_run or _bool_value(spec.get("dry_run"), False)
    args.max_synthetic = int(_coalesce(spec.get("max_synthetic"), args.max_synthetic, default=args.max_synthetic))
    args.synthetic_count_button = _bool_value(_coalesce(spec.get("synthetic_count_button"), args.synthetic_count_button, default=True), True)
    args.fixed_synthetic_count = int(_coalesce(spec.get("fixed_synthetic_count"), args.fixed_synthetic_count, default=args.fixed_synthetic_count))
    args.produce_count_button = _bool_value(_coalesce(spec.get("produce_count_button"), ctx.get("produce_count_button"), args.produce_count_button, default=True), True)
    args.fixed_produce_synthetic_count = int(_coalesce(spec.get("fixed_produce_synthetic_count"), ctx.get("fixed_produce_synthetic_count"), args.fixed_produce_synthetic_count, default=args.fixed_produce_synthetic_count))
    args.split_requested = args.split_requested or _bool_value(spec.get("split_requested"), False)
    args.planner_llm = spec.get("planner_llm") if isinstance(spec.get("planner_llm"), dict) else {}
    args.split_train = float(_coalesce(split.get("train"), dataset.get("split_train"), args.split_train, default=args.split_train))
    args.split_val = float(_coalesce(split.get("val"), dataset.get("split_val"), args.split_val, default=args.split_val))
    args.split_test = float(_coalesce(split.get("test"), dataset.get("split_test"), args.split_test, default=args.split_test))
    training = spec.get("training") if isinstance(spec.get("training"), dict) else {}
    args.training_task = _normalize_yolo_task(_coalesce(training.get("task"), spec.get("training_task"), args.training_task, default=args.training_task))
    return args


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare YOLO dataset: inspect, annotate, compose synthetic images, merge, split.")
    parser.add_argument("--input-json", default="")
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--coco-json", default="")
    parser.add_argument("--task", default="")
    parser.add_argument("--generation-prompt", default="")
    parser.add_argument("--labels", nargs="*", default=[])
    parser.add_argument("--annotation-prompts", nargs="*", default=[])
    parser.add_argument("--annotation-prompt-map-json", default="")
    parser.add_argument("--annotation-provider", default="", choices=["", "sam3", "locate_sam3"])
    parser.add_argument("--work-dir", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--ref-image", default=None)
    parser.add_argument("--image1", default=None)
    parser.add_argument("--image2", default=None)
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--generation-skip-reason", default="")
    parser.add_argument("--reuse-synthetic-data", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-synthetic", type=int, default=2000)
    parser.add_argument("--synthetic-count-button", default=True)
    parser.add_argument("--fixed-synthetic-count", type=int, default=20)
    parser.add_argument("--produce-count-button", default=True)
    parser.add_argument("--fixed-produce-synthetic-count", type=int, default=5)
    parser.add_argument("--split-requested", action="store_true")
    parser.set_defaults(planner_llm={})
    parser.add_argument("--split-train", type=float, default=0.7)
    parser.add_argument("--split-val", type=float, default=0.2)
    parser.add_argument("--split-test", type=float, default=0.1)
    parser.add_argument("--training-task", default="detect", choices=["detect", "segment"])
    parser.set_defaults(annotation_prompt_map={})
    parser.set_defaults(intent_items=[])
    args = parser.parse_args()
    if args.annotation_prompt_map_json:
        args.annotation_prompt_map = _dict_value(args.annotation_prompt_map_json)
    args = _apply_input_json(args)

    if not args.dataset_root:
        raise ValueError("dataset_root is required")
    if not args.work_dir:
        raise ValueError("work_dir is required")
    if not args.output_dir:
        raise ValueError("output_dir is required")

    work_dir = Path(args.work_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    _initialize_log_dir(work_dir)

    dataset_root = Path(args.dataset_root).resolve()
    inspection = _inspect(dataset_root, work_dir, args.dry_run)
    if args.dry_run:
        _safe_print("[data-prep] dry-run completed after command planning.")
        return
    inspection = _sanitize_inspection_images(inspection, work_dir, args.dry_run)

    using_uploaded_coco = False
    if args.coco_json:
        real_coco = Path(args.coco_json).resolve()
        if not real_coco.exists():
            raise FileNotFoundError(f"Provided coco_json does not exist: {real_coco}")
        if int(inspection.get("invalid_image_count") or 0) > 0:
            clean_images_dir = Path(str(inspection["images_dir"])).resolve()
            real_coco = _filter_coco_for_valid_images(
                real_coco,
                clean_images_dir,
                _collect_images_for_validation(clean_images_dir),
                work_dir / "provided_filtered_valid_images.coco.json",
            )
        using_uploaded_coco = True
    else:
        real_coco = _prepare_real_coco(
            inspection,
            work_dir,
            args.labels,
            args.annotation_prompts,
            args.annotation_prompt_map,
            args.intent_items,
            args.annotation_provider,
            args.dry_run,
        )
        using_uploaded_coco = str(inspection.get("format") or "").lower() == "coco"

    requested_labels = list(args.labels)
    image_validation = {
        "original_image_count": inspection.get("original_image_count", inspection.get("image_count")),
        "valid_image_count": inspection.get("valid_image_count", inspection.get("image_count")),
        "invalid_image_count": inspection.get("invalid_image_count", 0),
        "invalid_images_report": inspection.get("invalid_images_report"),
        "original_images_dir": inspection.get("original_images_dir"),
        "validated_images_dir": inspection.get("images_dir"),
    }
    label_policy: dict[str, Any] = {
        "mode": "auto",
        "requested_labels": requested_labels,
        "label_source": "llm_labels",
        "coco_labels": [],
        "final_labels": requested_labels,
        "llm_only": [],
        "coco_only": [],
    }
    if using_uploaded_coco:
        coco_labels = _coco_category_names(real_coco)
        if coco_labels:
            args.labels = coco_labels
            requested_set = set(requested_labels)
            coco_set = set(coco_labels)
            label_policy.update({
                "label_source": "coco_categories",
                "coco_labels": coco_labels,
                "final_labels": coco_labels,
                "llm_only": [label for label in requested_labels if label not in coco_set],
                "coco_only": [label for label in coco_labels if label not in requested_set],
                "reason": "uploaded_coco_categories_take_precedence",
            })
            _safe_print(
                "[data-prep] label_policy=auto; using uploaded COCO categories as final labels: "
                + ",".join(coco_labels),
            )

    training_coco = real_coco
    training_root = Path(inspection["images_dir"]).resolve()
    synthetic_policy_enabled = False
    plan_path = ""
    skipped_for_sufficient_data = args.skip_generation and args.generation_skip_reason == "sufficient_data_samples"
    synthetic_status: Dict[str, Any] = {
        "synthetic_generation_status": "skipped_sufficient_data" if skipped_for_sufficient_data else ("not_requested" if args.skip_generation else "not_planned"),
        "synthetic_generation_error": "",
        "synthetic_generation_fallback": "",
        "synthetic_pending_inputs": False,
        "synthetic_generation_skip_reason": args.generation_skip_reason if args.skip_generation else "",
        "synthetic_generation_skip_description": "Data samples are sufficient." if skipped_for_sufficient_data else "",
    }

    if skipped_for_sufficient_data and not args.reuse_synthetic_data:
        _safe_print("[data-prep] Synthetic generation skipped. Data samples are sufficient.")
        _emit_synthetic_generation_progress(
            "skipped",
            planned=0,
            completed=0,
            success=0,
            failed=0,
            reason_code="sufficient_data_samples",
            reason="Data samples are sufficient.",
        )

    if args.reuse_synthetic_data:
        synthetic_policy_enabled = True
        synthetic_status.update({
            "synthetic_generation_status": "reused",
            "synthetic_generation_fallback": "",
            "synthetic_pending_inputs": False,
        })

    if not args.skip_generation:
        plan = _plan_synthetic(
            real_coco,
            args.task,
            args.generation_prompt,
            args.split_train,
            work_dir,
            args.max_synthetic,
            args.planner_llm,
            args.synthetic_count_button,
            args.fixed_synthetic_count,
            args.dry_run,
        )
        plan_path = str(plan)
        recommended_count = _read_recommended_synthetic_count(plan)
        _safe_print(f"[data-prep] recommended_synthetic_count={recommended_count}")
        if recommended_count > 0:
            if not args.image1 or not args.image2:
                _safe_print("[data-prep] Synthetic generation skipped. Data samples are sufficient.")
                synthetic_status.update({
                    "synthetic_generation_status": "skipped_sufficient_data",
                    "synthetic_pending_inputs": False,
                    "synthetic_generation_skip_reason": "sufficient_data_samples",
                    "synthetic_generation_skip_description": "Data samples are sufficient.",
                })
                _emit_synthetic_generation_progress(
                    "skipped",
                    planned=0,
                    completed=0,
                    success=0,
                    failed=0,
                    reason_code="sufficient_data_samples",
                    reason="Data samples are sufficient.",
                )
            else:
                _emit_synthetic_generation_progress("running", planned=recommended_count, completed=0)
                try:
                    synthetic_root, synthetic_coco = _generate_and_annotate_synthetic(
                        plan,
                        Path(args.image1).resolve(),
                        Path(args.image2).resolve(),
                        args.labels,
                        args.annotation_prompts,
                        args.annotation_prompt_map,
                        args.intent_items,
                        args.annotation_provider,
                        work_dir,
                        args.dry_run,
                    )
                    training_coco = _merge(real_coco, synthetic_coco, Path(inspection["images_dir"]).resolve(), synthetic_root, work_dir, args.dry_run)
                    training_root = work_dir / "merged_images"
                    synthetic_policy_enabled = True
                    synthetic_status.update({
                        "synthetic_generation_status": "merged",
                        "synthetic_images": str(synthetic_root),
                        "synthetic_coco": str(synthetic_coco),
                    })
                    generated_count = _count_generated_images(synthetic_root)
                    _emit_synthetic_generation_progress(
                        "completed",
                        planned=recommended_count,
                        completed=generated_count,
                        success=generated_count,
                        failed=0,
                    )
                except Exception as exc:
                    if _looks_like_generation_api_unavailable(exc):
                        primary_payload = _synthetic_generation_failure_payload(exc)
                        message = primary_payload.get("synthetic_generation_error") or str(exc)
                        _safe_print(f"[data-prep] Composite generation unavailable; falling back to image-dataset-produce: {message}", file=sys.stderr)
                        try:
                            synthetic_root, synthetic_coco = _generate_and_annotate_synthetic_with_produce(
                                plan,
                                Path(args.image1).resolve(),
                                args.labels,
                                args.annotation_prompts,
                                args.annotation_prompt_map,
                                args.intent_items,
                                args.annotation_provider,
                                work_dir,
                                args.dry_run,
                                args.planner_llm,
                                args.produce_count_button,
                                args.fixed_produce_synthetic_count,
                                args.max_synthetic,
                            )
                            training_coco = _merge(real_coco, synthetic_coco, Path(inspection["images_dir"]).resolve(), synthetic_root, work_dir, args.dry_run)
                            training_root = work_dir / "merged_images"
                            synthetic_policy_enabled = True
                            synthetic_status.update({
                                "synthetic_generation_status": "merged",
                                "synthetic_generation_fallback": "image-dataset-produce",
                                "synthetic_generation_primary_error": primary_payload.get("synthetic_generation_error", ""),
                                "synthetic_images": str(synthetic_root),
                                "synthetic_coco": str(synthetic_coco),
                            })
                            generated_count = _count_generated_images(synthetic_root)
                            _emit_synthetic_generation_progress(
                                "completed",
                                planned=recommended_count,
                                completed=generated_count,
                                success=generated_count,
                                failed=0,
                                fallback_used=True,
                            )
                        except Exception as fallback_exc:
                            synthetic_status.update(_synthetic_generation_failure_payload(fallback_exc))
                            synthetic_status["synthetic_generation_primary_error"] = primary_payload.get("synthetic_generation_error", "")
                            message = synthetic_status.get("synthetic_generation_error") or str(fallback_exc)
                            _safe_print(f"[data-prep] image-dataset-produce fallback failed; continuing with real dataset only: {message}", file=sys.stderr)
                            _emit_synthetic_generation_progress(
                                _synthetic_generation_terminal_status(synthetic_status),
                                planned=recommended_count,
                                completed=0,
                                success=0,
                                failed=recommended_count,
                                error=message,
                                fallback_used=True,
                            )
                    else:
                        synthetic_status.update(_synthetic_generation_failure_payload(exc))
                        message = synthetic_status.get("synthetic_generation_error") or str(exc)
                        _safe_print(f"[data-prep] Synthetic generation failed; continuing with real dataset only: {message}", file=sys.stderr)
                        _emit_synthetic_generation_progress(
                            _synthetic_generation_terminal_status(synthetic_status),
                            planned=recommended_count,
                            completed=0,
                            success=0,
                            failed=recommended_count,
                            error=message,
                        )
        else:
            _safe_print("[data-prep] Synthetic generation skipped because planner recommended 0 images.")
            synthetic_status.update({"synthetic_generation_status": "skipped_zero_recommendation"})

    if not args.split_requested:
        combined_dir = _export_combined_dataset(training_coco, training_root, output_dir, args.dry_run)
        summary = {
            "mode": "combined_dataset",
            "real_coco": str(real_coco),
            "training_coco": str(training_coco),
            "training_root": str(training_root),
            "combined_dataset": str(combined_dir),
            "synthetic_plan": plan_path,
            "work_dir": str(work_dir),
            "output_dir": str(output_dir),
            "label_policy": label_policy,
            "image_validation": image_validation,
            "labels_requested": requested_labels,
            "labels_final": args.labels,
            **synthetic_status,
        }
        summary_path = output_dir / "data_preparation_summary.json"
        _write_json(summary_path, summary)
        _safe_print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    class_names = ",".join(args.labels)
    prep_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "prepare_yolo_dataset.py"),
        "--dataset-root",
        str(training_root),
        "--coco-json",
        str(training_coco),
        "--output-dir",
        str(output_dir),
        "--class-names",
        class_names,
        "--task",
        args.training_task,
        "--split-train",
        str(args.split_train),
        "--split-val",
        str(args.split_val),
        "--split-test",
        str(args.split_test),
    ]
    if synthetic_policy_enabled:
        prep_cmd.append("--synthetic-policy")
    _run(prep_cmd, args.dry_run)

    summary_path = output_dir / "data_preparation_summary.json"
    summary = _load_json(summary_path)
    summary.update({
        "real_coco": str(real_coco),
        "training_coco": str(training_coco),
        "training_root": str(training_root),
        "synthetic_plan": plan_path,
        "work_dir": str(work_dir),
        "output_dir": str(output_dir),
        "label_policy": label_policy,
        "image_validation": image_validation,
        "labels_requested": requested_labels,
        "labels_final": args.labels,
        **synthetic_status,
    })
    _write_json(summary_path, summary)
    _safe_print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
