import argparse
import json
import os
import random
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from urllib import request

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_ROOT = SCRIPT_DIR.parent
DATASET_PROCESS_ROOT = SKILL_ROOT.parent
AUTO_ANNOTATION_SCRIPT = SCRIPT_DIR / "sam3-predict.py"
GENERATION_SCRIPT = DATASET_PROCESS_ROOT / "image-dataset-generation" / "scripts" / "run_composite.py"
PRODUCE_GENERATION_SCRIPT = DATASET_PROCESS_ROOT / "image-dataset-produce" / "scripts" / "run_generation.py"
LOG_DIR: Path | None = None

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def _write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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
    print("[data-prep] " + " ".join(cmd))
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
        print(message)
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
                print(text, end="" if text.endswith("\n") else "\n", file=target, flush=True)
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


def _prepare_real_coco(
    inspection: Dict,
    work_dir: Path,
    task_labels: List[str],
    annotation_prompts: List[str],
    prompt_label_map: Dict[str, str],
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
        prompts = _annotation_prompts_for_sam3(task_labels, annotation_prompts, prompt_label_map)
        try:
            cmd = [
                sys.executable,
                str(AUTO_ANNOTATION_SCRIPT),
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
    print(f"[data-prep] per-image real coco written: {sidecar}", flush=True)
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
) -> List[str]:
    prompts = _dedupe_text([str(item).strip() for item in (annotation_prompts or []) if str(item).strip()])
    prompt_map = {str(prompt).strip(): str(label).strip() for prompt, label in (prompt_label_map or {}).items() if str(prompt).strip() and str(label).strip()}
    selected: List[str] = []
    for label in _dedupe_text(labels):
        prompt = _best_annotation_prompt_for_label(label, labels, prompts, prompt_map)
        if prompt:
            selected.append(prompt)
    return _dedupe_text(selected or prompts or labels)


def _best_annotation_prompt_for_label(
    label: str,
    labels: List[str],
    prompts: List[str],
    prompt_label_map: Dict[str, str],
) -> str:
    class_key = str(label or "").strip().lower()
    if not class_key:
        return ""
    # 默认每个训练 label 只取一个最代表性的自然语言 prompt 去请求 SAM3。
    # 多 prompt 虽能提高召回，但会让同一个目标被重复标注；这里从源头降低重复框。
    for prompt, mapped_label in prompt_label_map.items():
        if mapped_label == label:
            return prompt
    for prompt in prompts:
        if prompt.lower() != class_key and (len(labels) == 1 or _looks_related_prompt(prompt, label)):
            return prompt
    return _label_to_sam3_prompt(label) or label


def _prompt_label_map_for_sam3(labels: List[str], prompts: List[str], prompt_label_map: Dict[str, str] | None) -> Dict[str, str]:
    class_names = _dedupe_text(labels)
    result: Dict[str, str] = {}
    for prompt, label in (prompt_label_map or {}).items():
        prompt_text = str(prompt or "").strip()
        label_text = str(label or "").strip()
        if prompt_text and label_text in class_names:
            result[prompt_text] = label_text
    if len(class_names) == 1:
        for prompt in prompts:
            result.setdefault(prompt, class_names[0])
    for label in class_names:
        result.setdefault(_label_to_sam3_prompt(label), label)
        result.setdefault(label, label)
    return result


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
    output_json: Path,
    dry_run: bool,
) -> Path:
    if not labels:
        raise ValueError("Synthetic auto-annotation requires labels")
    prompts = _annotation_prompts_for_sam3(labels, annotation_prompts, prompt_label_map)
    _run([
        sys.executable,
        str(AUTO_ANNOTATION_SCRIPT),
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
        print(f"[data-prep] synthetic coco written: {output_json}", flush=True)
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
            print(f"[data-prep] annotating composite image immediately: {generated_image}")
            partial_coco = annotation_root / f"{generated_image.stem}_coco.json"
            _annotate_one_synthetic(generated_image, labels, annotation_prompts, prompt_label_map, partial_coco, dry_run)
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
        print(f"[data-prep] image-dataset-produce prompt planner unavailable; using local prompt template: {exc}", file=sys.stderr)
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
        print(f"[data-prep] annotating image-dataset-produce image immediately: {generated_image}")
        partial_coco = annotation_root / f"{generated_image.stem}_coco.json"
        _annotate_one_synthetic(generated_image, labels, annotation_prompts, prompt_label_map, partial_coco, dry_run)
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
    args.work_dir = _coalesce(args.work_dir, ctx.get("work_dir"), spec.get("work_dir"), output.get("work_dir"), default="")
    args.output_dir = _coalesce(args.output_dir, ctx.get("output_dir"), spec.get("output_dir"), output.get("output_dir"), default="")
    args.skip_generation = args.skip_generation or _bool_value(spec.get("skip_generation"), False)
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
    parser.add_argument("--work-dir", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--ref-image", default=None)
    parser.add_argument("--image1", default=None)
    parser.add_argument("--image2", default=None)
    parser.add_argument("--skip-generation", action="store_true")
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
        print("[data-prep] dry-run completed after command planning.")
        return

    if args.coco_json:
        real_coco = Path(args.coco_json).resolve()
        if not real_coco.exists():
            raise FileNotFoundError(f"Provided coco_json does not exist: {real_coco}")
    else:
        real_coco = _prepare_real_coco(
            inspection,
            work_dir,
            args.labels,
            args.annotation_prompts,
            args.annotation_prompt_map,
            args.dry_run,
        )

    training_coco = real_coco
    training_root = Path(inspection["images_dir"]).resolve()
    synthetic_policy_enabled = False
    plan_path = ""
    synthetic_status: Dict[str, Any] = {
        "synthetic_generation_status": "not_requested" if args.skip_generation else "not_planned",
        "synthetic_generation_error": "",
        "synthetic_generation_fallback": "",
        "synthetic_pending_inputs": False,
    }

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
        print(f"[data-prep] recommended_synthetic_count={recommended_count}")
        if recommended_count > 0:
            if not args.image1 or not args.image2:
                print("[data-prep] Synthetic generation is pending because image1/image2 inputs were not provided.")
                synthetic_status.update({
                    "synthetic_generation_status": "pending_inputs",
                    "synthetic_pending_inputs": True,
                })
            else:
                try:
                    synthetic_root, synthetic_coco = _generate_and_annotate_synthetic(
                        plan,
                        Path(args.image1).resolve(),
                        Path(args.image2).resolve(),
                        args.labels,
                        args.annotation_prompts,
                        args.annotation_prompt_map,
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
                except Exception as exc:
                    if _looks_like_generation_api_unavailable(exc):
                        primary_payload = _synthetic_generation_failure_payload(exc)
                        message = primary_payload.get("synthetic_generation_error") or str(exc)
                        print(f"[data-prep] Composite generation unavailable; falling back to image-dataset-produce: {message}", file=sys.stderr)
                        try:
                            synthetic_root, synthetic_coco = _generate_and_annotate_synthetic_with_produce(
                                plan,
                                Path(args.image1).resolve(),
                                args.labels,
                                args.annotation_prompts,
                                args.annotation_prompt_map,
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
                        except Exception as fallback_exc:
                            synthetic_status.update(_synthetic_generation_failure_payload(fallback_exc))
                            synthetic_status["synthetic_generation_primary_error"] = primary_payload.get("synthetic_generation_error", "")
                            message = synthetic_status.get("synthetic_generation_error") or str(fallback_exc)
                            print(f"[data-prep] image-dataset-produce fallback failed; continuing with real dataset only: {message}", file=sys.stderr)
                    else:
                        synthetic_status.update(_synthetic_generation_failure_payload(exc))
                        message = synthetic_status.get("synthetic_generation_error") or str(exc)
                        print(f"[data-prep] Synthetic generation failed; continuing with real dataset only: {message}", file=sys.stderr)
        else:
            print("[data-prep] Synthetic generation skipped because planner recommended 0 images.")
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
            **synthetic_status,
        }
        summary_path = output_dir / "data_preparation_summary.json"
        _write_json(summary_path, summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
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
        **synthetic_status,
    })
    _write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
