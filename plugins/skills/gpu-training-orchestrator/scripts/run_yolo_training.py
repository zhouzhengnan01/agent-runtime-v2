import argparse
import json
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

_log_file = None  # module-level tee target


class _TeeWriter:
    """Wraps a stream so every write also goes to a log file."""

    def __init__(self, original, log_path: Path):
        self._original = original
        self._log_fh = log_path.open("a", encoding="utf-8")

    def write(self, data):
        self._original.write(data)
        if self._log_fh and data:
            self._log_fh.write(data)
            self._log_fh.flush()
        return len(data) if data else 0

    def flush(self):
        self._original.flush()
        if self._log_fh:
            self._log_fh.flush()

    def fileno(self):
        return self._original.fileno()

    def __getattr__(self, name):
        return getattr(self._original, name)


def _install_tee(log_path: Path) -> None:
    global _log_file
    _log_file = log_path
    sys.stdout = _TeeWriter(sys.stdout, log_path)
    sys.stderr = _TeeWriter(sys.stderr, log_path)


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def log_info(msg: str) -> None:
    print(f"[INFO] {msg}")


def log_warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _validate_split(split: Dict[str, float]) -> None:
    for key in ("train", "val", "test"):
        if key not in split:
            raise ValueError(f"dataset.split 缺少字段: {key}")
        if split[key] < 0:
            raise ValueError(f"dataset.split.{key} 不能小于 0")
    total = float(split["train"]) + float(split["val"]) + float(split["test"])
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"dataset.split 的和必须为 1.0，当前为 {total}")


def _check_conda_runtime(cfg: Dict) -> str:
    runtime_cfg = cfg.get("runtime", {})
    conda_env_name = str(runtime_cfg.get("conda_env_name", "")).strip()
    enforce = bool(runtime_cfg.get("enforce_conda_env", True))

    if not conda_env_name:
        raise ValueError("input.json 缺少 runtime.conda_env_name")

    current_env = str(os.environ.get("CONDA_DEFAULT_ENV", "")).strip()
    if current_env != conda_env_name:
        msg = (
            f"当前环境为 '{current_env or 'N/A'}'，但 input.json 要求 conda 环境 '{conda_env_name}'。"
            f"建议使用 conda run -n {conda_env_name} python ... 运行。"
        )
        if enforce:
            raise EnvironmentError(msg)
        log_warn(msg)

    return conda_env_name


def _load_coco(coco_json_path: Path) -> Dict:
    coco = _load_json(coco_json_path)
    for key in ("images", "annotations", "categories"):
        if key not in coco:
            raise ValueError(f"COCO 文件缺少字段: {key}")
    return coco


def _resolve_image_path(dataset_root: Path, file_name: str, cache: Dict[str, Path]) -> Path:
    direct = dataset_root / file_name
    if direct.exists() and direct.suffix.lower() in IMAGE_EXTS:
        return direct

    basename = Path(file_name).name
    if basename in cache:
        return cache[basename]

    candidate = dataset_root / basename
    if candidate.exists() and candidate.suffix.lower() in IMAGE_EXTS:
        cache[basename] = candidate
        return candidate

    for path in dataset_root.rglob(basename):
        if path.suffix.lower() in IMAGE_EXTS:
            cache[basename] = path
            return path

    raise FileNotFoundError(f"未找到图片文件: {file_name}（dataset.root_dir={dataset_root}）")


def _build_category_mapping(coco: Dict, forced_class_names: List[str]) -> Tuple[List[str], Dict[int, int]]:
    categories = sorted(coco["categories"], key=lambda x: int(x["id"]))
    coco_cat_ids = [int(c["id"]) for c in categories]
    coco_cat_names = [str(c["name"]) for c in categories]

    if forced_class_names:
        class_names = [str(x) for x in forced_class_names]
        name_to_index = {name: idx for idx, name in enumerate(class_names)}
        cat_id_to_index: Dict[int, int] = {}

        for cat in categories:
            name = str(cat["name"])
            if name not in name_to_index:
                raise ValueError(
                    f"dataset.class_names 中缺少 COCO 类别 '{name}'。"
                    "请补齐 class_names 或移除强制映射。"
                )
            cat_id_to_index[int(cat["id"])] = name_to_index[name]
        return class_names, cat_id_to_index

    class_names = coco_cat_names
    cat_id_to_index = {cat_id: idx for idx, cat_id in enumerate(coco_cat_ids)}
    return class_names, cat_id_to_index


def _split_image_ids(image_ids: List[int], split: Dict[str, float], seed: int) -> Dict[str, List[int]]:
    random.seed(seed)
    ids = image_ids[:]
    random.shuffle(ids)

    total = len(ids)
    n_train = int(total * float(split["train"]))
    n_val = int(total * float(split["val"]))

    train_ids = ids[:n_train]
    val_ids = ids[n_train : n_train + n_val]
    test_ids = ids[n_train + n_val :]

    if len(train_ids) + len(val_ids) + len(test_ids) != total:
        raise RuntimeError("数据集划分计数异常")

    return {"train": train_ids, "val": val_ids, "test": test_ids}


def _to_yolo_detect_line(bbox: List[float], width: int, height: int, cls_idx: int) -> str:
    x, y, w, h = [float(v) for v in bbox]
    xc = (x + w / 2.0) / float(width)
    yc = (y + h / 2.0) / float(height)
    wn = w / float(width)
    hn = h / float(height)
    return f"{cls_idx} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}"


def _to_yolo_segment_line(segmentation, bbox: List[float], width: int, height: int, cls_idx: int) -> str:
    points: List[float] = []

    if isinstance(segmentation, list) and segmentation:
        polygon = segmentation[0]
        if isinstance(polygon, list) and len(polygon) >= 6 and len(polygon) % 2 == 0:
            points = [float(v) for v in polygon]

    if not points:
        x, y, w, h = [float(v) for v in bbox]
        points = [
            x,
            y,
            x + w,
            y,
            x + w,
            y + h,
            x,
            y + h,
        ]

    normalized = []
    for i, value in enumerate(points):
        if i % 2 == 0:
            normalized.append(value / float(width))
        else:
            normalized.append(value / float(height))

    coords = " ".join(f"{v:.6f}" for v in normalized)
    return f"{cls_idx} {coords}"


def _prepare_yolo_dataset(
    dataset_root: Path,
    coco: Dict,
    class_names: List[str],
    cat_id_to_index: Dict[int, int],
    split_image_ids: Dict[str, List[int]],
    prepared_root: Path,
    task: str,
    copy_images: bool,
) -> Dict[str, int]:
    images_by_id = {int(img["id"]): img for img in coco["images"]}
    anns_by_image_id: Dict[int, List[Dict]] = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image_id[int(ann["image_id"])].append(ann)

    _ensure_dir(prepared_root)
    for split_name in ("train", "val", "test"):
        _ensure_dir(prepared_root / "images" / split_name)
        _ensure_dir(prepared_root / "labels" / split_name)

    cache: Dict[str, Path] = {}
    split_counts = {"train": 0, "val": 0, "test": 0}

    for split_name, ids in split_image_ids.items():
        for image_id in ids:
            image_obj = images_by_id.get(image_id)
            if not image_obj:
                log_warn(f"跳过未知 image_id: {image_id}")
                continue

            file_name = str(image_obj["file_name"])
            width = int(image_obj["width"])
            height = int(image_obj["height"])

            src_img = _resolve_image_path(dataset_root, file_name, cache)
            dst_img_name = Path(file_name).name
            dst_img = prepared_root / "images" / split_name / dst_img_name
            dst_lbl = prepared_root / "labels" / split_name / f"{Path(dst_img_name).stem}.txt"

            if copy_images:
                shutil.copy2(src_img, dst_img)
            else:
                if not dst_img.exists():
                    os.link(src_img, dst_img)

            lines = []
            for ann in anns_by_image_id.get(image_id, []):
                cat_id = int(ann["category_id"])
                if cat_id not in cat_id_to_index:
                    continue
                cls_idx = cat_id_to_index[cat_id]
                bbox = ann.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue

                if task == "segment":
                    lines.append(
                        _to_yolo_segment_line(
                            ann.get("segmentation"), bbox, width, height, cls_idx
                        )
                    )
                else:
                    lines.append(_to_yolo_detect_line(bbox, width, height, cls_idx))

            dst_lbl.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            split_counts[split_name] += 1

    return split_counts


def _write_dataset_yaml(prepared_root: Path, class_names: List[str], yaml_path: Path) -> None:
    names_str = "[" + ", ".join([f'"{x}"' for x in class_names]) + "]"
    content = "\n".join(
        [
            f"path: {prepared_root.as_posix()}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            f"nc: {len(class_names)}",
            f"names: {names_str}",
            "",
        ]
    )
    yaml_path.write_text(content, encoding="utf-8")


def run(config_path: Path) -> None:
    cfg = _load_json(config_path)
    conda_env_name = _check_conda_runtime(cfg)

    dataset_cfg = cfg.get("dataset", {})
    training_cfg = cfg.get("training", {})
    output_cfg = cfg.get("output", {})
    dataset_root = Path(dataset_cfg.get("root_dir", "")).expanduser()
    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset.root_dir 不存在: {dataset_root}")

    split = dataset_cfg.get("split", {})
    _validate_split(split)

    project_dir = Path(output_cfg.get("project_dir", "")).expanduser()
    run_name = str(output_cfg.get("run_name", "exp"))
    if not project_dir:
        raise ValueError("output.project_dir 不能为空")

    run_root = project_dir / run_name
    _ensure_dir(run_root)

    coco_json_raw = str(dataset_cfg.get("coco_json", "")).strip()
    if not coco_json_raw:
        raise ValueError("dataset.coco_json 不能为空（请提供已标注好的 COCO JSON 文件路径）")
    coco_json_path = Path(coco_json_raw).expanduser()
    if not coco_json_path.exists():
        raise FileNotFoundError(
            f"未找到 COCO 标注文件: {coco_json_path}。"
            "请先准备好标注后的 COCO 文件并确认路径正确。"
        )

    log_info(f"读取 COCO 标注: {coco_json_path}")
    coco = _load_coco(coco_json_path)

    forced_class_names = dataset_cfg.get("class_names") or []
    class_names, cat_id_to_index = _build_category_mapping(coco, forced_class_names)

    image_ids = [int(x["id"]) for x in coco["images"]]
    if not image_ids:
        raise ValueError("COCO images 为空，无法训练")

    seed = int(cfg.get("seed", 42))
    split_image_ids = _split_image_ids(image_ids, split, seed)

    prepared_root = run_root / "prepared_dataset"
    dataset_yaml = run_root / "dataset.yaml"

    task = str(training_cfg.get("task", "detect")).strip().lower()
    if task not in {"detect", "segment"}:
        raise ValueError(f"training.task 仅支持 detect/segment，当前: {task}")

    copy_images = bool(dataset_cfg.get("copy_images", True))
    split_counts = _prepare_yolo_dataset(
        dataset_root=dataset_root,
        coco=coco,
        class_names=class_names,
        cat_id_to_index=cat_id_to_index,
        split_image_ids=split_image_ids,
        prepared_root=prepared_root,
        task=task,
        copy_images=copy_images,
    )

    _write_dataset_yaml(prepared_root, class_names, dataset_yaml)
    log_info(f"完成数据集划分与转换: {prepared_root}")

    model_name = str(training_cfg.get("model", "yolov8n.pt"))
    epochs = int(training_cfg.get("epochs", 100))
    imgsz = int(training_cfg.get("imgsz", 640))
    batch = int(training_cfg.get("batch", 16))
    import torch
    _has_cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0
    _default_device = '0' if _has_cuda else 'cpu'
    device = str(training_cfg.get('device', _default_device))
    if device not in ('cpu',) and not _has_cuda:
        log_warn(f"input.json 指定 device='{device}' 但未检测到 GPU，自动回退为 CPU")
        device = 'cpu'
    log_info(f"设备选择: {device} (CUDA available={_has_cuda}, device_count={torch.cuda.device_count() if _has_cuda else 0})")
    workers = int(training_cfg.get("workers", 8))
    patience = int(training_cfg.get("patience", 50))

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("请在 conda 环境内安装依赖: pip install ultralytics") from exc

    log_info(f"加载模型(自动下载): {model_name}")
    model = YOLO(model_name, task=task)

    train_project = run_root / "runs"
    _ensure_dir(train_project)

    log_info("开始训练...日志将实时输出")
    train_results = model.train(
        data=str(dataset_yaml),
        task=task,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        patience=patience,
        project=str(train_project),
        name="train",
        exist_ok=True,
    )

    log_info("开始在 test split 上评估...")
    eval_results = model.val(
        data=str(dataset_yaml),
        split="test",
        task=task,
        device=device,
        project=str(train_project),
        name="test_eval",
        exist_ok=True,
    )

    summary = {
        "config_path": str(config_path),
        "conda_env_name": conda_env_name,
        "dataset_root": str(dataset_root),
        "coco_json": str(coco_json_path),
        "task": task,
        "model": model_name,
        "num_images": len(image_ids),
        "num_categories": len(class_names),
        "class_names": class_names,
        "split_counts": split_counts,
        "run_root": str(run_root),
        "runs_dir": str(train_project),
        "train_save_dir": str(getattr(train_results, "save_dir", "")),
        "eval_results": str(eval_results),
    }

    summary_path = run_root / "run_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log_info(f"训练完成。摘要已写入: {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="GPU training orchestrator driven by input.json")
    parser.add_argument("--input", required=True, help="Path to input.json")
    parser.add_argument("--log-file", default=None, help="Tee stdout/stderr to this file for tail streaming")
    args = parser.parse_args()

    config_path = Path(args.input).resolve()

    if args.log_file:
        log_path = Path(args.log_file).resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        _install_tee(log_path)
        log_info(f"日志文件: {log_path}")

    try:
        run(config_path)
    except Exception as exc:
        log_info(f"[ERROR] 训练失败: {exc}")
        raise
    finally:
        # Write completion sentinel so tail loop knows to stop
        if _log_file is not None:
            marker = "=== TRAINING_PROCESS_DONE ===\n"
            sys.stdout.write(marker)
            sys.stdout.flush()


if __name__ == "__main__":
    main()
