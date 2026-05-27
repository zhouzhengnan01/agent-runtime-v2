import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _collect_images(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def _find_images_dir(dataset_root: Path) -> Optional[Path]:
    candidates = [dataset_root / "images", dataset_root / "image", dataset_root / "JPEGImages"]
    for candidate in candidates:
        if _collect_images(candidate):
            return candidate
    if _collect_images(dataset_root):
        return dataset_root
    return None


def _find_labels_dir(dataset_root: Path) -> Optional[Path]:
    for name in ("labels", "label", "annotations", "annotation"):
        candidate = dataset_root / name
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


def _is_coco_json(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return isinstance(payload, dict) and all(k in payload for k in ("images", "annotations", "categories"))
    except Exception:
        return False


def _looks_like_yolo_txt(path: Path) -> bool:
    try:
        lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
    except Exception:
        return False
    if not lines:
        return True
    checked = 0
    for line in lines[:20]:
        line = line.lstrip("\ufeff")
        parts = line.split()
        if len(parts) not in (5,) and len(parts) < 7:
            return False
        try:
            int(float(parts[0]))
            coords = [float(x) for x in parts[1:]]
        except ValueError:
            return False
        if any(v < -0.001 or v > 1.001 for v in coords):
            return False
        checked += 1
    return checked > 0 or not lines


def _detect_label_format(labels_dir: Optional[Path]) -> Dict:
    if labels_dir is None or not labels_dir.exists():
        return {"format": "unlabeled", "label_count": 0, "coco_json": None}

    label_files = [p for p in labels_dir.rglob("*") if p.is_file()]
    if not label_files:
        return {"format": "unlabeled", "label_count": 0, "coco_json": None}

    json_files = sorted(labels_dir.rglob("*.json"))
    coco_files = [p for p in json_files if _is_coco_json(p)]
    if coco_files:
        return {
            "format": "coco",
            "label_count": len(coco_files),
            "coco_json": str(coco_files[0]),
            "all_coco_jsons": [str(p) for p in coco_files],
        }

    txt_files = sorted(
        p
        for p in labels_dir.rglob("*.txt")
        if p.name.lower() not in {"classes.txt", "obj.names"}
    )
    if txt_files:
        sample = txt_files[: min(20, len(txt_files))]
        yolo_like = sum(1 for p in sample if _looks_like_yolo_txt(p))
        if yolo_like >= max(1, len(sample) // 2):
            classes_file = labels_dir / "classes.txt"
            return {
                "format": "yolo",
                "label_count": len(txt_files),
                "classes_file": str(classes_file) if classes_file.exists() else None,
            }

    xml_files = sorted(labels_dir.rglob("*.xml"))
    if xml_files:
        return {"format": "pascal_voc", "label_count": len(xml_files), "supported": False}

    return {"format": "unknown", "label_count": len(list(labels_dir.rglob("*"))), "supported": False}


def inspect_dataset(dataset_root: Path) -> Dict:
    if not dataset_root.exists() or not dataset_root.is_dir():
        return {"status": "error", "message": f"dataset_root 不存在或不是目录: {dataset_root}"}

    images_dir = _find_images_dir(dataset_root)
    if images_dir is None:
        return {"status": "error", "message": f"未找到 images 目录或图片文件: {dataset_root}"}

    labels_dir = _find_labels_dir(dataset_root)
    images = _collect_images(images_dir)
    label_info = _detect_label_format(labels_dir)
    fmt = label_info["format"]

    if fmt == "unlabeled":
        case = "images_only"
        next_step = "auto_annotation"
        message = "检测到只有 images，没有 labels，需要先自动标注生成 COCO。"
    elif fmt == "coco":
        case = "images_and_labels"
        next_step = "train_or_plan_synthetic"
        message = "检测到 COCO JSON 标注，可直接进入训练或合成数据规划。"
    elif fmt == "yolo":
        case = "images_and_labels"
        next_step = "convert_yolo_to_coco"
        message = "检测到 YOLO txt 标注，建议先转换为 COCO 后进入统一训练流程。"
    else:
        case = "images_and_labels"
        next_step = "unsupported_or_manual_convert"
        message = "检测到 labels，但格式暂不支持自动处理，请转换为 COCO 或 YOLO。"

    return {
        "status": "ok",
        "dataset_root": str(dataset_root),
        "case": case,
        "format": fmt,
        "images_dir": str(images_dir),
        "labels_dir": str(labels_dir) if labels_dir else None,
        "image_count": len(images),
        "recommended_next_step": next_step,
        "message": message,
        **label_info,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect uploaded object-detection dataset structure.")
    parser.add_argument("--dataset-root", required=True, help="Dataset root path")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    result = inspect_dataset(Path(args.dataset_root).resolve())
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
