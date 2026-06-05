import argparse
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def _write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _validate_split(split: Dict[str, float]) -> None:
    for key in ("train", "val", "test"):
        if key not in split:
            raise ValueError(f"dataset.split missing field: {key}")
        if float(split[key]) < 0:
            raise ValueError(f"dataset.split.{key} must be >= 0")
    total = float(split["train"]) + float(split["val"]) + float(split["test"])
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"dataset.split must sum to 1.0, got {total}")


def _load_coco(coco_json_path: Path) -> Dict:
    coco = _load_json(coco_json_path)
    for key in ("images", "annotations", "categories"):
        if key not in coco:
            raise ValueError(f"COCO file missing field: {key}")
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

    raise FileNotFoundError(f"Image file not found: {file_name}, dataset_root={dataset_root}")


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
                raise ValueError(f"dataset.class_names is missing COCO category: {name}")
            cat_id_to_index[int(cat["id"])] = name_to_index[name]
        return class_names, cat_id_to_index

    class_names = coco_cat_names
    cat_id_to_index = {cat_id: idx for idx, cat_id in enumerate(coco_cat_ids)}
    return class_names, cat_id_to_index


def _is_synthetic_image(image_obj: Dict, synthetic_policy: Dict) -> bool:
    source_field = str(synthetic_policy.get("source_field", "source") or "source")
    synthetic_values = {str(x).strip().lower() for x in synthetic_policy.get("synthetic_values", ["synthetic", "generated", "gen"])}
    is_synthetic_fields = synthetic_policy.get("is_synthetic_fields", ["is_synthetic", "synthetic"])
    for field in is_synthetic_fields:
        if bool(image_obj.get(str(field), False)):
            return True
    source = str(image_obj.get(source_field, "real")).strip().lower()
    return source in synthetic_values


def _holdout_counts(total: int, split: Dict[str, float]) -> Tuple[int, int]:
    val_ratio = max(0.0, float(split["val"]))
    test_ratio = max(0.0, float(split["test"]))
    n_val = int(total * val_ratio)
    n_test = int(total * test_ratio)

    if test_ratio > 0 and n_test == 0 and total >= 2:
        n_test = 1
    if val_ratio > 0 and n_val == 0 and total - n_test >= 2:
        n_val = 1

    while n_val + n_test >= total and total > 0:
        if n_val >= n_test and n_val > 0:
            n_val -= 1
        elif n_test > 0:
            n_test -= 1
        else:
            break
    return n_val, n_test


def _split_image_ids(image_ids: List[int], split: Dict[str, float], seed: int) -> Dict[str, List[int]]:
    random.seed(seed)
    ids = image_ids[:]
    random.shuffle(ids)
    total = len(ids)
    n_val, n_test = _holdout_counts(total, split)
    n_train = total - n_val - n_test
    return {"train": ids[:n_train], "val": ids[n_train:n_train + n_val], "test": ids[n_train + n_val:]}


def _split_image_ids_with_source(coco: Dict, split: Dict[str, float], seed: int, synthetic_policy: Dict) -> Tuple[Dict[str, List[int]], Dict[str, Dict[str, int]]]:
    real_ids: List[int] = []
    synthetic_ids: List[int] = []
    for image_obj in coco["images"]:
        image_id = int(image_obj["id"])
        if _is_synthetic_image(image_obj, synthetic_policy):
            synthetic_ids.append(image_id)
        else:
            real_ids.append(image_id)

    if not real_ids:
        raise ValueError("synthetic_policy is enabled but no real images were found for val/test")

    random.seed(seed)
    random.shuffle(real_ids)
    random.shuffle(synthetic_ids)
    real_total = len(real_ids)
    n_val, n_test = _holdout_counts(real_total, split)
    val_ids = real_ids[:n_val]
    test_ids = real_ids[n_val:n_val + n_test]
    train_real_ids = real_ids[n_val + n_test:]
    train_ids = train_real_ids + synthetic_ids
    return (
        {"train": train_ids, "val": val_ids, "test": test_ids},
        {
            "train": {"real": len(train_real_ids), "synthetic": len(synthetic_ids)},
            "val": {"real": len(val_ids), "synthetic": 0},
            "test": {"real": len(test_ids), "synthetic": 0},
        },
    )


def _clip_bbox_to_image(bbox: List[float], width: int, height: int) -> List[float] | None:
    x, y, w, h = [float(v) for v in bbox]
    if width <= 0 or height <= 0 or w <= 0 or h <= 0:
        return None
    x1 = max(0.0, min(float(width), x))
    y1 = max(0.0, min(float(height), y))
    x2 = max(0.0, min(float(width), x + w))
    y2 = max(0.0, min(float(height), y + h))
    clipped_w = x2 - x1
    clipped_h = y2 - y1
    if clipped_w <= 1.0 or clipped_h <= 1.0:
        return None
    return [x1, y1, clipped_w, clipped_h]


def _to_yolo_detect_line(bbox: List[float], width: int, height: int, cls_idx: int) -> str | None:
    clipped = _clip_bbox_to_image(bbox, width, height)
    if clipped is None:
        return None
    x, y, w, h = clipped
    return f"{cls_idx} {(x + w / 2.0) / width:.6f} {(y + h / 2.0) / height:.6f} {w / width:.6f} {h / height:.6f}"


def _to_yolo_segment_line(segmentation, bbox: List[float], width: int, height: int, cls_idx: int) -> str | None:
    points: List[float] = []
    if isinstance(segmentation, list) and segmentation:
        polygon = segmentation[0]
        if isinstance(polygon, list) and len(polygon) >= 6 and len(polygon) % 2 == 0:
            points = [float(v) for v in polygon]
    clipped_bbox = _clip_bbox_to_image(bbox, width, height)
    if clipped_bbox is None:
        return None
    if not points:
        x, y, w, h = clipped_bbox
        points = [x, y, x + w, y, x + w, y + h, x, y + h]
    normalized = []
    for i, value in enumerate(points):
        if i % 2 == 0:
            normalized.append(max(0.0, min(float(width), value)) / float(width))
        else:
            normalized.append(max(0.0, min(float(height), value)) / float(height))
    return f"{cls_idx} " + " ".join(f"{v:.6f}" for v in normalized)


def _prepare_yolo_dataset(dataset_root: Path, coco: Dict, class_names: List[str], cat_id_to_index: Dict[int, int], split_image_ids: Dict[str, List[int]], prepared_root: Path, task: str, copy_images: bool) -> Dict[str, int]:
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
                bbox = ann.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue
                line = _to_yolo_segment_line(ann.get("segmentation"), bbox, width, height, cat_id_to_index[cat_id]) if task == "segment" else _to_yolo_detect_line(bbox, width, height, cat_id_to_index[cat_id])
                if line:
                    lines.append(line)
            dst_lbl.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            split_counts[split_name] += 1
    return split_counts


def _write_dataset_yaml(prepared_root: Path, class_names: List[str], yaml_path: Path) -> None:
    names_str = "[" + ", ".join([f'"{x}"' for x in class_names]) + "]"
    content = "\n".join([
        f"path: {prepared_root.as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        f"nc: {len(class_names)}",
        f"names: {names_str}",
        "",
    ])
    yaml_path.write_text(content, encoding="utf-8")


def prepare_dataset(*, dataset_root: Path, coco_json: Path, output_dir: Path, split: Dict[str, float], class_names: List[str], task: str, synthetic_policy: Dict, copy_images: bool = True, seed: int = 42) -> Dict:
    _validate_split(split)
    coco = _load_coco(coco_json)
    class_names, cat_id_to_index = _build_category_mapping(coco, class_names)
    image_ids = [int(x["id"]) for x in coco["images"]]
    if not image_ids:
        raise ValueError("COCO images is empty")

    if bool(synthetic_policy.get("enabled", False)):
        split_image_ids, source_counts = _split_image_ids_with_source(coco, split, seed, synthetic_policy)
    else:
        split_image_ids = _split_image_ids(image_ids, split, seed)
        source_counts = None

    prepared_root = output_dir / "prepared_dataset"
    dataset_yaml = output_dir / "dataset.yaml"
    split_counts = _prepare_yolo_dataset(dataset_root, coco, class_names, cat_id_to_index, split_image_ids, prepared_root, task, copy_images)
    _write_dataset_yaml(prepared_root, class_names, dataset_yaml)

    summary = {
        "dataset_root": str(dataset_root),
        "coco_json": str(coco_json),
        "prepared_dataset": str(prepared_root),
        "dataset_yaml": str(dataset_yaml),
        "task": task,
        "num_images": len(image_ids),
        "num_categories": len(class_names),
        "class_names": class_names,
        "split_counts": split_counts,
        "source_counts": source_counts,
        "synthetic_policy": synthetic_policy,
    }
    _write_json(output_dir / "data_preparation_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare split YOLO dataset from COCO.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--coco-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--class-names", default="")
    parser.add_argument("--task", default="detect", choices=["detect", "segment"])
    parser.add_argument("--split-train", type=float, default=0.7)
    parser.add_argument("--split-val", type=float, default=0.2)
    parser.add_argument("--split-test", type=float, default=0.1)
    parser.add_argument("--synthetic-policy", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    policy = {
        "enabled": args.synthetic_policy,
        "source_field": "source",
        "synthetic_values": ["synthetic", "generated", "gen"],
        "is_synthetic_fields": ["is_synthetic", "synthetic"],
        "val_real_only": True,
        "test_real_only": True,
        "synthetic_to_train_only": True,
    }
    class_names = [x.strip() for x in args.class_names.split(",") if x.strip()]
    summary = prepare_dataset(
        dataset_root=Path(args.dataset_root).resolve(),
        coco_json=Path(args.coco_json).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        split={"train": args.split_train, "val": args.split_val, "test": args.split_test},
        class_names=class_names,
        task=args.task,
        synthetic_policy=policy,
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
