from __future__ import annotations

import argparse
import json
import random
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON is not an object: {path}")
    for key in ("images", "annotations", "categories"):
        if key not in payload:
            raise ValueError(f"{path} is not COCO: missing {key}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def image_is_synthetic(image: dict[str, Any]) -> bool:
    if bool(image.get("is_synthetic") or image.get("synthetic")):
        return True
    return str(image.get("source") or "").strip().lower() in {"synthetic", "generated", "gen"}


def normalize_categories(coco: dict[str, Any], class_names: list[str]) -> tuple[list[dict[str, Any]], dict[int, int]]:
    categories = [cat for cat in coco.get("categories", []) if isinstance(cat, dict)]
    original_names = {int(cat.get("id", 0)): str(cat.get("name") or "").strip() for cat in categories}
    ordered_names = [name for name in class_names if name]
    for cat in categories:
        name = str(cat.get("name") or "").strip()
        if name and name not in ordered_names:
            ordered_names.append(name)
    if not ordered_names:
        raise ValueError("No class names found in COCO categories or spec")
    name_to_new_id = {name: index for index, name in enumerate(ordered_names)}
    id_map: dict[int, int] = {}
    for old_id, name in original_names.items():
        if name in name_to_new_id:
            id_map[old_id] = name_to_new_id[name]
    normalized = [
        {"id": index, "name": name, "supercategory": "object"}
        for index, name in enumerate(ordered_names)
    ]
    return normalized, id_map


def split_real_images(images: list[dict[str, Any]], split: dict[str, Any], seed: int) -> dict[str, set[int]]:
    real_ids = [int(img["id"]) for img in images if not image_is_synthetic(img)]
    if not real_ids:
        raise ValueError("DEIMv2 requires real images for validation/test splits")
    train_ratio = float(split.get("train", 0.7))
    val_ratio = float(split.get("val", 0.2))
    test_ratio = float(split.get("test", 0.1))
    if min(train_ratio, val_ratio, test_ratio) < 0:
        raise ValueError("split ratios must be non-negative")
    total_ratio = train_ratio + val_ratio + test_ratio
    if total_ratio <= 0:
        raise ValueError("split ratios sum to zero")
    val_ratio /= total_ratio
    test_ratio /= total_ratio
    ids = list(real_ids)
    random.Random(seed).shuffle(ids)
    total = len(ids)
    val_n = int(total * val_ratio)
    test_n = int(total * test_ratio)
    if val_ratio > 0 and val_n == 0 and total >= 2:
        val_n = 1
    if test_ratio > 0 and test_n == 0 and total >= 3:
        test_n = 1
    while val_n + test_n >= total and total > 1:
        if test_n > 0:
            test_n -= 1
        elif val_n > 0:
            val_n -= 1
        else:
            break
    train_n = total - val_n - test_n
    return {
        "train": set(ids[:train_n]),
        "val": set(ids[train_n : train_n + val_n]),
        "test": set(ids[train_n + val_n :]),
    }


def resolve_image(root: Path, file_name: str, cache: dict[str, Path]) -> Path:
    normalized = file_name.replace("\\", "/")
    candidates = [root / normalized, root / Path(normalized).name]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    key = Path(normalized).name.lower()
    if key in cache:
        return cache[key]
    matches = [p for p in root.rglob(Path(normalized).name) if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    if not matches:
        raise FileNotFoundError(f"Image not found under {root}: {file_name}")
    cache[key] = matches[0].resolve()
    return cache[key]


def prepare_dataset(spec: dict[str, Any]) -> dict[str, Any]:
    dataset_root = Path(str(spec["dataset_root"])).resolve()
    coco_json = Path(str(spec["coco_json"])).resolve()
    output_root = Path(str(spec["output_dir"])).resolve()
    class_names = [str(item).strip() for item in spec.get("class_names", []) if str(item).strip()]
    split = spec.get("split") if isinstance(spec.get("split"), dict) else {}
    seed = int(spec.get("seed", 42))
    coco = load_json(coco_json)
    categories, category_id_map = normalize_categories(coco, class_names)
    real_splits = split_real_images(coco.get("images", []), split, seed)
    annotations_by_image: dict[int, list[dict[str, Any]]] = {}
    for ann in coco.get("annotations", []):
        if not isinstance(ann, dict):
            continue
        annotations_by_image.setdefault(int(ann.get("image_id", 0)), []).append(ann)

    prepared = {
        split_name: {
            "info": {"description": f"DEIMv2 {split_name} split"},
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": deepcopy(categories),
        }
        for split_name in ("train", "val", "test")
    }
    counters = {name: {"image": 1, "annotation": 1} for name in prepared}
    source_counts = {name: {"real": 0, "synthetic": 0} for name in prepared}
    cache: dict[str, Path] = {}

    for image in coco.get("images", []):
        if not isinstance(image, dict):
            continue
        image_id = int(image.get("id", 0))
        synthetic = image_is_synthetic(image)
        if synthetic:
            split_name = "train"
        else:
            split_name = next(name for name, ids in real_splits.items() if image_id in ids)
        src = resolve_image(dataset_root, str(image.get("file_name") or ""), cache)
        dst_dir = output_root / "images" / split_name
        dst_dir.mkdir(parents=True, exist_ok=True)
        new_image_id = counters[split_name]["image"]
        counters[split_name]["image"] += 1
        prefix = "synthetic" if synthetic else "real"
        dst_name = f"{prefix}_{new_image_id:06d}_{src.name}"
        shutil.copy2(src, dst_dir / dst_name)
        copied = deepcopy(image)
        copied.update(
            {
                "id": new_image_id,
                "file_name": dst_name,
                "source": prefix,
                "is_synthetic": synthetic,
            }
        )
        prepared[split_name]["images"].append(copied)
        source_counts[split_name]["synthetic" if synthetic else "real"] += 1

        for ann in annotations_by_image.get(image_id, []):
            old_category = int(ann.get("category_id", 0))
            if old_category not in category_id_map:
                continue
            copied_ann = deepcopy(ann)
            copied_ann.update(
                {
                    "id": counters[split_name]["annotation"],
                    "image_id": new_image_id,
                    "category_id": category_id_map[old_category],
                    "iscrowd": int(ann.get("iscrowd", 0)),
                }
            )
            counters[split_name]["annotation"] += 1
            bbox = copied_ann.get("bbox")
            if isinstance(bbox, list) and len(bbox) == 4:
                copied_ann["area"] = float(max(0.0, float(bbox[2])) * max(0.0, float(bbox[3])))
            prepared[split_name]["annotations"].append(copied_ann)

    annotations_dir = output_root / "annotations"
    for split_name, payload in prepared.items():
        write_json(annotations_dir / f"instances_{split_name}.json", payload)

    summary = {
        "dataset_root": str(output_root),
        "source_dataset_root": str(dataset_root),
        "source_coco_json": str(coco_json),
        "num_classes": len(categories),
        "class_names": [cat["name"] for cat in categories],
        "categories": categories,
        "split_counts": {
            split_name: {
                "images": len(payload["images"]),
                "annotations": len(payload["annotations"]),
            }
            for split_name, payload in prepared.items()
        },
        "source_counts": source_counts,
    }
    write_json(output_root / "dataset_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a DEIMv2 COCO dataset from a merged workflow COCO file.")
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    spec = json.loads(Path(args.input).resolve().read_text(encoding="utf-8-sig"))
    print(json.dumps(prepare_dataset(spec), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
