import argparse
import json
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _validate_coco(coco: Dict, name: str) -> None:
    for key in ("images", "annotations", "categories"):
        if key not in coco:
            raise ValueError(f"{name} 缺少 COCO 字段: {key}")


def _category_key(category: Dict) -> Tuple[str, str]:
    return (str(category.get("name", "")).strip(), str(category.get("supercategory", "")).strip())


def _merge_categories(real_categories: Iterable[Dict], synthetic_categories: Iterable[Dict]) -> Tuple[List[Dict], Dict[Tuple[str, int], int]]:
    merged: List[Dict] = []
    key_to_new_id: Dict[Tuple[str, str], int] = {}
    old_to_new: Dict[Tuple[str, int], int] = {}

    for source_name, categories in (("real", real_categories), ("synthetic", synthetic_categories)):
        for category in categories:
            key = _category_key(category)
            if not key[0]:
                raise ValueError(f"{source_name} categories 中存在空 name: {category}")
            if key not in key_to_new_id:
                new_id = len(merged) + 1
                key_to_new_id[key] = new_id
                new_category = deepcopy(category)
                new_category["id"] = new_id
                merged.append(new_category)
            old_to_new[(source_name, int(category["id"]))] = key_to_new_id[key]

    return merged, old_to_new


def _resolve_image_path(root: Path, file_name: str) -> Path:
    candidate = root / file_name
    if candidate.exists() and candidate.suffix.lower() in IMAGE_EXTS:
        return candidate

    basename = Path(file_name).name
    candidate = root / basename
    if candidate.exists() and candidate.suffix.lower() in IMAGE_EXTS:
        return candidate

    matches = [p for p in root.rglob(basename) if p.suffix.lower() in IMAGE_EXTS]
    if matches:
        return matches[0]

    raise FileNotFoundError(f"未找到图片: {file_name} (root={root})")


def _copy_images(coco: Dict, source_root: Path, output_root: Path, source_label: str, image_id_map: Dict[int, int]) -> Dict[int, str]:
    target_dir = output_root / source_label
    target_dir.mkdir(parents=True, exist_ok=True)
    copied_file_names: Dict[int, str] = {}

    for image in coco["images"]:
        old_id = int(image["id"])
        src = _resolve_image_path(source_root, str(image["file_name"]))
        new_id = image_id_map[old_id]
        dst_name = f"{new_id:08d}_{src.name}"
        shutil.copy2(src, target_dir / dst_name)
        copied_file_names[new_id] = f"{source_label}/{dst_name}"

    return copied_file_names


def _append_dataset(
    merged_images: List[Dict],
    merged_annotations: List[Dict],
    coco: Dict,
    source_name: str,
    old_category_to_new: Dict[Tuple[str, int], int],
    next_image_id: int,
    next_annotation_id: int,
) -> Tuple[int, int, Dict[int, int]]:
    image_id_map: Dict[int, int] = {}

    for image in coco["images"]:
        old_id = int(image["id"])
        new_id = next_image_id
        next_image_id += 1
        image_id_map[old_id] = new_id

        new_image = deepcopy(image)
        new_image["id"] = new_id
        new_image["source"] = source_name
        new_image["is_synthetic"] = source_name == "synthetic"
        merged_images.append(new_image)

    for ann in coco["annotations"]:
        old_image_id = int(ann["image_id"])
        if old_image_id not in image_id_map:
            continue
        old_cat_id = int(ann["category_id"])
        new_ann = deepcopy(ann)
        new_ann["id"] = next_annotation_id
        next_annotation_id += 1
        new_ann["image_id"] = image_id_map[old_image_id]
        new_ann["category_id"] = old_category_to_new[(source_name, old_cat_id)]
        merged_annotations.append(new_ann)

    return next_image_id, next_annotation_id, image_id_map


def merge_coco(
    real_coco_path: Path,
    synthetic_coco_path: Path,
    output_coco_path: Path,
    real_root: Path | None = None,
    synthetic_root: Path | None = None,
    output_image_root: Path | None = None,
) -> Dict:
    real_coco = _load_json(real_coco_path)
    synthetic_coco = _load_json(synthetic_coco_path)
    _validate_coco(real_coco, "real_coco")
    _validate_coco(synthetic_coco, "synthetic_coco")

    categories, category_map = _merge_categories(real_coco["categories"], synthetic_coco["categories"])
    merged_images: List[Dict] = []
    merged_annotations: List[Dict] = []

    next_image_id, next_ann_id, real_image_id_map = _append_dataset(
        merged_images, merged_annotations, real_coco, "real", category_map, 1, 1
    )
    next_image_id, next_ann_id, synthetic_image_id_map = _append_dataset(
        merged_images, merged_annotations, synthetic_coco, "synthetic", category_map, next_image_id, next_ann_id
    )

    if output_image_root:
        if real_root is None or synthetic_root is None:
            raise ValueError("启用 --output-image-root 时必须同时提供 --real-root 和 --synthetic-root")
        output_image_root.mkdir(parents=True, exist_ok=True)
        copied_file_names = {}
        copied_file_names.update(_copy_images(real_coco, real_root, output_image_root, "real", real_image_id_map))
        copied_file_names.update(_copy_images(synthetic_coco, synthetic_root, output_image_root, "synthetic", synthetic_image_id_map))
        for image in merged_images:
            copied_file_name = copied_file_names.get(int(image["id"]))
            if copied_file_name:
                image["file_name"] = copied_file_name

    merged = {
        "info": real_coco.get("info", {"description": "merged real and synthetic COCO dataset"}),
        "licenses": real_coco.get("licenses", []),
        "images": merged_images,
        "annotations": merged_annotations,
        "categories": categories,
    }
    _write_json(output_coco_path, merged)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge real and synthetic COCO files and tag image source.")
    parser.add_argument("--real-coco", required=True, help="Path to real COCO JSON")
    parser.add_argument("--synthetic-coco", required=True, help="Path to synthetic COCO JSON")
    parser.add_argument("--output", required=True, help="Path to merged COCO JSON")
    parser.add_argument("--real-root", default=None, help="Real image root, required when copying images")
    parser.add_argument("--synthetic-root", default=None, help="Synthetic image root, required when copying images")
    parser.add_argument("--output-image-root", default=None, help="Optional merged image root. Images are copied into real/ and synthetic/")
    args = parser.parse_args()

    merged = merge_coco(
        real_coco_path=Path(args.real_coco).resolve(),
        synthetic_coco_path=Path(args.synthetic_coco).resolve(),
        output_coco_path=Path(args.output).resolve(),
        real_root=Path(args.real_root).resolve() if args.real_root else None,
        synthetic_root=Path(args.synthetic_root).resolve() if args.synthetic_root else None,
        output_image_root=Path(args.output_image_root).resolve() if args.output_image_root else None,
    )
    real_count = sum(1 for img in merged["images"] if img.get("source") == "real")
    synthetic_count = sum(1 for img in merged["images"] if img.get("source") == "synthetic")
    print(json.dumps({"output": str(Path(args.output).resolve()), "real_images": real_count, "synthetic_images": synthetic_count}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
