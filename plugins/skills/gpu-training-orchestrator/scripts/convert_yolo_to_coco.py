import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _collect_images(images_dir: Path) -> List[Path]:
    return sorted([p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def _read_class_names(value: str, labels_dir: Path) -> List[str]:
    if value:
        path = Path(value)
        if path.exists():
            return [line.strip().lstrip("\ufeff") for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        return [item.strip().lstrip("\ufeff") for item in value.split(",") if item.strip()]

    for candidate in (labels_dir / "classes.txt", labels_dir.parent / "classes.txt", labels_dir.parent / "obj.names"):
        if candidate.exists():
            return [line.strip().lstrip("\ufeff") for line in candidate.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    return []


def _image_size(path: Path) -> Tuple[int, int]:
    with Image.open(path) as img:
        return img.size


def _yolo_bbox_to_coco(parts: List[str], width: int, height: int) -> List[float] | None:
    if len(parts) != 5:
        return None
    _, xc, yc, bw, bh = [float(x) for x in parts]
    x = (xc - bw / 2.0) * width
    y = (yc - bh / 2.0) * height
    w = bw * width
    h = bh * height
    x = max(0.0, min(float(width), x))
    y = max(0.0, min(float(height), y))
    w = max(0.0, min(float(width) - x, w))
    h = max(0.0, min(float(height) - y, h))
    if w <= 1.0 or h <= 1.0:
        return None
    return [x, y, w, h]


def _polygon_to_bbox(coords: List[float], width: int, height: int) -> List[float] | None:
    if len(coords) < 6 or len(coords) % 2 != 0:
        return None
    xs = [max(0.0, min(float(width), coords[i] * width)) for i in range(0, len(coords), 2)]
    ys = [max(0.0, min(float(height), coords[i] * height)) for i in range(1, len(coords), 2)]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    if x2 - x1 <= 1.0 or y2 - y1 <= 1.0:
        return None
    return [x1, y1, x2 - x1, y2 - y1]


def convert_yolo_to_coco(images_dir: Path, labels_dir: Path, output: Path, class_names_arg: str = "") -> Dict:
    images = _collect_images(images_dir)
    if not images:
        raise ValueError(f"images_dir 中没有图片: {images_dir}")

    class_names = _read_class_names(class_names_arg, labels_dir)
    categories: Dict[int, str] = {idx: name for idx, name in enumerate(class_names)}
    coco_images = []
    coco_annotations = []
    next_ann_id = 1

    for image_id, image_path in enumerate(images, start=1):
        width, height = _image_size(image_path)
        rel_file = image_path.relative_to(images_dir).as_posix()
        coco_images.append(
            {
                "id": image_id,
                "file_name": rel_file,
                "width": width,
                "height": height,
                "source": "real",
                "is_synthetic": False,
            }
        )

        label_path = labels_dir / image_path.relative_to(images_dir).with_suffix(".txt")
        if not label_path.exists():
            label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue

        for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip().lstrip("\ufeff")
            if not line:
                continue
            parts = line.split()
            cls_id = int(float(parts[0]))
            categories.setdefault(cls_id, f"class_{cls_id}")
            bbox = _yolo_bbox_to_coco(parts, width, height)
            segmentation = []
            if bbox is None and len(parts) >= 7:
                coords = [float(x) for x in parts[1:]]
                bbox = _polygon_to_bbox(coords, width, height)
                if bbox is not None:
                    segmentation = [[coords[i] * (width if i % 2 == 0 else height) for i in range(len(coords))]]
            if bbox is None:
                continue
            coco_annotations.append(
                {
                    "id": next_ann_id,
                    "image_id": image_id,
                    "category_id": cls_id + 1,
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0,
                    "segmentation": segmentation,
                }
            )
            next_ann_id += 1

    coco_categories = [
        {"id": cls_id + 1, "name": name, "supercategory": "object"}
        for cls_id, name in sorted(categories.items())
    ]
    coco = {
        "info": {"description": "Converted from YOLO labels"},
        "licenses": [],
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": coco_categories,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(coco, ensure_ascii=False, indent=2), encoding="utf-8")
    return coco


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert YOLO txt labels to COCO JSON.")
    parser.add_argument("--images-dir", required=True, help="Images directory")
    parser.add_argument("--labels-dir", required=True, help="YOLO labels directory")
    parser.add_argument("--output", required=True, help="Output COCO JSON path")
    parser.add_argument("--class-names", default="", help="Comma-separated class names or path to classes.txt")
    args = parser.parse_args()

    coco = convert_yolo_to_coco(
        images_dir=Path(args.images_dir).resolve(),
        labels_dir=Path(args.labels_dir).resolve(),
        output=Path(args.output).resolve(),
        class_names_arg=args.class_names,
    )
    print(
        json.dumps(
            {
                "output": str(Path(args.output).resolve()),
                "images": len(coco["images"]),
                "annotations": len(coco["annotations"]),
                "categories": len(coco["categories"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
