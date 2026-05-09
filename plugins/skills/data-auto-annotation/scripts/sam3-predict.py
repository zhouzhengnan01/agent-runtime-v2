"""Call the SAM3 prediction API for one or more image sources and save COCO format.

Example:
    python scripts/sam3-predict.py --input-dir ./images --output coco.json --text-prompts person monitor

You can also pass JSON input for compatibility:
    python scripts/sam3-predict.py --input-json '{"image_path":"./images"}' --output coco.json
"""

from __future__ import annotations

import argparse
import json
import mimetypes
from pathlib import Path
from typing import Any

import requests
from PIL import Image

DEFAULT_URL = "http://192.168.33.140:8800/sam3/predict"
DEFAULT_TOKEN = "abc@123"
DEFAULT_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send images in a folder to the SAM3 prediction API.")
    parser.add_argument("--url", default=DEFAULT_URL, help="SAM3 prediction endpoint")
    parser.add_argument("--token", default=DEFAULT_TOKEN, help="Bearer token for Authorization header")
    parser.add_argument("--input-dir", default=None, help="Directory containing images to annotate")
    parser.add_argument(
        "--input-json",
        default=None,
        help='JSON string containing image_path(s) and optional labels, e.g. "{\"image_path\":\"./test.jpg\",\"labels\":[\"person\"]}"',
    )
    parser.add_argument(
        "--text-prompts",
        nargs="*",
        default=None,
        help="Text prompts as space-separated values, e.g. person monitor",
    )
    parser.add_argument(
        "--text-prompts-json",
        default=None,
        help='Text prompts as a JSON array string, e.g. "[\"person\", \"monitor\"]"',
    )
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold")
    parser.add_argument("--timeout", type=int, default=120, help="Request timeout in seconds")
    parser.add_argument("--workers", type=int, default=4, help="Number of concurrent request workers (reserved)")
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save COCO JSON output. If omitted, prints COCO JSON to stdout.",
    )
    return parser.parse_args()


def resolve_prompts(args: argparse.Namespace) -> list[str]:
    if args.text_prompts_json:
        prompts = json.loads(args.text_prompts_json)
        if not isinstance(prompts, list) or not all(isinstance(item, str) for item in prompts):
            raise ValueError("--text-prompts-json must be a JSON array of strings")
        return prompts
    if args.text_prompts:
        return args.text_prompts
    if args.input_json:
        parsed = json.loads(args.input_json)
        if isinstance(parsed, dict):
            labels = parsed.get("labels")
            if isinstance(labels, list) and all(isinstance(item, str) for item in labels):
                return labels
    return ["person", "monitor"]


def _collect_image_paths(value: Any) -> list[Path]:
    paths: list[Path] = []
    if isinstance(value, str):
        p = Path(value)
        if p.is_dir():
            for child in sorted(p.iterdir()):
                if child.is_file() and child.suffix.lower() in DEFAULT_IMAGE_EXTENSIONS:
                    paths.append(child)
        elif p.is_file() and p.suffix.lower() in DEFAULT_IMAGE_EXTENSIONS:
            paths.append(p)
    elif isinstance(value, list):
        for item in value:
            paths.extend(_collect_image_paths(item))
    return paths


def resolve_image_paths(args: argparse.Namespace) -> list[Path]:
    if args.input_dir:
        paths = _collect_image_paths(args.input_dir)
        if not paths:
            raise ValueError(f"--input-dir has no supported images: {args.input_dir}")
        return paths

    if args.input_json:
        parsed = json.loads(args.input_json)
        if not isinstance(parsed, dict):
            raise ValueError("--input-json must be a JSON object")

        for key in ("image_path", "image_paths", "path", "paths", "input_dir", "input_path"):
            if key in parsed:
                paths = _collect_image_paths(parsed[key])
                if paths:
                    return paths
        raise ValueError("--input-json must contain image_path, image_paths, path, paths, input_dir, or input_path")

    raise ValueError("You must provide either --input-dir or --input-json")


def _extract_boxes(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        for key in ("boxes", "results", "data", "predictions", "annotations"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def _normalize_box(item: dict[str, Any]) -> dict[str, Any] | None:
    score = item.get("score", item.get("conf", item.get("confidence", 1.0)))
    label = item.get("label", item.get("category", item.get("class", item.get("name", "object"))))

    bbox = item.get("bbox", item.get("box"))
    if isinstance(bbox, dict):
        bbox = [bbox.get("x1"), bbox.get("y1"), bbox.get("x2"), bbox.get("y2")]

    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None

    try:
        x1, y1, x2, y2 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    except (TypeError, ValueError):
        return None

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    return {"label": str(label), "score": float(score), "bbox": [x1, y1, x2 - x1, y2 - y1]}


def get_category_id(category_map: dict[str, int], categories: list[dict[str, Any]], label: str) -> int:
    if label in category_map:
        return category_map[label]
    category_id = len(categories) + 1
    category_map[label] = category_id
    categories.append({"id": category_id, "name": label, "supercategory": "object"})
    return category_id


def response_to_coco(
    payload: Any,
    image_path: Path,
    image_id: int,
    category_map: dict[str, int],
    categories: list[dict[str, Any]],
    annotation_start_id: int = 1,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with Image.open(image_path) as img:
        width, height = img.size

    image = {"id": image_id, "file_name": image_path.name, "width": width, "height": height}
    annotations: list[dict[str, Any]] = []

    for idx, item in enumerate(_extract_boxes(payload), start=annotation_start_id):
        normalized = _normalize_box(item)
        if normalized is None:
            continue
        category_id = get_category_id(category_map, categories, normalized["label"])
        x, y, w, h = normalized["bbox"]
        annotations.append(
            {
                "id": idx,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0,
                "segmentation": [],
                "score": normalized["score"],
            }
        )

    return image, annotations


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def post_image(args: argparse.Namespace, headers: dict[str, str], data: dict[str, str], image_path: Path) -> Any:
    content_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
    with image_path.open("rb") as f:
        files = {"file": (image_path.name, f, content_type)}
        response = requests.post(args.url, headers=headers, data=data, files=files, timeout=args.timeout)
    response.raise_for_status()
    return response.json()


def main() -> None:
    args = build_args()
    image_paths = resolve_image_paths(args)

    for image_path in image_paths:
        if not image_path.is_file():
            raise FileNotFoundError(f"Image file not found: {image_path}")

    prompts = resolve_prompts(args)
    headers = {"Authorization": f"Bearer {args.token}"}
    data = {"text_prompts": json.dumps(prompts, ensure_ascii=False), "conf": str(args.conf), "iou": str(args.iou)}

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    categories: list[dict[str, Any]] = []
    category_map: dict[str, int] = {}
    next_annotation_id = 1

    for image_id, image_path in enumerate(image_paths, start=1):
        payload = post_image(args, headers, data, image_path)
        image_info, image_annotations = response_to_coco(
            payload,
            image_path,
            image_id=image_id,
            category_map=category_map,
            categories=categories,
            annotation_start_id=next_annotation_id,
        )
        images.append(image_info)
        annotations.extend(image_annotations)
        next_annotation_id += len(image_annotations)

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
        "licenses": [],
        "info": {"description": "SAM3 auto-annotation export"},
    }

    if args.output:
        save_json(Path(args.output), coco)
    else:
        print(json.dumps(coco, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
