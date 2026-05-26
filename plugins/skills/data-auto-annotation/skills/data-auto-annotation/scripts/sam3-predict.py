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
import os
import sys
from pathlib import Path
from typing import Any

import requests
from PIL import Image

DEFAULT_URL = "http://218.67.242.10:58800/sam3/predict"
URL_ENV_NAMES = ("SAM3_PREDICT_URL", "SAM3_URL")
DEFAULT_TOKEN = "abc@123"
DEFAULT_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_CONNECT_TIMEOUT = 5
DEFAULT_READ_TIMEOUT = 12
DEFAULT_MIN_IMAGE_SIZE = 16


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
    parser.add_argument("--min-image-size", type=int, default=DEFAULT_MIN_IMAGE_SIZE, help="Minimum accepted image width/height")
    parser.add_argument("--connect-timeout", type=int, default=DEFAULT_CONNECT_TIMEOUT, help="Connection timeout in seconds")
    parser.add_argument("--timeout", type=int, default=DEFAULT_READ_TIMEOUT, help="Read timeout in seconds")
    parser.add_argument("--workers", type=int, default=4, help="Number of concurrent request workers (reserved)")
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save COCO JSON output. If omitted, prints COCO JSON to stdout.",
    )
    parser.add_argument(
        "--source",
        default="real",
        choices=["real", "synthetic", "generated"],
        help="Dataset source tag written to each COCO image.",
    )
    parser.add_argument(
        "--is-synthetic",
        action="store_true",
        help="Mark exported COCO images as synthetic. This is used by training split policy.",
    )
    return parser.parse_args()


def effective_url(value: str) -> str:
    normalized = str(value or "").strip()
    if normalized in {"$SAM3_PREDICT_URL", "${SAM3_PREDICT_URL}", "$SAM3_URL", "${SAM3_URL}", ""}:
        for env_name in URL_ENV_NAMES:
            env_value = os.getenv(env_name, "").strip()
            if env_value:
                return env_value
        return DEFAULT_URL
    return normalized


def resolve_prompts(args: argparse.Namespace) -> list[str]:
    if args.text_prompts_json:
        prompts = json.loads(args.text_prompts_json)
        if not isinstance(prompts, list) or not all(isinstance(item, str) for item in prompts):
            raise ValueError("--text-prompts-json must be a JSON array of strings")
        return prompts
    if args.text_prompts:
        return args.text_prompts
    if args.input_json:
        parsed = _unwrap_payload(_parse_input_json_arg(args.input_json))
        if isinstance(parsed, dict):
            labels = parsed.get("labels")
            if isinstance(labels, list) and all(isinstance(item, str) for item in labels):
                return labels
    raise ValueError(
        "data-auto-annotation requires labels. Provide labels in input JSON, "
        'for example: {"image_path":"./images","labels":["person","cigarette"]}'
    )


def _collect_image_paths(value: Any) -> list[Path]:
    paths: list[Path] = []
    if isinstance(value, str):
        p = Path(value)
        if p.is_dir():
            for child in sorted(p.rglob("*")):
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
        parsed = _unwrap_payload(_parse_input_json_arg(args.input_json))
        if not isinstance(parsed, dict):
            raise ValueError("--input-json must be a JSON object")

        for key in ("image_path", "image_paths", "path", "paths", "input_dir", "input_path"):
            if key in parsed:
                paths = _collect_image_paths(parsed[key])
                if paths:
                    return paths
        # recursive fallback for nested request envelopes
        nested = _find_image_field_recursive(parsed)
        if nested is not None:
            paths = _collect_image_paths(nested)
            if paths:
                return paths
        raise ValueError("--input-json must contain image_path, image_paths, path, paths, input_dir, or input_path")

    raise ValueError("You must provide either --input-dir or --input-json")


def _parse_input_json_arg(raw_value: str) -> Any:
    value = (raw_value or "").strip()
    if value in {"$spec_json", "${spec_json}"}:
        value = ""
    try:
        return json.loads(value)
    except Exception:
        pass
    candidate = Path(value)
    if candidate.exists() and candidate.is_file():
        with candidate.open("r", encoding="utf-8") as f:
            return json.load(f)
    fallback = Path.cwd() / "data-auto-annotation-input.json"
    if fallback.exists() and fallback.is_file():
        with fallback.open("r", encoding="utf-8") as f:
            return json.load(f)
    raise ValueError("--input-json must be a JSON string or a JSON file path")


def _unwrap_payload(parsed: Any) -> Any:
    if not isinstance(parsed, dict):
        return parsed
    workflow_ctx = parsed.get("workflow_context")
    if isinstance(workflow_ctx, dict) and ("image_path" in workflow_ctx or "input_dir" in workflow_ctx):
        merged = dict(parsed)
        merged.update(workflow_ctx)
        return merged
    for key in ("spec", "input", "payload", "data"):
        value = parsed.get(key)
        if isinstance(value, dict):
            workflow_ctx = value.get("workflow_context")
            if isinstance(workflow_ctx, dict) and ("image_path" in workflow_ctx or "input_dir" in workflow_ctx):
                merged = dict(value)
                merged.update(workflow_ctx)
                return merged
            return value
    return parsed


def _find_image_field_recursive(node: Any) -> Any:
    keys = ("image_path", "image_paths", "path", "paths", "input_dir", "input_path")
    if isinstance(node, dict):
        for k in keys:
            if k in node:
                return node[k]
        for v in node.values():
            found = _find_image_field_recursive(v)
            if found is not None:
                return found
    elif isinstance(node, list):
        for item in node:
            found = _find_image_field_recursive(item)
            if found is not None:
                return found
    return None


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


def seed_categories(labels: list[str], category_map: dict[str, int], categories: list[dict[str, Any]]) -> None:
    for label in labels:
        normalized = label.strip()
        if normalized:
            get_category_id(category_map, categories, normalized)


def image_size(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as img:
        return img.size


def response_to_coco(
    payload: Any,
    image_path: Path,
    image_id: int,
    category_map: dict[str, int],
    categories: list[dict[str, Any]],
    source: str = "real",
    is_synthetic: bool = False,
    annotation_start_id: int = 1,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with Image.open(image_path) as img:
        width, height = img.size

    image = {
        "id": image_id,
        "file_name": image_path.name,
        "width": width,
        "height": height,
        "source": source,
        "is_synthetic": is_synthetic,
    }
    annotations: list[dict[str, Any]] = []

    for idx, item in enumerate(_extract_boxes(payload), start=annotation_start_id):
        normalized = _normalize_box(item)
        if normalized is None:
            continue
        canonical_label = next((name for name in category_map if name.lower() == str(normalized["label"]).lower()), "")
        if not canonical_label:
            continue
        normalized["label"] = canonical_label
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
        response = requests.post(
            effective_url(args.url),
            headers=headers,
            data=data,
            files=files,
            timeout=(args.connect_timeout, args.timeout),
        )
    response.raise_for_status()
    return response.json()


def _error_payload(error_type: str, message: str, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "ok": False,
        "error_type": error_type,
        "message": message,
        "url": effective_url(args.url),
        "hint": "SAM3 annotation endpoint is unreachable. Check the endpoint host/port, VPN/network route, or configure SAM3_PREDICT_URL.",
    }


def _http_error_payload(exc: requests.exceptions.HTTPError, args: argparse.Namespace) -> dict[str, Any]:
    response = exc.response
    status_code = response.status_code if response is not None else None
    response_text = response.text[:2000] if response is not None else ""
    payload = _error_payload("sam3_http_error", str(exc), args)
    payload.update(
        {
            "status_code": status_code,
            "response_body": response_text,
            "hint": "SAM3 endpoint returned an HTTP error. Check the endpoint contract, request fields, labels, and model server logs.",
        }
    )
    if response_text:
        payload["message"] = f"{exc}: {response_text}"
    return payload


def _print_error(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(payload["message"], file=sys.stderr)


def main() -> int:
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
    seed_categories(prompts, category_map, categories)
    next_annotation_id = 1

    try:
        for image_id, image_path in enumerate(image_paths, start=1):
            width, height = image_size(image_path)
            if width < args.min_image_size or height < args.min_image_size:
                payload = _error_payload(
                    "image_too_small",
                    f"Image is too small for SAM3 inference: {image_path.name} ({width}x{height}), minimum is {args.min_image_size}x{args.min_image_size}.",
                    args,
                )
                payload["image"] = {"path": str(image_path), "width": width, "height": height}
                _print_error(payload)
                return 2
            payload = post_image(args, headers, data, image_path)
            image_info, image_annotations = response_to_coco(
                payload,
                image_path,
                image_id=image_id,
                category_map=category_map,
                categories=categories,
                source=args.source,
                is_synthetic=bool(args.is_synthetic or args.source in {"synthetic", "generated"}),
                annotation_start_id=next_annotation_id,
            )
            images.append(image_info)
            annotations.extend(image_annotations)
            next_annotation_id += len(image_annotations)
    except requests.exceptions.ConnectTimeout as exc:
        _print_error(_error_payload("sam3_connect_timeout", str(exc), args))
        return 2
    except requests.exceptions.ReadTimeout as exc:
        _print_error(_error_payload("sam3_read_timeout", str(exc), args))
        return 2
    except requests.exceptions.ConnectionError as exc:
        _print_error(_error_payload("sam3_connection_error", str(exc), args))
        return 2
    except requests.exceptions.HTTPError as exc:
        _print_error(_http_error_payload(exc, args))
        return 2
    except requests.exceptions.RequestException as exc:
        _print_error(_error_payload("sam3_request_error", str(exc), args))
        return 2

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
