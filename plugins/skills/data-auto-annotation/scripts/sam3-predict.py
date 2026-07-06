"""Call the SAM3 prediction API for one or more image sources and save COCO format.

Example:
    python scripts/sam3-predict.py --input-dir ./images --output coco.json --text-prompts person monitor

You can also pass JSON input for compatibility:
    python scripts/sam3-predict.py --input-json '{"image_path":"./images"}' --output coco.json
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import sys
import time
from pathlib import Path
from typing import Any

import requests
from PIL import Image

DEFAULT_URL = "http://192.168.33.92:8800/v1/sam3/predict"
URL_ENV_NAMES = ("SAM3_PREDICT_URL", "SAM3_URL")
DEFAULT_TOKEN = "abc@123"
DEFAULT_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_CONNECT_TIMEOUT = 5
DEFAULT_READ_TIMEOUT = 12
DEFAULT_MIN_IMAGE_SIZE = 16
RETRYABLE_HTTP_STATUS = {502, 503, 504}


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
    parser.add_argument(
        "--class-names",
        nargs="*",
        default=None,
        help="Training class names written to COCO categories. Defaults to text prompts.",
    )
    parser.add_argument(
        "--class-names-json",
        default=None,
        help='Training class names as a JSON array string, e.g. "[\"person_fall\"]"',
    )
    parser.add_argument(
        "--prompt-label-map-json",
        default=None,
        help='JSON object mapping SAM3 prompt text to training class name, e.g. {"fallen person":"person_fall"}',
    )
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold")
    parser.add_argument(
        "--dedupe-iou",
        type=float,
        default=0.9,
        help="Drop duplicate boxes with the same final class when IoU is greater than or equal to this value",
    )
    parser.add_argument("--min-image-size", type=int, default=DEFAULT_MIN_IMAGE_SIZE, help="Minimum accepted image width/height")
    parser.add_argument("--connect-timeout", type=int, default=DEFAULT_CONNECT_TIMEOUT, help="Connection timeout in seconds")
    parser.add_argument("--timeout", type=int, default=DEFAULT_READ_TIMEOUT, help="Read timeout in seconds")
    parser.add_argument("--retries", type=int, default=3, help="Retries for transient SAM3 HTTP 502/503/504, connection, and timeout errors")
    parser.add_argument("--retry-sleep", type=float, default=2.0, help="Base sleep seconds between SAM3 retry attempts")
    parser.add_argument("--workers", type=int, default=4, help="Number of concurrent request workers (reserved)")
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save COCO JSON output. If omitted, prints COCO JSON to stdout.",
    )
    parser.add_argument(
        "--per-image-output-dir",
        default=None,
        help="Optional directory for streaming one COCO JSON sidecar per annotated image.",
    )
    parser.add_argument(
        "--per-image-base-dir",
        default=None,
        help="Optional image root used to mirror relative paths under --per-image-output-dir.",
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


def resolve_class_names(args: argparse.Namespace, prompts: list[str]) -> list[str]:
    if args.class_names_json:
        class_names = json.loads(args.class_names_json)
        if not isinstance(class_names, list) or not all(isinstance(item, str) for item in class_names):
            raise ValueError("--class-names-json must be a JSON array of strings")
        return _dedupe_text(class_names)
    if args.class_names:
        return _dedupe_text(args.class_names)
    if args.input_json:
        parsed = _unwrap_payload(_parse_input_json_arg(args.input_json))
        if isinstance(parsed, dict):
            for key in ("class_names", "classes"):
                value = parsed.get(key)
                if isinstance(value, list) and all(isinstance(item, str) for item in value):
                    return _dedupe_text(value)
    return _dedupe_text(prompts)


def resolve_prompt_label_map(args: argparse.Namespace, prompts: list[str], class_names: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if args.prompt_label_map_json:
        parsed = json.loads(args.prompt_label_map_json)
        if not isinstance(parsed, dict):
            raise ValueError("--prompt-label-map-json must be a JSON object")
        mapping.update({str(key).strip(): str(value).strip() for key, value in parsed.items() if str(key).strip() and str(value).strip()})
    elif args.input_json:
        parsed = _unwrap_payload(_parse_input_json_arg(args.input_json))
        if isinstance(parsed, dict):
            value = parsed.get("prompt_label_map") or parsed.get("annotation_prompt_map")
            if isinstance(value, dict):
                mapping.update({str(key).strip(): str(item).strip() for key, item in value.items() if str(key).strip() and str(item).strip()})

    valid_classes = set(class_names)
    mapping = {prompt: label for prompt, label in mapping.items() if label in valid_classes}
    if len(class_names) == 1:
        for prompt in prompts:
            mapping.setdefault(prompt, class_names[0])
    for class_name in class_names:
        mapping.setdefault(class_name, class_name)
    return mapping


def _dedupe_text(values: list[str]) -> list[str]:
    result: list[str] = []
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

    try:
        score_value = 1.0 if score is None else float(score)
    except (TypeError, ValueError):
        score_value = 1.0

    return {"label": str(label), "score": score_value, "bbox": [x1, y1, x2 - x1, y2 - y1]}


def _bbox_iou_xywh(a: list[float], b: list[float]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    area_a = max(0.0, aw) * max(0.0, ah)
    area_b = max(0.0, bw) * max(0.0, bh)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def _dedupe_same_class_boxes(annotations: list[dict[str, Any]], iou_threshold: float) -> list[dict[str, Any]]:
    if not annotations or iou_threshold <= 0:
        return annotations
    kept: list[dict[str, Any]] = []
    # 多 prompt 会让同一个目标以不同提示词返回多次。这里在 prompt 映射成最终训练
    # 类别之后做同类 NMS，只清理重复框，不削弱多提示词带来的召回能力。
    for annotation in sorted(annotations, key=lambda item: float(item.get("score", 0.0)), reverse=True):
        category_id = annotation.get("category_id")
        bbox = annotation.get("bbox")
        if not isinstance(bbox, list) or len(bbox) < 4:
            kept.append(annotation)
            continue
        duplicated = False
        for existing in kept:
            if existing.get("category_id") != category_id:
                continue
            existing_bbox = existing.get("bbox")
            if isinstance(existing_bbox, list) and len(existing_bbox) >= 4 and _bbox_iou_xywh(bbox, existing_bbox) >= iou_threshold:
                duplicated = True
                break
        if not duplicated:
            kept.append(annotation)
    return kept


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
    prompt_label_map: dict[str, str] | None = None,
    source: str = "real",
    is_synthetic: bool = False,
    annotation_start_id: int = 1,
    dedupe_iou: float = 0.9,
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
    candidates: list[dict[str, Any]] = []

    for item in _extract_boxes(payload):
        normalized = _normalize_box(item)
        if normalized is None:
            continue
        raw_label = str(normalized["label"])
        mapped_label = _lookup_prompt_label(raw_label, prompt_label_map or {})
        canonical_label = next((name for name in category_map if name.lower() == mapped_label.lower()), "")
        if not canonical_label:
            continue
        normalized["label"] = canonical_label
        category_id = get_category_id(category_map, categories, normalized["label"])
        x, y, w, h = normalized["bbox"]
        candidates.append(
            {
                "id": 0,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0,
                "segmentation": [],
                "score": normalized["score"],
            }
        )

    annotations = _dedupe_same_class_boxes(candidates, dedupe_iou)
    for idx, annotation in enumerate(annotations, start=annotation_start_id):
        annotation["id"] = idx

    return image, annotations


def _lookup_prompt_label(raw_label: str, prompt_label_map: dict[str, str]) -> str:
    text = str(raw_label or "").strip()
    if not text:
        return text
    lowered = text.lower()
    for prompt, label in prompt_label_map.items():
        if lowered == str(prompt).strip().lower():
            return str(label).strip()
    return text


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _single_image_coco(image_info: dict[str, Any], annotations: list[dict[str, Any]], categories: list[dict[str, Any]]) -> dict[str, Any]:
    local_image = dict(image_info)
    original_image_id = int(local_image.get("id", 1) or 1)
    local_image["id"] = 1
    local_annotations: list[dict[str, Any]] = []
    for annotation_id, annotation in enumerate(annotations, start=1):
        local_annotation = dict(annotation)
        local_annotation["id"] = annotation_id
        local_annotation["image_id"] = 1
        local_annotations.append(local_annotation)
    return {
        "images": [local_image],
        "annotations": local_annotations,
        "categories": [dict(category) for category in categories],
        "licenses": [],
        "info": {
            "description": "SAM3 per-image auto-annotation export",
            "original_image_id": original_image_id,
        },
    }


def save_per_image_coco(
    output_dir: str | None,
    base_dir: str | None,
    image_path: Path,
    image_info: dict[str, Any],
    annotations: list[dict[str, Any]],
    categories: list[dict[str, Any]],
) -> Path | None:
    if not output_dir:
        return None
    root = Path(output_dir).resolve()
    try:
        relative_image = image_path.resolve().relative_to(Path(base_dir).resolve()) if base_dir else Path(image_path.name)
    except ValueError:
        relative_image = Path(image_path.name)
    sidecar = root / relative_image.parent / f"{image_path.stem}_coco.json"
    save_json(sidecar, _single_image_coco(image_info, annotations, categories))
    print(f"per_image_coco_path: {sidecar}", flush=True)
    return sidecar


def image_to_data_url(image_path: Path) -> str:
    content_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{content_type};base64,{encoded}"


def build_sam3_payload(image_path: Path, prompts: list[str], conf: float, iou: float) -> dict[str, Any]:
    return {
        "model": "sam3",
        "input": {
            "image": image_to_data_url(image_path),
            "text_prompts": prompts,
        },
        "parameters": {
            "conf": float(conf),
            "iou": float(iou),
        },
    }


def post_image(args: argparse.Namespace, headers: dict[str, str], prompts: list[str], image_path: Path) -> Any:
    attempts = max(1, int(args.retries) + 1)
    last_error: requests.exceptions.RequestException | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = requests.post(
                effective_url(args.url),
                headers=headers,
                json=build_sam3_payload(image_path, prompts, args.conf, args.iou),
                timeout=(args.connect_timeout, args.timeout),
            )
            if response.status_code in RETRYABLE_HTTP_STATUS and attempt < attempts:
                print(
                    f"[sam3-predict] transient HTTP {response.status_code} for {image_path.name}; retry {attempt}/{attempts - 1}",
                    file=sys.stderr,
                )
                time.sleep(float(args.retry_sleep) * attempt)
                continue
            response.raise_for_status()
            return response.json()
        except (requests.exceptions.ConnectTimeout, requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as exc:
            last_error = exc
            if attempt >= attempts:
                raise
            print(f"[sam3-predict] transient request error for {image_path.name}: {exc}; retry {attempt}/{attempts - 1}", file=sys.stderr)
            time.sleep(float(args.retry_sleep) * attempt)
        except requests.exceptions.HTTPError as exc:
            last_error = exc
            status_code = exc.response.status_code if exc.response is not None else None
            if status_code not in RETRYABLE_HTTP_STATUS or attempt >= attempts:
                raise
            print(f"[sam3-predict] transient HTTP {status_code} for {image_path.name}; retry {attempt}/{attempts - 1}", file=sys.stderr)
            time.sleep(float(args.retry_sleep) * attempt)
    if last_error:
        raise last_error
    raise RuntimeError(f"SAM3 request failed without response for {image_path}")


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


def _print_request_error(error_type: str, exc: Exception, args: argparse.Namespace, image_path: Path | None = None) -> None:
    payload = _http_error_payload(exc, args) if isinstance(exc, requests.exceptions.HTTPError) else _error_payload(error_type, str(exc), args)
    if image_path is not None:
        payload["image_path"] = str(image_path)
    _print_error(payload)


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
    class_names = resolve_class_names(args, prompts)
    prompt_label_map = resolve_prompt_label_map(args, prompts, class_names)
    headers = {"Authorization": f"Bearer {args.token}", "Content-Type": "application/json"}

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    categories: list[dict[str, Any]] = []
    category_map: dict[str, int] = {}
    seed_categories(class_names, category_map, categories)
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
            payload = post_image(args, headers, prompts, image_path)
            image_info, image_annotations = response_to_coco(
                payload,
                image_path,
                image_id=image_id,
                category_map=category_map,
                categories=categories,
                prompt_label_map=prompt_label_map,
                source=args.source,
                is_synthetic=bool(args.is_synthetic or args.source in {"synthetic", "generated"}),
                annotation_start_id=next_annotation_id,
                dedupe_iou=float(args.dedupe_iou),
            )
            images.append(image_info)
            annotations.extend(image_annotations)
            next_annotation_id += len(image_annotations)
            save_per_image_coco(
                args.per_image_output_dir,
                args.per_image_base_dir or args.input_dir,
                image_path,
                image_info,
                image_annotations,
                categories,
            )
    except requests.exceptions.ConnectTimeout as exc:
        _print_request_error("sam3_connect_timeout", exc, args, locals().get("image_path"))
        return 2
    except requests.exceptions.ReadTimeout as exc:
        _print_request_error("sam3_read_timeout", exc, args, locals().get("image_path"))
        return 2
    except requests.exceptions.ConnectionError as exc:
        _print_request_error("sam3_connection_error", exc, args, locals().get("image_path"))
        return 2
    except requests.exceptions.HTTPError as exc:
        _print_request_error("sam3_http_error", exc, args, locals().get("image_path"))
        return 2
    except requests.exceptions.RequestException as exc:
        _print_request_error("sam3_request_error", exc, args, locals().get("image_path"))
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
