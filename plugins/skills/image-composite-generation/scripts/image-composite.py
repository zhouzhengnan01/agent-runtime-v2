import argparse
import base64
import json
import mimetypes
import random
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


BASE_URL = "https://www.tokencloud.yun/v1/images/generations"
API_KEY = "sk-6qbqlTAZV7qVszLplZhxZH2yvQwSMRotrOT8wr9aVPII8BYo"
MODEL_NAME = "wan2.7-image-pro"

DEFAULT_NEGATIVE_PROMPT = (
    "pasted photo, cutout, sticker, collage, fake composite, hard outline, halo edge, "
    "wrong perspective, wrong scale, wrong lighting, wrong shadow direction, floating object, "
    "missing contact shadow, oversharp object on blurry background, cartoon, illustration, CGI, "
    "3D render, text, watermark, logo"
)
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def is_remote_or_data_url(value: str) -> bool:
    return value.strip().startswith(("http://", "https://", "data:"))


def collect_candidate_images(image_input: str) -> list[str]:
    image_input = image_input.strip()
    if is_remote_or_data_url(image_input):
        return [image_input]

    path = Path(image_input)
    if path.is_file():
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {path}")
        return [str(path.resolve())]

    if path.is_dir():
        candidates = [
            str(item.resolve())
            for item in path.rglob("*")
            if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
        ]
        candidates.sort()
        if not candidates:
            raise FileNotFoundError(f"No supported images found in folder: {path}")
        return candidates

    raise FileNotFoundError(f"Image path not found: {path}")


def choose_image(image_input: str, field_name: str) -> str:
    candidates = collect_candidate_images(image_input)
    selected = random.choice(candidates)
    if len(candidates) > 1:
        print(f"[sample] {field_name}: selected {selected} from {len(candidates)} images")
    return selected


def image_to_data_url(image_path: str) -> str:
    path = Path(image_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    mime_type, _ = mimetypes.guess_type(path.name)
    if not mime_type:
        mime_type = "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def image_value(image: str) -> str:
    image = image.strip()
    if image.startswith(("http://", "https://", "data:")):
        return image
    return image_to_data_url(image)


def build_payload(
    image1: str,
    image2: str,
    prompt: str,
    negative_prompt: str,
    model: str,
    size: str,
    prompt_extend: bool,
    watermark: bool,
) -> dict[str, Any]:
    return {
        "model": model,
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"image": image_value(image1)},
                        {"image": image_value(image2)},
                        {"text": prompt},
                    ],
                }
            ]
        },
        "parameters": {
            "negative_prompt": negative_prompt,
            "prompt_extend": prompt_extend,
            "watermark": watermark,
            "size": size,
        },
    }


def request_one(api_url: str, api_key: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    response = requests.post(api_url, headers=headers, json=payload, timeout=timeout)
    try:
        data = response.json()
    except ValueError:
        response.raise_for_status()
        raise RuntimeError(f"Non-JSON response: {response.text[:500]}")
    if response.status_code >= 400:
        raise RuntimeError(json.dumps(data, ensure_ascii=False, indent=2))
    return data


def iter_image_urls(obj: Any):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in {"url", "image_url", "image", "orig_url", "actual_url"} and isinstance(value, str):
                if value.startswith(("http://", "https://")):
                    yield value
            yield from iter_image_urls(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from iter_image_urls(item)


def download_file(url: str, output_path: Path, timeout: int) -> None:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        output_path.write_bytes(response.read())


def save_b64_image(b64_text: str, output_path: Path) -> None:
    cleaned = b64_text.strip()
    if cleaned.startswith("data:"):
        cleaned = cleaned.split(",", 1)[1]
    output_path.write_bytes(base64.b64decode(cleaned))


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")


def elapsed_seconds(start: float) -> float:
    return round(time.perf_counter() - start, 3)


def write_request_preview(payload: dict[str, Any], output_path: Path, selected_inputs: dict[str, str] | None = None) -> None:
    preview = json.loads(json.dumps(payload, ensure_ascii=False))
    if selected_inputs:
        preview["_local_selected_inputs"] = selected_inputs
    for item in preview["input"]["messages"][0]["content"]:
        image_field = item.get("image")
        if isinstance(image_field, str) and image_field.startswith("data:"):
            item["image"] = image_field[:80] + "...<base64 omitted>"
    output_path.write_text(json.dumps(preview, ensure_ascii=False, indent=2), encoding="utf-8")


def save_images_from_response(data: dict[str, Any], output_dir: Path, stem: str, timeout: int) -> list[Path]:
    saved_paths: list[Path] = []
    data_items = data.get("data", []) if isinstance(data, dict) else []
    if isinstance(data_items, list):
        for index, item in enumerate(data_items, start=1):
            if not isinstance(item, dict):
                continue
            suffix = f"_{index:02d}" if len(data_items) > 1 else ""
            image_path = output_dir / f"{stem}{suffix}.png"
            b64_json = item.get("b64_json")
            url = item.get("url")
            if isinstance(b64_json, str) and b64_json.strip():
                save_b64_image(b64_json, image_path)
                saved_paths.append(image_path)
            elif isinstance(url, str) and url.strip():
                download_file(url, image_path, timeout)
                saved_paths.append(image_path)
    if saved_paths:
        return saved_paths

    urls = list(dict.fromkeys(iter_image_urls(data)))
    for index, url in enumerate(urls, start=1):
        suffix = f"_{index:02d}" if len(urls) > 1 else ""
        image_path = output_dir / f"{stem}{suffix}.png"
        download_file(url, image_path, timeout)
        saved_paths.append(image_path)
    return saved_paths


def composite_images(
    *,
    image1: str,
    image2: str,
    prompt: str,
    output_dir: str,
    api_url: str = BASE_URL,
    api_key: str = API_KEY,
    model: str = MODEL_NAME,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    size: str = "768*768",
    count: int = 1,
    timeout: int = 180,
    sleep: float = 1.0,
    prompt_extend: bool = False,
    watermark: bool = False,
    retries: int = 3,
    retry_sleep: float = 5.0,
) -> list[str]:
    if not image1.strip():
        raise ValueError("image1 must not be empty")
    if not image2.strip():
        raise ValueError("image2 must not be empty")
    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    if not api_url.strip():
        raise ValueError("api_url must not be empty")
    if not api_key.strip():
        raise ValueError("api_key must not be empty")
    if not model.strip():
        raise ValueError("model must not be empty")

    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    for index in range(1, max(1, int(count)) + 1):
        stem = f"composite_{index:03d}"
        selected_image1 = choose_image(image1, "image1")
        selected_image2 = choose_image(image2, "image2")
        payload = build_payload(
            image1=selected_image1,
            image2=selected_image2,
            prompt=prompt,
            negative_prompt=negative_prompt,
            model=model,
            size=size,
            prompt_extend=prompt_extend,
            watermark=watermark,
        )
        selected_inputs = {"image1": selected_image1, "image2": selected_image2}
        write_request_preview(payload, out_dir / f"request_{index:03d}.json", selected_inputs)
        print(f"[{index}/{count}] requesting {model} with 2 images...")
        total_start = time.perf_counter()
        attempts = max(1, int(retries))
        for attempt in range(1, attempts + 1):
            try:
                data = request_one(api_url, api_key, payload, timeout)
                saved_paths = save_images_from_response(data, out_dir, stem, timeout)
                data["_local_timing"] = {
                    "request_started_at": now_iso(),
                    "total_seconds": elapsed_seconds(total_start),
                    "selected_inputs": selected_inputs,
                    "saved_images": [str(path.resolve()) for path in saved_paths],
                    "finished_at": now_iso(),
                    "attempt": attempt,
                }
                (out_dir / f"response_{index:03d}.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
                if not saved_paths:
                    print(f"[{index}/{count}] no image found; response saved.")
                for image_path in saved_paths:
                    saved.append(str(image_path.resolve()))
                    print(f"[{index}/{count}] saved {image_path.resolve()}")
                break
            except Exception as exc:
                (out_dir / f"error_{index:03d}.txt").write_text(str(exc), encoding="utf-8")
                print(f"[{index}/{count}] attempt {attempt}/{attempts} failed: {exc}")
                if attempt >= attempts:
                    raise
                time.sleep(max(0.0, float(retry_sleep)))
        if index < count and sleep > 0:
            time.sleep(sleep)
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Composite two images through TokenCloud wan2.7-image-pro.")
    parser.add_argument("--url", default=BASE_URL)
    parser.add_argument("--token", default=API_KEY)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--image1", required=True)
    parser.add_argument("--image2", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--size", default="768*768")
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--prompt-extend", action="store_true")
    parser.add_argument("--watermark", action="store_true")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=5.0)
    args = parser.parse_args()

    saved = composite_images(
        api_url=args.url,
        api_key=args.token,
        model=args.model,
        image1=args.image1,
        image2=args.image2,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        output_dir=args.output_dir,
        size=args.size,
        count=args.count,
        timeout=args.timeout,
        sleep=args.sleep,
        prompt_extend=args.prompt_extend,
        watermark=args.watermark,
        retries=args.retries,
        retry_sleep=args.retry_sleep,
    )
    print(f"Composite success. saved: {saved[0] if saved else ''}")


if __name__ == "__main__":
    main()
