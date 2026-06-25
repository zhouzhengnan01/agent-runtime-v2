import argparse
import base64
import json
import mimetypes
from datetime import datetime
from pathlib import Path
from typing import Any

import requests


DEFAULT_API_URL = "http://218.67.242.10:58801/v1/flux2/generate"
DEFAULT_MODEL = "flux2"
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 640
DEFAULT_STEPS = 8


def _guess_ext_from_bytes(content: bytes) -> str:
    """Guess a file extension from image magic bytes."""
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if content.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if content[:4] == b"RIFF" and content[8:12] == b"WEBP":
        return ".webp"
    if content.startswith(b"GIF87a") or content.startswith(b"GIF89a"):
        return ".gif"
    return ".bin"


def _save_bytes(out_dir: Path, content: bytes, ext: str, prefix: str = "generated") -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = out_dir / f"{prefix}_{timestamp}{ext}"
    with open(save_path, "wb") as wf:
        wf.write(content)
    return str(save_path)


def _image_to_data_url(input_path: Path) -> str:
    content = input_path.read_bytes()
    mime_type = mimetypes.guess_type(input_path.name)[0] or "image/png"
    encoded = base64.b64encode(content).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _decode_base64_image(value: str) -> bytes:
    b64_value = value
    if "base64," in b64_value:
        b64_value = b64_value.split("base64,", 1)[1]
    return base64.b64decode(b64_value, validate=False)


def _iter_response_candidates(data: Any):
    if isinstance(data, str):
        yield data
        return
    if isinstance(data, list):
        for item in data:
            yield from _iter_response_candidates(item)
        return
    if not isinstance(data, dict):
        return

    preferred_keys = (
        "b64_json",
        "image",
        "image_base64",
        "base64",
        "url",
        "data",
        "result",
        "output",
    )
    for key in preferred_keys:
        if key in data:
            value = data.get(key)
            if isinstance(value, str):
                yield value
            else:
                yield from _iter_response_candidates(value)

    for key, value in data.items():
        if key not in preferred_keys:
            yield from _iter_response_candidates(value)


def _save_json_image_response(data: dict[str, Any], out_dir: Path, timeout: int) -> str:
    for candidate in _iter_response_candidates(data):
        if not isinstance(candidate, str) or len(candidate) <= 30:
            continue

        if candidate.startswith(("http://", "https://")):
            image_resp = requests.get(candidate, timeout=timeout)
            if image_resp.status_code != 200:
                continue
            content = image_resp.content or b""
            ext = _guess_ext_from_bytes(content)
            if ext == ".bin":
                content_type = image_resp.headers.get("Content-Type", "").lower()
                if "png" in content_type:
                    ext = ".png"
                elif "jpeg" in content_type or "jpg" in content_type:
                    ext = ".jpg"
                elif "webp" in content_type:
                    ext = ".webp"
            return _save_bytes(out_dir, content, ext if ext != ".bin" else ".png")

        if "base64," in candidate or len(candidate) > 30:
            try:
                img_bytes = _decode_base64_image(candidate)
            except Exception:
                continue
            ext = _guess_ext_from_bytes(img_bytes)
            return _save_bytes(out_dir, img_bytes, ext if ext != ".bin" else ".png")

    raise RuntimeError("JSON response did not contain a recognized image field")


def generate_image(
    api_url: str,
    token: str,
    input_image: str,
    prompt: str,
    output_dir: str,
    timeout: int = 120,
    model: str = DEFAULT_MODEL,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    steps: int = DEFAULT_STEPS,
) -> str:
    """Call the Flux2 OpenAI-style JSON API and save the generated image."""
    input_path = Path(input_image).resolve()
    if not input_path.exists() or not input_path.is_file():
        raise FileNotFoundError(f"Input image does not exist: {input_path}")

    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model or DEFAULT_MODEL,
        "input": {
            "prompt": prompt,
            "image": _image_to_data_url(input_path),
        },
        "parameters": {
            "height": int(height),
            "width": int(width),
            "steps": int(steps),
        },
    }

    resp = requests.post(
        api_url,
        headers=headers,
        json=payload,
        timeout=timeout,
    )

    if resp.status_code != 200:
        raw_path = _save_bytes(out_dir, resp.content or b"", ".raw", prefix="error_response")
        raise RuntimeError(f"Request failed, HTTP {resp.status_code}. Response saved: {raw_path}")

    content_type = resp.headers.get("Content-Type", "").lower()
    body = resp.content or b""

    if body and (
        "image/" in content_type
        or body.startswith(b"\x89PNG\r\n\x1a\n")
        or body.startswith(b"\xff\xd8\xff")
        or (body[:4] == b"RIFF" and body[8:12] == b"WEBP")
    ):
        ext = _guess_ext_from_bytes(body)
        if ext == ".bin":
            if "png" in content_type:
                ext = ".png"
            elif "jpeg" in content_type or "jpg" in content_type:
                ext = ".jpg"
            elif "webp" in content_type:
                ext = ".webp"
        return _save_bytes(out_dir, body, ext)

    text = resp.text or ""
    if "application/json" in content_type or text.lstrip().startswith(("{", "[")):
        try:
            data = resp.json()
        except Exception:
            data = json.loads(text)
        if isinstance(data, dict):
            return _save_json_image_response(data, out_dir, timeout)
        if isinstance(data, list):
            return _save_json_image_response({"data": data}, out_dir, timeout)

    raw_path = _save_bytes(out_dir, body, ".raw", prefix="unknown_response")
    snippet = text[:200].replace("\n", " ")
    raise RuntimeError(
        f"Response is not a recognized image format. Raw response saved: {raw_path}. "
        f"Content-Type={content_type or 'N/A'}; snippet={snippet}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Call the Flux2 image generation API and save the generated image"
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_API_URL,
        help="Flux2 model API endpoint",
    )
    parser.add_argument(
        "--token",
        default="abc@123",
        help="Bearer token value without the Bearer prefix",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model name in the JSON payload",
    )
    parser.add_argument(
        "--input",
        default="./cat.jpg",
        help="Input image path",
    )
    parser.add_argument(
        "--prompt",
        default="turn it into watercolor style",
        help="Image generation prompt",
    )
    parser.add_argument(
        "--output-dir",
        default="./outputs",
        help="Output image directory",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Request timeout in seconds",
    )
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)

    args = parser.parse_args()

    try:
        saved = generate_image(
            api_url=args.url,
            token=args.token,
            input_image=args.input,
            prompt=args.prompt,
            output_dir=args.output_dir,
            timeout=args.timeout,
            model=args.model,
            width=args.width,
            height=args.height,
            steps=args.steps,
        )
        print(f"Successfully saved: {saved}")
    except Exception as e:
        print(f"Generation failed: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
