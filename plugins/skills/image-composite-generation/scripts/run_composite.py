import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


SAME_DIR = Path(__file__).resolve().parent
COMPOSITE_SCRIPT = os.environ.get("IMAGE_COMPOSITE_SCRIPT", "") or str(SAME_DIR / "image-composite.py")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def _require_abs_path(path_str: str, field_name: str) -> Path:
    if path_str.startswith(("http://", "https://", "data:")):
        return Path(path_str)
    path = Path(path_str)
    if not path.is_absolute():
        raise ValueError(f"{field_name} must be an absolute file/folder path, URL, or data URL, got: {path_str}")
    return path


def _load_input(input_path: Path) -> dict:
    if not input_path.exists():
        raise FileNotFoundError(f"input.json not found: {input_path}")
    with input_path.open("r", encoding="utf-8-sig") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("input.json must contain a JSON object")
    return _normalize_runtime_payload(payload)


def _normalize_runtime_payload(payload: dict) -> dict:
    if isinstance(payload.get("task"), dict):
        model = payload.get("model") if isinstance(payload.get("model"), dict) else {}
        task = dict(payload["task"])
        return {"model": model, "task": task}
    spec = payload.get("spec") if isinstance(payload.get("spec"), dict) else payload
    ctx = spec.get("workflow_context") if isinstance(spec.get("workflow_context"), dict) else {}
    task_spec = spec.get("task") if isinstance(spec.get("task"), dict) else {}
    images = _attachment_image_paths(spec)
    model = {
        "api_url": spec.get("api_url") or "https://www.tokencloud.yun/v1/images/generations",
        "token": spec.get("token") or spec.get("api_key") or "sk-6qbqlTAZV7qVszLplZhxZH2yvQwSMRotrOT8wr9aVPII8BYo",
        "model": spec.get("model") or "wan2.7-image-pro",
        "timeout": spec.get("timeout") or 180,
        "size": spec.get("size") or "768*768",
        "count": spec.get("count") or 1,
        "sleep": spec.get("sleep") or 1.0,
        "retries": spec.get("retries") or 3,
        "retry_sleep": spec.get("retry_sleep") or spec.get("retrySleep") or 5.0,
        "negative_prompt": spec.get("negative_prompt") or spec.get("negativePrompt") or "",
        "prompt_extend": spec.get("prompt_extend") or spec.get("promptExtend") or False,
        "watermark": spec.get("watermark") or False,
    }
    task = {
        "image1": task_spec.get("image1")
        or spec.get("image1")
        or spec.get("input_image1")
        or spec.get("background_image")
        or (images[0] if len(images) > 0 else ""),
        "image2": task_spec.get("image2")
        or spec.get("image2")
        or spec.get("input_image2")
        or spec.get("foreground_image")
        or (images[1] if len(images) > 1 else ""),
        "prompt": task_spec.get("prompt")
        or spec.get("prompt")
        or spec.get("composite_prompt")
        or spec.get("generation_prompt")
        or _prompt_from_text(str(spec.get("overrides_text") or ""))
        or "",
        "output_dir": task_spec.get("output_dir") or spec.get("output_dir") or ctx.get("output_dir") or str(payload.get("outputs_dir") or "").strip(),
    }
    return {"model": model, "task": task}


def _attachment_image_paths(spec: dict) -> list[str]:
    attachments = spec.get("attachments")
    if not isinstance(attachments, list):
        return []
    paths: list[str] = []
    for item in attachments:
        if not isinstance(item, dict):
            continue
        mime_type = str(item.get("mime_type") or item.get("content_type") or "").lower()
        name = str(item.get("name") or item.get("filename") or "").lower()
        path = str(item.get("path") or item.get("local_path") or item.get("file_path") or "").strip()
        path_lower = path.lower()
        if path and (mime_type.startswith("image/") or name.endswith(IMAGE_EXTENSIONS) or path_lower.endswith(IMAGE_EXTENSIONS)):
            paths.append(path)
    return paths


def _prompt_from_text(text: str) -> str:
    match = re.search(r"(?:prompt|提示词|composite_prompt|合成提示词)\s*[:：=]\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""


def _decode_output(value: bytes | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    for encoding in ("utf-8", "gb18030", "gbk"):
        try:
            return value.decode(encoding)
        except UnicodeDecodeError:
            continue
    return value.decode("utf-8", errors="replace")


def _extract_result_path(stdout: str) -> str:
    match = re.search(r"Composite success\. saved:\s*(.+)", stdout)
    return match.group(1).strip() if match else ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Read input.json and call image-composite.py.")
    parser.add_argument("--input", default="input.json")
    args = parser.parse_args()

    cfg = _load_input(Path(args.input).resolve())
    model = cfg.get("model", {})
    task = cfg.get("task", {})

    image1 = str(task.get("image1", "")).strip()
    image2 = str(task.get("image2", "")).strip()
    prompt = str(task.get("prompt", "")).strip()
    output_dir = str(task.get("output_dir", "")).strip()

    if not image1:
        raise ValueError("task.image1 must not be empty")
    if not image2:
        raise ValueError("task.image2 must not be empty")
    if not prompt:
        raise ValueError("task.prompt must not be empty")
    if not output_dir:
        raise ValueError("task.output_dir must not be empty")

    for field_name, value in (("task.image1", image1), ("task.image2", image2)):
        path = _require_abs_path(value, field_name)
        if not value.startswith(("http://", "https://", "data:")) and not path.exists():
            raise FileNotFoundError(f"{field_name} does not exist: {path}")
        if not value.startswith(("http://", "https://", "data:")) and not (path.is_file() or path.is_dir()):
            raise ValueError(f"{field_name} must be a file or folder path: {path}")

    output_path = _require_abs_path(output_dir, "task.output_dir")
    output_path.mkdir(parents=True, exist_ok=True)

    composite_script = Path(COMPOSITE_SCRIPT).resolve()
    if not composite_script.exists():
        raise FileNotFoundError(f"image-composite.py not found: {composite_script}")

    cmd = [
        sys.executable,
        str(composite_script),
        "--url",
        str(model.get("api_url", "") or "https://www.tokencloud.yun/v1/images/generations"),
        "--token",
        str(model.get("token", "") or model.get("api_key", "") or "sk-6qbqlTAZV7qVszLplZhxZH2yvQwSMRotrOT8wr9aVPII8BYo"),
        "--model",
        str(model.get("model", "") or "wan2.7-image-pro"),
        "--image1",
        image1,
        "--image2",
        image2,
        "--prompt",
        prompt,
        "--output-dir",
        str(output_path),
        "--size",
        str(model.get("size", "") or "768*768"),
        "--count",
        str(int(model.get("count", 1) or 1)),
        "--timeout",
        str(int(model.get("timeout", 180) or 180)),
        "--sleep",
        str(float(model.get("sleep", 1.0) or 1.0)),
        "--retries",
        str(int(model.get("retries", 3) or 3)),
        "--retry-sleep",
        str(float(model.get("retry_sleep", 5.0) or 5.0)),
    ]
    negative_prompt = str(model.get("negative_prompt", "") or "").strip()
    if negative_prompt:
        cmd.extend(["--negative-prompt", negative_prompt])
    if bool(model.get("prompt_extend", False)):
        cmd.append("--prompt-extend")
    if bool(model.get("watermark", False)):
        cmd.append("--watermark")

    print("[image-composite-generation] Starting image composition task...")
    start = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=False)
    elapsed = round(time.time() - start, 2)

    stdout = _decode_output(proc.stdout).strip()
    stderr = _decode_output(proc.stderr).strip()
    if stdout:
        print(f"\n--- child stdout ---\n{stdout}")
    if stderr:
        print(f"\n--- child stderr ---\n{stderr}")

    result_image_path = _extract_result_path(stdout)
    status = "success" if proc.returncode == 0 else "failed"
    print("\n--- summary ---")
    print(f"api_url: {model.get('api_url') or 'https://www.tokencloud.yun/v1/images/generations'}")
    print(f"model: {model.get('model') or 'wan2.7-image-pro'}")
    print(f"image1: {image1}")
    print(f"image2: {image2}")
    print(f"prompt: {prompt}")
    print(f"output_dir: {output_path}")
    print(f"result_image_path: {result_image_path}")
    print(f"elapsed_seconds: {elapsed}")
    print(f"status: {status}")
    if status == "failed":
        print(f"error_message: {stderr or stdout or f'child process exit code: {proc.returncode}'}")
        raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
