import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

# --- Configurable: set this to the location of your image-genereate.py ---
# Default: look for image-genereate.py in the same parent directory as this skill.
import os

# Resolve image-genereate.py: same dir > plugin root > env var
_SAME_DIR = str(Path(__file__).resolve().parent / "image-genereate.py")
_PLUGIN_ROOT = str(Path(__file__).resolve().parents[3] / "image-genereate.py")
IMAGE_GENERATE_SCRIPT = (
    os.environ.get("IMAGE_GENERATE_SCRIPT", "")
    or (Path(_SAME_DIR).as_posix() if Path(_SAME_DIR).is_file() else "")
    or (Path(_PLUGIN_ROOT).as_posix() if Path(_PLUGIN_ROOT).is_file() else "")
    or _PLUGIN_ROOT  # fallback: will raise FileNotFoundError with the expected path
)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def _require_abs_path(path_str: str, field_name: str) -> Path:
    p = Path(path_str)
    if not p.is_absolute():
        raise ValueError(f"{field_name} must be an absolute path, got: {path_str}")
    return p


def _load_input(input_path: Path) -> dict:
    if not input_path.exists():
        raise FileNotFoundError(f"input.json not found: {input_path}")
    with input_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("input.json must contain a JSON object")
    return _normalize_runtime_payload(payload)


def _normalize_runtime_payload(payload: dict) -> dict:
    if isinstance(payload.get("model"), dict) and isinstance(payload.get("task"), dict):
        return payload

    spec = payload.get("spec") if isinstance(payload.get("spec"), dict) else payload
    ctx = spec.get("workflow_context") if isinstance(spec.get("workflow_context"), dict) else {}
    outputs_dir = str(payload.get("outputs_dir") or "").strip()
    model = {
        "api_url": spec.get("api_url") or "http://192.168.33.25:8801/flux2/generate",
        "token": spec.get("token") or os.environ.get("IMAGE_GEN_TOKEN", "") or "abc@123",
        "timeout": spec.get("timeout") or 120,
    }
    task = {
        "input_image": spec.get("input_image") or spec.get("image_path") or _first_attachment_path(spec),
        "prompt": spec.get("prompt") or spec.get("generation_prompt") or _prompt_from_text(str(spec.get("overrides_text") or "")) or spec.get("task") or "",
        "output_dir": spec.get("output_dir") or ctx.get("output_dir") or (str(Path(outputs_dir) / "generated_images") if outputs_dir else "") or str(Path(__file__).resolve().parents[2] / "outputs"),
    }
    return {"model": model, "task": task}


def _first_attachment_path(spec: dict) -> str:
    attachments = spec.get("attachments")
    if not isinstance(attachments, list):
        return ""
    for item in attachments:
        if not isinstance(item, dict):
            continue
        mime_type = str(item.get("mime_type") or item.get("content_type") or "").lower()
        name = str(item.get("name") or item.get("filename") or "").lower()
        path = str(item.get("path") or item.get("local_path") or item.get("file_path") or "").strip()
        path_lower = path.lower()
        if path and (mime_type.startswith("image/") or name.endswith(IMAGE_EXTENSIONS) or path_lower.endswith(IMAGE_EXTENSIONS)):
            return path
    return ""


def _prompt_from_text(text: str) -> str:
    match = re.search(r"(?:prompt|提示词)\s*[:：]\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def _extract_result_path(stdout: str) -> str:
    m = re.search(r"saved:\s*(.+)", stdout)
    if not m:
        m = re.search(r"Successfully saved:\s*(.+)", stdout)
    if not m:
        m = re.search(r"鐢熸垚鎴愬姛锛屽凡淇濆瓨:\s*(.+)", stdout)
    return m.group(1).strip() if m else ""


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


def main():
    parser = argparse.ArgumentParser(
        description="Read input.json and call image-genereate.py to generate images"
    )
    parser.add_argument(
        "--input",
        default="input.json",
        help="Path to the input.json config file",
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    cfg = _load_input(input_path)

    model = cfg.get("model", {})
    task = cfg.get("task", {})

    api_url = str(model.get("api_url", "")).strip()
    token = str(model.get("token", "")).strip()
    timeout = int(model.get("timeout", 120))

    input_image = str(task.get("input_image", "")).strip()
    prompt = str(task.get("prompt", "")).strip()
    output_dir = str(task.get("output_dir", "")).strip()

    if not api_url:
        raise ValueError("model.api_url must not be empty")
    if not token:
        raise ValueError("model.token must not be empty")
    if not prompt:
        raise ValueError("task.prompt must not be empty")

    if not output_dir:
        raise ValueError("task.output_dir must not be empty - user must specify the output directory")

    input_image_path = _require_abs_path(input_image, "task.input_image")
    output_dir_path = _require_abs_path(output_dir, "task.output_dir")

    if not input_image_path.exists() or not input_image_path.is_file():
        raise FileNotFoundError(f"Input image does not exist: {input_image_path}")

    output_dir_path.mkdir(parents=True, exist_ok=True)

    image_script = Path(IMAGE_GENERATE_SCRIPT).resolve()
    if not image_script.exists():
        raise FileNotFoundError(
            f"image-genereate.py not found at: {image_script}. "
            f"Set IMAGE_GENERATE_SCRIPT env var or edit IMAGE_GENERATE_SCRIPT in this script."
        )

    cmd = [
        sys.executable,
        str(image_script),
        "--url", api_url,
        "--token", token,
        "--input", str(input_image_path),
        "--prompt", prompt,
        "--output-dir", str(output_dir_path),
        "--timeout", str(timeout),
    ]

    print("[image-dataset-generation] Starting image generation task...")
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
    print(f"api_url: {api_url}")
    print(f"input_image: {input_image_path}")
    print(f"prompt: {prompt}")
    print(f"output_dir: {output_dir_path}")
    print(f"result_image_path: {result_image_path}")
    print(f"elapsed_seconds: {elapsed}")
    print(f"status: {status}")
    if status == "failed":
        err_msg = stderr or stdout or f"child process exit code: {proc.returncode}"
        print(f"error_message: {err_msg}")
        raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
