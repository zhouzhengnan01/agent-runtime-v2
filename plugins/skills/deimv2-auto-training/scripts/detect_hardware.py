from __future__ import annotations

import argparse
import json
import os
import platform
from typing import Any


def _safe_call(obj: Any, name: str, default: Any) -> Any:
    try:
        value = getattr(obj, name)
        return value() if callable(value) else value
    except Exception:
        return default


def detect_hardware() -> dict[str, Any]:
    result: dict[str, Any] = {
        "priority": ["cuda", "npu", "cpu"],
        "selected": "cpu",
        "device": "cpu",
        "device_count": 1,
        "device_name": platform.processor() or "CPU",
        "cuda_available": False,
        "npu_available": False,
    }

    try:
        import torch
    except Exception as exc:
        result["torch_error"] = str(exc)
        return result

    try:
        cuda = getattr(torch, "cuda", None)
        if cuda is not None and bool(cuda.is_available()) and int(cuda.device_count()) > 0:
            result.update(
                {
                    "selected": "cuda",
                    "device": "cuda",
                    "device_count": int(cuda.device_count()),
                    "device_name": str(cuda.get_device_name(0)),
                    "cuda_available": True,
                    "torch_version": str(torch.__version__),
                }
            )
            return result
    except Exception as exc:
        result["cuda_error"] = str(exc)

    try:
        import torch_npu  # noqa: F401

        npu = getattr(torch, "npu", None)
        if npu is not None and bool(npu.is_available()):
            count = int(_safe_call(npu, "device_count", 1) or 1)
            name = _safe_call(npu, "get_device_name", "Ascend NPU")
            if callable(getattr(npu, "get_device_name", None)):
                try:
                    name = npu.get_device_name(0)
                except Exception:
                    name = "Ascend NPU"
            result.update(
                {
                    "selected": "npu",
                    "device": "npu:0",
                    "device_count": count,
                    "device_name": str(name),
                    "npu_available": True,
                    "torch_version": str(torch.__version__),
                }
            )
            return result
    except Exception as exc:
        result["npu_error"] = str(exc)

    result["torch_version"] = str(torch.__version__)
    result["cpu_threads"] = int(os.cpu_count() or 1)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Select training hardware in CUDA GPU > Ascend NPU > CPU order.")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    args = parser.parse_args()
    result = detect_hardware()
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        from pathlib import Path

        path = Path(args.output).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
