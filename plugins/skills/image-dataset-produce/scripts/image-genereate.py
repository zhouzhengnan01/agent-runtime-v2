import argparse
import base64
import json
import os
from datetime import datetime
from pathlib import Path

import requests


def _guess_ext_from_bytes(content: bytes) -> str:
    """通过文件魔数判断扩展名。"""
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


def generate_image(
    api_url: str,
    token: str,
    input_image: str,
    prompt: str,
    output_dir: str,
    timeout: int = 120,
) -> str:
    """
    上传图片到模型服务，并将返回的图片保存到本地。
    兼容三种返回：
    1) 直接二进制图片；
    2) JSON 内含 base64 图片；
    3) 非图片错误内容（会落盘 raw 便于排查）。
    """
    input_path = Path(input_image).resolve()
    if not input_path.exists() or not input_path.is_file():
        raise FileNotFoundError(f"输入图片不存在: {input_path}")

    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    headers = {
        "Authorization": f"Bearer {token}",
    }

    with open(input_path, "rb") as f:
        files = {
            "file": (input_path.name, f, "application/octet-stream"),
        }
        data = {
            "prompt": prompt,
        }

        resp = requests.post(
            api_url,
            headers=headers,
            files=files,
            data=data,
            timeout=timeout,
        )

    if resp.status_code != 200:
        # 即使失败，也把响应落盘，便于你直接看服务端返回了什么
        raw_path = _save_bytes(out_dir, resp.content or b"", ".raw", prefix="error_response")
        raise RuntimeError(
            f"请求失败，HTTP {resp.status_code}。响应已保存: {raw_path}"
        )

    content_type = resp.headers.get("Content-Type", "").lower()
    body = resp.content or b""

    # 1) 直接返回图片二进制
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

    # 2) 返回 JSON，里面包含 base64 图片
    is_json = "application/json" in content_type
    text = resp.text or ""
    if is_json or text.lstrip().startswith("{"):
        try:
            data = resp.json()
        except Exception:
            try:
                data = json.loads(text)
            except Exception:
                data = None

        if isinstance(data, dict):
            # 常见字段兜底
            candidates = [
                data.get("image"),
                data.get("image_base64"),
                data.get("base64"),
                data.get("data"),
                data.get("result"),
                data.get("output"),
            ]

            # 兼容 output: {image: ...}
            out_obj = data.get("output")
            if isinstance(out_obj, dict):
                candidates.extend([
                    out_obj.get("image"),
                    out_obj.get("image_base64"),
                    out_obj.get("base64"),
                ])

            b64_str = None
            for x in candidates:
                if isinstance(x, str) and len(x) > 30:
                    b64_str = x
                    break

            if b64_str:
                # 去掉 data URL 前缀
                if "base64," in b64_str:
                    b64_str = b64_str.split("base64,", 1)[1]
                try:
                    img_bytes = base64.b64decode(b64_str, validate=False)
                except Exception as e:
                    raise RuntimeError(f"返回了 base64 字段，但解码失败: {e}")

                ext = _guess_ext_from_bytes(img_bytes)
                if ext == ".bin":
                    ext = ".png"
                return _save_bytes(out_dir, img_bytes, ext)

    # 3) 既不是图片也不是可解析的 JSON 图片：保存原始响应
    raw_path = _save_bytes(out_dir, body, ".raw", prefix="unknown_response")
    snippet = (resp.text or "")[:200].replace("\n", " ")
    raise RuntimeError(
        f"响应不是可识别图片格式，已保存原始响应到: {raw_path}；"
        f"Content-Type={content_type or 'N/A'}；片段: {snippet}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="调用本地生图模型接口，上传图片并保存生成结果"
    )
    parser.add_argument(
        "--url",
        default="http://218.67.242.10:58801/flux2/generate",
        help="模型接口地址",
    )
    parser.add_argument(
        "--token",
        default="abc@123",
        help="Bearer Token（仅填 token，不要写 Bearer 前缀）",
    )
    parser.add_argument(
        "--input",
        default="./cat.jpg",
        help="输入图片路径",
    )
    parser.add_argument(
        "--prompt",
        default="把猫改成宇航员",
        help="生成提示词",
    )
    parser.add_argument(
        "--output-dir",
        default="./outputs",
        help="结果图片保存目录",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="请求超时时间（秒）",
    )

    args = parser.parse_args()

    try:
        saved = generate_image(
            api_url=args.url,
            token=args.token,
            input_image=args.input,
            prompt=args.prompt,
            output_dir=args.output_dir,
            timeout=args.timeout,
        )
        print(f"生成成功，已保存: {saved}")
    except Exception as e:
        print(f"生成失败: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
