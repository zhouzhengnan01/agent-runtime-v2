from __future__ import annotations
'''
Author: 13594053100@163.com
Date: 2025-11-12 16:32:39
LastEditTime: 2025-11-12 16:32:43
'''
import os
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Generator

import httpx

from app.shared.jetlinks_video.utils import logger_utils
logger = logger_utils.get_logger(__name__)

class LocalASRClient:
    """
    本地 ASR 占位客户端。
    接口对齐 CloudASRClient：
      infer() -> (mode, iter_pair, nonstream_pair)
        - mode == "stream":  iter_pair 为生成器，yield (delta:str, usage_part:dict|None)
        - mode == "nonstream": nonstream_pair 为 (full_text:str, usage:dict|None)
    """
    def __init__(self, *, model_name: str, audio_uri: str, q_ctrl=None, stop=None, asr_options: Optional[Dict[str, Any]] = None):
        self.model_name = model_name
        self.audio_uri = audio_uri
        self.q_ctrl = q_ctrl
        self.stop = stop
        self.opt = asr_options or {}

    def infer(self) -> Tuple[str, Optional[Generator[Tuple[str, Optional[dict]], None, None]], Optional[Tuple[str, Optional[dict]]]]:
        text, usage = self._run_nonstream()
        return "nonstream", None, (text, usage)

    def _run_nonstream(self) -> Tuple[str, Optional[dict]]:
        base_url = self._resolve_base_url()
        model = self._resolve_model_name()
        audio_path = self._to_local_path(self.audio_uri)
        if not os.path.exists(audio_path):
            raise RuntimeError(f"audio file not found: {audio_path}")

        response_format = str(self.opt.get("response_format") or "json").strip()
        timeout_sec = float(os.getenv("ASR_HTTP_TIMEOUT_SEC") or "120")
        headers = {}
        api_key = (os.getenv("ASR_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        endpoint = f"{base_url}/audio/transcriptions"
        data = {"model": model, "response_format": response_format}

        with open(audio_path, "rb") as f:
            files = {"file": (Path(audio_path).name, f, "application/octet-stream")}
            resp = httpx.post(endpoint, data=data, files=files, headers=headers, timeout=timeout_sec)

        if resp.status_code >= 400:
            raise RuntimeError(f"local ASR failed: HTTP {resp.status_code} {resp.text}")

        if response_format.lower() == "text" or resp.headers.get("content-type", "").startswith("text/"):
            text = resp.text.strip()
        else:
            payload = resp.json()
            text = (payload.get("text") or payload.get("result") or payload.get("sentence") or "").strip()

        return text, {"asr_backend": "local"}

    @staticmethod
    def _normalize_base_url(base_url: str) -> str:
        u = (base_url or "").strip().rstrip("/")
        for suffix in ("/chat/completions", "/completions", "/v1/chat/completions", "/v1/completions"):
            if u.endswith(suffix):
                u = u[: -len(suffix)].rstrip("/")
                break
        if u and not u.endswith("/v1"):
            u = f"{u}/v1"
        return u

    def _resolve_base_url(self) -> str:
        raw = (
            os.getenv("ASR_API_BASE")
            or os.getenv("ASR_BASE_URL")
            or os.getenv("ASR_MODEL_SERVICE_BASE_URL")
            or os.getenv("MODEL_SERVICE_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("OPENAI_API_BASE")
            or ""
        )
        base = self._normalize_base_url(raw)
        if not base:
            raise RuntimeError("ASR base url not configured (set ASR_API_BASE)")
        return base

    def _resolve_model_name(self) -> str:
        env_model = (os.getenv("ASR_MODEL") or os.getenv("ASR_LOCAL_MODEL") or "").strip()
        if env_model:
            return env_model
        name = (self.model_name or "").strip()
        if not name or name.lower() == "whisper":
            return "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
        return name

    @staticmethod
    def _to_local_path(maybe_uri: str) -> str:
        if not maybe_uri:
            raise RuntimeError("audio_uri is empty")
        if maybe_uri.lower().startswith("file://"):
            return maybe_uri[len("file://"):]
        return maybe_uri
