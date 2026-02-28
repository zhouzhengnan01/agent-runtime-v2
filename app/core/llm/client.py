"""
LLM 客户端 - 支持多种LLM提供商
"""
import os
from typing import Dict, Any, List, Optional, Tuple
import logging
from openai import AsyncOpenAI, OpenAI
import httpx
import asyncio

from app.core.llm.model_config_manager import get_model_config_manager
from app.config import settings

logger = logging.getLogger(__name__)


class LLMClient:
    """统一的LLM客户端"""

    # 当 VLM_BASE_URL 指向本地 OpenAI 兼容服务（如 vLLM）时，DB 里的云端模型名可能不存在。
    # 这里缓存一次“模型不存在”的结果，后续直接回退到 env/config 的 VLM 模型，避免重复 404。
    _VLM_MODEL_FALLBACK_CACHE: dict[str, str] = {}
    _LLM_MODEL_FALLBACK_CACHE: dict[str, str] = {}
    
    def __init__(self, provider: str = "unified", **kwargs):
        """
        初始化LLM客户端
        
        Args:
            provider: LLM提供商 (qwen, openai, deepseek, glm等)
            **kwargs: 额外配置参数
        """
        self.provider = provider
        self.config = kwargs
        self.llm_fallback_client: Optional[Any] = None
        self.vlm_fallback_client: Optional[Any] = None
        self.embedding_fallback_client: Optional[Any] = None
        self.llm_fallback_model: Optional[str] = None
        self.vlm_fallback_model: Optional[str] = None
        self.embedding_fallback_model: Optional[str] = None
        self.last_usage: Optional[Dict[str, Any]] = None

        # 初始化客户端
        if provider == "unified":
            # 使用统一配置系统（LLM/VLM/Embedding 可分离配置）
            self._ensure_model_config()
            try:
                from .client_factory import create_llm_client, create_vlm_client, create_embedding_client

                self.llm_client = create_llm_client(async_client=False)
                self.vlm_client = create_vlm_client(async_client=False)
                self.embedding_client = create_embedding_client(async_client=False)
                self._init_fallback_clients()

                # 默认 client 仍指向 LLM，保持向后兼容
                self.client = self.llm_client
                self.provider = "unified"
            except Exception as e:
                logger.warning(f"⚠️ 统一配置初始化失败，回退到qwen: {e}")
                self.provider = "qwen"
                self.client = self._init_qwen_client()
        elif provider == "qwen":
            self.client = self._init_qwen_client()
        elif provider == "openai":
            self.client = self._init_openai_client()
        elif provider == "deepseek":
            self.client = self._init_deepseek_client()
        elif provider == "glm":
            self.client = self._init_glm_client()
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def _ensure_model_config(self) -> None:
        """保证模型配置管理器已初始化，防止 fallback 走错误的 base_url/model"""
        try:
            manager = get_model_config_manager()
            if not getattr(manager, "_config", None):
                manager.initialize()
        except Exception as e:
            logger.warning(f"⚠️ 模型配置初始化失败，将回退到单一 provider: {e}")

    def _init_fallback_clients(self) -> None:
        """初始化备用模型服务客户端（基于 *_BASE_URL_BACKUP 环境变量）。"""
        llm_client, llm_model = self._build_fallback_client("LLM")
        vlm_client, vlm_model = self._build_fallback_client("VLM")
        embedding_client, embedding_model = self._build_fallback_client("EMBEDDING")

        self.llm_fallback_client = llm_client
        self.vlm_fallback_client = vlm_client
        self.embedding_fallback_client = embedding_client
        self.llm_fallback_model = llm_model
        self.vlm_fallback_model = vlm_model
        self.embedding_fallback_model = embedding_model

    @staticmethod
    def _build_fallback_client(prefix: str) -> Tuple[Optional[Any], Optional[str]]:
        backup_base_url = os.getenv(f"{prefix}_BASE_URL_BACKUP") or os.getenv("OPENAI_BASE_URL_BACKUP")
        if not backup_base_url:
            return None, None

        backup_api_key = (
            os.getenv(f"{prefix}_API_KEY_BACKUP")
            or os.getenv("OPENAI_API_KEY_BACKUP")
            or os.getenv(f"{prefix}_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        if not backup_api_key:
            logger.warning(f"⚠️ 未配置 {prefix}_API_KEY_BACKUP/OPENAI_API_KEY_BACKUP，备用客户端跳过")
            return None, None

        backup_model = os.getenv(f"{prefix}_MODEL_BACKUP")
        logger.info(f"🔁 已启用 {prefix} 备用服务: {backup_base_url}")
        return OpenAI(api_key=backup_api_key, base_url=backup_base_url), backup_model

    def _get_fallback_chat_client(self, client: Any) -> Tuple[Optional[Any], Optional[str]]:
        if self._is_vlm_client(client):
            return self.vlm_fallback_client, self.vlm_fallback_model
        if self._is_llm_client(client):
            return self.llm_fallback_client, self.llm_fallback_model
        return None, None

    @staticmethod
    def _is_connection_error(error: Exception) -> bool:
        try:
            from openai import APITimeoutError, APIConnectionError  # type: ignore

            if isinstance(error, (APITimeoutError, APIConnectionError)):
                return True
        except Exception:
            pass

        if isinstance(error, (httpx.TimeoutException, httpx.ConnectError, httpx.ConnectTimeout)):
            return True

        message = str(error).lower()
        return any(
            key in message
            for key in (
                "connection error",
                "connect timeout",
                "timed out",
                "connection refused",
                "network is unreachable",
            )
        )

    def _get_chat_client(self, messages: List[Dict[str, Any]], model: Optional[str] = None):
        # 根据是否多模态请求，选择 LLM 或 VLM client。
        if self.provider != "unified":
            return self.client

        use_vlm = False
        try:
            if self._has_image_content(messages):
                use_vlm = True
        except Exception:
            # 兜底：不让检测失败影响正常对话
            use_vlm = False

        if model:
            env_vlm_model = os.getenv("VLM_MODEL")
            if env_vlm_model and model == env_vlm_model:
                use_vlm = True
            try:
                cfg = get_model_config_manager().get_model_config("vlm")
                if cfg and model == cfg.model_name:
                    use_vlm = True
            except Exception:
                pass

        if use_vlm:
            return getattr(self, "vlm_client", None) or self.client

        return getattr(self, "llm_client", None) or self.client

    def _get_embedding_client(self):
        if self.provider == "unified":
            return getattr(self, "embedding_client", None) or self.client
        return self.client

    def _is_vlm_client(self, client: Any) -> bool:
        if self.provider != "unified":
            return False
        vlm_client = getattr(self, "vlm_client", None)
        return bool(vlm_client and client is vlm_client)

    def _get_vlm_fallback_model(self) -> Optional[str]:
        target_model: Optional[str] = None
        try:
            cfg = get_model_config_manager().get_model_config("vlm")
            target_model = getattr(cfg, "model_name", None)
        except Exception:
            target_model = None
        return target_model or os.getenv("VLM_MODEL")

    def _get_llm_fallback_model(self) -> Optional[str]:
        target_model: Optional[str] = None
        try:
            cfg = get_model_config_manager().get_model_config("llm")
            target_model = getattr(cfg, "model_name", None)
        except Exception:
            target_model = None
        return target_model or os.getenv("LLM_MODEL")

    def _is_llm_client(self, client: Any) -> bool:
        if self.provider != "unified":
            return False
        llm_client = getattr(self, "llm_client", None)
        return bool(llm_client and client is llm_client)

    @staticmethod
    def _is_transport_closed_runtime_error(e: BaseException) -> bool:
        """uvloop/httpx 关闭连接时偶发 RuntimeError（closed transport / handler is closed）。"""
        if not isinstance(e, RuntimeError):
            return False
        msg = str(e)
        return (
            "TCPTransport closed=True" in msg
            or "handler is closed" in msg
            or ("write_eof" in msg and "uvloop" in msg)
        )

    def _extract_usage(self, response: Any, *, model: Optional[str] = None) -> Optional[Dict[str, Any]]:
        usage_obj = getattr(response, "usage", None)
        if usage_obj is None:
            return None

        if isinstance(usage_obj, dict):
            # Chat Completions: prompt_tokens/completion_tokens
            # Responses API: input_tokens/output_tokens
            prompt_tokens = usage_obj.get("prompt_tokens") or usage_obj.get("input_tokens")
            completion_tokens = usage_obj.get("completion_tokens") or usage_obj.get("output_tokens")
            total_tokens = usage_obj.get("total_tokens")
            details = (
                usage_obj.get("prompt_tokens_details")
                or usage_obj.get("completion_tokens_details")
                or usage_obj.get("input_tokens_details")
                or usage_obj.get("output_tokens_details")
            )
        else:
            prompt_tokens = getattr(usage_obj, "prompt_tokens", None) or getattr(usage_obj, "input_tokens", None)
            completion_tokens = getattr(usage_obj, "completion_tokens", None) or getattr(usage_obj, "output_tokens", None)
            total_tokens = getattr(usage_obj, "total_tokens", None)
            details = (
                getattr(usage_obj, "prompt_tokens_details", None)
                or getattr(usage_obj, "completion_tokens_details", None)
                or getattr(usage_obj, "input_tokens_details", None)
                or getattr(usage_obj, "output_tokens_details", None)
            )

        try:
            prompt_tokens_i = int(prompt_tokens or 0)
        except Exception:
            prompt_tokens_i = 0
        try:
            completion_tokens_i = int(completion_tokens or 0)
        except Exception:
            completion_tokens_i = 0
        try:
            total_tokens_i = int(total_tokens or (prompt_tokens_i + completion_tokens_i))
        except Exception:
            total_tokens_i = prompt_tokens_i + completion_tokens_i

        if prompt_tokens_i <= 0 and completion_tokens_i <= 0 and total_tokens_i <= 0:
            return None

        model_name = getattr(response, "model", None) or model

        usage: Dict[str, Any] = {
            "prompt_tokens": prompt_tokens_i,
            "completion_tokens": completion_tokens_i,
            "total_tokens": total_tokens_i,
            "provider": self.provider,
        }
        if model_name:
            usage["model"] = model_name
        if details:
            # Ensure JSON-serializable (FastAPI response + review record persistence).
            safe_details: Any = None
            if isinstance(details, dict):
                safe_details = details
            else:
                try:
                    safe_details = details.model_dump()  # type: ignore[attr-defined]
                except Exception:
                    try:
                        safe_details = details.dict()  # type: ignore[attr-defined]
                    except Exception:
                        safe_details = None
            if safe_details:
                usage["details"] = safe_details
        return usage

    @staticmethod
    def _is_chat_completions_endpoint_missing(error: Exception) -> bool:
        """Detect providers that only support Responses API (no /chat/completions)."""
        status_code = getattr(error, "status_code", None) or getattr(error, "status", None)
        if status_code not in (404, 405):
            return False

        msg = str(error).lower()
        if "chat/completions" in msg:
            return True
        if "cannot post" in msg and "chat" in msg and "completions" in msg:
            return True
        return False

    @staticmethod
    def _is_responses_stream_required(error: Exception) -> bool:
        msg = str(error).lower()
        return "stream must be set to true" in msg or "stream must be true" in msg

    @staticmethod
    def _is_responses_system_messages_not_allowed(error: Exception) -> bool:
        msg = str(error).lower()
        return "system messages are not allowed" in msg

    @staticmethod
    def _extract_unsupported_parameter_name(error: Exception) -> Optional[str]:
        import re

        msg = str(error)
        m = re.search(r"unsupported parameter\\s*[:：]\\s*([\\w.]+)", msg, re.IGNORECASE)
        if m:
            return m.group(1)
        return None

    @staticmethod
    def _coerce_messages_to_responses_input_and_instructions(
        messages: List[Dict[str, Any]],
        *,
        use_instructions: bool = False,
    ) -> Tuple[List[Dict[str, Any]], str]:
        """Convert Chat Completions messages to Responses API (input + instructions)."""
        out: List[Dict[str, Any]] = []
        instructions_parts: List[str] = []

        def _role(raw: Any) -> str:
            r = str(raw or "user").strip().lower()
            if r in {"user", "assistant", "system", "developer"}:
                return r
            return "user"

        def _append_text(parts: List[Dict[str, Any]], text: Any) -> None:
            s = str(text or "")
            if not s:
                return
            parts.append({"type": "input_text", "text": s})

        def _append_image(parts: List[Dict[str, Any]], url: Any) -> None:
            s = str(url or "").strip()
            if not s:
                return
            parts.append({"type": "input_image", "image_url": s})

        def _content_to_plain_text(content: Any) -> str:
            if content is None:
                return ""
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                texts: List[str] = []
                for part in content:
                    if isinstance(part, str) and part.strip():
                        texts.append(part.strip())
                    elif isinstance(part, dict):
                        t = part.get("text")
                        if isinstance(t, str) and t.strip():
                            texts.append(t.strip())
                return "\n".join(texts)
            if isinstance(content, dict):
                t = content.get("text")
                if isinstance(t, str) and t.strip():
                    return t.strip()
            return str(content)

        def _content_to_responses_parts(content: Any) -> List[Dict[str, Any]]:
            parts: List[Dict[str, Any]] = []

            if isinstance(content, list):
                for part in content:
                    if isinstance(part, str):
                        _append_text(parts, part)
                        continue
                    if not isinstance(part, dict):
                        continue

                    ptype = str(part.get("type") or "").strip().lower()
                    if ptype in {"text", "input_text"} or "text" in part:
                        _append_text(parts, part.get("text"))
                        continue

                    if ptype in {"image_url", "image", "input_image"} or "image_url" in part or "image" in part:
                        if isinstance(part.get("image_url"), dict):
                            _append_image(parts, (part.get("image_url") or {}).get("url"))
                        else:
                            _append_image(parts, part.get("image_url") or part.get("image"))
                        continue

                return parts or [{"type": "input_text", "text": ""}]

            if isinstance(content, dict):
                ptype = str(content.get("type") or "").strip().lower()
                if ptype in {"text", "input_text"} or "text" in content:
                    _append_text(parts, content.get("text"))
                    return parts or [{"type": "input_text", "text": ""}]
                if ptype in {"image_url", "image", "input_image"} or "image_url" in content or "image" in content:
                    if isinstance(content.get("image_url"), dict):
                        _append_image(parts, (content.get("image_url") or {}).get("url"))
                    else:
                        _append_image(parts, content.get("image_url") or content.get("image"))
                    return parts or [{"type": "input_text", "text": ""}]

            # Fallback: stringify to input_text.
            _append_text(parts, content)
            return parts or [{"type": "input_text", "text": ""}]

        for m in messages or []:
            if not isinstance(m, dict):
                continue
            role = _role(m.get("role"))
            if role in {"system", "developer"} and use_instructions:
                t = _content_to_plain_text(m.get("content")).strip()
                if t:
                    instructions_parts.append(t)
                continue

            parts = _content_to_responses_parts(m.get("content"))

            out.append(
                {
                    "role": role,
                    "content": parts,
                }
            )

        instructions = "\n\n".join([p for p in instructions_parts if p.strip()])
        return out, instructions

    @staticmethod
    def _coerce_response_format_to_responses_text(response_format: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not response_format or not isinstance(response_format, dict):
            return None

        t = str(response_format.get("type") or "").strip().lower()
        if not t:
            return None

        if t in {"json_object", "object"}:
            return {"format": {"type": "json_object"}}

        if t in {"json_schema", "schema"}:
            inner = response_format.get("json_schema") if isinstance(response_format.get("json_schema"), dict) else None
            if not inner:
                return None
            schema = inner.get("schema")
            if not isinstance(schema, dict) or not schema:
                return None
            return {
                "format": {
                    "type": "json_schema",
                    "name": str(inner.get("name") or "AnalysisResult"),
                    "schema": schema,
                    "strict": bool(inner.get("strict", True)),
                    "description": str(inner.get("description") or ""),
                }
            }

        return None

    def _is_model_not_found_error(self, error: Exception) -> bool:
        try:
            from openai import NotFoundError  # type: ignore

            if isinstance(error, NotFoundError):
                return True
        except Exception:
            pass

        status_code = getattr(error, "status_code", None)
        message = str(error).lower()
        if status_code == 404 and "model" in message:
            return True
        if "does not exist" in message and "model" in message:
            return True
        if "model_not_found" in message:
            return True
        return False

    @staticmethod
    def _is_response_format_unsupported_error(error: Exception) -> bool:
        """
        Some OpenAI-compatible providers don't support `response_format` (or only support
        parts of it). Detect common error messages so we can retry without it.
        """
        try:
            body = getattr(error, "body", None)
            body_text = str(body).lower() if body is not None else ""
        except Exception:
            body_text = ""

        message = (str(error) + " " + body_text).lower()
        if ("response_format" not in message) and ("json_schema" not in message):
            return False

        keywords = [
            "unknown",
            "unrecognized",
            "unexpected",
            "unsupported",
            "not supported",
            "invalid",
            "extra fields",
            "additional properties",
        ]
        return any(k in message for k in keywords)

    def _has_image_content(self, messages: List[Dict[str, Any]]) -> bool:
        # 检测消息中是否包含图片内容（兼容 OpenAI 多模态 content=list 格式）
        if not messages:
            return False

        # 1) 结构化多模态判断
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = (message.get("role") or "").lower().strip()
            if role and role != "user":
                continue
            message_content = message.get("content")
            if not message_content:
                continue

            # OpenAI 多模态：content=[{type:text,...},{type:image_url,...}]
            if isinstance(message_content, list):
                for part in message_content:
                    if isinstance(part, dict):
                        ptype = part.get("type")
                        if ptype in {"image_url", "image", "video"}:
                            return True
                        if "image_url" in part or "image" in part or "video" in part:
                            return True
            elif isinstance(message_content, dict):
                ptype = message_content.get("type")
                if ptype in {"image_url", "image", "video"}:
                    return True
                if "image_url" in message_content or "image" in message_content or "video" in message_content:
                    return True

        # 2) 兜底：仅检测明确的媒体标记，避免 system prompt 误判
        image_keywords = [
            "imageurl", "image_url",
            "data:image", "data:video",
        ]

        image_patterns = [
            r"imageurl[:：]\s*(https?://[^\s]+)",
            r"image_url[:：]\s*(https?://[^\s]+)",
            r"!\[.*?\]\(.*?\)",  # markdown图片格式
            r"https?://[^\s]*\.(png|jpg|jpeg|gif|webp|bmp)",
            r"https?://[^\s]*\.(mp4|mov|avi|mkv|flv|wmv|webm|m3u8)",
            r"data:image/[^;]+;base64,",
            r"data:video/[^;]+;base64,",
        ]

        chunks: list[str] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = (message.get("role") or "").lower().strip()
            if role and role != "user":
                continue
            message_content = message.get("content")
            if not message_content:
                continue

            if isinstance(message_content, str):
                chunks.append(message_content)
            elif isinstance(message_content, list):
                for part in message_content:
                    if isinstance(part, dict):
                        t = part.get("text")
                        if isinstance(t, str) and t:
                            chunks.append(t)
                        image_url = part.get("image_url")
                        if isinstance(image_url, dict):
                            url = image_url.get("url")
                            if isinstance(url, str) and url:
                                chunks.append(url)
                    elif isinstance(part, str) and part:
                        chunks.append(part)
            elif isinstance(message_content, dict):
                t = message_content.get("text")
                if isinstance(t, str) and t:
                    chunks.append(t)

        content = " ".join(chunks)
        content_lower = content.lower()

        for keyword in image_keywords:
            if keyword.lower() in content_lower:
                logger.debug(f"🔍 检测到图片关键词: {keyword}")
                return True

        import re
        for pattern in image_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                logger.debug(f"🔍 检测到图片模式: {pattern}")
                return True

        return False

    def _init_qwen_client(self) -> OpenAI:
        """初始化通义千问客户端 - DashScope不支持真正的异步，使用同步客户端"""
        # 使用新的统一配置
        api_key = (
            os.getenv("LLM_API_KEY")
            or os.getenv("DASHSCOPE_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or getattr(settings, "LLM_API_KEY", None)
            or getattr(settings, "DASHSCOPE_API_KEY", None)
            or getattr(settings, "OPENAI_API_KEY", None)
            or self.config.get("api_key")
        )
        if not api_key:
            raise ValueError("LLM_API_KEY not found")

        # DashScope的OpenAI兼容模式不支持真正的异步，使用同步客户端
        return OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            http_client=httpx.Client(verify=False)
        )
    
    def _init_openai_client(self) -> AsyncOpenAI:
        """初始化OpenAI客户端"""
        api_key = os.getenv("OPENAI_API_KEY", self.config.get("api_key"))
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found")
        
        return AsyncOpenAI(api_key=api_key)
    
    def _init_deepseek_client(self) -> AsyncOpenAI:
        """初始化DeepSeek客户端"""
        api_key = os.getenv("DEEPSEEK_API_KEY", self.config.get("api_key"))
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY not found")

        return AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )

    def _init_glm_client(self) -> AsyncOpenAI:
        """初始化智谱GLM客户端"""
        api_key = os.getenv("OPENAI_API_KEY", self.config.get("api_key"))
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found for GLM")

        return AsyncOpenAI(
            api_key=api_key,
            base_url="https://open.bigmodel.cn/api/paas/v4/"
        )
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        response_format: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """
        发送聊天请求

        Args:
            messages: 消息列表
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大token数
            response_format: 响应格式（支持JSON Schema）
            **kwargs: 额外参数

        Returns:
            模型响应文本
        """
        self.last_usage = None
        # 智能模型选择：根据消息内容自动选择合适的模型
        if not model:
            if self.provider == "qwen":
                # 检查消息中是否包含图片信息，智能选择模型
                if self._has_image_content(messages):
                    model = "qwen-vl-max"  # 多模态模型，支持图片分析
                    logger.info("🖼️ 检测到图片内容，使用视觉模型 qwen-vl-max")
                else:
                    # 使用环境变量配置的默认模型，而不是硬编码
                    import os
                    model = os.getenv("LLM_MODEL", "qwen-turbo")
                    logger.info(f"📝 纯文本内容，使用配置的文本模型 {model}")
            elif self.provider == "openai":
                model = "gpt-3.5-turbo"
            elif self.provider == "deepseek":
                model = "deepseek-chat"
        
        try:
            # 构建API调用参数
            api_params = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                **kwargs
            }

            # 如果提供了response_format，添加到参数中
            if response_format:
                api_params["response_format"] = response_format
                logger.info(f"🎯 [LLM] 使用response_format: {response_format.get('type')}")

            client = self._get_chat_client(messages=messages, model=model)

            original_model = model
            model_to_use = model

            # 如果之前已确认该 model 在本 VLM 服务不存在，则直接使用 fallback。
            if original_model and self._is_vlm_client(client):
                cached_fallback = self._VLM_MODEL_FALLBACK_CACHE.get(original_model)
                if cached_fallback:
                    model_to_use = cached_fallback
                    api_params["model"] = cached_fallback
            # 如果之前已确认该 model 在本 LLM 服务不存在，则直接使用 fallback。
            if original_model and self._is_llm_client(client):
                cached_fallback = self._LLM_MODEL_FALLBACK_CACHE.get(original_model)
                if cached_fallback:
                    model_to_use = cached_fallback
                    api_params["model"] = cached_fallback

            attempts = 0
            response_format_removed = False
            used_fallback = False
            use_responses_api = False
            responses_text: Optional[Dict[str, Any]] = None
            responses_input: Optional[List[Dict[str, Any]]] = None
            responses_instructions: Optional[str] = None
            responses_force_stream = False
            responses_output_text: Optional[str] = None
            responses_use_instructions = False
            responses_disabled_params: set[str] = set()
            while True:
                attempts += 1
                try:
                    if use_responses_api:
                        if responses_input is None:
                            responses_input, responses_instructions = self._coerce_messages_to_responses_input_and_instructions(
                                messages,
                                use_instructions=responses_use_instructions,
                            )  # type: ignore[arg-type]
                        if responses_text is None:
                            responses_text = self._coerce_response_format_to_responses_text(api_params.get("response_format"))  # type: ignore[arg-type]

                        resp_params: Dict[str, Any] = {
                            "model": api_params.get("model"),
                            "input": responses_input,
                        }
                        if "max_output_tokens" not in responses_disabled_params:
                            resp_params["max_output_tokens"] = api_params.get("max_tokens")
                        if responses_instructions and "instructions" not in responses_disabled_params:
                            resp_params["instructions"] = responses_instructions
                        if responses_text and "text" not in responses_disabled_params:
                            resp_params["text"] = responses_text
                        if responses_force_stream and "stream" not in responses_disabled_params:
                            resp_params["stream"] = True

                        if responses_force_stream:
                            def _collect_stream_text(sync_client: Any, params: Dict[str, Any]) -> Tuple[str, Optional[Any]]:
                                stream_obj = sync_client.responses.create(**params)
                                chunks: List[str] = []
                                final_resp: Optional[Any] = None
                                try:
                                    for ev in stream_obj:
                                        try:
                                            ev_type = getattr(ev, "type", None) or (ev.get("type") if isinstance(ev, dict) else None)
                                        except Exception:
                                            ev_type = None
                                        if ev_type == "response.output_text.delta":
                                            delta = getattr(ev, "delta", None) if not isinstance(ev, dict) else ev.get("delta")
                                            if isinstance(delta, str) and delta:
                                                chunks.append(delta)
                                        elif ev_type == "response.completed":
                                            r = getattr(ev, "response", None) if not isinstance(ev, dict) else ev.get("response")
                                            if r is not None:
                                                final_resp = r
                                finally:
                                    try:
                                        stream_obj.close()
                                    except Exception:
                                        pass
                                return "".join(chunks), final_resp

                            if isinstance(client, OpenAI):
                                responses_output_text, response = await asyncio.to_thread(_collect_stream_text, client, resp_params)
                            else:
                                # AsyncOpenAI stream path isn't used in unified mode; keep a simple fallback.
                                stream_obj = await client.responses.create(**resp_params)
                                chunks: List[str] = []
                                final_resp: Optional[Any] = None
                                try:
                                    async for ev in stream_obj:
                                        ev_type = getattr(ev, "type", None)
                                        if ev_type == "response.output_text.delta":
                                            delta = getattr(ev, "delta", None)
                                            if isinstance(delta, str) and delta:
                                                chunks.append(delta)
                                        elif ev_type == "response.completed":
                                            r = getattr(ev, "response", None)
                                            if r is not None:
                                                final_resp = r
                                finally:
                                    try:
                                        await stream_obj.close()
                                    except Exception:
                                        pass
                                responses_output_text = "".join(chunks)
                                response = final_resp
                        else:
                            if isinstance(client, OpenAI):
                                response = await asyncio.to_thread(client.responses.create, **resp_params)
                            else:
                                response = await client.responses.create(**resp_params)
                            responses_output_text = getattr(response, "output_text", None)
                    else:
                        if isinstance(client, OpenAI):
                            response = await asyncio.to_thread(
                                client.chat.completions.create,
                                **api_params
                            )
                        else:
                            response = await client.chat.completions.create(**api_params)
                    break
                except Exception as e:
                    if (not use_responses_api) and self._is_chat_completions_endpoint_missing(e):
                        logger.warning("⚠️ [LLM] /chat/completions 不可用，切换到 Responses API 重试一次...")
                        use_responses_api = True
                        continue
                    if use_responses_api and (not responses_force_stream) and self._is_responses_stream_required(e):
                        logger.warning("⚠️ [LLM] Responses API 要求 stream=true，已启用后重试一次...")
                        responses_force_stream = True
                        continue
                    if use_responses_api and (not responses_use_instructions) and self._is_responses_system_messages_not_allowed(e):
                        logger.warning("⚠️ [LLM] Responses API 不允许 system messages，改用 instructions 重试一次...")
                        responses_use_instructions = True
                        responses_input = None
                        responses_instructions = None
                        continue
                    if use_responses_api:
                        unsupported = self._extract_unsupported_parameter_name(e)
                        if unsupported:
                            top = unsupported.split(".")[0].strip()
                            if top and top not in {"model", "input"} and top not in responses_disabled_params:
                                logger.warning("⚠️ [LLM] Responses API 不支持参数 %s，移除后重试一次...", top)
                                responses_disabled_params.add(top)
                                continue

                    # 只有当“走的是 VLM client 且模型不存在”时，才回退到 env/config 的 VLM_MODEL。
                    if (
                        original_model
                        and self._is_vlm_client(client)
                        and model_to_use == original_model
                        and self._is_model_not_found_error(e)
                    ):
                        fallback_model = self._get_vlm_fallback_model()
                        if fallback_model and fallback_model != original_model:
                            logger.info(f"🔁 [LLM] VLM模型不存在，回退到 VLM_MODEL: {original_model} -> {fallback_model}")
                            self._VLM_MODEL_FALLBACK_CACHE[original_model] = fallback_model
                            model_to_use = fallback_model
                            api_params["model"] = fallback_model
                            continue
                        raise
                    if (
                        original_model
                        and self._is_llm_client(client)
                        and model_to_use == original_model
                        and self._is_model_not_found_error(e)
                    ):
                        fallback_model = self._get_llm_fallback_model()
                        if fallback_model and fallback_model != original_model:
                            logger.info(f"🔁 [LLM] 模型不存在，回退到 LLM_MODEL: {original_model} -> {fallback_model}")
                            self._LLM_MODEL_FALLBACK_CACHE[original_model] = fallback_model
                            model_to_use = fallback_model
                            api_params["model"] = fallback_model
                            continue
                        raise

                    if (not used_fallback) and self._is_connection_error(e):
                        fallback_client, fallback_model = self._get_fallback_chat_client(client)
                        if fallback_client:
                            used_fallback = True
                            client = fallback_client
                            if fallback_model:
                                model_to_use = fallback_model
                                api_params["model"] = fallback_model
                            logger.warning("⚠️ [LLM] 主服务不可达，切换到备用服务重试...")
                            continue

                    if (
                        "response_format" in api_params
                        and (not response_format_removed)
                        and self._is_response_format_unsupported_error(e)
                    ):
                        logger.warning("⚠️ [LLM] response_format 该后端不支持，移除后重试一次...")
                        api_params.pop("response_format", None)
                        response_format_removed = True
                        responses_text = None
                        continue

                    # httpx/uvloop 偶发 closed transport：重试一次降低噪声
                    if attempts == 1 and self._is_transport_closed_runtime_error(e):
                        logger.warning("⚠️ [LLM] transport closed during request; retrying once...")
                        await asyncio.sleep(0.2)
                        continue

                    raise

            self.last_usage = self._extract_usage(response, model=model_to_use)
            if use_responses_api:
                if isinstance(responses_output_text, str) and responses_output_text.strip():
                    return responses_output_text
                text = getattr(response, "output_text", None)
                if isinstance(text, str) and text.strip():
                    return text
                return str(text or responses_output_text or "")
            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"LLM chat error: {e}")
            raise
    
    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        model: str = None,
        temperature: float = 0.7,
        **kwargs
    ):
        """
        流式聊天
        
        Args:
            messages: 消息列表
            model: 模型名称
            temperature: 温度参数
            **kwargs: 额外参数
            
        Yields:
            响应片段
        """
        # 智能模型选择：根据消息内容自动选择合适的模型
        if not model:
            if self.provider == "qwen":
                # 检查消息中是否包含图片信息，智能选择模型
                if self._has_image_content(messages):
                    model = "qwen-vl-max"  # 多模态模型，支持图片分析
                    logger.info("🖼️ 检测到图片内容，使用视觉模型 qwen-vl-max")
                else:
                    # 使用环境变量配置的默认模型，而不是硬编码
                    import os
                    model = os.getenv("LLM_MODEL", "qwen-turbo")
                    logger.info(f"📝 纯文本内容，使用配置的文本模型 {model}")
            elif self.provider == "openai":
                model = "gpt-3.5-turbo"
            elif self.provider == "deepseek":
                model = "deepseek-chat"
        
        try:
            # create() 可能返回协程或Stream对象
            import inspect
            from collections.abc import AsyncIterator

            client = self._get_chat_client(messages=messages, model=model)

            original_model = model
            model_to_use = model

            if original_model and self._is_vlm_client(client):
                cached_fallback = self._VLM_MODEL_FALLBACK_CACHE.get(original_model)
                if cached_fallback:
                    model_to_use = cached_fallback

            retries_left = 1
            used_fallback = False
            yielded_any = False
            while True:
                try:
                    create_result = client.chat.completions.create(
                        model=model_to_use,
                        messages=messages,
                        temperature=temperature,
                        stream=True,
                        **kwargs
                    )
                    break
                except Exception as e:
                    if (
                        original_model
                        and self._is_vlm_client(client)
                        and model_to_use == original_model
                        and self._is_model_not_found_error(e)
                    ):
                        fallback_model = self._get_vlm_fallback_model()
                        if fallback_model and fallback_model != original_model:
                            logger.info(f"🔁 [LLM] VLM模型不存在，回退到 VLM_MODEL: {original_model} -> {fallback_model}")
                            self._VLM_MODEL_FALLBACK_CACHE[original_model] = fallback_model
                            model_to_use = fallback_model
                            continue
                    if (not used_fallback) and self._is_connection_error(e):
                        fallback_client, fallback_model = self._get_fallback_chat_client(client)
                        if fallback_client:
                            used_fallback = True
                            client = fallback_client
                            if fallback_model:
                                model_to_use = fallback_model
                            logger.warning("⚠️ [LLM] 主服务不可达，切换到备用服务重试流式请求...")
                            continue
                    if retries_left > 0 and self._is_transport_closed_runtime_error(e):
                        retries_left -= 1
                        logger.warning("⚠️ [LLM] transport closed during stream init; retrying once...")
                        await asyncio.sleep(0.2)
                        continue
                    raise

            # 检查返回类型
            stream_type = type(create_result).__name__
            logger.debug(f"Create result type: {stream_type}")

            # 1. 如果是coroutine，先await获取stream
            if inspect.iscoroutine(create_result):
                logger.debug("Result is coroutine, awaiting...")
                stream = await create_result
            else:
                stream = create_result

            # 2. 检查stream类型并相应迭代
            stream_type = type(stream).__name__
            logger.debug(f"Stream type: {stream_type}")

            # 判断是否是异步迭代器
            if hasattr(stream, '__aiter__'):
                # 异步流：使用 async for
                logger.debug("Using async iteration")
                try:
                    async for chunk in stream:
                        if chunk.choices and chunk.choices[0].delta.content:
                            yielded_any = True
                            yield chunk.choices[0].delta.content
                except RuntimeError as e:
                    if self._is_transport_closed_runtime_error(e) and yielded_any:
                        logger.warning("⚠️ [LLM] stream ended due to transport close (ignored): %s", e)
                        return
                    raise
            elif hasattr(stream, '__iter__'):
                # 同步流：使用 for（在async generator中需要特殊处理）
                logger.debug("Using sync iteration in async context")
                import asyncio
                try:
                    for chunk in stream:
                        if chunk.choices and chunk.choices[0].delta.content:
                            yielded_any = True
                            yield chunk.choices[0].delta.content
                            # 让出控制权给事件循环
                            await asyncio.sleep(0)
                except RuntimeError as e:
                    if self._is_transport_closed_runtime_error(e) and yielded_any:
                        logger.warning("⚠️ [LLM] stream ended due to transport close (ignored): %s", e)
                        return
                    raise
            else:
                raise TypeError(f"Unexpected stream type: {stream_type}")

        except Exception as e:
            logger.error(f"LLM stream error: {e}", exc_info=True)
            raise
    
    async def embedding(
        self,
        text: str,
        model: str = None
    ) -> List[float]:
        """
        获取文本嵌入向量
        
        Args:
            text: 输入文本
            model: 嵌入模型
            
        Returns:
            嵌入向量
        """
        if not model:
            if self.provider == "qwen":
                model = "text-embedding-v2"
            elif self.provider == "openai":
                model = "text-embedding-ada-002"
            else:
                raise ValueError(f"Provider {self.provider} does not support embeddings")
        
        try:
            client = self._get_embedding_client()

            if isinstance(client, OpenAI):
                try:
                    response = await asyncio.to_thread(
                        client.embeddings.create,
                        model=model,
                        input=text
                    )
                except Exception as e:
                    if self._is_connection_error(e) and self.embedding_fallback_client:
                        fallback_model = self.embedding_fallback_model or model
                        logger.warning("⚠️ [LLM] Embedding主服务不可达，切换到备用服务重试...")
                        response = await asyncio.to_thread(
                            self.embedding_fallback_client.embeddings.create,
                            model=fallback_model,
                            input=text
                        )
                    else:
                        raise
            else:
                try:
                    response = await client.embeddings.create(
                        model=model,
                        input=text
                    )
                except Exception as e:
                    if self._is_connection_error(e) and self.embedding_fallback_client:
                        fallback_model = self.embedding_fallback_model or model
                        logger.warning("⚠️ [LLM] Embedding主服务不可达，切换到备用服务重试...")
                        response = await self.embedding_fallback_client.embeddings.create(
                            model=fallback_model,
                            input=text
                        )
                    else:
                        raise
            
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            raise
