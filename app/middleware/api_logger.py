"""
API日志记录中间件
自动记录所有HTTP REST API的调用信息
"""
import time
import json
import logging
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.background import BackgroundTask
from starlette.responses import StreamingResponse

logger = logging.getLogger(__name__)


class APILoggerMiddleware(BaseHTTPMiddleware):
    """
    API调用日志中间件（简化版）

    功能:
    1. 自动拦截所有HTTP请求
    2. 记录请求信息(方法、路径、查询参数、IP、User-Agent)
    3. 记录响应信息(状态码、耗时)
    4. 异步写入数据库(不阻塞响应)

    注意：
    - 不记录请求体和响应体，避免性能问题
    - 只记录关键的审计信息（谁、何时、调用了什么、结果如何）
    """

    # ═══════════════════════════════════════════════════════════════
    # 配置：排除路径（不记录日志）
    # ═══════════════════════════════════════════════════════════════
    EXCLUDE_PATHS = [
        "/health",                          # 健康检查
        "/docs",                            # API文档
        "/redoc",                           # API文档
        "/openapi.json",                    # OpenAPI规范
        "/static",                          # 静态文件
        "/favicon.ico",                     # 图标
        "/session-files",                   # 会话文件
    ]


    def __init__(self, app):
        super().__init__(app)
        logger.info("✅ APILoggerMiddleware initialized")

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        拦截请求，记录日志（简化版：不读取body）

        Args:
            request: FastAPI请求对象
            call_next: 下一个中间件/路由处理器

        Returns:
            响应对象
        """
        # ───────────────────────────────────────────────────────────
        # 步骤1: 检查是否需要记录
        # ───────────────────────────────────────────────────────────
        if not self._should_log(request):
            return await call_next(request)

        # ───────────────────────────────────────────────────────────
        # 步骤2: 记录开始时间
        # ───────────────────────────────────────────────────────────
        start_time = time.time()

        # ───────────────────────────────────────────────────────────
        # 步骤3: 提取请求基本信息（不读取body）
        # ───────────────────────────────────────────────────────────
        log_data = self._extract_request_info_simple(request)

        # ───────────────────────────────────────────────────────────
        # 步骤4: 执行请求处理
        # ───────────────────────────────────────────────────────────
        try:
            response = await call_next(request)

            # 记录响应信息
            log_data["status_code"] = response.status_code
            log_data["duration_ms"] = round((time.time() - start_time) * 1000, 2)

            # 使用后台任务保存日志，不阻塞响应
            response.background = BackgroundTask(self._save_log, log_data)
            return response

        except Exception as e:
            # 记录错误信息
            log_data["status_code"] = 500
            log_data["error_message"] = str(e)
            log_data["duration_ms"] = round((time.time() - start_time) * 1000, 2)

            # 异步写入日志
            await self._save_log(log_data)

            # 重新抛出异常
            raise

    def _should_log(self, request: Request) -> bool:
        """
        判断是否需要记录日志

        Args:
            request: 请求对象

        Returns:
            是否记录
        """
        # 检查配置开关
        from app.config import settings
        if not getattr(settings, 'ENABLE_API_LOGGING', True):
            return False

        # 检查排除路径
        path = request.url.path
        for exclude in self.EXCLUDE_PATHS:
            if path.startswith(exclude):
                return False

        return True

    def _extract_request_info_simple(self, request: Request) -> dict:
        """
        提取请求基本信息（不读取body，避免性能问题）

        Args:
            request: 请求对象

        Returns:
            日志数据字典
        """
        # ═══════════════════════════════════════════════════════════
        # 基础信息
        # ═══════════════════════════════════════════════════════════
        log_data = {
            "method": request.method,
            "path": request.url.path,
            "query_params": str(request.url.query)[:500] if request.url.query else None,
            "ip_address": request.client.host if request.client else None,
            "user_agent": request.headers.get("User-Agent", "")[:500],
        }

        # ═══════════════════════════════════════════════════════════
        # 不读取请求体，标记为null
        # ═══════════════════════════════════════════════════════════
        log_data["request_body"] = None
        log_data["response_body"] = None

        # ═══════════════════════════════════════════════════════════
        # 提取业务关联字段
        # ═══════════════════════════════════════════════════════════
        # 从路径参数提取
        path_params = request.path_params
        log_data["agent_id"] = path_params.get("agent_id")
        log_data["session_id"] = path_params.get("session_id")

        # 从查询参数提取
        query_params = request.query_params
        if not log_data.get("agent_id"):
            log_data["agent_id"] = query_params.get("agent_id")
        if not log_data.get("session_id"):
            log_data["session_id"] = query_params.get("session_id")

        # 从请求头提取
        log_data["session_id"] = log_data.get("session_id") or request.headers.get("X-Session-ID")
        log_data["user_id"] = request.headers.get("X-User-ID")

        # ═══════════════════════════════════════════════════════════
        # 请求头（只保留关键头）
        # ═══════════════════════════════════════════════════════════
        important_headers = {
            "content-type": request.headers.get("content-type"),
            "user-agent": request.headers.get("user-agent"),
            "x-session-id": request.headers.get("x-session-id"),
            "x-user-id": request.headers.get("x-user-id"),
        }
        # 移除空值
        important_headers = {k: v for k, v in important_headers.items() if v}
        log_data["request_headers"] = json.dumps(important_headers, ensure_ascii=False) if important_headers else None

        return log_data

    async def _save_log(self, log_data: dict):
        """
        异步保存日志到数据库

        Args:
            log_data: 日志数据
        """
        try:
            # 动态导入服务（避免循环导入）
            from app.services.api_log_service import APILogService

            service = APILogService()
            await service.create_log(log_data)

            # logger.debug(f"✅ API log saved: {log_data['method']} {log_data['path']} - {log_data['status_code']}")

        except Exception as e:
            # 日志写入失败不影响主流程，只记录错误
            logger.error(f"❌ Failed to save API log: {e}")
