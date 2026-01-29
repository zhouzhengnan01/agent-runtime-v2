"""
JetLinks Agent 主入口
"""
import uvicorn
import asyncio
import base64
import hmac
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from contextlib import asynccontextmanager
import logging
import sys
import os
from dotenv import load_dotenv

# 添加项目路径并确保优先级 + 显式加载 .env（避免 python-dotenv 在某些启动形态下 find_dotenv 失败）
project_root = os.path.dirname(os.path.abspath(__file__))
try:
    # 强制覆盖环境变量，避免外部环境残留导致配置不生效
    load_dotenv(dotenv_path=os.path.join(project_root, ".env"), override=True)
except Exception:
    # 兜底：.env 加载失败则继续（由系统环境变量提供配置）
    pass

# 添加项目路径并确保优先级
if project_root not in sys.path:
    sys.path.insert(0, project_root)
else:
    # 如果已经在path中，移动到最前面
    sys.path.remove(project_root)
    sys.path.insert(0, project_root)

# 清理可能的冲突路径

from app.config import settings
from app.api.v1.router import api_router
from app.db.session import engine, Base

# 导入所有模型以确保创建表
from app.models import (
    Agent, Tool
)
# APILog暂时注释,等服务器代码同步后再启用
# from app.models import APILog

# 导入并注册所有工具（必须在应用启动前完成） 
# TODO 改成数据看配置
# from app.core.tools.agent import (
#     VideoAnalyzer
# )

# ═══════════════════════════════════════════════════════════════
# 日志配置：控制台输出模式
# ═══════════════════════════════════════════════════════════════

# 1️⃣ 创建根日志配置（控制台输出）
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)  # 输出到控制台
    ],
    force=True  # 强制重新配置日志
)


# 2️⃣ 创建控制台 handler（DEBUG 模式使用彩色输出）
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(getattr(logging, settings.LOG_LEVEL, logging.INFO))

if settings.DEBUG:
    class ColoredFormatter(logging.Formatter):
        """彩色日志格式化器"""
        COLORS = {
            'DEBUG': '\033[34m',    # 蓝色
            'INFO': '\033[32m',     # 绿色
            'WARNING': '\033[33m',  # 黄色
            'ERROR': '\033[31m',    # 红色
            'CRITICAL': '\033[35m'  # 紫色
        }
        RESET = '\033[0m'
        CYAN = '\033[36m'

        def format(self, record):
            # 给日志级别添加颜色
            levelname = record.levelname
            if levelname in self.COLORS:
                record.levelname = f"{self.COLORS[levelname]}{levelname}{self.RESET}"
            # 给模块名添加颜色
            record.name = f"{self.CYAN}{record.name}{self.RESET}"
            return super().format(record)

    console_handler.setFormatter(
        ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
    )
else:
    console_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )

# 3️⃣ 设置特定模块的日志级别（DEBUG以查看详细WebSocket通信）
handler_logger = logging.getLogger('app.core.jaip.handler')
handler_logger.setLevel(logging.DEBUG)
handler_logger.propagate = False  # ✅ 禁止传播到根日志器，避免重复输出
handler_logger.handlers.clear()  # ✅ 清空现有handlers，避免重复添加
if settings.DEBUG:
    handler_logger.addHandler(console_handler)  # 控制台输出

websocket_logger = logging.getLogger('app.api.v1.websocket')
websocket_logger.setLevel(logging.INFO)  # 🚀 固定为INFO级别，屏蔽DEBUG通信日志
websocket_logger.propagate = False  # ✅ 禁止传播到根日志器，避免重复输出
websocket_logger.handlers.clear()  # ✅ 清空现有handlers，避免重复添加
websocket_logger.addHandler(console_handler)  # 始终使用INFO级别

tool_manager_logger = logging.getLogger('app.core.tools.session_tool_manager')
tool_manager_logger.setLevel(logging.INFO)  # 🚀 改为INFO级别，减少冗余日志
tool_manager_logger.propagate = False  # ✅ 禁止传播到根日志器，避免重复输出
tool_manager_logger.handlers.clear()  # ✅ 清空现有handlers，避免重复添加
tool_manager_logger.addHandler(console_handler)  # 始终使用INFO级别

# 🆕 增加详细的工具执行日志
tool_router_logger = logging.getLogger('app.core.tools.tool_router')
tool_router_logger.setLevel(logging.DEBUG)
tool_router_logger.propagate = False
tool_router_logger.handlers.clear()
if settings.DEBUG:
    tool_router_logger.addHandler(console_handler)

# 🆕 增加详细的Agent执行日志
cognitive_agent_logger = logging.getLogger('app.core.agents.cognitive_agent')
cognitive_agent_logger.setLevel(logging.DEBUG)
cognitive_agent_logger.propagate = False
cognitive_agent_logger.handlers.clear()
if settings.DEBUG:
    cognitive_agent_logger.addHandler(console_handler)

# 🆕 增加详细的计划器日志
planner_logger = logging.getLogger('app.core.agents.planning.intelligent_planner')
planner_logger.setLevel(logging.DEBUG)
planner_logger.propagate = False
planner_logger.handlers.clear()
if settings.DEBUG:
    planner_logger.addHandler(console_handler)

# 🆕 增加参数填充器日志
parameter_filler_logger = logging.getLogger('app.core.tools.parameter_filler')
parameter_filler_logger.setLevel(logging.DEBUG)
parameter_filler_logger.propagate = False
parameter_filler_logger.handlers.clear()
if settings.DEBUG:
    parameter_filler_logger.addHandler(console_handler)

# 🆕 增加模板Agent日志
template_agent_logger = logging.getLogger('app.core.agents.template_agent')
template_agent_logger.setLevel(logging.DEBUG)
template_agent_logger.propagate = False
template_agent_logger.handlers.clear()
if settings.DEBUG:
    template_agent_logger.addHandler(console_handler)

# 🆕 增加 shared/jetlinks_video 模块日志
streaming_analyze_logger = logging.getLogger('app.shared.streaming_analyze')
streaming_analyze_logger.setLevel(logging.DEBUG)
streaming_analyze_logger.propagate = False
streaming_analyze_logger.handlers.clear()
if settings.DEBUG:
    streaming_analyze_logger.addHandler(console_handler)

# 增加 jetlinks_video workers 日志
worker_a_logger = logging.getLogger('app.shared.jetlinks_video.workers.worker_a_cut')
worker_a_logger.setLevel(logging.DEBUG)
worker_a_logger.propagate = False
worker_a_logger.handlers.clear()
if settings.DEBUG:
    worker_a_logger.addHandler(console_handler)

worker_b_logger = logging.getLogger('app.shared.jetlinks_video.workers.worker_b_vlm')
worker_b_logger.setLevel(logging.DEBUG)
worker_b_logger.propagate = False
worker_b_logger.handlers.clear()
if settings.DEBUG:
    worker_b_logger.addHandler(console_handler)

worker_c_logger = logging.getLogger('app.shared.jetlinks_video.workers.worker_c_asr')
worker_c_logger.setLevel(logging.DEBUG)
worker_c_logger.propagate = False
worker_c_logger.handlers.clear()
if settings.DEBUG:
    worker_c_logger.addHandler(console_handler)

# 禁用第三方库的DEBUG日志，减少干扰
logging.getLogger('dashscope').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
# 调试期间建议打开访问日志，便于排查“任务没触发/没响应”等问题
if settings.DEBUG:
    logging.getLogger('uvicorn.protocols.websockets').setLevel(logging.INFO)
    logging.getLogger('uvicorn.access').setLevel(logging.INFO)
else:
    logging.getLogger('uvicorn.protocols.websockets').setLevel(logging.WARNING)  # 禁用 WebSocket 协议层日志
    logging.getLogger('uvicorn.access').setLevel(logging.WARNING)  # 禁用访问日志
logging.getLogger('watchfiles').setLevel(logging.WARNING)  # 禁用watchfiles的DEBUG日志

# 🆕 屏蔽更多冗余日志
logging.getLogger('asyncio').setLevel(logging.WARNING)
logging.getLogger('urllib3.connectionpool').setLevel(logging.WARNING)
logging.getLogger('websockets').setLevel(logging.WARNING)
logging.getLogger('websockets.server').setLevel(logging.WARNING)
logging.getLogger('langchain').setLevel(logging.WARNING)
logging.getLogger('langchain_core').setLevel(logging.WARNING)
logging.getLogger('openai').setLevel(logging.WARNING)
logging.getLogger('openai._base_client').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)
logging.getLogger('sqlalchemy.pool').setLevel(logging.WARNING)
logging.getLogger('openai').setLevel(logging.WARNING)
logging.getLogger('openai._base_client').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)
logging.getLogger('sqlalchemy.pool').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Basic Auth middleware for protecting the whole site/API.
class BasicAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, *, user: str, password: str, realm: str = "JetLinks Agent", exclude_paths=None) -> None:
        super().__init__(app)
        self.user = user
        self.password = password
        self.realm = realm or "JetLinks Agent"
        self.exclude_paths = exclude_paths or []

    async def dispatch(self, request, call_next):
        if request.method == "OPTIONS":
            return await call_next(request)
        path = request.url.path
        for prefix in self.exclude_paths:
            if path.startswith(prefix):
                return await call_next(request)

        auth = request.headers.get("authorization")
        if auth and auth.lower().startswith("basic "):
            token = auth.split(" ", 1)[1].strip()
            try:
                decoded = base64.b64decode(token).decode("utf-8")
            except Exception:
                decoded = ""
            if ":" in decoded:
                user, pwd = decoded.split(":", 1)
                if hmac.compare_digest(user, self.user) and hmac.compare_digest(pwd, self.password):
                    return await call_next(request)

        return Response(
            status_code=401,
            content="Unauthorized",
            headers={"WWW-Authenticate": f'Basic realm="{self.realm}"'},
        )

# 将关键LLM密钥注入环境变量，便于第三方SDK（如 qwen-agent/dashscope）读取
if settings.DASHSCOPE_API_KEY and not os.getenv("DASHSCOPE_API_KEY"):
    os.environ["DASHSCOPE_API_KEY"] = settings.DASHSCOPE_API_KEY
if settings.OPENAI_API_KEY and not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时
    logger.info(f"Starting {settings.APP_NAME} v{settings.VERSION}")
    logger.info(f"Debug mode: {settings.DEBUG}")
    logger.info(f"API Key configured: {'Yes' if settings.DASHSCOPE_API_KEY else 'No'}")

    # 初始化统一模型配置管理器
    try:
        from app.core.llm import get_model_config_manager
        model_config_manager = get_model_config_manager()
        model_config_manager.initialize()
        logger.info("✅ 统一模型配置管理器初始化成功")
    except Exception as e:
        logger.error(f"❌ 统一模型配置管理器初始化失败: {e}")
        # 不退出程序，保持兼容性

    # 创建数据库表
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created/checked successfully")
    except Exception as e:
        logger.error(f"Failed to create database tables: {e}")

    # 预置默认工具/模板/智能体（新库可直接使用；重复调用安全）
    try:
        from app.db.seed_defaults import (
            seed_default_tools,
            seed_default_templates_and_agents,
        )

        seed_default_tools()
        seed_default_templates_and_agents()
    except Exception as e:
        logger.warning(f"⚠️ Default seed failed: {e}")

    # 初始化统一记忆管理器 (UnifiedMemoryManager)
    try:
        from app.core.memory import unified_memory
        await unified_memory.initialize()
        logger.info("✅ UnifiedMemoryManager initialized successfully")
    except Exception as e:
        logger.warning(f"⚠️ UnifiedMemoryManager initialization failed: {e}")
        logger.warning("⚠️ Long-term memory and vector search features will be unavailable")

    # 加载内部工具注册表（用于区分内部/外部工具）
    try:
        from app.core.tools import internal_tool_registry
        from app.db.session import SessionLocal

        db = SessionLocal()
        try:
            await internal_tool_registry.load_from_database(db)
        finally:
            db.close()
    except Exception as e:
        logger.warning(f"⚠️ Internal tool registry initialization failed: {e}")
        logger.warning("⚠️ Will treat all tools as external tools")

    report_scheduler_stop = asyncio.Event()
    report_scheduler_task = None
    if getattr(settings, "REVIEW_REPORTS_SCHEDULE_ENABLED", False):
        try:
            from app.services.review_report_scheduler import review_report_scheduler

            report_scheduler_task = asyncio.create_task(review_report_scheduler(report_scheduler_stop))
            logger.info("Review report scheduler started")
        except Exception as e:
            logger.warning("Review report scheduler init failed: %s", e)

    yield

    # 关闭时
    logger.info("Shutting down JetLinks Agent...")

    if report_scheduler_task:
        report_scheduler_stop.set()
        report_scheduler_task.cancel()
        try:
            await report_scheduler_task
        except Exception:
            pass

    # 关闭统一记忆管理器连接
    try:
        from app.core.memory import unified_memory
        await unified_memory.close()
        logger.info("UnifiedMemoryManager connections closed")
    except Exception as e:
        logger.error(f"Failed to close UnifiedMemoryManager: {e}")


# 创建FastAPI应用
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="JetLinks Agent - 智能体服务模块",
    lifespan=lifespan
)

# Basic Auth protection (enabled when credentials are configured).
basic_user = (settings.BASIC_AUTH_USER or "").strip()
basic_password = (settings.BASIC_AUTH_PASSWORD or "").strip()
basic_enabled = bool(basic_user and basic_password) or bool(settings.BASIC_AUTH_ENABLED)
if basic_enabled:
    if not (basic_user and basic_password):
        logger.warning("BASIC_AUTH_ENABLED is true but BASIC_AUTH_USER/PASSWORD missing; auth disabled.")
    else:
        exclude_paths = [
            p.strip()
            for p in str(settings.BASIC_AUTH_EXCLUDE_PATHS or "").split(",")
            if p.strip()
        ]
        app.add_middleware(
            BasicAuthMiddleware,
            user=basic_user,
            password=basic_password,
            realm=settings.BASIC_AUTH_REALM,
            exclude_paths=exclude_paths,
        )

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该设置具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册API日志中间件（在CORS之后）- 暂时注释,等服务器代码同步后再启用
# if settings.ENABLE_API_LOGGING:
#     from app.middleware.api_logger import APILoggerMiddleware
#     app.add_middleware(APILoggerMiddleware)
#     logger.info(f"✅ API Logging enabled (retention: {settings.API_LOG_RETENTION_DAYS} days)")
# else:
#     logger.info("⚠️  API Logging disabled")

# 注册路由
app.include_router(api_router, prefix="/api/v1")


# 挂载静态文件目录
# 1. 挂载用户上传的文件目录
if os.path.exists("storage/uploads"):
    app.mount("/static/uploads", StaticFiles(directory="storage/uploads"), name="uploads")
    logger.info("Mounted uploads directory at /static/uploads")

# 2. 挂载前端静态资源目录
static_root = os.path.join(project_root, "storage", "static")
try:
    os.makedirs(static_root, exist_ok=True)
except Exception:
    static_root = None

if static_root and os.path.exists(static_root):
    app.mount("/static", StaticFiles(directory=static_root), name="static")
    logger.info("Mounted static files at /static")

# 3. 挂载会话存储文件目录
if os.path.exists(settings.SESSION_STORAGE_PATH):
    app.mount("/session-files", StaticFiles(directory=settings.SESSION_STORAGE_PATH), name="session-files")
    logger.info(f"Mounted session files at /session-files from {settings.SESSION_STORAGE_PATH}")

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
# 4. 挂载视频巡检证据帧
evidence_images_path = os.path.join(BASE_PATH, "storage", "evidence_images")
if os.path.exists(evidence_images_path):
    app.mount(
        "/storage/evidence_images",
        StaticFiles(directory=evidence_images_path),
        name="evidence_images",
    )
    logger.info("Mounted evidence_images files at /storage/evidence_images")

# 5. 挂载视频巡检带目标框证据帧
evidence_images_box_path = os.path.join(BASE_PATH, "storage", "evidence_images_box")
if os.path.exists(evidence_images_box_path):
    app.mount(
        "/storage/evidence_images_box",
        StaticFiles(directory=evidence_images_box_path),
        name="evidence_images_box",
    )
    logger.info("Mounted evidence_images_box files at /storage/evidence_images_box")

# 6. 挂载 http url 下载后的临时本地视频
http_tmp_path = os.path.join(BASE_PATH, "storage", "http_tmp")
if os.path.exists(http_tmp_path):
    app.mount(
        "/storage/http_tmp",
        StaticFiles(directory=http_tmp_path),
        name="http_tmp",
    )
    logger.info("Mounted http_tmp files at /storage/http_tmp")

# 7. 挂载复判调用记录（record.json + video.mp4）
review_records_path = os.path.join(BASE_PATH, "storage", "review_records")
try:
    os.makedirs(review_records_path, exist_ok=True)
except Exception:
    review_records_path = None

if review_records_path and os.path.exists(review_records_path):
    app.mount(
        "/storage/review_records",
        StaticFiles(directory=review_records_path),
        name="review_records",
    )
    logger.info("Mounted review_records files at /storage/review_records")

# 4. 挂载网站前端页面目录（放在所有API路由之后）
# 注意：这个挂载需要放在所有API路由定义之后，所以暂时注释掉，在文件末尾添加

# 健康检查接口
@app.get("/health")
async def health():
    """健康检查"""
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "version": settings.VERSION
    }

# 挂载 website 目录（必须在所有API路由之后，但在根路径路由之前）
website_root = os.path.join(project_root, "website")
if os.path.exists(website_root):
    app.mount("/", StaticFiles(directory=website_root, html=True), name="website")
    logger.info("Mounted website directory at /")


if __name__ == "__main__":
    def _get_int_env(name: str, default: int) -> int:
        try:
            value = os.getenv(name)
            if value is None:
                return default
            value = value.strip()
            if not value:
                return default
            return int(value)
        except Exception:
            return default

    def _get_str_env(name: str, default: str) -> str:
        value = os.getenv(name)
        if value is None:
            return default
        value = value.strip()
        return value or default

    def _get_bool_env(name: str, default: bool) -> bool:
        value = os.getenv(name)
        if value is None:
            return default
        s = value.strip().lower()
        if s in ("1", "true", "yes", "y", "on"):
            return True
        if s in ("0", "false", "no", "n", "off", ""):
            return False
        return default

    uvicorn_log_level = _get_str_env("UVICORN_LOG_LEVEL", "info").lower()
    uvicorn_reload = _get_bool_env("UVICORN_RELOAD", False)
    ws_max_size = _get_int_env("UVICORN_WS_MAX_SIZE", 64 * 1024 * 1024)
    ws_max_queue = _get_int_env("UVICORN_WS_MAX_QUEUE", 256)
    ws_ping_interval = _get_int_env("UVICORN_WS_PING_INTERVAL", 20)
    ws_ping_timeout = _get_int_env("UVICORN_WS_PING_TIMEOUT", 120)
    timeout_keep_alive = _get_int_env("UVICORN_TIMEOUT_KEEP_ALIVE", 300)

    def _get_int_env(name: str, default: int) -> int:
        try:
            value = os.getenv(name)
            if value is None:
                return default
            value = value.strip()
            if not value:
                return default
            return int(value)
        except Exception:
            return default

    def _get_str_env(name: str, default: str) -> str:
        value = os.getenv(name)
        if value is None:
            return default
        value = value.strip()
        return value or default

    def _get_bool_env(name: str, default: bool) -> bool:
        value = os.getenv(name)
        if value is None:
            return default
        s = value.strip().lower()
        if s in ("1", "true", "yes", "y", "on"):
            return True
        if s in ("0", "false", "no", "n", "off", ""):
            return False
        return default

    uvicorn_log_level = _get_str_env("UVICORN_LOG_LEVEL", "info").lower()
    uvicorn_reload = _get_bool_env("UVICORN_RELOAD", False)
    ws_max_size = _get_int_env("UVICORN_WS_MAX_SIZE", 64 * 1024 * 1024)
    ws_max_queue = _get_int_env("UVICORN_WS_MAX_QUEUE", 256)
    ws_ping_interval = _get_int_env("UVICORN_WS_PING_INTERVAL", 20)
    ws_ping_timeout = _get_int_env("UVICORN_WS_PING_TIMEOUT", 120)
    timeout_keep_alive = _get_int_env("UVICORN_TIMEOUT_KEEP_ALIVE", 300)

    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=uvicorn_reload,
        log_level=uvicorn_log_level,
        ws_max_size=ws_max_size,           # WebSocket 消息限制（字节）
        ws_max_queue=ws_max_queue,         # 接收队列上限（避免高频小包/回包导致溢出断链）
        ws_ping_interval=ws_ping_interval, # 心跳间隔（秒）
        ws_ping_timeout=ws_ping_timeout,   # 心跳超时（秒）
        timeout_keep_alive=timeout_keep_alive,  # HTTP 连接保活时间（秒）
    )
