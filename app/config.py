"""
JetLinks Agent 配置文件
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


APP_ENV = os.getenv("APP_ENV", "").lower()
ENV_FILE = os.getenv(
    "ENV_FILE",
    ".env.production" if APP_ENV in {"prod", "production"} else ".env"
)


class Settings(BaseSettings):
    """应用配置"""

    # Pydantic Settings v2 配置（支持 .env 文件，允许额外环境变量）
    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        case_sensitive=True,
        extra="allow",  # 允许未声明的环境变量，避免 NO_PROXY 等阻塞加载
    )

    # 基础配置
    APP_NAME: str = "JetLinks Agent"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # 服务配置
    HOST: str = "0.0.0.0"
    PORT: int = 8005  # Agent服务端口（默认8005）
    WORKERS: int = 1
    
    # 数据库配置
    # DB_TYPE: mysql | postgresql
    DB_TYPE: str = "postgresql"
    DB_HOST: str = "localhost"
    # PostgreSQL 默认端口/用户（仍可通过环境变量覆盖）
    DB_PORT: int = 5432
    DB_USER: str = "postgres"
    DB_PASSWORD: str = ""
    DB_NAME: str = "jetlinks_agent"
    
    # Redis配置
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: Optional[str] = None
    REDIS_DB: int = 0
    
    # 知识库服务配置（知识库运行在8006端口）
    KNOWLEDGE_SERVICE_URL: str = "http://localhost:8006"
    KNOWLEDGE_SERVICE_TIMEOUT: int = 30
    
    # 视频巡检配置
    VIDEO_MAX_SIZE_MB: int = 500  # 最大视频文件大小
    VIDEO_FRAME_EXTRACT_INTERVAL: int = 30  # 帧提取间隔（帧数）
    VIDEO_ANALYSIS_TIMEOUT: int = 300  # 视频分析超时（秒）
    
    # LLM配置 - 传统模式（向后兼容）
    DASHSCOPE_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_API_BASE: str = "https://api.openai.com/v1"

    # 统一模型配置 - OpenAI兼容API
    OPENAI_BASE_URL: Optional[str] = None
    LLM_MODEL: Optional[str] = None
    VLM_MODEL: Optional[str] = None
    EMBEDDING_MODEL: Optional[str] = None

    # 🚀 新模式：分类配置（推荐）- 支持不同服务商的混合配置
    # LLM配置 - 用于对话、工具选择等
    LLM_API_KEY: Optional[str] = None
    LLM_BASE_URL: Optional[str] = None
    LLM_MODEL_NEW: Optional[str] = None  # 避免与上面的LLM_MODEL冲突

    # VLM配置 - 用于视觉模型（图片/视频分析）
    VLM_API_KEY: Optional[str] = None
    VLM_BASE_URL: Optional[str] = None
    VLM_MODEL_NEW: Optional[str] = None  # 避免与上面的VLM_MODEL冲突

    # EMBEDDING配置 - 用于嵌入模型
    EMBEDDING_API_KEY: Optional[str] = None
    EMBEDDING_BASE_URL: Optional[str] = None
    EMBEDDING_MODEL_NEW: Optional[str] = None  # 避免与上面的EMBEDDING_MODEL冲突

    # 统一LLM Provider配置
    TOOL_SELECTOR_PROVIDER: str = "openai"
    TOOL_SELECTOR_MODEL: str = "glm-4.6"
    VIDEO_ANALYZER_PROVIDER: str = "openai"
    DEFAULT_LLM_PROVIDER: str = "openai"
    DEFAULT_LLM_MODEL: str = "glm-4.6"
    
    # JAIP配置
    ENABLE_JAIP: bool = True
    JAIP_WS_TIMEOUT: int = 3600
    JAIP_MAX_CONNECTIONS: int = 100
    
    # 安全配置
    SECRET_KEY: str = "your-secret-key-here"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRATION_HOURS: int = 24

    # 基础认证（HTTP Basic）
    BASIC_AUTH_ENABLED: bool = False
    BASIC_AUTH_USER: Optional[str] = None
    BASIC_AUTH_PASSWORD: Optional[str] = None
    BASIC_AUTH_REALM: str = "JetLinks Agent"
    BASIC_AUTH_EXCLUDE_PATHS: str = ""
    
    # 日志配置
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Optional[str] = "storage/logs/app.log"

    # SQLAlchemy（避免 DEBUG 模式刷爆日志；需要时显式开启）
    SQLALCHEMY_ECHO: bool = False

    # 🆕 详细日志配置
    ENABLE_DETAILED_LOGGING: bool = True  # 启用详细日志记录
    SHOW_TOOL_CALLS: bool = True  # 显示工具调用详情
    SHOW_LLM_PROMPTS: bool = True  # 显示LLM提示词
    SHOW_LLM_RESPONSES: bool = True  # 显示LLM响应内容
    SHOW_PLAN_STEPS: bool = True  # 显示计划步骤
    SHOW_EXECUTION_FLOW: bool = True  # 显示执行流程
    LOG_TOOL_INPUTS: bool = True  # 记录工具输入参数
    LOG_TOOL_OUTPUTS: bool = True  # 记录工具输出结果
    MAX_LOG_MESSAGE_LENGTH: int = 2000  # 最大日志消息长度（字符）

    # API日志配置
    ENABLE_API_LOGGING: bool = True  # 是否启用API日志记录
    API_LOG_CAPTURE_RESPONSE: bool = True  # 是否记录响应体（关闭可提升性能）
    API_LOG_RETENTION_DAYS: int = 30  # API日志保留天数
    API_LOG_MAX_BODY_SIZE: int = 10000  # API日志请求体最大长度（字符）
    
    # 缓存配置
    CACHE_TTL: int = 300  # 5分钟
    CACHE_MAX_SIZE: int = 1000
    
    # Agent配置
    MAX_AGENT_SESSIONS: int = 100
    AGENT_TIMEOUT: int = 300
    DEFAULT_AGENT_MODEL: str = "glm-4.6"
    
    # Tool配置
    TOOL_EXECUTION_TIMEOUT: int = 60
    TOOL_MAX_RETRIES: int = 3

    # Milvus配置
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530
    MILVUS_USER: Optional[str] = None
    MILVUS_PASSWORD: Optional[str] = None
    MILVUS_DB: str = "default"

    # 记忆向量存储后端（Milvus / pgvector）
    MEMORY_VECTOR_BACKEND: str = "pgvector"  # milvus | pgvector
    MEMORY_VECTOR_DIM: int = 1536

    # 文件存储配置
    SESSION_STORAGE_PATH: str = "storage/sessions"
    MAX_FILE_SIZE: int = 500 * 1024 * 1024  # 500MB
    ALLOWED_FILE_TYPES: str = "pdf,docx,xlsx,pptx,jpg,jpeg,png,gif,mp4,avi,mov,txt,md,json,xml,yaml,html"
    
    # URL下载配置
    URL_DOWNLOAD_TIMEOUT: int = 300  # 5分钟
    URL_DOWNLOAD_ENABLED: bool = True
    
    # Markdown转换配置
    MARKDOWN_CONVERSION_ENABLED: bool = True
    MARKDOWN_CONVERSION_ASYNC: bool = True
    
    # 对话历史配置
    CONVERSATION_HISTORY_ENABLED: bool = True
    CONVERSATION_CACHE_SIZE: int = 1000
    CONVERSATION_CLEANUP_DAYS: int = 90
    
    # 自动总结配置
    AUTO_SUMMARY_ENABLED: bool = True
    SUMMARY_GENERATION_HOUR: int = 2  # 凌晨2点生成
    
    # 静态文件服务配置
    STATIC_FILES_ENABLED: bool = True
    STATIC_FILES_PATH: str = "static"

    # 复判/调用记录落盘配置（filesystem-based）
    REVIEW_RECORDS_STORAGE_PATH: str = "storage/review_records"
    REVIEW_RECORDS_MAX_RECORDS: int = 1000
    REVIEW_RECORDS_CLIP_SECONDS: int = 30
    REVIEW_RECORDS_FFMPEG_BIN: str = "ffmpeg"
    REVIEW_RECORDS_FFMPEG_TIMEOUT: int = 60
    # 复判/调用记录可选同步到 MySQL（不影响落盘；用于长期查询/审计）
    REVIEW_RECORDS_DB_ENABLED: bool = False

    # 复判结果同步到知识库（可选，异步最佳努力）
    REVIEW_KB_ENABLED: bool = False
    REVIEW_KB_BASE_URL: str = ""
    REVIEW_KB_TIMEOUT: int = 15
    REVIEW_KB_COLLECTION_ID: str = "video_search"
    REVIEW_KB_COLLECTION_NAME: str = "Video Search"
    REVIEW_KB_COLLECTION_DESCRIPTION: str = "Auto-created collection for review results"
    REVIEW_KB_COLLECTION_TYPE: str = "multimodal"
    REVIEW_KB_INCLUDE_FAILED: bool = False
    REVIEW_KB_MIN_HIT: int = 1
    REVIEW_KB_MAX_PAYLOAD_CHARS: int = 20000

    # 复判日报配置（默认落盘，可选同步到数据库）
    REVIEW_REPORTS_STORAGE_PATH: str = "storage/review_reports"
    REVIEW_REPORTS_DB_ENABLED: bool = False
    REVIEW_REPORTS_RETENTION_DAYS: int = 180
    REVIEW_REPORTS_MAX_RECORDS: int = 5000
    REVIEW_REPORTS_SAMPLE_SIZE: int = 6
    REVIEW_REPORTS_MAX_PAYLOAD_CHARS: int = 8500
    REVIEW_REPORTS_AGENT_ID: str = "review_cloud"
    REVIEW_REPORTS_GENERATE_EMPTY: bool = True

    # 复判日报定时生成（默认关闭，避免无人值守环境误触发）
    REVIEW_REPORTS_SCHEDULE_ENABLED: bool = False
    REVIEW_REPORTS_SCHEDULE_HOUR: int = 23
    REVIEW_REPORTS_SCHEDULE_MINUTE: int = 50
    REVIEW_REPORTS_SCHEDULE_DAY_OFFSET: int = 0
    REVIEW_REPORTS_SCHEDULE_FORCE: bool = False

    # GLM思考模式配置
    ENABLE_REASONING_MODE: bool = False
    REASONING_MODEL: str = "glm-4.6-reasoning"
    REASONING_TEMPERATURE: float = 0.3
    REASONING_MAX_TOKENS: int = 8000

    # 智能规划器专用配置
    PLANNER_API_KEY: Optional[str] = None
    PLANNER_BASE_URL: Optional[str] = None
    PLANNER_MODEL: Optional[str] = None
    PLANNER_TEMPERATURE: float = 0.05
    PLANNER_MAX_TOKENS: int = 800
    PLANNER_TIMEOUT: float = 60.0

    # 参数填充器专用配置
    PARAMETER_FILLER_API_KEY: Optional[str] = None
    PARAMETER_FILLER_BASE_URL: Optional[str] = None
    PARAMETER_FILLER_MODEL: Optional[str] = None
    PARAMETER_FILLER_TEMPERATURE: float = 0.1
    PARAMETER_FILLER_MAX_TOKENS: int = 1500

    # 视频生成工具配置
    VIDEO_GENERATOR_API_BASE: str = "duomiapi.com"
    VIDEO_GENERATOR_API_KEY: str = ""
    VIDEO_GENERATOR_TIMEOUT: int = 300
    VIDEO_GENERATOR_MAX_DURATION: int = 60
    VIDEO_GENERATOR_MAX_KEYFRAMES: int = 10
    VIDEO_GENERATOR_DEFAULT_ASPECT_RATIO: str = "16:9"
    VIDEO_GENERATOR_DEFAULT_IMAGE_SIZE: str = "2k"
    VIDEO_GENERATOR_ENABLE_MULTI_KEYFRAME: bool = True
    VIDEO_GENERATOR_DEFAULT_MOTION_STRENGTH: str = "moderate"

    # Google搜索工具配置
    SERPAPI_KEY: Optional[str] = None
    GOOGLE_SEARCH_ENGINE: str = "google"
    GOOGLE_SEARCH_NUM_RESULTS: int = 10
    GOOGLE_SEARCH_LANGUAGE: str = "en"
    GOOGLE_SEARCH_SAFE_SEARCH: str = "active"
    #百度搜索工具配置
    BAIDU_API_KEY: Optional[str] = None
    BAIDU_API_BASE_URL: str = "https://qianfan.baidubce.com/v2/ai_search/web_search"
    BAIDU_SEARCH_ENGINE: str = "baidu_search_v2"
    BAIDU_SEARCH_NUM_RESULTS: int = 10
    BAIDU_SEARCH_LANGUAGE: str = "zh"
    BAIDU_SEARCH_SAFE_SEARCH: str = "active"
    BAIDU_DEFAULT_TIMEOUT: int = 30
    BAIDU_MAX_RESULTS: int = 50

    # 搜索代理配置
    USE_SEARCH_PROXY: bool = False
    PROXY_HOST: Optional[str] = None
    PROXY_PORT: Optional[str] = None

    # Sora视频生成工具配置
    SORA_VIDEO_DURATION: int = 15
    SORA_VIDEO_ASPECT_RATIO: str = "16:9"

    # Gemini图片生成工具配置
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_IMAGE_SIZE: str = "2k"
    GEMINI_ASPECT_RATIO: str = "16:9"


# 创建全局配置实例
settings = Settings()
