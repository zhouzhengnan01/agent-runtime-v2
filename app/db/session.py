"""
数据库会话管理
"""
import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from urllib.parse import quote_plus
from app.config import settings
from app.db.utils import get_postgres_driver

# 构建数据库连接URL（对密码进行URL编码以处理特殊字符）
encoded_password = quote_plus(settings.DB_PASSWORD or "")

db_type = (getattr(settings, "DB_TYPE", "mysql") or "mysql").lower()
if db_type in {"sqlite", "sqlite3"}:
    sqlite_path = os.getenv("DB_SQLITE_PATH") or getattr(settings, "DB_SQLITE_PATH", None)
    if not sqlite_path:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        sqlite_path = os.path.join(base_dir, "storage", "jetlinks_agent.db")
    if sqlite_path == ":memory:":
        SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"
    else:
        if not os.path.isabs(sqlite_path):
            sqlite_path = os.path.abspath(sqlite_path)
        SQLALCHEMY_DATABASE_URL = f"sqlite:///{sqlite_path}"
elif db_type in {"postgres", "postgresql", "pg"}:
    pg_driver = get_postgres_driver()
    SQLALCHEMY_DATABASE_URL = (
        f"postgresql+{pg_driver}://{settings.DB_USER}:{encoded_password}"
        f"@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
    )
else:
    SQLALCHEMY_DATABASE_URL = (
        f"mysql+pymysql://{settings.DB_USER}:{encoded_password}"
        f"@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
    )

# 创建数据库引擎
if db_type in {"sqlite", "sqlite3"}:
    engine = create_engine(
        SQLALCHEMY_DATABASE_URL,
        connect_args={"check_same_thread": False},
        pool_pre_ping=True,  # 连接前ping检查
        echo=getattr(settings, "SQLALCHEMY_ECHO", False),  # 需要时显式开启，避免刷爆日志
    )
else:
    engine = create_engine(
        SQLALCHEMY_DATABASE_URL,
        pool_pre_ping=True,  # 连接前ping检查
        pool_size=10,  # 连接池大小
        max_overflow=20,  # 最大溢出连接数
        echo=getattr(settings, "SQLALCHEMY_ECHO", False),  # 需要时显式开启，避免刷爆日志
    )

# 创建会话工厂
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 创建基类
Base = declarative_base()


def get_db():
    """获取数据库会话"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
