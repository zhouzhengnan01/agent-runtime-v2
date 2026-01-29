"""
复判/视频巡检请求日志表
"""
from sqlalchemy import Column, Integer, String, Text, DateTime, Float
from sqlalchemy.sql import func

from app.db.session import Base


class ReviewLog(Base):
    """记录复判接口请求与结果，便于后期排查"""

    __tablename__ = "review_logs"

    id = Column(Integer, primary_key=True, index=True, comment="主键ID")

    # 业务维度
    kind = Column(String(20), nullable=False, comment="请求类型: chat/machine")
    session_id = Column(String(100), comment="会话ID（chat）")
    rpc_id = Column(String(100), comment="RPC ID（machine）")
    video_url = Column(Text, comment="视频/RTSP地址")
    mode = Column(String(50), comment="处理模式: OFFLINE/SECURITY_SINGLE/SECURITY_POLLING")

    # 模型信息
    vlm_backend = Column(String(20), comment="VLM 后端: cloud/local")
    vlm_model = Column(String(50), comment="VLM 模型名称")

    # 状态 & 时长
    status = Column(String(20), nullable=False, default="started", comment="状态: started/success/error/invalid")
    error_message = Column(Text, comment="错误信息")
    duration_ms = Column(Float, comment="耗时（毫秒）")
    started_at = Column(DateTime, server_default=func.now(), comment="开始时间")
    finished_at = Column(DateTime, comment="结束时间")

    # 请求/响应
    request_params = Column(Text, comment="请求参数（JSON字符串）")
    response_body = Column(Text, comment="响应内容（摘要/JSON）")

    # 记录时间
    created_at = Column(DateTime, server_default=func.now(), comment="创建时间")
    updated_at = Column(
        DateTime, server_default=func.now(), onupdate=func.now(), comment="更新时间"
    )
