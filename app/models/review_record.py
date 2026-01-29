"""
复判调用记录（用于 review.html 的历史记录持久化）
"""

from sqlalchemy import Column, String, DateTime, BigInteger, JSON
from sqlalchemy.sql import func

from app.db.session import Base


class ReviewRecord(Base):
    """复判调用记录表（建议用于长期查询/审计）"""

    __tablename__ = "review_records"

    record_id = Column(String(64), primary_key=True, index=True, comment="记录ID（uuid hex）")
    agent_id = Column(String(50), index=True, nullable=False, comment="智能体ID")
    created_at_ms = Column(BigInteger, index=True, nullable=False, comment="创建时间（毫秒）")

    # 完整记录（与 storage/review_records/{record_id}/record.json 保持一致）
    record = Column(JSON, nullable=False, comment="完整记录 JSON（request/response/meta/video）")

    created_at = Column(DateTime, server_default=func.now(), comment="创建时间")
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), comment="更新时间")

