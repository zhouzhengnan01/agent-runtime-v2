"""
Daily review report (optional DB mirror).
"""

from sqlalchemy import Column, String, DateTime, BigInteger, JSON, Text
from sqlalchemy.sql import func

from app.db.session import Base


class ReviewReport(Base):
    """Daily review report table (optional for long-term queries)."""

    __tablename__ = "review_reports"

    report_id = Column(String(64), primary_key=True, index=True, comment="Report ID (date or uuid)")
    date = Column(String(10), index=True, nullable=False, comment="Report date (YYYY-MM-DD)")
    agent_id = Column(String(50), index=True, nullable=False, comment="Agent ID")
    base_url = Column(String(255), nullable=True, comment="Base URL")
    report_markdown = Column(Text, nullable=False, comment="Markdown report")
    raw_data = Column(JSON, nullable=True, comment="LLM raw JSON")
    stats = Column(JSON, nullable=True, comment="Summary stats")
    meta = Column(JSON, nullable=True, comment="Meta info")
    created_at_ms = Column(BigInteger, index=True, nullable=False, comment="Created timestamp (ms)")

    created_at = Column(DateTime, server_default=func.now(), comment="Created at")
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), comment="Updated at")
