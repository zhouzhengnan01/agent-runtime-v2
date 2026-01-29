"""
Daily review report persistence (DB mirror).

This is optional and best-effort, so report generation won't fail if the table
is missing or temporarily unavailable.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, Optional

from sqlalchemy.orm import Session

from app.config import settings
from app.db.session import SessionLocal
from app.models.review_report import ReviewReport

logger = logging.getLogger(__name__)


class ReviewReportDBService:
    """Persist daily review reports into DB (optional)."""

    def __init__(self) -> None:
        self.enabled = bool(getattr(settings, "REVIEW_REPORTS_DB_ENABLED", False))

    def upsert_report(self, report: Dict[str, Any]) -> None:
        if not self.enabled:
            return

        report_id = str(report.get("report_id") or report.get("date") or "").strip()
        if not report_id:
            return

        date = str(report.get("date") or "").strip() or report_id
        agent_id = str(report.get("agent_id") or "").strip() or "unknown"
        base_url = report.get("base_url")
        report_markdown = report.get("report_markdown") or ""
        raw_data = report.get("raw_data")
        stats = report.get("stats")
        meta = report.get("meta")

        created_at_ms_raw = report.get("created_at_ms") or 0
        try:
            created_at_ms = int(created_at_ms_raw)
        except Exception:
            created_at_ms = 0

        db: Session = SessionLocal()
        try:
            existing: Optional[ReviewReport] = (
                db.query(ReviewReport).filter(ReviewReport.report_id == report_id).one_or_none()
            )
            if existing:
                existing.date = date or existing.date
                existing.agent_id = agent_id or existing.agent_id
                existing.base_url = base_url or existing.base_url
                existing.report_markdown = report_markdown or existing.report_markdown
                existing.raw_data = raw_data
                existing.stats = stats
                existing.meta = meta
                if created_at_ms:
                    existing.created_at_ms = created_at_ms
            else:
                db.add(
                    ReviewReport(
                        report_id=report_id,
                        date=date,
                        agent_id=agent_id,
                        base_url=base_url,
                        report_markdown=report_markdown,
                        raw_data=raw_data,
                        stats=stats,
                        meta=meta,
                        created_at_ms=created_at_ms,
                    )
                )
            db.commit()
        except Exception as e:
            try:
                db.rollback()
            except Exception:
                pass
            logger.warning("ReviewReport DB upsert failed (ignored): %s", e)
        finally:
            try:
                db.close()
            except Exception:
                pass

    def delete_reports(self, report_ids: Iterable[str]) -> None:
        if not self.enabled:
            return

        ids = [str(rid).strip() for rid in report_ids if str(rid or "").strip()]
        if not ids:
            return

        db: Session = SessionLocal()
        try:
            db.query(ReviewReport).filter(ReviewReport.report_id.in_(ids)).delete(synchronize_session=False)
            db.commit()
        except Exception as e:
            try:
                db.rollback()
            except Exception:
                pass
            logger.warning("ReviewReport DB delete failed (ignored): %s", e)
        finally:
            try:
                db.close()
            except Exception:
                pass


_review_report_db_service: Optional[ReviewReportDBService] = None


def get_review_report_db_service() -> ReviewReportDBService:
    global _review_report_db_service
    if _review_report_db_service is None:
        _review_report_db_service = ReviewReportDBService()
    return _review_report_db_service
