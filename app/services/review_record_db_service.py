"""
Review record persistence (MySQL-based).

This is optional and designed to be best-effort, so that review calls won't fail
even if the DB table is missing or temporarily unavailable.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional

from sqlalchemy.orm import Session

from app.config import settings
from app.db.session import SessionLocal
from app.models.review_record import ReviewRecord

logger = logging.getLogger(__name__)


class ReviewRecordDBService:
    """Persist review records into MySQL (optional)."""

    def __init__(self) -> None:
        self.enabled = bool(getattr(settings, "REVIEW_RECORDS_DB_ENABLED", False))

    def upsert_record(self, record: Dict[str, Any]) -> None:
        if not self.enabled:
            return

        record_id = str(record.get("record_id") or "").strip()
        if not record_id:
            return

        agent_id = str(record.get("agent_id") or "").strip()
        created_at_ms_raw = record.get("created_at_ms") or 0
        try:
            created_at_ms = int(created_at_ms_raw)
        except Exception:
            created_at_ms = 0

        db: Session = SessionLocal()
        try:
            existing: Optional[ReviewRecord] = (
                db.query(ReviewRecord).filter(ReviewRecord.record_id == record_id).one_or_none()
            )
            if existing:
                existing.agent_id = agent_id or existing.agent_id
                if created_at_ms:
                    existing.created_at_ms = created_at_ms
                existing.record = record
            else:
                db.add(
                    ReviewRecord(
                        record_id=record_id,
                        agent_id=agent_id or "unknown",
                        created_at_ms=created_at_ms,
                        record=record,
                    )
                )

            db.commit()
        except Exception as e:
            try:
                db.rollback()
            except Exception:
                pass
            logger.warning("⚠️ ReviewRecord 持久化到 MySQL 失败（将忽略，不影响接口返回）: %s", e)
        finally:
            try:
                db.close()
            except Exception:
                pass

    def delete_records(self, record_ids: Iterable[str]) -> None:
        if not self.enabled:
            return

        ids = [str(rid).strip() for rid in record_ids if str(rid or "").strip()]
        if not ids:
            return

        db: Session = SessionLocal()
        try:
            db.query(ReviewRecord).filter(ReviewRecord.record_id.in_(ids)).delete(synchronize_session=False)
            db.commit()
        except Exception as e:
            try:
                db.rollback()
            except Exception:
                pass
            logger.warning("⚠️ ReviewRecord MySQL 删除失败（将忽略）: %s", e)
        finally:
            try:
                db.close()
            except Exception:
                pass

    def list_records(
        self,
        *,
        agent_id: Optional[str],
        page_index: int,
        page_size: int,
    ) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None

        page_index = max(0, int(page_index))
        page_size = max(1, int(page_size))

        db: Session = SessionLocal()
        try:
            query = db.query(ReviewRecord)
            if agent_id:
                query = query.filter(ReviewRecord.agent_id == agent_id)

            total = int(query.count())
            records: List[ReviewRecord] = (
                query.order_by(ReviewRecord.created_at_ms.desc())
                .offset(page_index * page_size)
                .limit(page_size)
                .all()
            )

            payloads: List[Dict[str, Any]] = []
            for row in records:
                record = row.record if isinstance(row.record, dict) else {}
                if not record:
                    record = {}
                if not record.get("record_id"):
                    record["record_id"] = row.record_id
                if not record.get("agent_id"):
                    record["agent_id"] = row.agent_id
                if not record.get("created_at_ms"):
                    record["created_at_ms"] = row.created_at_ms
                payloads.append(record)

            return {"total": total, "records": payloads}
        except Exception as e:
            logger.warning("⚠️ ReviewRecord MySQL 查询失败（将回退到本地文件）: %s", e)
            return None
        finally:
            try:
                db.close()
            except Exception:
                pass


_review_record_db_service: Optional[ReviewRecordDBService] = None


def get_review_record_db_service() -> ReviewRecordDBService:
    global _review_record_db_service
    if _review_record_db_service is None:
        _review_record_db_service = ReviewRecordDBService()
    return _review_record_db_service
