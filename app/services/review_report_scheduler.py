"""
Background scheduler for daily review report generation.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, time as dt_time

from app.config import settings
from app.services.review_report_service import get_review_report_service

logger = logging.getLogger(__name__)


async def review_report_scheduler(stop_event: asyncio.Event) -> None:
    service = get_review_report_service()
    last_run_date = None
    check_interval = 30.0

    while not stop_event.is_set():
        now = datetime.now()
        try:
            hour = int(getattr(settings, "REVIEW_REPORTS_SCHEDULE_HOUR", 23))
            minute = int(getattr(settings, "REVIEW_REPORTS_SCHEDULE_MINUTE", 50))
        except Exception:
            hour = 23
            minute = 50

        scheduled_time = datetime.combine(now.date(), dt_time(hour=hour, minute=minute))
        should_run = now >= scheduled_time and last_run_date != now.date()

        if should_run:
            day_offset = int(getattr(settings, "REVIEW_REPORTS_SCHEDULE_DAY_OFFSET", 0))
            target_date = now.date() + timedelta(days=day_offset)
            target_date_str = target_date.isoformat()
            try:
                await service.generate_report(
                    date_value=target_date_str,
                    base_url=None,
                    agent_id=getattr(settings, "REVIEW_REPORTS_AGENT_ID", "review_cloud"),
                    sample_size=getattr(settings, "REVIEW_REPORTS_SAMPLE_SIZE", 6),
                    force=bool(getattr(settings, "REVIEW_REPORTS_SCHEDULE_FORCE", False)),
                    generated_by="schedule",
                )
                logger.info("Daily review report generated for %s", target_date_str)
            except Exception as e:
                logger.warning("Daily review report generation failed: %s", e)
            last_run_date = now.date()

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=check_interval)
        except asyncio.TimeoutError:
            continue
