from __future__ import annotations

import logging

from app.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="maintenance.cleanup", ignore_result=False)
def cleanup_task() -> dict:
    from app.services.cleanup_service import cleanup_expired

    report = cleanup_expired()
    logger.info(
        "maintenance.cleanup removed %d job(s) and %d dataset(s), freed %d byte(s)",
        len(report["deleted_jobs"]),
        len(report["deleted_datasets"]),
        report["freed_bytes"],
    )
    return report
