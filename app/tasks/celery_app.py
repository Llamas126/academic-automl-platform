from __future__ import annotations

from celery import Celery

from app.core.config import get_settings

settings = get_settings()

celery_app = Celery(
    "academic_automl",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["app.tasks.train_tasks", "app.tasks.maintenance_tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    broker_connection_retry_on_startup=True,
    result_expires=7 * 24 * 3600,
    worker_max_tasks_per_child=8,
)

if settings.cleanup_enabled:
    from datetime import timedelta

    celery_app.conf.beat_schedule = {
        "cleanup-expired-artifacts": {
            "task": "maintenance.cleanup",
            "schedule": timedelta(minutes=settings.cleanup_interval_minutes),
        }
    }

if settings.celery_eager:
    celery_app.conf.task_always_eager = True
    celery_app.conf.task_eager_propagates = False
