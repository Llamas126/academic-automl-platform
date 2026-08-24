"""Celery app factory with monorepo-friendly composite mode.

Production: each service builds its own independent Celery application.
Tests: setting ``AUTOML_CELERY_SHARED_APP`` collapses every service app into
one process-wide instance so eager-mode chains can resolve tasks from every
domain without a broker.
"""
from __future__ import annotations

import importlib
import threading

from celery import Celery

_apps: dict[str, Celery] = {}
_lock = threading.RLock()


def _base_conf(app: Celery, beat_schedule: dict | None) -> None:
    from automl_shared.core.config import get_settings

    settings = get_settings()
    app.conf.update(
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
    if beat_schedule:
        app.conf.beat_schedule = beat_schedule
    if settings.celery_eager:
        app.conf.task_always_eager = True
        app.conf.task_eager_propagates = False


def build_app(
    service_name: str,
    *,
    include: list[str],
    default_queue: str | None = None,
    beat_schedule: dict | None = None,
) -> Celery:
    import os

    from automl_shared.core.config import get_settings

    shared_id = os.environ.get("AUTOML_CELERY_SHARED_APP")
    key = f"shared_{shared_id}" if shared_id else service_name

    with _lock:
        app = _apps.get(key)
        if app is None:
            app = Celery(
                f"automl_{key}",
                broker=get_settings().redis_url,
                backend=get_settings().redis_url,
                include=list(include),
            )
            if default_queue:
                app.conf.task_default_queue = default_queue
            _base_conf(app, beat_schedule)
            _apps[key] = app
        else:
            existing = list(getattr(app.conf, "include", []) or [])
            for module in include:
                if module not in existing:
                    existing.append(module)
            app.conf.include = existing

    for module in include:
        try:
            importlib.import_module(module)
        except ImportError as exc:  # pragma: no cover - defensive for partial installs
            import logging

            logging.getLogger(__name__).warning("celery include %s not importable: %s", module, exc)
    return app
