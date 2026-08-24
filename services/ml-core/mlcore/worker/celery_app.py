"""Celery application for the ML core service."""
from __future__ import annotations

from automl_shared.celery_runtime import build_app

celery_app = build_app(
    "mlcore",
    include=["mlcore.worker.train_tasks", "mlcore.worker.maintenance_tasks"],
    default_queue="ml.training",
)
