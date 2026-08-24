"""Celery application for the exporter service."""
from __future__ import annotations

from automl_shared.celery_runtime import build_app

celery_app = build_app(
    "exporter",
    include=["exporter.worker.tasks"],
    default_queue="export.package",
)
