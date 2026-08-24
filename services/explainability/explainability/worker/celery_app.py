"""Celery application for the explainability service."""
from __future__ import annotations

from automl_shared.celery_runtime import build_app

celery_app = build_app(
    "explainability",
    include=["explainability.worker.tasks"],
    default_queue="explain.shap",
)
