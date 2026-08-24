"""Composite Celery app for the composable monolith (tests / single-node).

In production each service builds its own app; with AUTOML_CELERY_SHARED_APP
set this instance collapses into the shared registry so eager chains resolve
every task name.
"""
from automl_shared.celery_runtime import build_app

celery_app = build_app(
    "automl_composite",
    include=[
        "mlcore.worker.train_tasks",
        "mlcore.worker.maintenance_tasks",
        "explainability.worker.tasks",
        "exporter.worker.tasks",
    ],
)

