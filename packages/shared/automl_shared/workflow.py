"""Cross-service workflow constants and small guards shared by workers."""
from __future__ import annotations

QUEUE_TRAINING = "ml.training"
QUEUE_EXPLAIN = "explain.shap"
QUEUE_EXPORT = "export.package"
QUEUE_MAINTENANCE = "maintenance"

TASK_TRAIN = "train.run"
TASK_EXPLAIN = "explain.run"
TASK_EXPORT = "export.package"
TASK_CLEANUP = "maintenance.cleanup"


def stage_guard(record: dict | None) -> dict | None:
    """Return a skip-marker payload when a downstream stage must not run.

    A missing record or an already-failed job short-circuits the remaining
    links of the orchestration chain without raising, so Celery eager mode
    keeps the exact same semantics as production queues.
    """
    if record is None:
        return {"state": "MISSING", "skipped": True}
    if record.get("state") == "FAILURE":
        return {"state": "FAILURE", "skipped": True}
    return None
