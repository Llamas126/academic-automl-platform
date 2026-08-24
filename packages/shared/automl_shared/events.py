from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

EVENTS_CHANNEL = "automl.events"


class EventEnvelope:
    """Minimal typed envelope for domain events published over Redis Pub/Sub."""

    def __init__(self, event_type: str, job_id: str | None = None, payload: dict[str, Any] | None = None):
        self.event_type = event_type
        self.job_id = job_id
        self.payload = payload or {}
        self.occurred_at = datetime.now(UTC).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.event_type,
            "job_id": self.job_id,
            "payload": self.payload,
            "occurred_at": self.occurred_at,
        }


def publish_event(envelope: EventEnvelope) -> bool:
    """Best-effort publish; events are observability sugar, never critical path."""
    try:
        from automl_shared.core.config import get_settings

        settings = get_settings()
        if settings.job_store_backend != "redis":
            return False
        import redis

        client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
        client.publish(EVENTS_CHANNEL, json.dumps(envelope.to_dict()))
        return True
    except Exception as exc:  # noqa: BLE001 - observability must never break the flow
        logger.debug("event publish skipped (%s): %s", envelope.event_type, exc)
        return False


def dataset_ingested(dataset_id: str, filename: str, n_rows: int, n_cols: int) -> EventEnvelope:
    return EventEnvelope(
        "dataset.ingested",
        None,
        {"dataset_id": dataset_id, "filename": filename, "n_rows": n_rows, "n_cols": n_cols},
    )


def stage_completed(stage: str, job_id: str, extra: dict[str, Any] | None = None) -> EventEnvelope:
    return EventEnvelope(f"stage.{stage}", job_id, extra or {})


def job_failed(job_id: str, error: str) -> EventEnvelope:
    return EventEnvelope("job.failed", job_id, {"error": error[:500]})
