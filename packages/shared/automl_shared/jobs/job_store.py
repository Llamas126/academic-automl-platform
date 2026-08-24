from __future__ import annotations

import json
import logging
import threading
from datetime import UTC, datetime

from automl_shared.core.config import get_settings

logger = logging.getLogger(__name__)

JOB_TTL_SECONDS = 7 * 24 * 3600


class JobStore:
    def create(self, job_id: str, dataset_id: str, request_payload: dict) -> None:
        raise NotImplementedError

    def get(self, job_id: str) -> dict | None:
        raise NotImplementedError

    def update(self, job_id: str, **fields) -> None:
        raise NotImplementedError

    def all_ids(self) -> set[str]:
        raise NotImplementedError


class MemoryJobStore(JobStore):
    def __init__(self) -> None:
        self._jobs: dict[str, dict] = {}
        self._lock = threading.Lock()

    def create(self, job_id: str, dataset_id: str, request_payload: dict) -> None:
        with self._lock:
            self._jobs[job_id] = {
                "job_id": job_id,
                "dataset_id": dataset_id,
                "state": "PENDING",
                "stage": "",
                "progress": 0.0,
                "error": None,
                "download_ready": False,
                "request": request_payload,
                "created_at": datetime.now(UTC).isoformat(),
                "finished_at": None,
            }

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            record = self._jobs.get(job_id)
            return json.loads(json.dumps(record)) if record else None

    def update(self, job_id: str, **fields) -> None:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return
            record.update(fields)

    def all_ids(self) -> set[str]:
        with self._lock:
            return set(self._jobs.keys())


class RedisJobStore(JobStore):
    def __init__(self, url: str) -> None:
        import redis

        self._client = redis.Redis.from_url(url, decode_responses=True)

    def _key(self, job_id: str) -> str:
        return f"automl:job:{job_id}"

    def create(self, job_id: str, dataset_id: str, request_payload: dict) -> None:
        record = {
            "job_id": job_id,
            "dataset_id": dataset_id,
            "state": "PENDING",
            "stage": "",
            "progress": 0.0,
            "error": None,
            "download_ready": False,
            "request": request_payload,
            "created_at": datetime.now(UTC).isoformat(),
            "finished_at": None,
        }
        self._client.set(self._key(job_id), json.dumps(record), ex=JOB_TTL_SECONDS)

    def get(self, job_id: str) -> dict | None:
        raw = self._client.get(self._key(job_id))
        return json.loads(raw) if raw else None

    def update(self, job_id: str, **fields) -> None:
        key = self._key(job_id)
        raw = self._client.get(key)
        if not raw:
            logger.warning("update on missing job %s", job_id)
            return
        record = json.loads(raw)
        record.update(fields)
        self._client.set(key, json.dumps(record), ex=JOB_TTL_SECONDS)

    def all_ids(self) -> set[str]:
        prefix = "automl:job:"
        return {key[len(prefix) :] for key in self._client.scan_iter(match=f"{prefix}*")}


_store_instance: JobStore | None = None
_store_lock = threading.Lock()


def get_job_store() -> JobStore:
    global _store_instance
    with _store_lock:
        if _store_instance is not None:
            return _store_instance
        settings = get_settings()
        if settings.job_store_backend == "redis":
            try:
                store = RedisJobStore(settings.redis_url)
                store._client.ping()
                logger.info("job store: redis (%s)", settings.redis_url)
                _store_instance = store
                return _store_instance
            except Exception as exc:
                logger.warning("redis unavailable (%s); falling back to in-memory job store", exc)
        else:
            logger.info("job store: memory")
        _store_instance = MemoryJobStore()
        return _store_instance


def reset_job_store() -> None:
    global _store_instance
    with _store_lock:
        _store_instance = None


def mark_failure(store: JobStore, job_id: str, error: str) -> None:
    store.update(
        job_id,
        state="FAILURE",
        error=error[:2000],
        finished_at=datetime.now(UTC).isoformat(),
    )


def mark_success(store: JobStore, job_id: str) -> None:
    store.update(
        job_id,
        state="SUCCESS",
        stage="done",
        progress=1.0,
        download_ready=True,
        finished_at=datetime.now(UTC).isoformat(),
    )
