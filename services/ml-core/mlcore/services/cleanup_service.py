from __future__ import annotations

import logging
import shutil
from datetime import UTC, datetime
from pathlib import Path

from automl_shared.core.config import get_settings
from automl_shared.jobs.job_store import get_job_store

logger = logging.getLogger(__name__)

ACTIVE_STATES = {"PENDING", "STARTED", "PROGRESS"}


def _latest_mtime(path: Path) -> float:
    stamps = [path.stat().st_mtime]
    stamps.extend(child.stat().st_mtime for child in path.rglob("*"))
    return max(stamps)


def _tree_size(path: Path) -> int:
    return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())


def _active_job_ids() -> set[str]:
    store = get_job_store()
    active: set[str] = set()
    for job_id in store.all_ids():
        record = store.get(job_id)
        if record and record.get("state") in ACTIVE_STATES:
            active.add(job_id)
    return active


def _scan(
    root: Path,
    retention_hours: float,
    now_ts: float,
    protected: set[str] | None = None,
) -> tuple[list[tuple[str, Path]], list[str], int]:
    expired: list[tuple[str, Path]] = []
    skipped: list[str] = []
    freed = 0
    if not root.exists():
        return expired, skipped, freed
    max_age_s = retention_hours * 3600
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        if protected and entry.name in protected:
            skipped.append(entry.name)
            continue
        if now_ts - _latest_mtime(entry) > max_age_s:
            freed += _tree_size(entry)
            expired.append((entry.name, entry))
    return expired, skipped, freed


def cleanup_expired(now: datetime | None = None) -> dict:
    """Delete job and dataset directories older than their retention windows.

    Directories are dated by the newest modification time of any file inside,
    so partially written artifacts keep extending their own lifetime. Jobs still
    tracked as PENDING/STARTED/PROGRESS are never removed.
    """
    settings = get_settings()
    moment = now or datetime.now(UTC)
    now_ts = moment.timestamp()

    active = _active_job_ids()
    jobs_expired, jobs_skipped, jobs_freed = _scan(
        settings.jobs_dir, settings.job_retention_hours, now_ts, protected=active
    )
    datasets_expired, _skipped_ds, datasets_freed = _scan(
        settings.uploads_dir, settings.upload_retention_hours, now_ts
    )

    for name, path in jobs_expired:
        logger.info("cleanup: removing expired job dir '%s' (%s)", name, path)
        shutil.rmtree(path, ignore_errors=True)
    for name, path in datasets_expired:
        logger.info("cleanup: removing expired dataset dir '%s' (%s)", name, path)
        shutil.rmtree(path, ignore_errors=True)

    report = {
        "ran_at": moment.isoformat(),
        "job_retention_hours": settings.job_retention_hours,
        "upload_retention_hours": settings.upload_retention_hours,
        "deleted_jobs": [name for name, _path in jobs_expired],
        "skipped_active_jobs": jobs_skipped,
        "deleted_datasets": [name for name, _path in datasets_expired],
        "freed_bytes": jobs_freed + datasets_freed,
    }
    logger.info(
        "cleanup report: %d job(s), %d dataset(s), %d byte(s) freed",
        len(report["deleted_jobs"]),
        len(report["deleted_datasets"]),
        report["freed_bytes"],
    )
    return report
