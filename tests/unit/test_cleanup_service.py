import os
import time

from app.core.config import get_settings
from app.services.cleanup_service import cleanup_expired
from app.services.job_service import get_job_store


def _backdate(path, hours: float) -> None:
    stamp = time.time() - hours * 3600
    os.utime(path, (stamp, stamp))
    for child in path.rglob("*"):
        os.utime(child, (stamp, stamp))


def test_cleanup_deletes_expired_and_keeps_recent(isolated_data_dir):
    settings = get_settings()

    old_job = settings.jobs_dir / "job_old"
    (old_job / "output" / "models").mkdir(parents=True)
    (old_job / "results.zip").write_bytes(b"x" * 100)
    new_job = settings.jobs_dir / "job_new"
    new_job.mkdir(parents=True)
    (new_job / "results.zip").write_bytes(b"y" * 50)

    old_ds = settings.uploads_dir / "ds_old"
    old_ds.mkdir(parents=True)
    (old_ds / "data.csv").write_bytes(b"z" * 30)
    new_ds = settings.uploads_dir / "ds_new"
    new_ds.mkdir(parents=True)
    (new_ds / "data.csv").write_bytes(b"w" * 10)

    _backdate(old_job, 200)
    _backdate(old_ds, 200)

    report = cleanup_expired()

    assert report["deleted_jobs"] == ["job_old"]
    assert report["deleted_datasets"] == ["ds_old"]
    assert report["freed_bytes"] >= 130
    assert not old_job.exists()
    assert not old_ds.exists()
    assert new_job.exists()
    assert new_ds.exists()


def test_cleanup_skips_active_jobs(isolated_data_dir):
    settings = get_settings()
    active_job = settings.jobs_dir / "job_active"
    (active_job / "work").mkdir(parents=True)
    (active_job / "work" / "partial.bin").write_bytes(b"a")
    _backdate(active_job, 500)

    get_job_store().create("job_active", "ds_active", {})

    report = cleanup_expired()

    assert active_job.exists()
    assert report["skipped_active_jobs"] == ["job_active"]
    assert report["deleted_jobs"] == []


def test_cleanup_finished_old_job_without_store_record(isolated_data_dir):
    settings = get_settings()
    orphan_job = settings.jobs_dir / "job_orphan"
    orphan_job.mkdir(parents=True)
    _backdate(orphan_job, 500)

    report = cleanup_expired()

    assert report["deleted_jobs"] == ["job_orphan"]
    assert not orphan_job.exists()


def test_cleanup_zero_retention_removes_everything_inactive(isolated_data_dir, monkeypatch):
    settings = get_settings()
    job = settings.jobs_dir / "job_done"
    job.mkdir(parents=True)
    (job / "f.txt").write_text("data")
    dataset = settings.uploads_dir / "ds_done"
    dataset.mkdir(parents=True)
    _backdate(job, 1)
    _backdate(dataset, 1)

    monkeypatch.setenv("JOB_RETENTION_HOURS", "0")
    monkeypatch.setenv("UPLOAD_RETENTION_HOURS", "0")
    get_settings.cache_clear()
    try:
        report = cleanup_expired()
    finally:
        get_settings.cache_clear()

    assert sorted(report["deleted_jobs"]) == ["job_done"]
    assert report["deleted_datasets"] == ["ds_done"]
    assert not job.exists()
    assert not dataset.exists()
