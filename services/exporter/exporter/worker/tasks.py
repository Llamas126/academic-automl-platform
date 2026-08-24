"""Stage 3 of the training chain: build and zip the scientific bundle."""
from __future__ import annotations

import logging
import traceback

from automl_shared.jobs.job_store import get_job_store, mark_failure, mark_success
from automl_shared.schemas.training import TrainRequest
from automl_shared.storage.local_storage import storage
from automl_shared.workflow import stage_guard
from exporter.packaging import package_from_contract
from exporter.worker.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="export.package", bind=True, ignore_result=False)
def run_export_task(self, job_id: str) -> dict:
    store = get_job_store()
    record = store.get(job_id)
    if record is None:
        logger.error("job %s not found in store", job_id)
        return {"job_id": job_id, "state": "FAILURE", "error": "job record missing"}

    skip = stage_guard(store.get(job_id))
    if skip:
        return {"job_id": job_id, **skip}

    store.update(job_id, state="STARTED")
    request = TrainRequest.model_validate(record["request"])

    def progress_cb(stage: str, fraction: float) -> None:
        store.update(job_id, state="PROGRESS", stage=stage, progress=float(fraction))

    try:
        package_from_contract(
            job_id=job_id,
            request=request,
            work_dir=storage.job_work_dir(job_id),
            out_dir=storage.job_out_dir(job_id),
            zip_path=storage.job_zip_path(job_id),
            progress_cb=progress_cb,
        )
        mark_success(store, job_id)
        from automl_shared.events import publish_event, stage_completed

        publish_event(stage_completed("export", job_id))
        logger.info("job %s packaged successfully", job_id)
        return {"job_id": job_id, "state": "SUCCESS"}
    except Exception as exc:
        logger.error("job %s export failed: %s\n%s", job_id, exc, traceback.format_exc())
        mark_failure(store, job_id, f"{type(exc).__name__}: {exc}")
        return {"job_id": job_id, "state": "FAILURE", "error": str(exc)}
