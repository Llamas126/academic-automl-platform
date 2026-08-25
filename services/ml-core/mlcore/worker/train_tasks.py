"""Stage 1 of the training chain: train, persist, hand off via files."""
from __future__ import annotations

import logging
import traceback

from automl_shared.jobs.job_store import get_job_store, mark_failure, mark_success
from automl_shared.schemas.training import TrainRequest
from automl_shared.storage.local_storage import storage
from automl_shared.workflow import stage_guard

from mlcore.contract import persist_training_output
from mlcore.datasets import load_dataset_frame
from mlcore.worker.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="train.run", bind=True, ignore_result=False)
def run_training_task(self, job_id: str) -> dict:
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
    dataset_id = record["dataset_id"]

    try:
        df, profile = load_dataset_frame(dataset_id)

        def progress_cb(stage: str, fraction: float) -> None:
            store.update(job_id, state="PROGRESS", stage=stage, progress=float(fraction))

        from mlcore.ml.pipeline import run_training

        output = run_training(request, dataset_id, profile.filename, df, job_id, progress_cb)

        work_dir = storage.job_work_dir(job_id)
        out_dir = storage.job_out_dir(job_id)
        persist_training_output(output, work_dir, out_dir)
        mark_success(store, job_id)
        from automl_shared.events import publish_event, stage_completed

        publish_event(stage_completed("train", job_id))
        logger.info("job %s trained and persisted", job_id)
        return {"job_id": job_id, "state": "SUCCESS"}
    except Exception as exc:
        logger.error("job %s failed: %s\n%s", job_id, exc, traceback.format_exc())
        mark_failure(store, job_id, f"{type(exc).__name__}: {exc}")
        from automl_shared.events import job_failed, publish_event

        publish_event(job_failed(job_id, str(exc)))
        return {"job_id": job_id, "state": "FAILURE", "error": str(exc)}
