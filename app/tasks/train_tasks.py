from __future__ import annotations

import logging
import traceback

from app.schemas.training import TrainRequest
from app.services import dataset_service, job_service
from app.services.package_service import export_and_package
from app.storage.local_storage import storage
from app.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="train.run", bind=True, ignore_result=False)
def run_training_task(self, job_id: str) -> dict:
    store = job_service.get_job_store()
    record = store.get(job_id)
    if record is None:
        logger.error("job %s not found in store", job_id)
        return {"job_id": job_id, "state": "FAILURE", "error": "job record missing"}

    store.update(job_id, state="STARTED")
    request = TrainRequest.model_validate(record["request"])
    dataset_id = record["dataset_id"]

    try:
        df = dataset_service.load_dataset_dataframe(dataset_id)
        profile = dataset_service.get_profile(dataset_id)

        def progress_cb(stage: str, fraction: float) -> None:
            store.update(job_id, state="PROGRESS", stage=stage, progress=float(fraction))

        from app.ml.pipeline import run_training

        output = run_training(request, dataset_id, profile.filename, df, job_id, progress_cb)

        work_dir = storage.job_work_dir(job_id)
        out_dir = storage.job_out_dir(job_id)
        export_and_package(
            job_id=job_id,
            output=output,
            request=request,
            work_dir=work_dir,
            out_dir=out_dir,
            zip_path=storage.job_zip_path(job_id),
            progress_cb=progress_cb,
        )
        job_service.mark_success(store, job_id)
        logger.info("job %s finished successfully", job_id)
        return {"job_id": job_id, "state": "SUCCESS"}
    except Exception as exc:
        logger.error("job %s failed: %s\n%s", job_id, exc, traceback.format_exc())
        job_service.mark_failure(store, job_id, f"{type(exc).__name__}: {exc}")
        return {"job_id": job_id, "state": "FAILURE", "error": str(exc)}
