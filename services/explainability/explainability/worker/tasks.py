"""Stage 2 of the training chain: SHAP figures from the persisted contract."""
from __future__ import annotations

import logging

from automl_shared.jobs.job_store import get_job_store, mark_failure
from automl_shared.schemas.training import TrainRequest
from automl_shared.storage.local_storage import storage
from automl_shared.workflow import stage_guard

from explainability.rendering import render_shap_figures_if_requested
from explainability.worker.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="explain.run", bind=True, ignore_result=False)
def run_explain_task(self, job_id: str) -> dict:
    store = get_job_store()
    record = store.get(job_id)
    skip = stage_guard(record)
    if skip:
        return {"job_id": job_id, **skip}

    request = TrainRequest.model_validate(record["request"])
    work_dir = storage.job_work_dir(job_id)
    out_dir = storage.job_out_dir(job_id)
    try:
        rendered = render_shap_figures_if_requested(work_dir, out_dir, request)
        logger.info("job %s: SHAP figures rendered for %d target(s)", job_id, len(rendered))
        return {"job_id": job_id, "state": "SUCCESS", "rendered": rendered}
    except Exception as exc:
        logger.error("job %s explain stage failed: %s", job_id, exc)
        mark_failure(store, job_id, f"{type(exc).__name__}: {exc}")
        raise
