"""Training endpoint: validates, creates the job record and dispatches the chain.

The producer side of the orchestration lives here: a Celery chain routes
train -> (explain) -> export across service queues. Downstream links are
addressed by task name only; each worker resolves its own app.
"""
from __future__ import annotations

from automl_shared.core.exceptions import ValidationFailedError
from automl_shared.schemas.jobs import JobCreated
from automl_shared.schemas.training import TrainRequest
from automl_shared.storage.local_storage import new_id
from fastapi import APIRouter, status

from mlcore.api.deps import job_store
from mlcore.datasets import load_profile
from mlcore.ml.models.registry import REGISTRY

router = APIRouter(tags=["training"])


@router.post("/train", response_model=JobCreated, status_code=status.HTTP_202_ACCEPTED)
def train(request: TrainRequest):
    profile = load_profile(request.dataset_id)
    column_names = {c.name for c in profile.columns}
    missing = [t.name for t in request.targets if t.name not in column_names]
    if missing:
        raise ValidationFailedError(f"target column(s) not in dataset: {missing}")

    if request.models:
        known = {entry.key for entry in REGISTRY}
        unknown = [key for key in request.models if key not in known]
        if unknown:
            raise ValidationFailedError(f"unknown model key(s): {unknown}. Valid keys: {sorted(known)}")

    job_id = new_id("job")
    store = job_store()
    store.create(job_id, request.dataset_id, request.model_dump())
    _dispatch_chain(job_id)

    return JobCreated(
        job_id=job_id,
        status_url=f"/api/v1/status/{job_id}",
        download_url=f"/api/v1/download/{job_id}",
    )


def _dispatch_chain(job_id: str) -> None:
    from automl_shared.workflow import (
        QUEUE_EXPLAIN,
        QUEUE_EXPORT,
        TASK_EXPLAIN,
        TASK_EXPORT,
        TASK_TRAIN,
    )
    from celery import chain

    from mlcore.worker.celery_app import celery_app

    head = celery_app.signature(TASK_TRAIN, args=(job_id,), queue="ml.training")
    links = []
    if _wants_shap(job_id):
        links.append(
            celery_app.signature(TASK_EXPLAIN, args=(job_id,), queue=QUEUE_EXPLAIN).set(immutable=True)
        )
    tail = celery_app.signature(TASK_EXPORT, args=(job_id,), queue=QUEUE_EXPORT).set(immutable=True)
    flow = chain([head, *links, tail])
    flow.apply_async()


def _wants_shap(job_id: str) -> bool:
    from automl_shared.jobs.job_store import get_job_store

    record = get_job_store().get(job_id)
    if not record:
        return False
    request = TrainRequest.model_validate(record["request"])
    return bool(request.explain.shap)
