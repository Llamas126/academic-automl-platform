from fastapi import APIRouter, status

from app.api.deps import job_store
from app.core.exceptions import ValidationFailedError
from app.ml.models.registry import REGISTRY
from app.schemas.jobs import JobCreated
from app.schemas.training import TrainRequest
from app.services import dataset_service
from app.storage.local_storage import new_id
from app.tasks.train_tasks import run_training_task

router = APIRouter(tags=["training"])


@router.post("/train", response_model=JobCreated, status_code=status.HTTP_202_ACCEPTED)
def train(request: TrainRequest):
    profile = dataset_service.get_profile(request.dataset_id)
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
    run_training_task.delay(job_id)

    return JobCreated(
        job_id=job_id,
        status_url=f"/api/v1/status/{job_id}",
        download_url=f"/api/v1/download/{job_id}",
    )
