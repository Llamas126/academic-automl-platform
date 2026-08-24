from fastapi import APIRouter
from fastapi.responses import FileResponse

from app.api.deps import job_store
from app.core.config import get_settings
from app.core.exceptions import ConflictError, NotFoundError
from app.core.validation import validate_resource_id
from app.schemas.jobs import JobState, JobStatus

router = APIRouter(tags=["jobs"])


@router.get("/status/{job_id}", response_model=JobStatus)
def get_status(job_id: str):
    validate_resource_id(job_id, "job id")
    record = job_store().get(job_id)
    if record is None:
        raise NotFoundError(f"job '{job_id}' not found")
    return JobStatus(
        job_id=record["job_id"],
        dataset_id=record.get("dataset_id"),
        state=JobState(record.get("state", "PENDING")),
        stage=record.get("stage", ""),
        progress=float(record.get("progress", 0.0)),
        error=record.get("error"),
        download_ready=bool(record.get("download_ready", False)),
    )


@router.get("/download/{job_id}")
def download_results(job_id: str):
    validate_resource_id(job_id, "job id")
    store = job_store()
    record = store.get(job_id)
    if record is None:
        raise NotFoundError(f"job '{job_id}' not found")
    if record.get("state") != JobState.SUCCESS.value:
        raise ConflictError(f"job '{job_id}' is not finished yet (state={record.get('state')})")
    zip_path = get_settings().jobs_dir / job_id / f"results_{job_id}.zip"
    if not zip_path.exists():
        raise NotFoundError(f"results package for job '{job_id}' is missing")
    return FileResponse(
        path=zip_path,
        media_type="application/zip",
        filename=f"results_{job_id}.zip",
    )
