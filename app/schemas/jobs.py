from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class JobState(StrEnum):
    PENDING = "PENDING"
    STARTED = "STARTED"
    PROGRESS = "PROGRESS"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"


class JobStatus(BaseModel):
    job_id: str
    dataset_id: str | None = None
    state: JobState = JobState.PENDING
    stage: str = ""
    progress: float = Field(0.0, ge=0.0, le=1.0)
    error: str | None = None
    download_ready: bool = False
    created_at: datetime | None = None
    finished_at: datetime | None = None


class JobCreated(BaseModel):
    job_id: str
    status_url: str
    download_url: str
