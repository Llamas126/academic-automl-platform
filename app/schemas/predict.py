from typing import Any

from pydantic import BaseModel, Field


class PredictRecordsRequest(BaseModel):
    records: list[dict[str, Any]] = Field(
        min_length=1,
        description="Feature rows keyed by column name; keys must cover the training feature columns",
    )


class PredictTargetMeta(BaseModel):
    target: str
    task: str
    model: str | None = None
    classes: list[str] | None = None


class PredictResponse(BaseModel):
    job_id: str
    n_rows: int
    targets: dict[str, PredictTargetMeta]
    predictions: list[dict[str, Any]]
    warnings: list[str] = Field(default_factory=list)
