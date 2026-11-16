from typing import Any, Literal

from pydantic import BaseModel, Field


class PredictRecordsRequest(BaseModel):
    records: list[dict[str, Any]] = Field(
        min_length=1,
        description=(
            "Rows keyed by column name. Send every training feature column for the classic "
            "precomputed path, or raw covariates (plus the date) so the run's saved temporal "
            "context rebuilds lags and rolling windows automatically."
        ),
    )
    history: list[dict[str, Any]] | None = Field(
        default=None,
        description=(
            "Optional raw rows used as lookback context. When omitted, the last "
            "`max_lookback` training rows persisted with the job are used."
        ),
    )
    horizon: int | None = Field(
        default=None,
        ge=1,
        description="Requested forecast horizon when target lags require recursive prediction.",
    )
    mode: Literal["auto", "one-step-ahead", "recursive"] = Field(
        default="auto",
        description=(
            "Temporal inference strategy. 'auto' recurses when target lags are active and "
            "intermediate target values are missing."
        ),
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
