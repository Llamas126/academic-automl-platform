import json

from automl_shared.core.config import get_settings
from automl_shared.core.exceptions import PayloadTooLargeError, ValidationFailedError
from automl_shared.core.validation import validate_resource_id
from automl_shared.schemas.predict import PredictRecordsRequest, PredictResponse
from fastapi import APIRouter, Query, Request
from inference.inference_service import run_inference_from_file, run_inference_from_records
from inference.upload_guard import read_upload_capped

router = APIRouter(tags=["inference"])


def _parse_targets(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    items = [chunk.strip() for chunk in raw.split(",") if chunk.strip()]
    return items or None


def content_length_exceeds(request: Request, max_bytes: int) -> bool:
    raw = request.headers.get("content-length")
    try:
        return bool(raw) and int(raw) > max_bytes
    except ValueError:
        return False


@router.post("/predict/{job_id}", response_model=PredictResponse)
async def predict(
    job_id: str,
    request: Request,
    targets: str | None = Query(default=None, description="Comma-separated subset of target names"),
    horizon: int | None = Query(
        default=None,
        ge=1,
        description="Forecast horizon used to enable recursive target-lag prediction",
    ),
    mode: str = Query(
        default="auto",
        description="Temporal inference mode: auto, one-step-ahead or recursive",
    ),
):
    """Batch or single-row inference with the trained models of a finished job.

    Accepts either an ``application/json`` body (``{"records": [...], "history": [...], ...}``)
    or a ``multipart/form-data`` upload with a CSV/XLSX file in the ``file`` field. Raw
    covariates are enough for runs with temporal features: the saved ``max_lookback``
    training history rebuilds the same causal lags and rolling windows.
    """
    validate_resource_id(job_id, "job id")
    settings = get_settings()

    content_type = request.headers.get("content-type", "")
    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        upload = form.get("file")
        if upload is None or isinstance(upload, str):
            raise ValidationFailedError("multipart request must include a 'file' field with CSV/XLSX content")
        content = await read_upload_capped(upload, settings.max_upload_mb * 1024 * 1024)
        filename = upload.filename or "inference.csv"
        form_history = form.get("history")
        history = None
        if isinstance(form_history, str) and form_history.strip():
            try:
                history = json.loads(form_history)
            except json.JSONDecodeError as exc:
                raise ValidationFailedError(f"invalid JSON in form field 'history': {exc}") from exc
            if not isinstance(history, list):
                raise ValidationFailedError("form field 'history' must be a JSON array of rows")
        form_horizon = form.get("horizon")
        form_mode = form.get("mode")
        request_horizon = horizon
        if isinstance(form_horizon, str) and form_horizon.isdigit():
            request_horizon = int(form_horizon)
        return run_inference_from_file(
            job_id,
            filename,
            content,
            _parse_targets(targets),
            history=history,
            horizon=request_horizon,
            mode=form_mode if isinstance(form_mode, str) and form_mode else mode,
        )

    max_body_bytes = settings.predict_max_body_mb * 1024 * 1024
    if content_length_exceeds(request, max_body_bytes):
        raise PayloadTooLargeError(f"request body exceeds limit of {settings.predict_max_body_mb} MB")
    try:
        body = await request.json()
        payload = PredictRecordsRequest.model_validate(body)
    except ValueError as exc:
        raise ValidationFailedError(f"invalid JSON body: {exc}") from exc
    return run_inference_from_records(
        job_id,
        payload.records,
        _parse_targets(targets),
        history=payload.history,
        horizon=payload.horizon or horizon,
        mode=payload.mode if payload.mode != "auto" else mode,
    )
