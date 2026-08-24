from fastapi import APIRouter, Query, Request

from app.core.exceptions import ValidationFailedError
from app.schemas.predict import PredictRecordsRequest, PredictResponse
from app.services.inference_service import run_inference_from_file, run_inference_from_records

router = APIRouter(tags=["inference"])


def _parse_targets(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    items = [chunk.strip() for chunk in raw.split(",") if chunk.strip()]
    return items or None


@router.post("/predict/{job_id}", response_model=PredictResponse)
async def predict(
    job_id: str,
    request: Request,
    targets: str | None = Query(default=None, description="Comma-separated subset of target names"),
):
    """Batch or single-row inference with the trained models of a finished job.

    Accepts either an ``application/json`` body (``{"records": [...]}``) or a
    ``multipart/form-data`` upload with a CSV/XLSX file in the ``file`` field.
    """
    content_type = request.headers.get("content-type", "")
    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        upload = form.get("file")
        if upload is None or isinstance(upload, str):
            raise ValidationFailedError("multipart request must include a 'file' field with CSV/XLSX content")
        content = await upload.read()
        filename = upload.filename or "inference.csv"
        return run_inference_from_file(job_id, filename, content, _parse_targets(targets))

    try:
        body = await request.json()
        payload = PredictRecordsRequest.model_validate(body)
    except ValueError as exc:
        raise ValidationFailedError(f"invalid JSON body: {exc}") from exc
    return run_inference_from_records(job_id, payload.records, _parse_targets(targets))
