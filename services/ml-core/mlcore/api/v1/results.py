from automl_shared.core.exceptions import ConflictError, NotFoundError, ValidationFailedError
from automl_shared.core.validation import validate_resource_id
from automl_shared.schemas.jobs import JobState
from automl_shared.schemas.results import RunSummary
from automl_shared.storage.local_storage import storage
from fastapi import APIRouter
from fastapi.responses import FileResponse

from mlcore.api.deps import job_store

router = APIRouter(tags=["results"])

ALLOWED_FIGURE_SUFFIX = ".png"


def _require_successful_job(job_id: str) -> None:
    validate_resource_id(job_id, "job id")
    record = job_store().get(job_id)
    if record is None:
        raise NotFoundError(f"job '{job_id}' not found")
    if record.get("state") != JobState.SUCCESS.value:
        raise ConflictError(f"job '{job_id}' is not finished yet (state={record.get('state')})")


def _summary_path(job_id: str):
    return storage.job_out_dir(job_id) / "metrics" / "results_summary.json"


def _figures_root(job_id: str):
    return storage.job_out_dir(job_id) / "figures"


@router.get("/results/{job_id}", response_model=RunSummary)
def get_results(job_id: str):
    _require_successful_job(job_id)
    path = _summary_path(job_id)
    if not path.exists():
        raise NotFoundError(f"results summary for job '{job_id}' is missing")
    try:
        return RunSummary.model_validate(storage.load_json(path))
    except ValidationFailedError:
        raise
    except Exception as exc:  # malformed stored JSON/schema drift
        raise ValidationFailedError(f"stored results summary for job '{job_id}' is malformed: {exc}") from exc


@router.get("/results/{job_id}/figures")
def list_figures(job_id: str):
    _require_successful_job(job_id)
    root = _figures_root(job_id)
    if not root.exists():
        return {"job_id": job_id, "figures": []}
    figures = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob(f"*{ALLOWED_FIGURE_SUFFIX}")
        if path.is_file()
    )
    return {"job_id": job_id, "figures": figures}


@router.get("/results/{job_id}/figures/{figure_path:path}")
def get_figure(job_id: str, figure_path: str):
    _require_successful_job(job_id)
    root = _figures_root(job_id).resolve()
    target = (root / figure_path).resolve()
    if not target.is_relative_to(root):
        raise ValidationFailedError("figure path escapes the figures directory")
    if target.suffix.lower() != ALLOWED_FIGURE_SUFFIX or not target.is_file():
        raise NotFoundError(f"figure '{figure_path}' not found for job '{job_id}'")
    return FileResponse(path=target, media_type="image/png", filename=target.name)
