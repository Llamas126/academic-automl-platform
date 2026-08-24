"""Legacy facade kept for backwards compatibility.

Delegates to the split services: the ML core contract (persist), the
explainability stage (SHAP figures) and the exporter stage (bundle). The
in-memory ``output`` mode persists to files first, then packages from them,
so both call paths exercise the exact same file-based contract.
"""
from __future__ import annotations

from pathlib import Path

from automl_shared.schemas.training import TrainRequest

__all__ = ["export_and_package", "new_job_id"]


def export_and_package(
    job_id: str,
    output=None,
    request: TrainRequest | None = None,
    work_dir: Path | None = None,
    out_dir: Path | None = None,
    zip_path: Path | None = None,
    progress_cb=None,
) -> Path:
    cb = progress_cb or (lambda stage, frac: None)
    if output is not None:
        cb("export:contract", 0.8)
        from mlcore.contract import ensure_persisted

        ensure_persisted(output, work_dir, out_dir)

    if request.explain.shap:
        # Legacy in-process callers expect SHAP figures rendered here; the
        # orchestrated chain runs this in the explainability service instead.
        from explainability.rendering import render_shap_figures_if_requested

        render_shap_figures_if_requested(work_dir, out_dir, request)

    from exporter.packaging import package_from_contract

    return package_from_contract(
        job_id=job_id,
        request=request,
        work_dir=work_dir,
        out_dir=out_dir,
        zip_path=zip_path,
        progress_cb=cb,
    )


def new_job_id() -> str:
    from automl_shared.storage.local_storage import new_id

    return new_id("job")
