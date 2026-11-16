"""HTTP surface for the managed mango dataset."""

from __future__ import annotations

import hashlib
import json

from automl_shared.core.config import get_settings
from automl_shared.core.exceptions import PayloadTooLargeError, ValidationFailedError
from fastapi import APIRouter, Request, status

from ingestion import source_service, sync_service
from ingestion.dataset_service import read_upload_capped
from ingestion.schemas.mango import MangoSyncRequest, TargetKind

router = APIRouter(prefix="/datasets/mango", tags=["mango"])


def _content_type(request: Request) -> str:
    return request.headers.get("content-type", "").split(";", 1)[0].strip().lower()


async def _read_json(request: Request) -> dict:
    raw = await request.body()
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except ValueError as exc:
        raise ValidationFailedError(f"invalid JSON body: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValidationFailedError("JSON body must be an object")
    return payload


async def _read_form(request: Request, *, limit_mb: int):
    length = request.headers.get("content-length")
    if length and length.isdigit() and int(length) > (limit_mb + 1) * 1024 * 1024:
        raise PayloadTooLargeError(f"source file exceeds the mango limit of {limit_mb} MB")
    try:
        return await request.form()
    except Exception as exc:  # noqa: BLE001 - surfaced as a domain error
        raise ValidationFailedError(f"could not parse multipart body: {exc}") from exc


def _parse_target(raw: object) -> TargetKind:
    try:
        return TargetKind(str(raw or TargetKind.auto))
    except ValueError as exc:
        raise ValidationFailedError(
            f"unknown target '{raw}'; expected one of {[item.value for item in TargetKind]}"
        ) from exc


@router.put("/sources/nasa", status_code=status.HTTP_201_CREATED)
async def put_nasa_source(request: Request):
    """Register a NASA POWER CSV, by multipart upload or by path under MANGO_SOURCE_DIR."""
    settings = get_settings()
    if _content_type(request) == "multipart/form-data":
        form = await _read_form(request, limit_mb=settings.mango_max_source_mb)
        upload = form.get("file")
        if upload is None or not hasattr(upload, "read"):
            raise ValidationFailedError("multipart request must include a 'file' field")
        max_bytes = settings.mango_max_source_mb * 1024 * 1024
        content = await read_upload_capped(
            upload, max_bytes, limit_mb=settings.mango_max_source_mb
        )
        expected = form.get("sha256")
        if expected:
            _verify_sha256(hashlib.sha256(content).hexdigest(), str(expected))
        return source_service.register_nasa_upload(
            getattr(upload, "filename", None) or "nasa.csv",
            content,
            settings=settings,
        )

    body = await _read_json(request)
    path = body.get("path")
    if not path:
        raise ValidationFailedError("JSON body must include 'path' or use multipart upload")
    return source_service.register_nasa_path(
        str(path),
        expected_sha256=body.get("sha256"),
        settings=settings,
    )


@router.put("/sources/local", status_code=status.HTTP_201_CREATED)
async def put_local_source(request: Request):
    """Register farm records, dispatching by target (``auto|collected|lost``)."""
    settings = get_settings()
    if _content_type(request) == "multipart/form-data":
        form = await _read_form(request, limit_mb=settings.mango_max_source_mb)
        upload = form.get("file")
        if upload is None or not hasattr(upload, "read"):
            raise ValidationFailedError("multipart request must include a 'file' field")
        target = _parse_target(form.get("target"))
        max_bytes = settings.mango_max_source_mb * 1024 * 1024
        content = await read_upload_capped(
            upload, max_bytes, limit_mb=settings.mango_max_source_mb
        )
        expected = form.get("sha256")
        if expected:
            _verify_sha256(hashlib.sha256(content).hexdigest(), str(expected))
        return source_service.register_local_upload(
            getattr(upload, "filename", None) or "local.csv",
            content,
            target=target,
            settings=settings,
        )

    body = await _read_json(request)
    path = body.get("path")
    if not path:
        raise ValidationFailedError("JSON body must include 'path' or use multipart upload")
    return source_service.register_local_path(
        str(path),
        target=_parse_target(body.get("target")),
        expected_sha256=body.get("sha256"),
        settings=settings,
    )


@router.post("/sync")
async def post_sync(request: Request):
    """Build (or refresh) the canonical dataset and publish it atomically."""
    body = await _read_json(request)
    try:
        payload = MangoSyncRequest.model_validate(body)
    except ValueError as exc:
        raise ValidationFailedError(str(exc)) from exc
    return sync_service.sync_dataset(payload)


@router.get("/manifest")
def get_manifest():
    """Published manifest: coverage, target availability, hashes and warnings."""
    return sync_service.read_manifest()


@router.get("/sources")
def get_sources():
    """Configured sources, resolved winners and all detected candidates."""
    return source_service.list_sources()


@router.get("/state")
def get_state():
    """Last sync run summary."""
    return sync_service.read_state()


def _verify_sha256(actual: str, expected: str) -> None:
    if actual.lower() != expected.strip().lower():
        raise ValidationFailedError("source sha256 does not match the expected digest")
