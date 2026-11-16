"""Registration of mango sources (`PUT /datasets/mango/sources/{nasa|local}`).

Uploads are validated before they are accepted: NASA files must parse as a
POWER export, local files must carry a date column and at least one target
alias. Files are then copied into the managed dataset and recorded in
``sources/registry.json``. Nothing is built here; the caller triggers
``POST /datasets/mango/sync`` when ready.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path

from automl_shared.core.config import Settings, get_settings
from automl_shared.core.exceptions import (
    PayloadTooLargeError,
    UnsupportedFileError,
    ValidationFailedError,
)
from automl_shared.storage.local_storage import sanitize_filename, storage
from automl_shared.tabular import ALLOWED_EXTENSIONS

from .domain.mango.local_records import inspect_local_source
from .domain.mango.nasa_power import parse_nasa_power_csv
from .domain.mango.sources import (
    RegisteredSource,
    SourceRegistry,
    detect_local_targets,
    read_header,
    resolve_source_path,
    resolve_sources,
    sha256_file,
)
from .domain.mango.spec import MangoDatasetSpec, normalize_header
from .mango_config import build_spec
from .schemas.mango import TargetKind

NASA_EXTENSIONS = {".csv"}


def _settings(settings: Settings | None) -> Settings:
    return settings or get_settings()


def _now() -> datetime:
    return datetime.now(UTC)


def _stamp(moment: datetime) -> str:
    return moment.strftime("%Y%m%dT%H%M%SZ")


def _check_extension(filename: str, allowed: set[str], kind: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in allowed:
        raise UnsupportedFileError(
            f"{kind} source type '{suffix}' not allowed; supported: {sorted(allowed)}"
        )
    return suffix


def _check_size(content: bytes, settings: Settings) -> None:
    limit = settings.mango_max_source_mb * 1024 * 1024
    if len(content) > limit:
        raise PayloadTooLargeError(
            f"source file exceeds the mango limit of {settings.mango_max_source_mb} MB"
        )


def _write_temp(content: bytes, suffix: str, settings: Settings, dataset_id: str) -> Path:
    temp_dir = storage.managed_cache_dir(dataset_id) / "tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / f"{uuid.uuid4().hex}{suffix}"
    path.write_bytes(content)
    return path


def _store_upload(
    temp_path: Path,
    filename: str,
    *,
    digest: str,
    settings: Settings,
    dataset_id: str,
    area: str,
    moment: datetime,
) -> tuple[str, Path]:
    safe_name = sanitize_filename(filename)
    relative = f"sources/{area}/{_stamp(moment)}-{digest[:8]}-{safe_name}"
    target = storage.dataset_dir(dataset_id) / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temp_path, target)
    return relative, target


def _prune_managed_sources(dataset_id: str, registry: SourceRegistry) -> None:
    """Delete managed source copies no registration points at any more."""
    dataset_dir = storage.dataset_dir(dataset_id)
    referenced: set[Path] = set()
    entries = [registry.nasa, *registry.local.values()]
    for entry in entries:
        if entry is None or entry.root != "managed":
            continue
        try:
            referenced.add(resolve_source_path(dataset_dir, entry.path))
        except ValidationFailedError:
            continue
    for area in ("nasa", "local"):
        folder = storage.managed_sources_dir(dataset_id) / area
        if not folder.is_dir():
            continue
        for path in folder.iterdir():
            if path.is_file() and path.resolve() not in referenced:
                path.unlink(missing_ok=True)


def _normalised_headers(path: Path) -> list[str]:
    return [normalize_header(column) for column in read_header(path)[:20]]


def _selected_targets(
    detected: list[str],
    kind: TargetKind,
    spec: MangoDatasetSpec,
    *,
    filename: str,
) -> list[str]:
    if kind == TargetKind.collected and spec.collected_target not in detected:
        raise ValidationFailedError(
            f"'{filename}' has no column matching '{spec.collected_target}'"
        )
    if kind == TargetKind.lost and spec.lost_target not in detected:
        raise ValidationFailedError(f"'{filename}' has no column matching '{spec.lost_target}'")
    if kind == TargetKind.auto:
        if not detected:
            raise ValidationFailedError(
                f"'{filename}' matches no mango target; headers: {_normalised_headers(Path(filename))}"
            )
        return detected
    return [spec.collected_target if kind == TargetKind.collected else spec.lost_target]


def register_nasa_upload(
    filename: str,
    content: bytes,
    *,
    settings: Settings | None = None,
) -> dict:
    settings = _settings(settings)
    spec = build_spec(settings)
    suffix = _check_extension(filename, NASA_EXTENSIONS, "nasa")
    _check_size(content, settings)
    storage.dataset_dir(spec.dataset_id).mkdir(parents=True, exist_ok=True)

    temp_path = _write_temp(content, suffix, settings, spec.dataset_id)
    try:
        climate = parse_nasa_power_csv(temp_path, spec=spec)
        digest = sha256_file(temp_path)
        moment = _now()
        relative, _target = _store_upload(
            temp_path,
            filename,
            digest=digest,
            settings=settings,
            dataset_id=spec.dataset_id,
            area="nasa",
            moment=moment,
        )
    except ValidationFailedError as exc:
        temp_path.unlink(missing_ok=True)
        raise ValidationFailedError(f"{sanitize_filename(filename)}: {exc}") from exc

    registry = SourceRegistry.load(storage.managed_registry_path(spec.dataset_id), spec.target_names)
    registry.nasa = RegisteredSource(
        root="managed",
        path=relative,
        origin="upload",
        sha256=digest,
        filename=sanitize_filename(filename),
        registered_at=moment,
    )
    registry.save(storage.managed_registry_path(spec.dataset_id))
    _prune_managed_sources(spec.dataset_id, registry)

    return {
        "source": "nasa",
        "registered_at": moment,
        "registration": {
            "root": "managed",
            "path": relative,
            "origin": "upload",
            "sha256": digest,
            "filename": sanitize_filename(filename),
            "registered_at": moment,
            "target": None,
        },
        "inspection": {
            "rows": climate.days,
            "span": climate.span,
            "parameters": list(climate.parameters),
            "invalid_rows": climate.invalid_rows,
            "duplicate_rows": 0,
            "invalid_dates": 0,
        },
        "warnings": [],
        "next": "POST /api/v1/datasets/mango/sync",
    }


def register_nasa_path(
    path: str,
    *,
    expected_sha256: str | None = None,
    settings: Settings | None = None,
) -> dict:
    settings = _settings(settings)
    spec = build_spec(settings)
    absolute = resolve_source_path(settings.mango_source_dir, path)
    if not absolute.is_file():
        raise ValidationFailedError(f"NASA source not found under MANGO_SOURCE_DIR: {path}")
    _check_extension(absolute.name, NASA_EXTENSIONS, "nasa")
    digest = sha256_file(absolute)
    if expected_sha256 and digest.lower() != expected_sha256.lower():
        raise ValidationFailedError("NASA source sha256 does not match the expected digest")
    try:
        climate = parse_nasa_power_csv(absolute, spec=spec)
    except ValidationFailedError as exc:
        raise ValidationFailedError(f"{absolute.name}: {exc}") from exc

    moment = _now()
    relative = absolute.relative_to(settings.mango_source_dir.resolve()).as_posix()
    registry = SourceRegistry.load(storage.managed_registry_path(spec.dataset_id), spec.target_names)
    registry.nasa = RegisteredSource(
        root="source_dir",
        path=relative,
        origin="source_dir",
        sha256=digest,
        filename=absolute.name,
        registered_at=moment,
    )
    registry.save(storage.managed_registry_path(spec.dataset_id))

    return {
        "source": "nasa",
        "registered_at": moment,
        "registration": {
            "root": "source_dir",
            "path": relative,
            "origin": "source_dir",
            "sha256": digest,
            "filename": absolute.name,
            "registered_at": moment,
            "target": None,
        },
        "inspection": {
            "rows": climate.days,
            "span": climate.span,
            "parameters": list(climate.parameters),
            "invalid_rows": climate.invalid_rows,
            "duplicate_rows": 0,
            "invalid_dates": 0,
        },
        "warnings": [],
        "next": "POST /api/v1/datasets/mango/sync",
    }


def register_local_upload(
    filename: str,
    content: bytes,
    *,
    target: TargetKind = TargetKind.auto,
    settings: Settings | None = None,
) -> dict:
    settings = _settings(settings)
    spec = build_spec(settings)
    suffix = _check_extension(filename, ALLOWED_EXTENSIONS, "local")
    _check_size(content, settings)
    storage.dataset_dir(spec.dataset_id).mkdir(parents=True, exist_ok=True)

    temp_path = _write_temp(content, suffix, settings, spec.dataset_id)
    try:
        detected = detect_local_targets(temp_path, spec)
        targets = _selected_targets(detected, target, spec, filename=filename)
        inspections = {}
        warnings: list[str] = []
        for name in targets:
            inspection, target_warnings = inspect_local_source(temp_path, target=name, spec=spec)
            inspections[name] = inspection
            warnings.extend(target_warnings)
        digest = sha256_file(temp_path)
        moment = _now()
        relative, _stored = _store_upload(
            temp_path,
            filename,
            digest=digest,
            settings=settings,
            dataset_id=spec.dataset_id,
            area="local",
            moment=moment,
        )
    except ValidationFailedError:
        temp_path.unlink(missing_ok=True)
        raise

    registry = SourceRegistry.load(storage.managed_registry_path(spec.dataset_id), spec.target_names)
    registrations = []
    for name in targets:
        inspection = inspections[name]
        registry.local[name] = RegisteredSource(
            root="managed",
            path=relative,
            origin="upload",
            sha256=digest,
            filename=sanitize_filename(filename),
            registered_at=moment,
            target=name,
        )
        registrations.append(
            {
                "target": name,
                "root": "managed",
                "path": relative,
                "origin": "upload",
                "sha256": digest,
                "filename": sanitize_filename(filename),
                "rows": inspection.rows,
                "span": inspection.span,
                "duplicate_rows": inspection.duplicate_rows,
                "invalid_dates": inspection.invalid_dates,
            }
        )
    registry.save(storage.managed_registry_path(spec.dataset_id))
    _prune_managed_sources(spec.dataset_id, registry)

    return {
        "source": "local",
        "registered_at": moment,
        "registrations": registrations,
        "unmatched_targets": [name for name in spec.target_names if name not in targets],
        "warnings": warnings,
        "next": "POST /api/v1/datasets/mango/sync",
    }


def register_local_path(
    path: str,
    *,
    target: TargetKind = TargetKind.auto,
    expected_sha256: str | None = None,
    settings: Settings | None = None,
) -> dict:
    settings = _settings(settings)
    spec = build_spec(settings)
    absolute = resolve_source_path(settings.mango_source_dir, path)
    if not absolute.is_file():
        raise ValidationFailedError(f"local source not found under MANGO_SOURCE_DIR: {path}")
    _check_extension(absolute.name, ALLOWED_EXTENSIONS, "local")
    digest = sha256_file(absolute)
    if expected_sha256 and digest.lower() != expected_sha256.lower():
        raise ValidationFailedError("local source sha256 does not match the expected digest")

    detected = detect_local_targets(absolute, spec)
    targets = _selected_targets(detected, target, spec, filename=absolute.name)
    inspections = {}
    warnings: list[str] = []
    for name in targets:
        inspection, target_warnings = inspect_local_source(absolute, target=name, spec=spec)
        inspections[name] = inspection
        warnings.extend(target_warnings)

    moment = _now()
    relative = absolute.relative_to(settings.mango_source_dir.resolve()).as_posix()
    registry = SourceRegistry.load(storage.managed_registry_path(spec.dataset_id), spec.target_names)
    registrations = []
    for name in targets:
        inspection = inspections[name]
        registry.local[name] = RegisteredSource(
            root="source_dir",
            path=relative,
            origin="source_dir",
            sha256=digest,
            filename=absolute.name,
            registered_at=moment,
            target=name,
        )
        registrations.append(
            {
                "target": name,
                "root": "source_dir",
                "path": relative,
                "origin": "source_dir",
                "sha256": digest,
                "filename": absolute.name,
                "rows": inspection.rows,
                "span": inspection.span,
                "duplicate_rows": inspection.duplicate_rows,
                "invalid_dates": inspection.invalid_dates,
            }
        )
    registry.save(storage.managed_registry_path(spec.dataset_id))

    return {
        "source": "local",
        "registered_at": moment,
        "registrations": registrations,
        "unmatched_targets": [name for name in spec.target_names if name not in targets],
        "warnings": warnings,
        "next": "POST /api/v1/datasets/mango/sync",
    }


def list_sources(*, settings: Settings | None = None) -> dict:
    settings = _settings(settings)
    spec = build_spec(settings)
    registry = SourceRegistry.load(storage.managed_registry_path(spec.dataset_id), spec.target_names)
    resolution = resolve_sources(
        spec,
        source_dir=settings.mango_source_dir,
        dataset_dir=storage.dataset_dir(spec.dataset_id),
        registry=registry,
        scan_limit=settings.mango_source_scan_limit,
        max_source_mb=settings.mango_max_source_mb,
    )
    return {
        "dataset_id": spec.dataset_id,
        "registry": registry.as_dict(),
        "resolved": resolution.as_dict(),
    }


def clear_managed_sources(dataset_id: str) -> None:
    """Remove every copied source (used by maintenance/tests)."""
    folder = storage.managed_sources_dir(dataset_id)
    if folder.exists():
        shutil.rmtree(folder, ignore_errors=True)


def hashlib_sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()
