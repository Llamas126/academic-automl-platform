"""Build and publish the managed mango dataset (`POST /datasets/mango/sync`).

The sync is deliberately synchronous: it resolves the configured sources,
parses them, builds the canonical frame in memory and flips the published file
atomically. A file lock serialises concurrent syncs without requiring Redis, so
the single-process monolith and the standalone ingestion service behave the
same way.

Idempotency is content based. The manifest stores an ``input_fingerprint`` that
covers the spec, the policy and the SHA-256 of every source file; a second sync
with the same inputs returns ``unchanged`` and does not rewrite anything.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from automl_shared.core.config import Settings, get_settings
from automl_shared.core.exceptions import ConflictError, NotFoundError, ValidationFailedError
from automl_shared.profiling.type_detector import profile_dataframe
from automl_shared.schemas.dataset import DatasetProfile
from automl_shared.storage.local_storage import storage

from .domain.mango.cache import NasaCache
from .domain.mango.canonical import CanonicalResult, build_canonical, frame_to_csv_bytes
from .domain.mango.local_records import build_local_records
from .domain.mango.nasa_api import (
    NasaFetchError,
    load_climate_source,
)
from .domain.mango.sources import SourceRegistry, resolve_sources
from .domain.mango.spec import MANAGED_DOMAIN, MangoDatasetSpec
from .mango_config import build_spec
from .schemas.mango import MANIFEST_SCHEMA_VERSION, MangoSyncRequest

if TYPE_CHECKING:
    from .domain.mango.nasa_api import ClimateSource
    from .domain.mango.sources import ResolvedSource, SourceResolution

#: Bumped whenever the canonical artefact contract changes. Part of the fingerprint.
SYNC_POLICY_VERSION = "mango-canonical-v1"
SYNC_STATE_FILENAME = "sync_state.json"
LOCK_RELPATH = Path("locks") / "sync.lock"
MAX_WARNING_ITEMS = 200


def _settings(settings: Settings | None) -> Settings:
    return settings or get_settings()


def _now() -> datetime:
    return datetime.now(UTC)


def _iso(moment: datetime) -> str:
    return moment.isoformat().replace("+00:00", "Z")


def _utc_stamp(moment: datetime) -> str:
    return moment.strftime("%Y%m%dT%H%M%SZ")


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


@contextmanager
def _sync_lock(path: Path, timeout_seconds: int) -> Iterator[None]:
    """Cross-process exclusive lock implemented with ``O_CREAT|O_EXCL``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + max(0, int(timeout_seconds))
    payload = json.dumps({"pid": os.getpid(), "acquired_at": _iso(_now())}).encode("utf-8")

    while True:
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                age = time.time() - path.stat().st_mtime
            except OSError:
                age = 0.0
            if age > max(1, int(timeout_seconds)):
                # The previous holder died without releasing; steal the lock.
                path.unlink(missing_ok=True)
                continue
            if time.monotonic() >= deadline:
                raise ConflictError(
                    "another mango sync is already running; retry once it finishes"
                ) from None
            time.sleep(0.1)
            continue
        try:
            os.write(descriptor, payload)
        finally:
            os.close(descriptor)
        break

    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def _fingerprint(
    spec: MangoDatasetSpec,
    *,
    climate_sha256: str | None,
    local_sha256: dict[str, str | None],
) -> str:
    payload = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "policy": SYNC_POLICY_VERSION,
        "spec": spec.as_dict(),
        "climate_sha256": climate_sha256,
        "local_sha256": {name: local_sha256[name] for name in sorted(local_sha256)},
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_json(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _load_climate(
    spec: MangoDatasetSpec,
    *,
    settings: Settings,
    resolved: ResolvedSource | None,
    refresh: bool,
) -> ClimateSource:
    """Load climate from the API/cache when enabled, else from the local CSV."""
    csv_path = resolved.absolute if resolved is not None else None
    cache = NasaCache(storage.managed_cache_dir(spec.dataset_id) / "nasa", settings.mango_cache_ttl_hours)
    online = settings.mango_enable_nasa_api
    try:
        return load_climate_source(
            spec,
            csv_path=csv_path,
            online=online,
            cache=cache,
            force=refresh and online,
            retries=max(1, settings.mango_nasa_retries),
            timeout=settings.mango_nasa_timeout_seconds,
        )
    except NasaFetchError as exc:
        raise ValidationFailedError(
            f"no usable NASA POWER climate source: {exc}. "
            f"Place a POWER CSV under MANGO_SOURCE_DIR ({settings.mango_source_dir}) "
            "or enable MANGO_ENABLE_NASA_API."
        ) from exc


def _source_payloads(
    resolution: SourceResolution,
    climate: ClimateSource,
) -> dict[str, object]:
    resolved_climate = resolution.climate
    climate_payload = {
        "origin": climate.origin,
        "root": resolved_climate.root if resolved_climate else "nasa_api",
        "path": resolved_climate.path if resolved_climate else "",
        "sha256": climate.sha256 or (resolved_climate.sha256 if resolved_climate else "") or "",
        "fetched_at": _iso(climate.fetched_at) if climate.fetched_at else None,
        "params": climate.params,
    }
    local_payload = {
        target: (
            {
                "root": source.root,
                "path": source.path,
                "sha256": source.sha256,
            }
            if source is not None
            else None
        )
        for target, source in resolution.local.items()
    }
    return {"climate": climate_payload, "local": local_payload}


def _manifest(
    *,
    spec: MangoDatasetSpec,
    result: CanonicalResult,
    resolution: SourceResolution,
    climate: ClimateSource,
    fingerprint: str,
    run_id: str,
    version: str,
    payload: bytes,
    generated_at: datetime,
    extra_warnings: list[str],
) -> dict[str, object]:
    sources = _source_payloads(resolution, climate)
    local = sources["local"]
    local_details = {
        target: entry
        for target, entry in local.items()
        if entry is not None
    }
    for target, status in result.targets.items():
        entry = local_details.get(target)
        if entry is None:
            continue
        entry["rows"] = status.rows
        entry["span"] = list(status.span) if status.span else None

    available = {name: status for name, status in result.targets.items() if status.available}
    unavailable = {
        name: str(status.reason or "unavailable")
        for name, status in result.targets.items()
        if not status.available
    }
    warnings = _dedupe(
        [
            *resolution.warnings,
            *climate.warnings,
            *result.warnings,
            *extra_warnings,
        ]
    )

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset_id": spec.dataset_id,
        "domain": MANAGED_DOMAIN,
        "managed": True,
        "state": "ready" if result.coverage.is_complete else "partial",
        "generated_at": _iso(generated_at),
        "run_id": run_id,
        "generator": "ingestion.mango.sync",
        "spec": spec.as_dict(),
        "sources": sources,
        "coverage": result.coverage.as_dict(),
        "targets": {name: status.as_dict() for name, status in result.targets.items()},
        "available_targets": list(available),
        "unavailable_targets": unavailable,
        "canonical": {
            "filename": spec.canonical_filename,
            "version": version,
            "path": spec.canonical_filename,
            "sha256": result.sha256,
            "rows": int(len(result.frame)),
            "columns": int(result.frame.shape[1]),
            "bytes": len(payload),
            "float_format": "%.4f",
        },
        "imputation": result.imputation,
        "audit": result.audit,
        "input_fingerprint": fingerprint,
        "warnings": warnings[:MAX_WARNING_ITEMS],
    }


def _build_profile(
    spec: MangoDatasetSpec,
    result: CanonicalResult,
    *,
    available: list[str],
    unavailable: dict[str, str],
) -> DatasetProfile:
    profile = profile_dataframe(
        result.frame,
        dataset_id=spec.dataset_id,
        filename=spec.canonical_filename,
    )
    profile.managed = True
    profile.domain = MANAGED_DOMAIN
    profile.available_targets = available
    profile.unavailable_targets = unavailable
    profile.canonical_sha256 = result.sha256
    profile.manifest_version = MANIFEST_SCHEMA_VERSION
    return profile


def _publish(
    spec: MangoDatasetSpec,
    *,
    payload: bytes,
    version: str,
) -> Path:
    """Write the immutable version copy, then flip the published file."""
    dataset_dir = storage.dataset_dir(spec.dataset_id)
    version_dir = storage.managed_versions_dir(spec.dataset_id) / version
    version_path = version_dir / spec.canonical_filename
    storage.write_bytes_atomic(version_path, payload)
    published = storage.dataset_file(spec.dataset_id, spec.canonical_filename)
    storage.write_bytes_atomic(published, payload)
    return dataset_dir


def _prune_versions(dataset_id: str, keep: int) -> None:
    if keep <= 0:
        return
    versions_dir = storage.managed_versions_dir(dataset_id)
    if not versions_dir.is_dir():
        return
    entries = sorted((entry for entry in versions_dir.iterdir() if entry.is_dir()), key=lambda p: p.name)
    for entry in entries[:-keep] if len(entries) > keep else []:
        import shutil

        shutil.rmtree(entry, ignore_errors=True)


def _write_state(
    spec: MangoDatasetSpec,
    *,
    status: str,
    changed: bool,
    run_id: str,
    version: str | None,
    fingerprint: str,
    sha256: str | None,
    coverage_status: str | None,
    warnings: list[str],
    elapsed_ms: int,
) -> None:
    state = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "status": status,
        "changed": changed,
        "version": version,
        "input_fingerprint": fingerprint,
        "canonical_sha256": sha256,
        "coverage_status": coverage_status,
        "synced_at": _iso(_now()),
        "elapsed_ms": elapsed_ms,
        "warnings": warnings[:MAX_WARNING_ITEMS],
    }
    storage.write_json_atomic(
        storage.managed_cache_dir(spec.dataset_id) / SYNC_STATE_FILENAME,
        state,
    )


def _unchanged_response(
    *,
    spec: MangoDatasetSpec,
    manifest: dict,
    run_id: str,
    fingerprint: str,
    elapsed_ms: int,
    dry_run: bool,
) -> dict[str, object]:
    canonical = manifest.get("canonical") or {}
    return {
        "dataset_id": spec.dataset_id,
        "status": "dry_run" if dry_run else "unchanged",
        "changed": False,
        "dry_run": dry_run,
        "run_id": run_id,
        "version": canonical.get("version"),
        "canonical": canonical,
        "previous_sha256": canonical.get("sha256"),
        "coverage": manifest.get("coverage") or {},
        "targets": manifest.get("targets") or {},
        "sources": manifest.get("sources") or {},
        "caching": {"nasa_cache": "disabled" if not manifest else "unchanged"},
        "warnings": (manifest.get("warnings") or [])[:MAX_WARNING_ITEMS],
        "elapsed_ms": elapsed_ms,
    }


def sync_dataset(
    request: MangoSyncRequest | None = None,
    *,
    settings: Settings | None = None,
) -> dict[str, object]:
    """Run one full sync of the managed mango dataset."""
    request = request or MangoSyncRequest()
    settings = _settings(settings)
    started = time.monotonic()
    spec = build_spec(
        settings,
        requested_start=request.requested_start,
        requested_end=request.requested_end,
        include_uncovered_rows=request.include_uncovered_rows,
    )
    run_id = f"run_{_utc_stamp(_now())}_{uuid.uuid4().hex[:8]}"
    lock_path = storage.managed_cache_dir(spec.dataset_id) / LOCK_RELPATH

    with _sync_lock(lock_path, settings.mango_sync_lock_timeout_seconds):
        extra_warnings: list[str] = []
        if request.refresh_nasa and not settings.mango_enable_nasa_api:
            extra_warnings.append(
                "refresh_nasa was requested but MANGO_ENABLE_NASA_API is disabled; "
                "using the local NASA CSV"
            )

        registry = SourceRegistry.load(
            storage.managed_registry_path(spec.dataset_id), spec.target_names
        )
        resolution = resolve_sources(
            spec,
            source_dir=settings.mango_source_dir,
            dataset_dir=storage.dataset_dir(spec.dataset_id),
            registry=registry,
            scan_limit=settings.mango_source_scan_limit,
            max_source_mb=settings.mango_max_source_mb,
        )
        climate = _load_climate(
            spec,
            settings=settings,
            resolved=resolution.climate,
            refresh=bool(request.refresh_nasa),
        )
        local_paths = {
            target: (source.absolute if source is not None else None)
            for target, source in resolution.local.items()
        }
        records = build_local_records(local_paths, spec=spec)

        fingerprint = _fingerprint(
            spec,
            climate_sha256=climate.sha256,
            local_sha256={
                target: (source.sha256 if source is not None else None)
                for target, source in resolution.local.items()
            },
        )

        manifest_path = storage.managed_manifest_path(spec.dataset_id)
        published = storage.dataset_file(spec.dataset_id, spec.canonical_filename)
        existing = _load_json(manifest_path)

        unchanged = (
            not request.force
            and existing is not None
            and existing.get("input_fingerprint") == fingerprint
            and published.is_file()
            and (existing.get("canonical") or {}).get("sha256") == _sha256_file(published)
        )
        if unchanged:
            elapsed_ms = int((time.monotonic() - started) * 1000)
            return _unchanged_response(
                spec=spec,
                manifest=existing,
                run_id=run_id,
                fingerprint=fingerprint,
                elapsed_ms=elapsed_ms,
                dry_run=request.dry_run,
            )

        result = build_canonical(climate.climate, records, spec=spec)
        payload = frame_to_csv_bytes(result.frame)
        generated_at = _now()
        version = f"{_utc_stamp(generated_at)}-{result.sha256[:12]}"
        manifest = _manifest(
            spec=spec,
            result=result,
            resolution=resolution,
            climate=climate,
            fingerprint=fingerprint,
            run_id=run_id,
            version=version,
            payload=payload,
            generated_at=generated_at,
            extra_warnings=extra_warnings,
        )
        available = list(manifest["available_targets"])
        unavailable = dict(manifest["unavailable_targets"])
        previous_sha256 = ((existing or {}).get("canonical") or {}).get("sha256")
        changed = previous_sha256 != result.sha256

        elapsed_ms = int((time.monotonic() - started) * 1000)
        cache_status = {
            "nasa_cache": {
                "origin": climate.origin,
                "ttl_hours": settings.mango_cache_ttl_hours,
                "fetched_at": _iso(climate.fetched_at) if climate.fetched_at else None,
            }
        }

        if not request.dry_run:
            _publish(spec, payload=payload, version=version)
            storage.write_json_atomic(manifest_path, manifest)
            profile = _build_profile(
                spec, result, available=available, unavailable=unavailable
            )
            storage.save_json(
                storage.profile_path(spec.dataset_id), profile.model_dump(mode="json")
            )
            marker = storage.managed_marker_path(spec.dataset_id)
            if not marker.exists():
                storage.write_bytes_atomic(marker, b"managed by ingestion.mango.sync\n")
            _prune_versions(spec.dataset_id, settings.mango_keep_versions)
            _write_state(
                spec,
                status=str(manifest["state"]),
                changed=changed,
                run_id=run_id,
                version=version,
                fingerprint=fingerprint,
                sha256=result.sha256,
                coverage_status=str((manifest["coverage"] or {}).get("status")),
                warnings=list(manifest["warnings"]),
                elapsed_ms=elapsed_ms,
            )

        return {
            "dataset_id": spec.dataset_id,
            "status": "dry_run" if request.dry_run else str(manifest["state"]),
            "changed": changed,
            "dry_run": request.dry_run,
            "run_id": run_id,
            "version": version,
            "canonical": manifest["canonical"],
            "previous_sha256": previous_sha256,
            "coverage": manifest["coverage"],
            "targets": manifest["targets"],
            "sources": manifest["sources"],
            "caching": cache_status,
            "warnings": manifest["warnings"],
            "elapsed_ms": elapsed_ms,
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(*, settings: Settings | None = None) -> dict[str, object]:
    """Return the published manifest or raise ``NotFoundError``."""
    settings = _settings(settings)
    spec = build_spec(settings)
    path = storage.managed_manifest_path(spec.dataset_id)
    payload = _load_json(path)
    if payload is None:
        raise NotFoundError(
            f"manifest for '{spec.dataset_id}' not found; run POST /datasets/mango/sync first"
        )
    return payload


def read_state(*, settings: Settings | None = None) -> dict[str, object]:
    """Return the last sync state, or an empty payload before the first sync."""
    settings = _settings(settings)
    spec = build_spec(settings)
    path = storage.managed_cache_dir(spec.dataset_id) / SYNC_STATE_FILENAME
    return _load_json(path) or {}
