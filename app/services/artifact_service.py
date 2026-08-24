from __future__ import annotations

import hashlib
import hmac
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from app.core.config import get_settings
from app.core.exceptions import ExportError
from app.schemas.results import RunSummary

logger = logging.getLogger(__name__)

MANIFEST_NAME = "models.sha256"
_HASH_CHUNK_BYTES = 65536


def save_final_models(artifacts: dict, models_dir: Path) -> None:
    import joblib

    for target_name, artifact in artifacts.items():
        if artifact.final_pipeline is None:
            continue
        safe_target = _safe_name(target_name)
        payload = {
            "pipeline": artifact.final_pipeline,
            "target": target_name,
            "task": artifact.task,
            "model_key": artifact.best_entry_key,
            "hyperparameters": artifact.best_params,
            "label_classes": artifact.label_classes,
        }
        joblib.dump(payload, models_dir / f"{safe_target}_best_model.joblib")


def model_digest(path: Path, signing_key: str | None = None) -> str:
    """Streaming SHA-256 of the artifact; HMAC-SHA256 when a key is configured."""
    if signing_key:
        digest = hmac.new(signing_key.encode("utf-8"), digestmod=hashlib.sha256)
    else:
        digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_model_manifest(models_dir: Path) -> Path:
    settings = get_settings()
    signing_key = settings.artifact_signing_key or None
    lines = [
        f"{model_digest(path, signing_key)}  {path.name}" for path in sorted(models_dir.glob("*.joblib"))
    ]
    manifest_path = models_dir / MANIFEST_NAME
    manifest_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return manifest_path


def verify_model_integrity(models_dir: Path) -> None:
    """Verify every .joblib against the manifest before deserialization.

    Modes (settings.artifact_integrity):
      - ``off``: skip verification entirely.
      - ``if_present``: verify when the manifest exists; legacy artifacts
        without one are still accepted.
      - ``strict``: a missing manifest is itself a failure.
    """
    settings = get_settings()
    mode = settings.artifact_integrity
    if mode == "off":
        return
    manifest_path = models_dir / MANIFEST_NAME
    if not manifest_path.exists():
        if mode == "strict":
            raise ExportError("model artifact integrity manifest is missing")
        logger.info("no integrity manifest in %s; skipping verification", models_dir.name)
        return

    expected: dict[str, str] = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        digest, separator, name = line.partition("  ")
        if not separator or not digest or not name:
            raise ExportError("model artifact integrity manifest is malformed")
        expected[name] = digest

    actual_files = {path.name for path in models_dir.glob("*.joblib")}
    untracked = sorted(actual_files - set(expected))
    if untracked:
        raise ExportError(f"model artifact(s) not covered by integrity manifest: {untracked}")
    absent = sorted(set(expected) - actual_files)
    if absent:
        raise ExportError(f"integrity manifest references missing artifact(s): {absent}")

    signing_key = settings.artifact_signing_key or None
    for name, digest in expected.items():
        actual = model_digest(models_dir / name, signing_key)
        if not hmac.compare_digest(actual, digest):
            logger.error("artifact integrity mismatch for %s", name)
            raise ExportError("model artifact integrity verification failed")


def save_run_summary(summary: RunSummary, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary.model_dump(mode="json"), indent=2), encoding="utf-8")


def load_model(path: Path) -> dict:
    import joblib

    return joblib.load(path)


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)
