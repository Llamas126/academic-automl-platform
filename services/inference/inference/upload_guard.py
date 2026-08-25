"""Upload body guards for the inference service."""
from __future__ import annotations

from automl_shared.core.config import get_settings
from automl_shared.core.exceptions import PayloadTooLargeError
from fastapi import UploadFile

UPLOAD_CHUNK_BYTES = 1024 * 1024

__all__ = ["UPLOAD_CHUNK_BYTES", "read_upload_capped"]


async def read_upload_capped(upload: UploadFile, max_bytes: int) -> bytes:
    """Read an UploadFile in chunks, aborting as soon as the cap is exceeded.

    Prevents full-in-memory buffering of oversized bodies before the size
    check happens (memory-exhaustion DoS).
    """
    buffer = bytearray()
    while chunk := await upload.read(UPLOAD_CHUNK_BYTES):
        buffer.extend(chunk)
        if len(buffer) > max_bytes:
            settings = get_settings()
            raise PayloadTooLargeError(f"file exceeds upload limit of {settings.max_upload_mb} MB")
    return bytes(buffer)
