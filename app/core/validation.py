from __future__ import annotations

import re

from app.core.exceptions import ValidationFailedError

_RESOURCE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def validate_resource_id(value: str, kind: str = "identifier") -> str:
    """Reject identifiers that could escape storage paths (traversal, separators)."""
    if not _RESOURCE_ID_RE.match(value):
        raise ValidationFailedError(f"invalid {kind} format")
    return value
