from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from app.core.config import get_settings

logger = logging.getLogger(__name__)

API_KEY_HEADER = "X-API-Key"


def _error(status_code: int, error: str, detail: str, headers: dict | None = None) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error": error, "detail": detail}, headers=headers)


class AuthMiddleware(BaseHTTPMiddleware):
    """Rejects unauthenticated mutations when auth is enabled.

    Only write endpoints (dataset upload, train, predict) require the API key;
    reads (health, status, download) stay public.
    """

    def __init__(self, app, api_prefix: str):
        super().__init__(app)
        self._protected_prefixes = (
            ("POST", f"{api_prefix}/datasets"),
            ("POST", f"{api_prefix}/train"),
            ("POST", f"{api_prefix}/predict"),
        )

    async def dispatch(self, request: Request, call_next):
        settings = get_settings()
        if not settings.auth_enabled or request.method == "OPTIONS":
            return await call_next(request)
        if not any(
            request.method == method and request.url.path.startswith(prefix)
            for method, prefix in self._protected_prefixes
        ):
            return await call_next(request)

        configured_keys = [key.strip() for key in settings.api_keys.split(",") if key.strip()]
        if not configured_keys:
            logger.warning("AUTH_ENABLED is set but no API keys are configured; rejecting request")
            return _error(503, "AuthNotConfigured", "authentication enabled but no API keys configured")

        provided = request.headers.get(API_KEY_HEADER)
        if not provided:
            return _error(401, "Unauthorized", f"missing {API_KEY_HEADER} header")
        if provided not in configured_keys:
            return _error(403, "Forbidden", "invalid API key")
        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """In-process sliding-window rate limiter keyed by API key or client IP.

    Note: counters live in the current process; behind multiple replicas each
    instance enforces its own budget (upgrade path: Redis INCR+EXPIRE).
    """

    def __init__(self, app, exempt_paths: frozenset[str]):
        super().__init__(app)
        self._exempt_paths = exempt_paths
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    async def dispatch(self, request: Request, call_next):
        settings = get_settings()
        if (
            not settings.rate_limit_enabled
            or request.method == "OPTIONS"
            or request.url.path in self._exempt_paths
        ):
            return await call_next(request)

        identity = request.headers.get(API_KEY_HEADER) or (
            request.client.host if request.client else "anonymous"
        )
        now = time.monotonic()
        window_s = float(settings.rate_limit_window_seconds)
        limit = int(settings.rate_limit_max_requests)

        with self._lock:
            hits = self._hits[identity]
            cutoff = now - window_s
            while hits and hits[0] <= cutoff:
                hits.popleft()
            if len(hits) >= limit:
                retry_after = max(1, int(hits[0] + window_s - now) + 1)
                limited = True
            else:
                hits.append(now)
                limited = False

        if limited:
            return _error(
                429,
                "RateLimitExceeded",
                f"rate limit of {limit} requests per {int(window_s)}s exceeded",
                headers={"Retry-After": str(retry_after)},
            )
        return await call_next(request)


def register_security_middlewares(app) -> None:
    """Attach auth and rate limiting. Rate limiting ends up outermost."""
    settings = get_settings()
    app.add_middleware(AuthMiddleware, api_prefix=settings.api_v1_prefix)
    app.add_middleware(RateLimitMiddleware, exempt_paths=_exempt_paths(settings))


def _exempt_paths(settings) -> frozenset[str]:
    return frozenset(
        {
            "/docs",
            "/redoc",
            "/openapi.json",
            f"{settings.api_v1_prefix}/health",
        }
    )
