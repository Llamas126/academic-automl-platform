"""Gateway security middlewares: headers, API-key auth, rate limiting.

The rate limiter uses a Redis-backed sliding window when the shared store is
Redis, so the budget is global across gateway replicas. With the in-memory
store (tests / single node) it falls back to a process-local deque window.
"""
from __future__ import annotations

import hmac
import logging
import threading
import time
from collections import defaultdict, deque

from automl_shared.core.config import get_settings
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

API_KEY_HEADER = "X-API-Key"


def _error(status_code: int, error: str, detail: str, headers: dict | None = None) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error": error, "detail": detail}, headers=headers)


def _api_key_matches(provided: str, configured_keys: list[str]) -> bool:
    """Constant-time comparison to prevent timing oracles on key matching."""
    return any(hmac.compare_digest(provided, key) for key in configured_keys)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Adds defensive HTTP headers to API responses (docs pages are exempt)."""

    def __init__(self, app, api_prefix: str):
        super().__init__(app)
        self._api_prefix = api_prefix

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        if request.url.path.startswith(self._api_prefix):
            response.headers.setdefault("X-Content-Type-Options", "nosniff")
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["Referrer-Policy"] = "no-referrer"
            response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"
            if get_settings().hsts_enabled:
                response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response


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
        if not _api_key_matches(provided, configured_keys):
            return _error(403, "Forbidden", "invalid API key")
        return await call_next(request)


class MemoryRateLimiter:
    """In-process sliding-window counter (single replica)."""

    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, identity: str, window_s: float, limit: int) -> tuple[bool, int]:
        now = time.monotonic()
        with self._lock:
            hits = self._hits[identity]
            cutoff = now - window_s
            while hits and hits[0] <= cutoff:
                hits.popleft()
            if len(hits) >= limit:
                return False, max(1, int(hits[0] + window_s - now) + 1)
            hits.append(now)
            return True, 0


class RedisRateLimiter:
    """Sliding-window counter on a Redis ZSET, global across replicas.

    Falls back to an in-memory limiter when Redis is unreachable so the
    gateway degrades instead of failing closed.
    """

    def __init__(self) -> None:
        self._fallback = MemoryRateLimiter()
        self._warned = False

    def check(self, identity: str, window_s: float, limit: int) -> tuple[bool, int]:
        try:
            from automl_shared.core.config import get_settings

            client = _redis_client(get_settings().redis_url)
            now_ms = int(time.time() * 1000)
            window_ms = int(window_s * 1000)
            member = f"{now_ms}:{_rand_suffix()}"
            key = f"ratelimit:{identity}"
            pipe = client.pipeline()
            pipe.zremrangebyscore(key, "-inf", now_ms - window_ms)
            pipe.zcard(key)
            _, count = pipe.execute()
            if int(count) >= limit:
                oldest = client.zrange(key, 0, 0, withscores=True)
                retry_after = 1
                if oldest:
                    retry_after = max(1, int((oldest[0][1] + window_ms - now_ms) / 1000) + 1)
                return False, retry_after
            pipe = client.pipeline()
            pipe.zadd(key, {member: now_ms})
            pipe.pexpire(key, window_ms + 1000)
            pipe.execute()
            return True, 0
        except Exception as exc:
            if not self._warned:
                logger.warning("Redis rate limiter unavailable (%s); using in-process fallback", exc)
                self._warned = True
            return self._fallback.check(identity, window_s, limit)


def _redis_client(url: str):
    import redis

    if not hasattr(_redis_client, "_cache"):
        _redis_client._cache = {}
    clients = _redis_client._cache
    if url not in clients:
        clients[url] = redis.Redis.from_url(url, decode_responses=True)
    return clients[url]


def _rand_suffix() -> str:
    import uuid

    return uuid.uuid4().hex[:12]


def _limiter():
    settings = get_settings()
    if settings.job_store_backend == "redis":
        return RedisRateLimiter()
    return MemoryRateLimiter()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding-window rate limiter keyed by API key or client IP."""

    def __init__(self, app, exempt_paths: frozenset[str]):
        super().__init__(app)
        self._exempt_paths = exempt_paths
        self._limiter = _limiter()

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
        allowed, retry_after = self._limiter.check(
            identity, float(settings.rate_limit_window_seconds), int(settings.rate_limit_max_requests)
        )
        if not allowed:
            return _error(
                429,
                "RateLimitExceeded",
                f"rate limit of {settings.rate_limit_max_requests} requests per "
                f"{int(settings.rate_limit_window_seconds)}s exceeded",
                headers={"Retry-After": str(retry_after)},
            )
        return await call_next(request)


def register_security_middlewares(app) -> None:
    """Attach security headers, auth and rate limiting (rate limiting outermost)."""
    settings = get_settings()
    app.add_middleware(AuthMiddleware, api_prefix=settings.api_v1_prefix)
    app.add_middleware(RateLimitMiddleware, exempt_paths=_exempt_paths(settings))
    app.add_middleware(SecurityHeadersMiddleware, api_prefix=settings.api_v1_prefix)


def _exempt_paths(settings) -> frozenset[str]:
    return frozenset(
        {
            "/docs",
            "/redoc",
            "/openapi.json",
            "/health",
            f"{settings.api_v1_prefix}/health",
        }
    )


__all__ = [
    "API_KEY_HEADER",
    "AuthMiddleware",
    "MemoryRateLimiter",
    "RateLimitMiddleware",
    "RedisRateLimiter",
    "SecurityHeadersMiddleware",
    "register_security_middlewares",
]
