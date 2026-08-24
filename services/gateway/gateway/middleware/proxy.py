"""HTTP proxy routing API paths to the backing microservices.

The gateway owns no domain logic: it terminates security middlewares and
forwards requests with ``httpx``. Route resolution is by path prefix; each
upstream URL comes from settings (env-overridable per deployment).
"""
from __future__ import annotations

import logging

import httpx
from fastapi import APIRouter, Request, Response
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
}

router = APIRouter(tags=["gateway"])


def _route_table() -> list[tuple[str, str, str]]:
    from automl_shared.core.config import get_settings

    settings = get_settings()
    return [
        (f"{settings.api_v1_prefix}/datasets", "POST", settings.ingestion_service_url),
        (f"{settings.api_v1_prefix}/datasets", "*", settings.ingestion_service_url),
        (f"{settings.api_v1_prefix}/train", "*", settings.mlcore_service_url),
        (f"{settings.api_v1_prefix}/status", "*", settings.mlcore_service_url),
        (f"{settings.api_v1_prefix}/jobs", "*", settings.mlcore_service_url),
        (f"{settings.api_v1_prefix}/results", "*", settings.mlcore_service_url),
        (f"{settings.api_v1_prefix}/download", "*", settings.mlcore_service_url),
        (f"{settings.api_v1_prefix}/predict", "*", settings.inference_service_url),
    ]


def _resolve(path: str, method: str) -> str | None:
    for prefix, route_method, upstream in _route_table():
        if path.startswith(prefix) and (route_method == "*" or method == route_method):
            return upstream
    return None


@router.api_route("/{path:path}", methods=["GET", "POST", "DELETE", "PUT", "PATCH", "OPTIONS"])
async def proxy(request: Request, path: str) -> Response:
    from automl_shared.core.config import get_settings

    full_path = f"/{path}"
    upstream_base = _resolve(full_path, request.method)
    if upstream_base is None:
        return JSONResponse(status_code=404, content={"error": "NotFound", "detail": "no route"})

    target = httpx.URL(upstream_base).join(full_path)
    if request.url.query:
        target = target.copy_merge_params(query=request.url.query)

    headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_BY_HOP}
    body = await request.body()

    try:
        timeout = httpx.Timeout(60.0, connect=5.0)
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
            upstream_response = await client.request(
                request.method,
                str(target),
                headers=headers,
                content=body,
            )
    except httpx.HTTPError as exc:
        logger.error("upstream %s unreachable: %s", upstream_base, exc)
        return JSONResponse(
            status_code=502,
            content={"error": "BadGateway", "detail": f"service unavailable: {upstream_base}"},
        )

    response_headers = {
        k: v for k, v in upstream_response.headers.items() if k.lower() not in HOP_BY_HOP
    }
    return Response(
        content=upstream_response.content,
        status_code=upstream_response.status_code,
        headers=response_headers,
        media_type=upstream_response.headers.get("content-type"),
    )
