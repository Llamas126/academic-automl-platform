"""HTTP entrypoint for the API gateway."""
from __future__ import annotations

from contextlib import asynccontextmanager

from automl_shared.core.config import get_settings
from automl_shared.core.exceptions import register_exception_handlers
from automl_shared.core.logging import configure_logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


@asynccontextmanager
async def lifespan(_app: FastAPI):
    configure_logging()
    get_settings().ensure_dirs()
    yield


def _configure_cors(app: FastAPI, settings) -> None:
    origins = [origin.strip() for origin in settings.cors_origins.split(",") if origin.strip()]
    if not origins:
        return
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials="*" not in origins,
    )


def create_app() -> FastAPI:
    from gateway.middleware.proxy import router as proxy_router
    from gateway.middleware.security import register_security_middlewares

    settings = get_settings()
    configure_logging()
    app = FastAPI(
        title="Academic AutoML - Gateway",
        version=settings.app_version,
        lifespan=lifespan,
    )
    _configure_cors(app, settings)
    register_exception_handlers(app)
    register_security_middlewares(app)

    @app.get("/health")
    def health():
        return {"status": "ok", "service": "gateway"}

    @app.get(f"{settings.api_v1_prefix}/health")
    def api_health():
        return {"status": "ok", "service": "gateway"}

    app.include_router(proxy_router)
    return app


app = create_app()
