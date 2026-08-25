"""HTTP entrypoint for the ML core service (API + producer of the chain)."""
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
    from mlcore.api.v1.jobs import router as jobs_router
    from mlcore.api.v1.results import router as results_router
    from mlcore.api.v1.training import router as training_router

    settings = get_settings()
    configure_logging()
    app = FastAPI(
        title="Academic AutoML - ML Core",
        version=settings.app_version,
        lifespan=lifespan,
    )
    _configure_cors(app, settings)
    register_exception_handlers(app)

    @app.get("/health")
    def health():
        return {"status": "ok", "service": "ml-core"}

    prefix = settings.api_v1_prefix
    app.include_router(training_router, prefix=prefix)
    app.include_router(jobs_router, prefix=prefix)
    app.include_router(results_router, prefix=prefix)
    return app


app = create_app()
