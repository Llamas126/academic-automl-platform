from fastapi import APIRouter

from app.api.v1 import datasets, health, jobs, predict, results, training

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(datasets.router)
api_router.include_router(training.router)
api_router.include_router(jobs.router)
api_router.include_router(results.router)
api_router.include_router(predict.router)
