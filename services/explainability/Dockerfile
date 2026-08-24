FROM python:3.13-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN useradd --create-home appuser

COPY --chown=appuser:appuser requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=appuser:appuser packages/shared ./packages/shared
COPY --chown=appuser:appuser services/explainability ./services/explainability
RUN pip install --no-cache-dir --no-deps \
        ./packages/shared \
        ./services/explainability

USER appuser

CMD ["celery", "-A", "explainability.worker.celery_app.celery_app", "worker", "--loglevel=INFO", "--concurrency=1", "-Q", "explain.shap"]
