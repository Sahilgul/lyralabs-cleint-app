FROM python:3.14-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpango-1.0-0 libpangoft2-1.0-0 libcairo2 libgdk-pixbuf-2.0-0 \
    libffi-dev libssl-dev fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
RUN pip install --upgrade pip && pip install -e .

COPY packages /app/packages
COPY apps /app/apps
COPY alembic.ini /app/alembic.ini

ENV PYTHONPATH=/app:/app/packages

# Note: Celery has no HTTP listener, so this container does NOT satisfy
# Cloud Run's startup probe out of the box. Deploy the API service first;
# the worker needs either (a) a sidecar healthz HTTP server, (b) a refactor
# to Cloud Tasks push-based jobs, or (c) a GCE micro VM. See README.md.
CMD ["celery", "-A", "apps.worker.celery_app:celery", "worker", "--loglevel=INFO", "--concurrency=4", "-Q", "agent,default"]
