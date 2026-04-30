# Single image for both Lyralabs services. Same code, different runtime command:
#   - lyralabs-api    -> default CMD (uvicorn FastAPI on $PORT)
#   - lyralabs-worker -> override CMD to: celery -A apps.worker.celery_app:celery worker
#
# Cloud Run sets the command via "Container command" / "Container arguments"
# fields in the wizard. docker-compose sets it via the `command:` key.
FROM python:3.14-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# WeasyPrint native deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpango-1.0-0 libpangoft2-1.0-0 libcairo2 libgdk-pixbuf-2.0-0 \
    libffi-dev libssl-dev fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy ALL install inputs before running pip:
#   - pyproject.toml: declares deps + hatchling build config
#   - README.md:      hatchling validates the `readme = "README.md"` field
#   - packages/, apps/: editable install (`pip install -e .`) needs the actual
#                       package directories listed in [tool.hatch.build.targets.wheel]
#   - alembic.ini:    used by migrations at runtime, ride along here for simplicity
#
# Trade-off: any code change re-runs `pip install` (~2-3 min in Cloud Build vs
# ~5s with a deps-only layer cache). Optimize later by generating a
# requirements.txt and splitting deps from project install.
COPY pyproject.toml README.md alembic.ini ./
COPY packages /app/packages
COPY apps /app/apps

RUN pip install --upgrade pip && pip install -e .

ENV PYTHONPATH=/app:/app/packages

# Cloud Run injects $PORT (defaults to 8080). Locally we run on 8000.
# Shell form lets us read $PORT; `exec` makes uvicorn PID 1 so SIGTERM is handled.
EXPOSE 8080
CMD exec uvicorn apps.api.main:app --host 0.0.0.0 --port ${PORT:-8080}
