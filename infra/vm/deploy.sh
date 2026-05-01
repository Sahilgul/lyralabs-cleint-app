#!/usr/bin/env bash
# Lyralabs VM deploy helper.
#
# Run on the VM after editing .env or after Cloud Build pushes a new image:
#   cd /opt/lyralabs && ./deploy.sh
#
# It just authenticates Docker against Artifact Registry, pulls the latest
# image referenced by $WORKER_IMAGE in .env, and recreates the worker
# container. Redis is left running with its volume intact.

set -euo pipefail

cd "$(dirname "$0")"

if [[ ! -f .env ]]; then
  echo "ERROR: .env missing. Copy .env.example and fill it in." >&2
  exit 1
fi

# shellcheck disable=SC1091
source .env

if [[ -z "${WORKER_IMAGE:-}" ]]; then
  echo "ERROR: WORKER_IMAGE not set in .env" >&2
  exit 1
fi

# Authenticate Docker against Artifact Registry. The host is the part of
# WORKER_IMAGE before the first slash, e.g. us-east1-docker.pkg.dev.
REGISTRY_HOST="${WORKER_IMAGE%%/*}"
echo ">> auth docker against ${REGISTRY_HOST}"
gcloud auth configure-docker "${REGISTRY_HOST}" --quiet

echo ">> pulling ${WORKER_IMAGE}"
docker pull "${WORKER_IMAGE}"

echo ">> recreating worker (redis untouched)"
docker compose up -d --no-deps worker

echo ">> tailing worker logs (Ctrl-C to exit)"
docker compose logs -f worker
