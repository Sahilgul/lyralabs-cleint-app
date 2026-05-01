"""FastAPI entrypoint.

Mounts:
  - /slack/events       Slack Bolt request handler
  - /oauth/slack/...    Slack OAuth (install + callback) via Bolt
  - /oauth/google/...   Google OAuth callback
  - /oauth/ghl/...      GoHighLevel OAuth callback
  - /webhooks/stripe    Stripe billing webhook
  - /admin/...          Admin REST API consumed by the Vite SPA
                        (separate repo: lyralabs-admin-ui, deployed as its own
                        Cloud Run service. Cross-origin requests are allowed
                        via the CORS allow-list below.)
  - /healthz, /readyz   liveness / readiness
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from lyra_core.channels.slack.adapter import build_slack_app
from lyra_core.common.config import get_settings
from lyra_core.common.logging import configure_logging, get_logger

from .admin.llm import router as admin_llm_router
from .admin.routes import router as admin_router
from .oauth.ghl import router as ghl_router
from .oauth.google import router as google_router
from .stripe_webhook import router as stripe_router

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    settings = get_settings()
    configure_logging(level=settings.log_level, json_logs=settings.is_prod)
    log.info("api.startup", env=settings.app_env)
    yield
    log.info("api.shutdown")


app = FastAPI(
    title="Lyralabs API",
    version="0.1.0",
    lifespan=lifespan,
    default_response_class=JSONResponse,
)

_settings = get_settings()
# The admin SPA (lyralabs-admin-ui repo) is served from its own origin in
# production. Set ADMIN_BASE_URL to that origin (e.g. https://admin.your-domain
# .com) so its fetch() calls are allowed. In dev the Vite server proxies API
# calls server-side, but we still allow http://localhost:5173 for direct calls.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[_settings.admin_base_url],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Slack routes -------------------------------------------------------------

slack_app, slack_handler = build_slack_app()


@app.post("/slack/events")
async def slack_events(req: Request):
    return await slack_handler.handle(req)


@app.get("/oauth/slack/install")
async def slack_install(req: Request):
    return await slack_handler.handle(req)


@app.get("/oauth/slack/callback")
async def slack_callback(req: Request):
    return await slack_handler.handle(req)


# --- Other routers ------------------------------------------------------------

app.include_router(google_router, prefix="/oauth/google", tags=["oauth-google"])
app.include_router(ghl_router, prefix="/oauth/ghl", tags=["oauth-ghl"])
app.include_router(stripe_router, prefix="/webhooks", tags=["webhooks"])
app.include_router(admin_router, prefix="/admin", tags=["admin"])
# Super-admin (platform operator) routes for runtime LLM model switching.
app.include_router(admin_llm_router, prefix="/admin/llm", tags=["admin-llm"])


# --- Health -------------------------------------------------------------------


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
async def readyz() -> dict[str, str]:
    # TODO: check Postgres + Redis connectivity for true readiness.
    return {"status": "ready"}
