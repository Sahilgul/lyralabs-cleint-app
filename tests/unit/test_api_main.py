"""apps.api.main — health checks + route mounting."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient


@pytest.fixture
def app():
    from apps.api.main import app as fastapi_app
    return fastapi_app


@pytest.mark.asyncio
async def test_healthz(app) -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_readyz(app) -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/readyz")
    assert r.status_code == 200
    assert r.json() == {"status": "ready"}


@pytest.mark.asyncio
async def test_admin_unauthorized_without_token(app) -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/admin/me")
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_oauth_install_endpoints_require_tenant(app) -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        for path in ("/oauth/google/install", "/oauth/ghl/install"):
            r = await c.get(path)
            assert r.status_code == 422  # missing query


@pytest.mark.asyncio
async def test_unknown_path_404(app) -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/no-such-thing")
    assert r.status_code == 404


def test_app_metadata(app) -> None:
    assert app.title == "Lyralabs API"
    assert app.version == "0.1.0"


def test_app_includes_expected_routers(app) -> None:
    paths = {route.path for route in app.routes}
    assert "/healthz" in paths
    assert "/readyz" in paths
    assert "/slack/events" in paths
    assert "/oauth/google/install" in paths
    assert "/oauth/ghl/install" in paths
    assert "/webhooks/stripe" in paths
    assert "/admin/me" in paths
