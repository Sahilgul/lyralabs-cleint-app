"""apps.api.oauth.ghl."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import respx
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from apps.api.oauth import ghl as ghl_mod
from apps.api.oauth._state import encode_state


def _app() -> FastAPI:
    app = FastAPI()
    app.include_router(ghl_mod.router, prefix="/oauth/ghl")
    return app


@pytest.mark.asyncio
async def test_install_redirects_to_marketplace() -> None:
    app = _app()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://t", follow_redirects=False
    ) as c:
        r = await c.get("/oauth/ghl/install", params={"tenant_id": "tenant-1"})
    assert r.status_code in (302, 307)
    assert r.headers["location"].startswith(
        "https://marketplace.gohighlevel.com/oauth/chooselocation"
    )


@pytest.mark.asyncio
async def test_callback_invalid_state_400() -> None:
    app = _app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/oauth/ghl/callback", params={"code": "c", "state": "garbage"})
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_callback_token_exchange_fail() -> None:
    state = encode_state("tenant-1")
    with respx.mock as mock:
        mock.post("https://services.leadconnectorhq.com/oauth/token").respond(401, text="no")
        app = _app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            r = await c.get("/oauth/ghl/callback", params={"code": "c", "state": state})
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_callback_success_persists_and_redirects(monkeypatch) -> None:
    state = encode_state("tenant-1")
    captured = {}

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

        async def execute(self, _stmt):
            captured["executed"] = True
            return MagicMock()

        async def commit(self):
            captured["committed"] = True

    monkeypatch.setattr(ghl_mod, "async_session", FakeSession)

    with respx.mock as mock:
        mock.post("https://services.leadconnectorhq.com/oauth/token").respond(
            200,
            json={
                "access_token": "AT",
                "refresh_token": "RT",
                "expires_in": 86400,
                "scope": "contacts.readonly contacts.write",
                "locationId": "loc-XYZ",
                "userType": "Location",
            },
        )
        app = _app()
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://t", follow_redirects=False
        ) as c:
            r = await c.get("/oauth/ghl/callback", params={"code": "c", "state": state})

    assert r.status_code in (302, 307)
    assert "ghl=connected" in r.headers["location"]
    assert captured.get("committed") is True


@pytest.mark.asyncio
async def test_callback_falls_back_to_company_id_when_no_location(monkeypatch) -> None:
    state = encode_state("tenant-1")
    inserted_values = {}

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

        async def execute(self, stmt):
            # capture compiled values on the insert statement
            try:
                vals = dict(stmt.compile().params)
                inserted_values.update(vals)
            except Exception:
                pass
            return MagicMock()

        async def commit(self):
            return None

    monkeypatch.setattr(ghl_mod, "async_session", FakeSession)

    with respx.mock as mock:
        mock.post("https://services.leadconnectorhq.com/oauth/token").respond(
            200,
            json={
                "access_token": "AT",
                "refresh_token": "RT",
                "expires_in": 86400,
                "companyId": "co-Z",
            },
        )
        app = _app()
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://t", follow_redirects=False
        ) as c:
            r = await c.get("/oauth/ghl/callback", params={"code": "c", "state": state})

    assert r.status_code in (302, 307)
    assert inserted_values.get("external_account_id") == "co-Z"
