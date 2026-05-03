"""apps.api.oauth.google."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import respx
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from apps.api.oauth import google as google_mod
from apps.api.oauth._state import encode_state


def _build_app() -> FastAPI:
    app = FastAPI()
    app.include_router(google_mod.router, prefix="/oauth/google")
    return app


@pytest.fixture
def client():
    app = _build_app()

    async def _get():
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test", follow_redirects=False
        ) as c:
            yield c

    return _get


@pytest.mark.asyncio
async def test_install_redirects_to_google(monkeypatch) -> None:
    s = google_mod.get_settings()
    monkeypatch.setattr(s, "google_oauth_client_id", "test-client", raising=False)
    monkeypatch.setattr(s, "google_oauth_redirect_uri", "https://x/cb", raising=False)
    monkeypatch.setattr(s, "google_oauth_scopes", "https://scope.a https://scope.b", raising=False)

    app = _build_app()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://t", follow_redirects=False
    ) as c:
        r = await c.get("/oauth/google/install", params={"tenant_id": "tenant-1"})
    assert r.status_code in (302, 307)
    loc = r.headers["location"]
    assert loc.startswith("https://accounts.google.com/o/oauth2/v2/auth")
    assert "client_id=test-client" in loc
    assert "state=" in loc


@pytest.mark.asyncio
async def test_install_requires_tenant_id() -> None:
    app = _build_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/oauth/google/install")
    assert r.status_code == 422  # missing query param


@pytest.mark.asyncio
async def test_callback_invalid_state_400() -> None:
    app = _build_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/oauth/google/callback", params={"code": "x", "state": "garbage"})
    assert r.status_code == 400
    assert "invalid state" in r.text


@pytest.mark.asyncio
async def test_callback_token_exchange_failure_400(monkeypatch) -> None:
    state = encode_state("tenant-1")
    with respx.mock as mock:
        mock.post("https://oauth2.googleapis.com/token").respond(400, text="bad")
        app = _build_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            r = await c.get("/oauth/google/callback", params={"code": "x", "state": state})
    assert r.status_code == 400
    assert "token exchange failed" in r.text


@pytest.mark.asyncio
async def test_callback_success_persists_and_redirects(monkeypatch) -> None:
    state = encode_state("tenant-1")
    captured = {}

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

        async def execute(self, stmt):
            captured["stmt"] = stmt
            return MagicMock()

        async def commit(self):
            captured["committed"] = True

    monkeypatch.setattr(google_mod, "async_session", FakeSession)

    with respx.mock as mock:
        mock.post("https://oauth2.googleapis.com/token").respond(
            200,
            json={
                "access_token": "AT",
                "refresh_token": "RT",
                "expires_in": 3600,
                "scope": "https://x https://y",
            },
        )
        mock.get("https://openidconnect.googleapis.com/v1/userinfo").respond(
            200, json={"sub": "user-1", "email": "a@x.com", "name": "A"}
        )

        app = _build_app()
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://t", follow_redirects=False
        ) as c:
            r = await c.get("/oauth/google/callback", params={"code": "x", "state": state})

    assert r.status_code in (302, 307)
    assert "google=connected" in r.headers["location"]
    assert captured.get("committed") is True


@pytest.mark.asyncio
async def test_callback_handles_userinfo_failure_gracefully(monkeypatch) -> None:
    state = encode_state("tenant-1")

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

        async def execute(self, _):
            return MagicMock()

        async def commit(self):
            return None

    monkeypatch.setattr(google_mod, "async_session", FakeSession)

    with respx.mock as mock:
        mock.post("https://oauth2.googleapis.com/token").respond(
            200, json={"access_token": "AT", "expires_in": 3600}
        )
        mock.get("https://openidconnect.googleapis.com/v1/userinfo").respond(500)

        app = _build_app()
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://t", follow_redirects=False
        ) as c:
            r = await c.get("/oauth/google/callback", params={"code": "x", "state": state})

    assert r.status_code in (302, 307)  # still redirects to admin
