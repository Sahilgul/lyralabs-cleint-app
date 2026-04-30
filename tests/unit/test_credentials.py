"""lyra_core.tools.credentials."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import httpx
import pytest
import respx

from lyra_core.common.crypto import encrypt_for_tenant
from lyra_core.db.models import IntegrationConnection
from lyra_core.tools import credentials as creds_mod


def _make_connection(
    tenant_id: str = "t-1",
    provider: str = "google",
    expires_in_seconds: int | None = 3600,
    access: str = "access-XYZ",
    refresh: str | None = "refresh-XYZ",
) -> IntegrationConnection:
    expires = (
        datetime.now(UTC) + timedelta(seconds=expires_in_seconds)
        if expires_in_seconds is not None
        else None
    )
    row = IntegrationConnection(
        tenant_id=tenant_id,
        provider=provider,
        external_account_id="acct-1",
        scopes="scope.a scope.b",
        access_token_encrypted=encrypt_for_tenant(tenant_id, access),
        refresh_token_encrypted=encrypt_for_tenant(tenant_id, refresh) if refresh else None,
        expires_at=expires,
        metadata_={"location_id": "loc-1"},
        status="active",
    )
    row.id = "conn-1"
    return row


class _SessionWith:
    def __init__(self, row, captured: dict | None = None):
        self.row = row
        self.captured = captured if captured is not None else {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False

    async def execute(self, _stmt):
        r = MagicMock()
        r.scalar_one_or_none.return_value = self.row
        return r

    async def commit(self):
        self.captured["committed"] = True
        return None


@pytest.mark.asyncio
async def test_get_credentials_returns_decrypted_when_fresh(monkeypatch) -> None:
    row = _make_connection()
    monkeypatch.setattr(creds_mod, "async_session", lambda: _SessionWith(row))
    c = await creds_mod.get_credentials("t-1", "google")
    assert c.access_token == "access-XYZ"
    assert c.refresh_token == "refresh-XYZ"
    assert c.metadata == {"location_id": "loc-1"}
    assert c.scopes == "scope.a scope.b"


@pytest.mark.asyncio
async def test_get_credentials_raises_when_no_active(monkeypatch) -> None:
    monkeypatch.setattr(creds_mod, "async_session", lambda: _SessionWith(None))
    with pytest.raises(RuntimeError, match="No active 'google' integration"):
        await creds_mod.get_credentials("t-1", "google")


@pytest.mark.asyncio
async def test_get_credentials_refreshes_when_within_window(monkeypatch) -> None:
    """If expires_at is within 5 min, the token is refreshed and re-persisted."""
    # Row is about to expire (60s out)
    row = _make_connection(expires_in_seconds=60)
    captured = {}
    monkeypatch.setattr(creds_mod, "async_session", lambda: _SessionWith(row, captured))

    with respx.mock(assert_all_called=True) as mock:
        mock.post("https://oauth2.googleapis.com/token").respond(
            200,
            json={
                "access_token": "new-access-token",
                "refresh_token": "new-refresh-token",
                "expires_in": 3600,
                "token_type": "Bearer",
            },
        )
        c = await creds_mod.get_credentials("t-1", "google")

    assert c.access_token == "new-access-token"
    # Persisted (committed)
    assert captured.get("committed") is True
    # Underlying ciphertext was updated
    from lyra_core.common.crypto import decrypt_for_tenant

    assert decrypt_for_tenant("t-1", row.access_token_encrypted) == "new-access-token"


@pytest.mark.asyncio
async def test_get_credentials_refresh_keeps_old_refresh_when_google_omits(monkeypatch) -> None:
    row = _make_connection(expires_in_seconds=10)
    monkeypatch.setattr(creds_mod, "async_session", lambda: _SessionWith(row))

    with respx.mock as mock:
        mock.post("https://oauth2.googleapis.com/token").respond(
            200, json={"access_token": "AT2", "expires_in": 3600}
        )
        c = await creds_mod.get_credentials("t-1", "google")

    assert c.access_token == "AT2"
    # Refresh ciphertext should be unchanged (still original)
    from lyra_core.common.crypto import decrypt_for_tenant

    assert decrypt_for_tenant("t-1", row.refresh_token_encrypted) == "refresh-XYZ"


@pytest.mark.asyncio
async def test_get_credentials_raises_when_expired_no_refresh(monkeypatch) -> None:
    row = _make_connection(expires_in_seconds=10, refresh=None)
    monkeypatch.setattr(creds_mod, "async_session", lambda: _SessionWith(row))
    with pytest.raises(RuntimeError, match="no refresh token"):
        await creds_mod.get_credentials("t-1", "google")


@pytest.mark.asyncio
async def test_refresh_ghl(monkeypatch) -> None:
    row = _make_connection(provider="ghl", expires_in_seconds=10)
    monkeypatch.setattr(creds_mod, "async_session", lambda: _SessionWith(row))

    with respx.mock as mock:
        mock.post("https://services.leadconnectorhq.com/oauth/token").respond(
            200, json={"access_token": "ghl-AT2", "refresh_token": "ghl-RT2", "expires_in": 86400}
        )
        c = await creds_mod.get_credentials("t-1", "ghl")

    assert c.access_token == "ghl-AT2"


@pytest.mark.asyncio
async def test_refresh_unsupported_provider_raises() -> None:
    with pytest.raises(ValueError, match="refresh not implemented"):
        await creds_mod._refresh_token(provider="snowflake", refresh_token="x")


@pytest.mark.asyncio
async def test_refresh_google_failure_raises_runtime(monkeypatch) -> None:
    with respx.mock as mock:
        mock.post("https://oauth2.googleapis.com/token").respond(400, json={"err": "bad"})
        with pytest.raises(RuntimeError, match="google token refresh failed"):
            await creds_mod._refresh_token(provider="google", refresh_token="x")


@pytest.mark.asyncio
async def test_refresh_ghl_failure_raises_runtime() -> None:
    with respx.mock as mock:
        mock.post("https://services.leadconnectorhq.com/oauth/token").respond(401, json={})
        with pytest.raises(RuntimeError, match="ghl token refresh failed"):
            await creds_mod._refresh_token(provider="ghl", refresh_token="x")
