"""lyra_core.channels.slack.install_store.PostgresInstallationStore.

We mock the async_session context manager and exercise the AsyncInstallationStore
methods directly. slack_bolt's async OAuth flow calls `async_save` /
`async_find_bot` / `async_find_installation` — those are what we cover here.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from slack_sdk.oauth.installation_store import Installation

from lyra_core.channels.slack.install_store import PostgresInstallationStore
from lyra_core.db.models import SlackInstallation, Tenant


def _installation() -> Installation:
    return Installation(
        app_id="A1",
        enterprise_id=None,
        team_id="T123",
        team_name="Acme Inc",
        user_id="U1",
        bot_token="xoxb-test-bot",
        bot_id="B1",
        bot_user_id="UB1",
        bot_scopes=["chat:write", "channels:history"],
        bot_refresh_token="xoxe-1-refresh",
        bot_token_expires_at=int(datetime.now(UTC).timestamp()) + 3600,
        is_enterprise_install=False,
        token_type="bot",
    )


@pytest.mark.asyncio
async def test_ensure_tenant_creates_when_missing(monkeypatch) -> None:
    """First-time install: a Tenant row is created."""
    from lyra_core.channels.slack import install_store as mod

    captured: dict = {}

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

        async def execute(self, _stmt):
            r = MagicMock()
            r.scalar_one_or_none.return_value = None
            return r

        def add(self, obj):
            captured["added"] = obj

        async def commit(self):
            captured["committed"] = True
            captured["added"].id = "new-uuid-1"

        async def refresh(self, obj):
            obj.id = "new-uuid-1"

    monkeypatch.setattr(mod, "async_session", FakeSession)

    tenant_id = await PostgresInstallationStore._ensure_tenant("T999", "Acme")

    assert tenant_id == "new-uuid-1"
    assert isinstance(captured["added"], Tenant)
    assert captured["added"].external_team_id == "T999"
    assert captured["added"].channel == "slack"
    assert captured["added"].name == "Acme"


@pytest.mark.asyncio
async def test_ensure_tenant_returns_existing_id(monkeypatch) -> None:
    from lyra_core.channels.slack import install_store as mod

    existing = Tenant(
        external_team_id="T1", channel="slack", name="Acme", plan="trial", status="active"
    )
    existing.id = "existing-uuid"

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

        async def execute(self, _stmt):
            r = MagicMock()
            r.scalar_one_or_none.return_value = existing
            return r

        def add(self, obj):
            raise AssertionError("should not add when tenant exists")

        async def commit(self):
            return None

    monkeypatch.setattr(mod, "async_session", FakeSession)
    tid = await PostgresInstallationStore._ensure_tenant("T1", None)
    assert tid == "existing-uuid"


@pytest.mark.asyncio
async def test_async_save_encrypts_tokens(monkeypatch) -> None:
    """`async_save` should encrypt bot + refresh tokens before persisting."""
    from lyra_core.channels.slack import install_store as mod

    inserted: list[SlackInstallation] = []

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

        async def execute(self, _stmt):
            return MagicMock()

        def add(self, obj):
            inserted.append(obj)

        async def commit(self):
            return None

    monkeypatch.setattr(mod, "async_session", FakeSession)
    monkeypatch.setattr(
        PostgresInstallationStore,
        "_ensure_tenant",
        AsyncMock(return_value="tenant-1"),
    )

    store = PostgresInstallationStore()
    await store.async_save(_installation())

    assert len(inserted) == 1
    row = inserted[0]
    assert row.tenant_id == "tenant-1"
    assert row.team_id == "T123"
    assert row.team_name == "Acme Inc"
    # Confirm encryption happened (ciphertext is not the plaintext)
    assert row.bot_token_encrypted is not None
    assert "xoxb-test-bot" not in row.bot_token_encrypted
    assert row.bot_refresh_token_encrypted is not None
    assert "xoxe-1-refresh" not in row.bot_refresh_token_encrypted

    # Confirm we can decrypt back
    from lyra_core.common.crypto import decrypt_for_tenant

    assert decrypt_for_tenant("tenant-1", row.bot_token_encrypted) == "xoxb-test-bot"
    assert (
        decrypt_for_tenant("tenant-1", row.bot_refresh_token_encrypted) == "xoxe-1-refresh"
    )


@pytest.mark.asyncio
async def test_async_save_raises_without_team_id() -> None:
    store = PostgresInstallationStore()
    bad = Installation(
        app_id="A1",
        enterprise_id=None,
        team_id=None,
        user_id="U1",
        bot_token="x",
    )
    with pytest.raises(ValueError, match="team_id"):
        await store.async_save(bad)


@pytest.mark.asyncio
async def test_async_find_returns_none_when_missing(monkeypatch) -> None:
    from lyra_core.channels.slack import install_store as mod

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

        async def execute(self, _stmt):
            r = MagicMock()
            r.scalar_one_or_none.return_value = None
            return r

    monkeypatch.setattr(mod, "async_session", FakeSession)
    store = PostgresInstallationStore()
    row, tid = await store._async_find(team_id="T-missing", enterprise_id=None)
    assert row is None and tid is None


@pytest.mark.asyncio
async def test_async_find_returns_row(monkeypatch) -> None:
    from lyra_core.channels.slack import install_store as mod

    si = SlackInstallation(tenant_id="t-1", team_id="T1", bot_token_encrypted="ct")

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

        async def execute(self, _stmt):
            r = MagicMock()
            r.scalar_one_or_none.return_value = si
            return r

    monkeypatch.setattr(mod, "async_session", FakeSession)
    store = PostgresInstallationStore()
    row, tid = await store._async_find(team_id="T1", enterprise_id=None)
    assert row is si
    assert tid == "t-1"
