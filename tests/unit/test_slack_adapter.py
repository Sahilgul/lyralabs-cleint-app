"""lyra_core.channels.slack.adapter."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lyra_core.channels.slack import adapter as adapter_mod
from lyra_core.channels.slack.adapter import (
    _disable_workspace,
    _enqueue_from_event,
    build_slack_app,
)


def test_build_slack_app_falls_back_when_creds_missing() -> None:
    """No env -> stub app, no crash."""
    app, handler = build_slack_app()
    assert app is not None
    assert handler is not None


def test_build_slack_app_with_oauth(monkeypatch) -> None:
    from lyra_core.common import config as cfg

    s = cfg.get_settings()
    monkeypatch.setattr(s, "slack_client_id", "1.2", raising=False)
    monkeypatch.setattr(s, "slack_client_secret", "secret", raising=False)
    monkeypatch.setattr(s, "slack_signing_secret", "sign", raising=False)
    monkeypatch.setattr(s, "slack_install_redirect_url", "http://x/cb", raising=False)

    app, handler = build_slack_app()
    assert app is not None
    assert handler is not None


@pytest.mark.asyncio
async def test_enqueue_from_event_dispatches_to_celery(monkeypatch) -> None:
    """Slack message events are normalized + sent to run_agent.delay."""
    fake_run_agent = MagicMock()
    fake_module = MagicMock()
    fake_module.run_agent = fake_run_agent

    with patch.dict("sys.modules", {"apps.worker.tasks.run_agent": fake_module}):
        body = {
            "team_id": "T-XYZ",
            "event": {
                "type": "message",
                "channel": "C123",
                "thread_ts": "1234.5678",
                "user": "U1",
                "text": "hello bot",
                "ts": "1234.5678",
            },
        }
        await _enqueue_from_event(body)

    assert fake_run_agent.delay.called
    payload = fake_run_agent.delay.call_args.kwargs["message_json"]
    assert "T-XYZ" in payload
    assert "hello bot" in payload


@pytest.mark.asyncio
async def test_enqueue_skips_empty_text(monkeypatch) -> None:
    fake_run_agent = MagicMock()
    fake_module = MagicMock()
    fake_module.run_agent = fake_run_agent

    with patch.dict("sys.modules", {"apps.worker.tasks.run_agent": fake_module}):
        await _enqueue_from_event(
            {
                "team_id": "T1",
                "event": {"channel": "C", "user": "U", "text": "   ", "ts": "1"},
            }
        )

    assert not fake_run_agent.delay.called


@pytest.mark.asyncio
async def test_enqueue_falls_back_thread_ts_to_ts(monkeypatch) -> None:
    fake_run_agent = MagicMock()
    fake_module = MagicMock()
    fake_module.run_agent = fake_run_agent

    with patch.dict("sys.modules", {"apps.worker.tasks.run_agent": fake_module}):
        await _enqueue_from_event(
            {
                "team_id": "T1",
                "event": {
                    "channel": "C",
                    "user": "U",
                    "text": "hello",
                    "ts": "9999.0000",
                    # no thread_ts -> should use ts
                },
            }
        )

    payload = fake_run_agent.delay.call_args.kwargs["message_json"]
    assert "9999.0000" in payload


@pytest.mark.asyncio
async def test_disable_workspace_marks_tenant_cancelled(monkeypatch) -> None:
    """`_disable_workspace` must clear bot tokens + cancel the tenant."""
    from lyra_core.db.models import SlackInstallation, Tenant

    tenant = Tenant(external_team_id="T1", channel="slack", name="Acme")
    tenant.id = "tenant-uuid"
    tenant.status = "active"

    captured_updates = []

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

        async def execute(self, stmt):
            # First call: SELECT Tenant; second call: UPDATE SlackInstallation
            from sqlalchemy.sql.dml import Update
            from sqlalchemy.sql.selectable import Select

            r = MagicMock()
            if isinstance(stmt, Select):
                r.scalar_one_or_none.return_value = tenant
            elif isinstance(stmt, Update):
                captured_updates.append(stmt)
            return r

        async def commit(self):
            return None

    # _disable_workspace imports inside function
    import lyra_core.db.session as session_mod

    monkeypatch.setattr(session_mod, "async_session", FakeSession)

    await _disable_workspace("T1")

    assert tenant.status == "cancelled"
    assert len(captured_updates) == 1


@pytest.mark.asyncio
async def test_disable_workspace_noop_for_blank_team_id() -> None:
    # Should not raise and not perform any DB work
    await _disable_workspace("")
    await _disable_workspace(None)


@pytest.mark.asyncio
async def test_disable_workspace_noop_for_unknown_tenant(monkeypatch) -> None:
    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

        async def execute(self, _stmt):
            r = MagicMock()
            r.scalar_one_or_none.return_value = None
            return r

        async def commit(self):
            return None

    import lyra_core.db.session as session_mod

    monkeypatch.setattr(session_mod, "async_session", FakeSession)
    await _disable_workspace("T-unknown")  # no exception
