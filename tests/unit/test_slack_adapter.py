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
async def test_enqueue_from_event_dispatches_to_arq(monkeypatch) -> None:
    """Slack message events are normalized + sent to enqueue_run_agent."""
    fake_enqueue = AsyncMock()
    with patch("lyra_core.worker.queue.enqueue_run_agent", fake_enqueue):
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

    fake_enqueue.assert_awaited_once()
    payload = fake_enqueue.call_args.args[0]
    assert "T-XYZ" in payload
    assert "hello bot" in payload


@pytest.mark.asyncio
async def test_enqueue_skips_empty_text(monkeypatch) -> None:
    fake_enqueue = AsyncMock()
    with patch("lyra_core.worker.queue.enqueue_run_agent", fake_enqueue):
        await _enqueue_from_event(
            {
                "team_id": "T1",
                "event": {"channel": "C", "user": "U", "text": "   ", "ts": "1"},
            }
        )

    fake_enqueue.assert_not_awaited()


@pytest.mark.asyncio
async def test_enqueue_falls_back_thread_ts_to_ts(monkeypatch) -> None:
    fake_enqueue = AsyncMock()
    with patch("lyra_core.worker.queue.enqueue_run_agent", fake_enqueue):
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

    payload = fake_enqueue.call_args.args[0]
    assert "9999.0000" in payload


@pytest.mark.asyncio
async def test_enqueue_dm_top_level_does_not_thread_reply() -> None:
    """User DMs ARLO with a fresh top-level message: bot must reply in the
    main DM (reply_thread_ts=None), not as a thread on the user's message.
    Threading every DM reply is what made every bot response appear nested
    under the user's first message.
    """
    import json

    fake_enqueue = AsyncMock()
    with patch("lyra_core.worker.queue.enqueue_run_agent", fake_enqueue):
        await _enqueue_from_event(
            {
                "team_id": "T1",
                "event": {
                    "type": "message",
                    "channel": "D-dm",
                    "channel_type": "im",
                    "user": "U",
                    "text": "hello",
                    "ts": "9999.0000",
                },
            }
        )

    payload = json.loads(fake_enqueue.call_args.args[0])
    assert payload["is_dm"] is True
    assert payload["reply_thread_ts"] is None
    assert payload["thread_id"] == "9999.0000"


@pytest.mark.asyncio
async def test_enqueue_dm_inside_thread_keeps_thread_reply() -> None:
    """If the user explicitly threaded their DM reply, mirror that and
    reply in the same thread.
    """
    import json

    fake_enqueue = AsyncMock()
    with patch("lyra_core.worker.queue.enqueue_run_agent", fake_enqueue):
        await _enqueue_from_event(
            {
                "team_id": "T1",
                "event": {
                    "type": "message",
                    "channel": "D-dm",
                    "channel_type": "im",
                    "thread_ts": "1111.1111",
                    "user": "U",
                    "text": "follow-up in thread",
                    "ts": "2222.2222",
                },
            }
        )

    payload = json.loads(fake_enqueue.call_args.args[0])
    assert payload["is_dm"] is True
    assert payload["reply_thread_ts"] == "1111.1111"


@pytest.mark.asyncio
async def test_enqueue_dm_uses_continuous_agent_thread_id() -> None:
    """DM top-level messages must share one agent_thread_id per (team, channel,
    user), so the LangGraph checkpointer keeps memory across messages.
    Regression for: ARLO forgetting the user's name between top-level DMs.
    """
    import json

    fake_enqueue = AsyncMock()
    payloads = []
    with patch("lyra_core.worker.queue.enqueue_run_agent", fake_enqueue):
        for ts in ("1.0", "2.0", "3.0"):
            await _enqueue_from_event(
                {
                    "team_id": "T1",
                    "event": {
                        "type": "message",
                        "channel": "D-dm",
                        "channel_type": "im",
                        "user": "U-sahil",
                        "text": f"msg @ {ts}",
                        "ts": ts,
                    },
                }
            )
            payloads.append(json.loads(fake_enqueue.call_args.args[0]))

    keys = {p["agent_thread_id"] for p in payloads}
    assert keys == {"slack:dm:T1:D-dm:U-sahil"}, (
        f"All three DMs should share one agent thread; got {keys}"
    )
    # And the Slack-side `thread_id` still tracks per-message ts so existing
    # consumers (audit, status indicator) keep working.
    assert [p["thread_id"] for p in payloads] == ["1.0", "2.0", "3.0"]


@pytest.mark.asyncio
async def test_enqueue_channel_mention_scopes_agent_thread_to_slack_thread() -> None:
    """Channel @-mentions must scope agent memory per Slack thread, so two
    mentions in the same channel don't bleed context.
    """
    import json

    fake_enqueue = AsyncMock()
    with patch("lyra_core.worker.queue.enqueue_run_agent", fake_enqueue):
        # First mention: brand-new top-level message in #general.
        await _enqueue_from_event(
            {
                "team_id": "T1",
                "event": {
                    "type": "app_mention",
                    "channel": "C-general",
                    "channel_type": "channel",
                    "user": "U1",
                    "text": "<@bot> status",
                    "ts": "100.0",
                },
            }
        )
        first = json.loads(fake_enqueue.call_args.args[0])
        # Second mention: a reply inside an existing thread (different thread_ts).
        await _enqueue_from_event(
            {
                "team_id": "T1",
                "event": {
                    "type": "message",
                    "channel": "C-general",
                    "channel_type": "channel",
                    "thread_ts": "999.0",
                    "user": "U1",
                    "text": "<@bot> follow up",
                    "ts": "1000.0",
                },
            }
        )
        second = json.loads(fake_enqueue.call_args.args[0])

    assert first["agent_thread_id"] == "slack:ch:T1:C-general:100.0"
    assert second["agent_thread_id"] == "slack:ch:T1:C-general:999.0"
    assert first["agent_thread_id"] != second["agent_thread_id"]


@pytest.mark.asyncio
async def test_enqueue_channel_mention_threads_on_user_message() -> None:
    """Channel @-mentions should still get threaded responses so the
    channel doesn't get noisy with bot replies.
    """
    import json

    fake_enqueue = AsyncMock()
    with patch("lyra_core.worker.queue.enqueue_run_agent", fake_enqueue):
        await _enqueue_from_event(
            {
                "team_id": "T1",
                "event": {
                    "type": "app_mention",
                    "channel": "C-general",
                    "channel_type": "channel",
                    "user": "U",
                    "text": "<@bot> status",
                    "ts": "3333.3333",
                },
            }
        )

    payload = json.loads(fake_enqueue.call_args.args[0])
    assert payload["is_dm"] is False
    assert payload["reply_thread_ts"] == "3333.3333"


@pytest.mark.asyncio
async def test_enqueue_dm_with_client_fires_assistant_status() -> None:
    """DM events with a client get a native 'Thinking…' status indicator."""
    fake_enqueue = AsyncMock()
    client = MagicMock()
    client.assistant_threads_setStatus = AsyncMock()
    client.reactions_add = AsyncMock()

    with patch("lyra_core.worker.queue.enqueue_run_agent", fake_enqueue):
        await _enqueue_from_event(
            {
                "team_id": "T1",
                "event": {
                    "type": "message",
                    "channel": "D-dm",
                    "channel_type": "im",
                    "user": "U",
                    "text": "hello",
                    "ts": "9999.0000",
                },
            },
            client,
        )

    client.assistant_threads_setStatus.assert_awaited_once()
    kwargs = client.assistant_threads_setStatus.await_args.kwargs
    assert kwargs["channel_id"] == "D-dm"
    assert kwargs["thread_ts"] == "9999.0000"
    assert kwargs["status"] == "Thinking…"
    client.reactions_add.assert_not_called()
    fake_enqueue.assert_awaited_once()  # still enqueues


@pytest.mark.asyncio
async def test_enqueue_channel_with_client_fires_reaction() -> None:
    """Channel @-mentions fall back to an :eyes: reaction since
    assistant.threads.setStatus only works in assistant threads."""
    fake_enqueue = AsyncMock()
    client = MagicMock()
    client.assistant_threads_setStatus = AsyncMock()
    client.reactions_add = AsyncMock()

    with patch("lyra_core.worker.queue.enqueue_run_agent", fake_enqueue):
        await _enqueue_from_event(
            {
                "team_id": "T1",
                "event": {
                    "type": "app_mention",
                    "channel": "C-general",
                    "channel_type": "channel",
                    "user": "U",
                    "text": "<@bot> status",
                    "ts": "3333.3333",
                },
            },
            client,
        )

    client.reactions_add.assert_awaited_once()
    kwargs = client.reactions_add.await_args.kwargs
    assert kwargs["channel"] == "C-general"
    assert kwargs["timestamp"] == "3333.3333"
    assert kwargs["name"] == "eyes"
    client.assistant_threads_setStatus.assert_not_called()


@pytest.mark.asyncio
async def test_enqueue_indicator_failure_is_swallowed() -> None:
    """If Slack rejects setStatus (e.g. app missing assistant scope) we
    still enqueue the agent task -- the indicator is best-effort feedback."""
    from slack_sdk.errors import SlackApiError

    fake_enqueue = AsyncMock()
    fake_response = MagicMock()
    fake_response.data = {"error": "missing_scope"}
    client = MagicMock()
    client.assistant_threads_setStatus = AsyncMock(
        side_effect=SlackApiError("nope", fake_response)
    )

    with patch("lyra_core.worker.queue.enqueue_run_agent", fake_enqueue):
        await _enqueue_from_event(
            {
                "team_id": "T1",
                "event": {
                    "type": "message",
                    "channel": "D-dm",
                    "channel_type": "im",
                    "user": "U",
                    "text": "hi",
                    "ts": "1.0",
                },
            },
            client,
        )

    fake_enqueue.assert_awaited_once()


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
