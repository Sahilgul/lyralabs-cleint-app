"""lyra_core.tools.slack -- registration + happy-path mocks."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Import side-effect: registers all slack tools on default_registry.
from lyra_core.tools import slack as _slack  # noqa: F401
from lyra_core.tools.base import ToolContext
from lyra_core.tools.registry import default_registry
from lyra_core.tools.slack._client import SlackTokenMissing


def _ctx() -> ToolContext:
    return ToolContext(tenant_id="tenant-1", job_id="j-1", user_id="U1")


def _patch_token(
    monkeypatch, *, bot: str | None = "xoxb-test", user: str | None = "xoxp-test"
) -> None:
    """Stub _bot_token_for / _user_token_for so we don't touch Postgres."""
    from lyra_core.tools.slack import _client, canvas, conversations, search, users

    async def fake_bot(_):
        if bot is None:
            raise SlackTokenMissing("no bot")
        return bot

    async def fake_user(_):
        if user is None:
            raise SlackTokenMissing("no user")
        return user

    for mod in (_client, conversations, users, search, canvas):
        monkeypatch.setattr(mod, "_bot_token_for", fake_bot, raising=False)
        monkeypatch.setattr(mod, "_user_token_for", fake_user, raising=False)


def test_all_six_slack_tools_registered() -> None:
    names = {t.name for t in default_registry.all() if t.provider == "slack"}
    assert names == {
        "slack.conversations.history",
        "slack.conversations.replies",
        "slack.users.info",
        "slack.users.list",
        "slack.search.messages",
        "slack.canvas.create",
    }


def test_canvas_is_only_write_tool_in_slack_namespace() -> None:
    write_slack = [
        t for t in default_registry.all() if t.provider == "slack" and t.requires_approval
    ]
    assert [t.name for t in write_slack] == ["slack.canvas.create"]


@pytest.mark.asyncio
async def test_conversations_history_returns_normalized_messages(monkeypatch) -> None:
    _patch_token(monkeypatch)
    fake_resp = MagicMock()
    fake_resp.data = {
        "messages": [
            {"user": "U1", "text": "hi", "ts": "1.0"},
            {"user": "U2", "text": "yo", "ts": "2.0", "thread_ts": "1.0"},
            {"bot_id": "B1", "text": "[bot]", "ts": "3.0", "subtype": "bot_message"},
        ],
        "has_more": True,
    }
    fake_client = MagicMock()
    fake_client.conversations_history = AsyncMock(return_value=fake_resp)

    with patch("lyra_core.tools.slack.conversations.AsyncWebClient", return_value=fake_client):
        tool = default_registry.get("slack.conversations.history")
        out = await tool.run(_ctx(), tool.Input(channel_id="C1", limit=20))

    assert out.channel_id == "C1"
    assert out.has_more is True
    assert len(out.messages) == 3
    assert out.messages[2].is_bot is True


@pytest.mark.asyncio
async def test_conversations_history_surfaces_token_missing_as_tool_error(
    monkeypatch,
) -> None:
    _patch_token(monkeypatch, bot=None)
    tool = default_registry.get("slack.conversations.history")
    res = await tool.safe_run(_ctx(), tool.Input(channel_id="C1"))
    assert res.ok is False
    assert "no bot" in (res.error or "")


@pytest.mark.asyncio
async def test_users_list_returns_paginated_members(monkeypatch) -> None:
    _patch_token(monkeypatch)
    fake_resp = MagicMock()
    fake_resp.data = {
        "members": [
            {
                "id": "U1",
                "name": "sahil",
                "real_name": "Muhammad Sahil",
                "profile": {"display_name": "sahil", "email": "s@x.com"},
            },
            {"id": "U2", "name": "alice", "deleted": True},
        ],
        "response_metadata": {"next_cursor": "ABC"},
    }
    fake_client = MagicMock()
    fake_client.users_list = AsyncMock(return_value=fake_resp)

    with patch("lyra_core.tools.slack.users.AsyncWebClient", return_value=fake_client):
        tool = default_registry.get("slack.users.list")
        out = await tool.run(_ctx(), tool.Input(limit=100))

    assert out.next_cursor == "ABC"
    assert [m.id for m in out.members] == ["U1", "U2"]
    assert out.members[0].email == "s@x.com"
    assert out.members[1].is_deleted is True


@pytest.mark.asyncio
async def test_search_messages_uses_user_token(monkeypatch) -> None:
    _patch_token(monkeypatch)
    fake_resp = MagicMock()
    fake_resp.data = {
        "messages": {
            "total": 2,
            "matches": [
                {
                    "channel": {"id": "C1", "name": "general"},
                    "user": "U1",
                    "username": "sahil",
                    "text": "decision A",
                    "ts": "1.0",
                    "permalink": "https://x/1",
                },
                {
                    "channel": {"id": "C1", "name": "general"},
                    "user": "U2",
                    "text": "decision B",
                    "ts": "2.0",
                },
            ],
        }
    }
    fake_client = MagicMock()
    fake_client.search_messages = AsyncMock(return_value=fake_resp)

    with patch("lyra_core.tools.slack.search.AsyncWebClient", return_value=fake_client) as ctor:
        tool = default_registry.get("slack.search.messages")
        out = await tool.run(_ctx(), tool.Input(query="decision", count=10))

    # Critical: must be constructed with the USER token, not the bot token.
    ctor.assert_called_once_with(token="xoxp-test")
    assert out.total == 2
    assert out.matches[0].permalink == "https://x/1"
    assert out.matches[1].permalink is None


@pytest.mark.asyncio
async def test_search_messages_missing_user_token_returns_clean_error(
    monkeypatch,
) -> None:
    _patch_token(monkeypatch, user=None)
    tool = default_registry.get("slack.search.messages")
    res = await tool.safe_run(_ctx(), tool.Input(query="x"))
    assert res.ok is False
    assert "no user" in (res.error or "")


@pytest.mark.asyncio
async def test_canvas_create_marked_as_write(monkeypatch) -> None:
    """Sanity: the model is told this is a write tool; the executor /
    tool_node block direct calls. We just confirm the metadata here."""
    tool = default_registry.get("slack.canvas.create")
    assert tool.requires_approval is True
