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
    from lyra_core.tools.slack import (
        _client,
        bookmarks,
        canvas,
        chat,
        conversations,
        files,
        pins,
        # reactions,  # disabled 2026-05
        reminders,
        search,
        users,
    )

    async def fake_bot(_):
        if bot is None:
            raise SlackTokenMissing("no bot")
        return bot

    async def fake_user(_):
        if user is None:
            raise SlackTokenMissing("no user")
        return user

    mods = (
        _client,
        bookmarks,
        canvas,
        chat,
        conversations,
        files,
        pins,
        # reactions,  # disabled 2026-05
        reminders,
        search,
        users,
    )
    for mod in mods:
        monkeypatch.setattr(mod, "_bot_token_for", fake_bot, raising=False)
        monkeypatch.setattr(mod, "_user_token_for", fake_user, raising=False)


# Canonical list — update this set when adding/removing Slack tools.
EXPECTED_SLACK_TOOLS = {
    # Communication
    "slack.chat.send_message",
    "slack.chat.schedule_message",
    # Threading
    "slack.conversations.history",
    "slack.conversations.replies",
    # Discovery / membership
    "slack.conversations.list",
    "slack.conversations.info",
    "slack.conversations.open",
    "slack.conversations.invite",
    "slack.conversations.create",
    # Users
    "slack.users.info",
    "slack.users.list",
    "slack.users.lookup_by_email",
    # Search
    "slack.search.messages",
    "slack.search.files",
    # Reactions — disabled 2026-05 (speak, don't react)
    # "slack.reactions.add",
    # "slack.reactions.remove",
    # Curation
    "slack.pins.add",
    "slack.bookmarks.add",
    # Files
    "slack.files.upload",
    # Canvas
    "slack.canvas.create",
    "slack.canvas.update",
    "slack.canvas.read",
    # Reminders
    "slack.reminders.add",
}

EXPECTED_WRITE_TOOLS = {
    "slack.chat.send_message",
    "slack.chat.schedule_message",
    "slack.conversations.invite",
    "slack.conversations.create",
    "slack.canvas.create",
    "slack.canvas.update",
    "slack.files.upload",
}


def test_all_slack_tools_registered() -> None:
    names = {t.name for t in default_registry.all() if t.provider == "slack"}
    assert names == EXPECTED_SLACK_TOOLS


def test_write_tools_match_permissive_policy() -> None:
    """Reactions, pins, bookmarks, reminders are LOW (no approval).
    Only message-sending, channel creation/invite, canvas writes,
    and file uploads gate through submit_plan_for_approval."""
    write_names = {
        t.name for t in default_registry.all() if t.provider == "slack" and t.requires_approval
    }
    assert write_names == EXPECTED_WRITE_TOOLS


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


# -----------------------------------------------------------------------------
# Phase 1: chat / reactions / users.lookup_by_email / conversations.open
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_send_message_posts_and_resolves_permalink(monkeypatch) -> None:
    _patch_token(monkeypatch)
    fake_post = MagicMock()
    fake_post.data = {"ts": "1717000000.000100", "ok": True}
    fake_link = MagicMock()
    fake_link.data = {"permalink": "https://slack/x/p100"}

    fake_client = MagicMock()
    fake_client.chat_postMessage = AsyncMock(return_value=fake_post)
    fake_client.chat_getPermalink = AsyncMock(return_value=fake_link)

    with patch("lyra_core.tools.slack.chat.AsyncWebClient", return_value=fake_client):
        tool = default_registry.get("slack.chat.send_message")
        out = await tool.run(
            _ctx(),
            tool.Input(channel_id="C-1", text="hi", thread_ts="T-1"),
        )

    fake_client.chat_postMessage.assert_awaited_once()
    kw = fake_client.chat_postMessage.await_args.kwargs
    assert kw["channel"] == "C-1"
    assert kw["thread_ts"] == "T-1"
    assert out.ts == "1717000000.000100"
    assert out.permalink == "https://slack/x/p100"


@pytest.mark.asyncio
async def test_chat_send_message_swallows_permalink_failure(monkeypatch) -> None:
    """A permalink lookup failure must not break the send."""
    from slack_sdk.errors import SlackApiError

    _patch_token(monkeypatch)
    fake_post = MagicMock()
    fake_post.data = {"ts": "9.0"}

    fake_client = MagicMock()
    fake_client.chat_postMessage = AsyncMock(return_value=fake_post)
    err_resp = MagicMock()
    err_resp.data = {"error": "permission_denied"}
    fake_client.chat_getPermalink = AsyncMock(
        side_effect=SlackApiError("denied", response=err_resp)
    )

    with patch("lyra_core.tools.slack.chat.AsyncWebClient", return_value=fake_client):
        tool = default_registry.get("slack.chat.send_message")
        out = await tool.run(_ctx(), tool.Input(channel_id="C-1", text="hi"))

    assert out.ts == "9.0"
    assert out.permalink is None


@pytest.mark.asyncio
async def test_chat_schedule_message_passes_post_at(monkeypatch) -> None:
    _patch_token(monkeypatch)
    fake_resp = MagicMock()
    fake_resp.data = {"scheduled_message_id": "Q123", "post_at": 1800000000}

    fake_client = MagicMock()
    fake_client.chat_scheduleMessage = AsyncMock(return_value=fake_resp)

    with patch("lyra_core.tools.slack.chat.AsyncWebClient", return_value=fake_client):
        tool = default_registry.get("slack.chat.schedule_message")
        out = await tool.run(
            _ctx(),
            tool.Input(channel_id="C-1", text="reminder", post_at=1800000000),
        )

    kw = fake_client.chat_scheduleMessage.await_args.kwargs
    assert kw["post_at"] == 1800000000
    assert out.scheduled_message_id == "Q123"


# Reactions tests disabled 2026-05 (feature commented out — speak, don't react).
# To re-enable: uncomment reactions registration in reactions.py + __init__.py,
# then uncomment these tests.
#
# @pytest.mark.asyncio
# async def test_reactions_add_swallows_already_reacted(monkeypatch) -> None:
#     from slack_sdk.errors import SlackApiError
#     _patch_token(monkeypatch)
#     err_resp = MagicMock()
#     err_resp.data = {"error": "already_reacted"}
#     fake_client = MagicMock()
#     fake_client.reactions_add = AsyncMock(side_effect=SlackApiError("dup", response=err_resp))
#     with patch("lyra_core.tools.slack.reactions.AsyncWebClient", return_value=fake_client):
#         tool = default_registry.get("slack.reactions.add")
#         out = await tool.run(_ctx(), tool.Input(channel_id="C-1", timestamp="1.0", name="eyes"))
#     assert out.ok is True
#
# @pytest.mark.asyncio
# async def test_reactions_add_surfaces_real_errors(monkeypatch) -> None:
#     from slack_sdk.errors import SlackApiError
#     _patch_token(monkeypatch)
#     err_resp = MagicMock()
#     err_resp.data = {"error": "channel_not_found"}
#     fake_client = MagicMock()
#     fake_client.reactions_add = AsyncMock(side_effect=SlackApiError("nope", response=err_resp))
#     with patch("lyra_core.tools.slack.reactions.AsyncWebClient", return_value=fake_client):
#         tool = default_registry.get("slack.reactions.add")
#         res = await tool.safe_run(_ctx(), tool.Input(channel_id="C-1", timestamp="1.0", name="eyes"))
#     assert res.ok is False
#     assert "channel_not_found" in (res.error or "")


@pytest.mark.asyncio
async def test_users_lookup_by_email_returns_found_false_on_missing(monkeypatch) -> None:
    """Slack returns `users_not_found` for unknown emails. The tool must
    surface this as a clean `found=False` result, not a tool error, so
    the planner can branch on it."""
    from slack_sdk.errors import SlackApiError

    _patch_token(monkeypatch)
    err_resp = MagicMock()
    err_resp.data = {"error": "users_not_found"}
    fake_client = MagicMock()
    fake_client.users_lookupByEmail = AsyncMock(side_effect=SlackApiError("nf", response=err_resp))

    with patch("lyra_core.tools.slack.users.AsyncWebClient", return_value=fake_client):
        tool = default_registry.get("slack.users.lookup_by_email")
        out = await tool.run(_ctx(), tool.Input(email="ghost@nowhere.com"))

    assert out.found is False
    assert out.user is None


@pytest.mark.asyncio
async def test_users_lookup_by_email_returns_user_on_match(monkeypatch) -> None:
    _patch_token(monkeypatch)
    fake_resp = MagicMock()
    fake_resp.data = {
        "user": {
            "id": "U99",
            "name": "alice",
            "real_name": "Alice A",
            "profile": {"email": "alice@x.com", "display_name": "alice"},
        }
    }
    fake_client = MagicMock()
    fake_client.users_lookupByEmail = AsyncMock(return_value=fake_resp)

    with patch("lyra_core.tools.slack.users.AsyncWebClient", return_value=fake_client):
        tool = default_registry.get("slack.users.lookup_by_email")
        out = await tool.run(_ctx(), tool.Input(email="alice@x.com"))

    assert out.found is True
    assert out.user is not None
    assert out.user.id == "U99"
    assert out.user.email == "alice@x.com"


@pytest.mark.asyncio
async def test_conversations_open_returns_channel_id(monkeypatch) -> None:
    _patch_token(monkeypatch)
    fake_resp = MagicMock()
    fake_resp.data = {"channel": {"id": "D-NEW"}, "already_open": False}
    fake_client = MagicMock()
    fake_client.conversations_open = AsyncMock(return_value=fake_resp)

    with patch("lyra_core.tools.slack.conversations.AsyncWebClient", return_value=fake_client):
        tool = default_registry.get("slack.conversations.open")
        out = await tool.run(_ctx(), tool.Input(user_ids=["U1", "U2"]))

    kw = fake_client.conversations_open.await_args.kwargs
    assert kw["users"] == "U1,U2"
    assert out.channel_id == "D-NEW"
    assert out.is_new is True


# -----------------------------------------------------------------------------
# Phase 2: conversations.list/info, canvas.update/read, search.files
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_conversations_list_filters_by_name_clientside(monkeypatch) -> None:
    _patch_token(monkeypatch)
    fake_resp = MagicMock()
    fake_resp.data = {
        "channels": [
            {"id": "C1", "name": "general", "is_channel": True},
            {"id": "C2", "name": "design-system", "is_channel": True},
            {"id": "C3", "name": "design-reviews", "is_channel": True},
        ],
        "response_metadata": {"next_cursor": ""},
    }
    fake_client = MagicMock()
    fake_client.conversations_list = AsyncMock(return_value=fake_resp)

    with patch("lyra_core.tools.slack.conversations.AsyncWebClient", return_value=fake_client):
        tool = default_registry.get("slack.conversations.list")
        out = await tool.run(_ctx(), tool.Input(name_filter="design"))

    assert {c.id for c in out.channels} == {"C2", "C3"}
    assert out.next_cursor is None


@pytest.mark.asyncio
async def test_conversations_info_returns_topic_purpose(monkeypatch) -> None:
    _patch_token(monkeypatch)
    fake_resp = MagicMock()
    fake_resp.data = {
        "channel": {
            "id": "C1",
            "name": "design",
            "is_private": True,
            "topic": {"value": "Where pixels live"},
            "purpose": {"value": "Design crit + reviews"},
            "num_members": 12,
        }
    }
    fake_client = MagicMock()
    fake_client.conversations_info = AsyncMock(return_value=fake_resp)

    with patch("lyra_core.tools.slack.conversations.AsyncWebClient", return_value=fake_client):
        tool = default_registry.get("slack.conversations.info")
        out = await tool.run(_ctx(), tool.Input(channel_id="C1"))

    assert out.channel.topic == "Where pixels live"
    assert out.channel.purpose == "Design crit + reviews"
    assert out.channel.is_private is True
    assert out.channel.num_members == 12


@pytest.mark.asyncio
async def test_canvas_update_sends_correct_change_payload(monkeypatch) -> None:
    _patch_token(monkeypatch)
    fake_client = MagicMock()
    fake_client.canvases_edit = AsyncMock(return_value=MagicMock(data={"ok": True}))

    with patch("lyra_core.tools.slack.canvas.AsyncWebClient", return_value=fake_client):
        tool = default_registry.get("slack.canvas.update")
        out = await tool.run(
            _ctx(),
            tool.Input(canvas_id="F1", operation="insert_at_end", markdown="## Notes"),
        )

    kw = fake_client.canvases_edit.await_args.kwargs
    assert kw["canvas_id"] == "F1"
    changes = kw["changes"]
    assert len(changes) == 1
    assert changes[0]["operation"] == "insert_at_end"
    assert changes[0]["document_content"]["markdown"] == "## Notes"
    assert out.ok is True


@pytest.mark.asyncio
async def test_search_files_uses_user_token(monkeypatch) -> None:
    _patch_token(monkeypatch)
    fake_resp = MagicMock()
    fake_resp.data = {
        "files": {
            "total": 1,
            "matches": [
                {
                    "id": "F1",
                    "name": "deck.pdf",
                    "title": "Q3 Deck",
                    "filetype": "pdf",
                    "user": "U1",
                    "permalink": "https://x/F1",
                }
            ],
        }
    }
    fake_client = MagicMock()
    fake_client.search_files = AsyncMock(return_value=fake_resp)

    with patch("lyra_core.tools.slack.search.AsyncWebClient", return_value=fake_client) as ctor:
        tool = default_registry.get("slack.search.files")
        out = await tool.run(_ctx(), tool.Input(query="filetype:pdf deck"))

    ctor.assert_called_once_with(token="xoxp-test")
    assert out.total == 1
    assert out.matches[0].filetype == "pdf"
    assert out.matches[0].permalink == "https://x/F1"


@pytest.mark.asyncio
async def test_search_files_missing_user_token_returns_clean_error(monkeypatch) -> None:
    _patch_token(monkeypatch, user=None)
    tool = default_registry.get("slack.search.files")
    res = await tool.safe_run(_ctx(), tool.Input(query="x"))
    assert res.ok is False
    assert "no user" in (res.error or "")


# -----------------------------------------------------------------------------
# Phase 3: files.upload, pins, bookmarks, conversations.invite/create, reminders
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_files_upload_decodes_base64_and_calls_v2(monkeypatch) -> None:
    import base64

    _patch_token(monkeypatch)
    fake_resp = MagicMock()
    fake_resp.data = {"file": {"id": "F123", "permalink": "https://x/F123"}}
    fake_client = MagicMock()
    fake_client.files_upload_v2 = AsyncMock(return_value=fake_resp)

    payload = b"%PDF-1.4 hello"
    with patch("lyra_core.tools.slack.files.AsyncWebClient", return_value=fake_client):
        tool = default_registry.get("slack.files.upload")
        out = await tool.run(
            _ctx(),
            tool.Input(
                channel_id="C-1",
                filename="r.pdf",
                content_b64=base64.b64encode(payload).decode(),
                title="Report",
            ),
        )

    kw = fake_client.files_upload_v2.await_args.kwargs
    assert kw["filename"] == "r.pdf"
    assert kw["content"] == payload
    assert kw["title"] == "Report"
    assert out.file_id == "F123"


@pytest.mark.asyncio
async def test_files_upload_rejects_invalid_base64(monkeypatch) -> None:
    _patch_token(monkeypatch)
    tool = default_registry.get("slack.files.upload")
    res = await tool.safe_run(
        _ctx(),
        tool.Input(channel_id="C-1", filename="x.bin", content_b64="not!!!base64"),
    )
    assert res.ok is False
    assert "base64" in (res.error or "")


@pytest.mark.asyncio
async def test_pins_add_swallows_already_pinned(monkeypatch) -> None:
    from slack_sdk.errors import SlackApiError

    _patch_token(monkeypatch)
    err_resp = MagicMock()
    err_resp.data = {"error": "already_pinned"}
    fake_client = MagicMock()
    fake_client.pins_add = AsyncMock(side_effect=SlackApiError("dup", response=err_resp))

    with patch("lyra_core.tools.slack.pins.AsyncWebClient", return_value=fake_client):
        tool = default_registry.get("slack.pins.add")
        out = await tool.run(_ctx(), tool.Input(channel_id="C-1", timestamp="1.0"))

    assert out.ok is True


@pytest.mark.asyncio
async def test_bookmarks_add_returns_bookmark_id(monkeypatch) -> None:
    _patch_token(monkeypatch)
    fake_resp = MagicMock()
    fake_resp.data = {"bookmark": {"id": "Bk1", "channel_id": "C-1"}}
    fake_client = MagicMock()
    fake_client.bookmarks_add = AsyncMock(return_value=fake_resp)

    with patch("lyra_core.tools.slack.bookmarks.AsyncWebClient", return_value=fake_client):
        tool = default_registry.get("slack.bookmarks.add")
        out = await tool.run(
            _ctx(),
            tool.Input(channel_id="C-1", title="Spec", link="https://docs/x"),
        )

    kw = fake_client.bookmarks_add.await_args.kwargs
    assert kw["type"] == "link"
    assert kw["link"] == "https://docs/x"
    assert out.bookmark_id == "Bk1"


@pytest.mark.asyncio
async def test_conversations_invite_passes_csv_user_ids(monkeypatch) -> None:
    _patch_token(monkeypatch)
    fake_client = MagicMock()
    fake_client.conversations_invite = AsyncMock(return_value=MagicMock(data={"ok": True}))

    with patch("lyra_core.tools.slack.conversations.AsyncWebClient", return_value=fake_client):
        tool = default_registry.get("slack.conversations.invite")
        out = await tool.run(_ctx(), tool.Input(channel_id="C-1", user_ids=["U1", "U2", "U3"]))

    kw = fake_client.conversations_invite.await_args.kwargs
    assert kw["users"] == "U1,U2,U3"
    assert out.invited == ["U1", "U2", "U3"]


@pytest.mark.asyncio
async def test_conversations_create_returns_new_channel(monkeypatch) -> None:
    _patch_token(monkeypatch)
    fake_resp = MagicMock()
    fake_resp.data = {
        "channel": {
            "id": "C-NEW",
            "name": "project-atlas",
            "is_channel": True,
            "is_private": False,
        }
    }
    fake_client = MagicMock()
    fake_client.conversations_create = AsyncMock(return_value=fake_resp)

    with patch("lyra_core.tools.slack.conversations.AsyncWebClient", return_value=fake_client):
        tool = default_registry.get("slack.conversations.create")
        out = await tool.run(_ctx(), tool.Input(name="project-atlas", is_private=False))

    assert out.channel.id == "C-NEW"
    assert out.channel.name == "project-atlas"
    assert out.channel.is_private is False


@pytest.mark.asyncio
async def test_reminders_add_returns_reminder_id(monkeypatch) -> None:
    _patch_token(monkeypatch)
    fake_resp = MagicMock()
    fake_resp.data = {"reminder": {"id": "Rm1"}}
    fake_client = MagicMock()
    fake_client.reminders_add = AsyncMock(return_value=fake_resp)

    with patch("lyra_core.tools.slack.reminders.AsyncWebClient", return_value=fake_client):
        tool = default_registry.get("slack.reminders.add")
        out = await tool.run(
            _ctx(),
            tool.Input(text="ship the PR", time="tomorrow at 9am", user_id="U1"),
        )

    kw = fake_client.reminders_add.await_args.kwargs
    assert kw["text"] == "ship the PR"
    assert kw["user"] == "U1"
    assert out.reminder_id == "Rm1"
