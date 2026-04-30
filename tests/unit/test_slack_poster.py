"""lyra_core.channels.slack.poster."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from lyra_core.channels.schema import Artifact, OutboundReply
from lyra_core.channels.slack import poster as poster_mod


@pytest.mark.asyncio
async def test_bot_token_for_returns_decrypted_token(monkeypatch) -> None:
    from lyra_core.common.crypto import encrypt_for_tenant
    from lyra_core.db.models import SlackInstallation

    cipher = encrypt_for_tenant("tenant-1", "xoxb-bot-real")
    si = SlackInstallation(tenant_id="tenant-1", team_id="T1", bot_token_encrypted=cipher)

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

        async def execute(self, _stmt):
            r = MagicMock()
            r.scalar_one_or_none.return_value = si
            return r

    monkeypatch.setattr(poster_mod, "async_session", FakeSession)
    token = await poster_mod._bot_token_for("tenant-1")
    assert token == "xoxb-bot-real"


@pytest.mark.asyncio
async def test_bot_token_for_raises_when_no_install(monkeypatch) -> None:
    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

        async def execute(self, _stmt):
            r = MagicMock()
            r.scalar_one_or_none.return_value = None
            return r

    monkeypatch.setattr(poster_mod, "async_session", FakeSession)
    with pytest.raises(RuntimeError, match="No Slack installation"):
        await poster_mod._bot_token_for("tenant-x")


@pytest.mark.asyncio
async def test_post_reply_sends_message_and_uploads_artifacts(monkeypatch) -> None:
    monkeypatch.setattr(poster_mod, "_bot_token_for", AsyncMock(return_value="xoxb-test"))

    fake_client = MagicMock()
    fake_client.chat_postMessage = AsyncMock(return_value={"ts": "9999.0001", "ok": True})
    fake_client.files_upload_v2 = AsyncMock(return_value={"ok": True})

    class FakeWebClient:
        def __init__(self, token: str) -> None:
            assert token == "xoxb-test"

        chat_postMessage = fake_client.chat_postMessage
        files_upload_v2 = fake_client.files_upload_v2

    monkeypatch.setattr(poster_mod, "AsyncWebClient", FakeWebClient)

    reply = OutboundReply(
        text="hello",
        thread_id="T-thread",
        channel_id="C-1",
        artifacts=[
            Artifact(kind="pdf", filename="r.pdf", content=b"%PDF", description="report"),
            Artifact(kind="png", filename="c.png", content=b"\x89PNG", description="chart"),
        ],
    )
    parent_ts = await poster_mod.post_reply("tenant-1", reply)

    assert parent_ts == "9999.0001"
    fake_client.chat_postMessage.assert_awaited_once()
    assert fake_client.files_upload_v2.await_count == 2
    # Verify file payloads
    calls = fake_client.files_upload_v2.await_args_list
    assert calls[0].kwargs["filename"] == "r.pdf"
    assert calls[0].kwargs["content"] == b"%PDF"
    assert calls[1].kwargs["filename"] == "c.png"


@pytest.mark.asyncio
async def test_post_reply_sends_blocks_when_provided(monkeypatch) -> None:
    monkeypatch.setattr(poster_mod, "_bot_token_for", AsyncMock(return_value="xoxb-test"))

    fake_post = AsyncMock(return_value={"ts": "1.0"})

    class FakeWebClient:
        def __init__(self, *_a, **_kw) -> None:
            pass

        chat_postMessage = fake_post

        async def files_upload_v2(self, **kw):
            return {"ok": True}

    monkeypatch.setattr(poster_mod, "AsyncWebClient", FakeWebClient)

    blocks = [{"type": "section", "text": {"type": "mrkdwn", "text": "x"}}]
    await poster_mod.post_reply(
        "tenant-1",
        OutboundReply(text="t", blocks=blocks, thread_id="thr", channel_id="ch"),
    )
    kwargs = fake_post.call_args.kwargs
    assert kwargs["blocks"] == blocks
    assert kwargs["thread_ts"] == "thr"
    assert kwargs["channel"] == "ch"


@pytest.mark.asyncio
async def test_post_reply_falls_back_to_space_when_text_none(monkeypatch) -> None:
    monkeypatch.setattr(poster_mod, "_bot_token_for", AsyncMock(return_value="xoxb-test"))

    fake_post = AsyncMock(return_value={"ts": "1.0"})

    class FakeWebClient:
        def __init__(self, *_a, **_kw) -> None:
            pass

        chat_postMessage = fake_post

        async def files_upload_v2(self, **kw):
            return {"ok": True}

    monkeypatch.setattr(poster_mod, "AsyncWebClient", FakeWebClient)

    await poster_mod.post_reply(
        "tenant-1",
        OutboundReply(thread_id="thr", channel_id="ch"),
    )
    assert fake_post.call_args.kwargs["text"] == " "
