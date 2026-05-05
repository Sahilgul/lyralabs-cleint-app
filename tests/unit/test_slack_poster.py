"""lyra_core.channels.slack.poster."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from lyra_core.channels.schema import Artifact, OutboundReply
from lyra_core.channels.slack import poster as poster_mod


@pytest.fixture(autouse=True)
def _allow_dedup(request, monkeypatch):
    """Neutralize the Redis-backed dedup for `post_reply` tests so the SET
    NX EX call doesn't try to open an arq pool against a non-existent test
    broker.

    Tests that need to exercise the REAL `_claim_dedup_slot` (e.g. to verify
    the SET NX EX semantics or fail-open behaviour) opt out by declaring
    `@pytest.mark.real_dedup`. Without this opt-out, the AsyncMock below
    would short-circuit the real function and the assertions inside the
    test would silently pass for the wrong reason (or fail with KeyError
    because the fake pool's `set` was never called).
    """
    if "real_dedup" in request.keywords:
        return
    monkeypatch.setattr(
        poster_mod,
        "_claim_dedup_slot",
        AsyncMock(return_value=True),
    )


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

        chat_postMessage = fake_client.chat_postMessage  # noqa: N815 Slack SDK API contract
        files_upload_v2 = fake_client.files_upload_v2

    monkeypatch.setattr(poster_mod, "AsyncWebClient", FakeWebClient)

    reply = OutboundReply(
        text="hello",
        channel_id="C-1",
        thread_ts="T-thread",
        artifacts=[
            Artifact(kind="pdf", filename="r.pdf", content=b"%PDF", description="report"),
            Artifact(kind="png", filename="c.png", content=b"\x89PNG", description="chart"),
        ],
    )
    parent_ts = await poster_mod.post_reply("tenant-1", reply)

    assert parent_ts == "9999.0001"
    fake_client.chat_postMessage.assert_awaited_once()
    assert fake_client.chat_postMessage.call_args.kwargs["thread_ts"] == "T-thread"
    assert fake_client.files_upload_v2.await_count == 2
    # Verify file payloads
    calls = fake_client.files_upload_v2.await_args_list
    assert calls[0].kwargs["filename"] == "r.pdf"
    assert calls[0].kwargs["content"] == b"%PDF"
    assert calls[0].kwargs["thread_ts"] == "T-thread"
    assert calls[1].kwargs["filename"] == "c.png"


@pytest.mark.asyncio
async def test_post_reply_clears_assistant_status(monkeypatch) -> None:
    monkeypatch.setattr(poster_mod, "_bot_token_for", AsyncMock(return_value="xoxb-test"))

    fake_post = AsyncMock(return_value={"ts": "1.0"})
    fake_clear = AsyncMock(return_value={"ok": True})

    class FakeWebClient:
        def __init__(self, *_a, **_kw) -> None:
            pass

        chat_postMessage = fake_post  # noqa: N815 Slack SDK API contract
        assistant_threads_setStatus = fake_clear  # noqa: N815 Slack SDK API contract

        async def files_upload_v2(self, **kw):
            return {"ok": True}

    monkeypatch.setattr(poster_mod, "AsyncWebClient", FakeWebClient)

    await poster_mod.post_reply(
        "tenant-1",
        OutboundReply(
            text="hi",
            channel_id="D-dm",
            thread_ts=None,
            assistant_status_thread_ts="user-msg-ts",
        ),
    )

    fake_clear.assert_awaited_once_with(
        channel_id="D-dm",
        thread_ts="user-msg-ts",
        status="",
    )


@pytest.mark.asyncio
async def test_post_reply_sends_blocks_when_provided(monkeypatch) -> None:
    monkeypatch.setattr(poster_mod, "_bot_token_for", AsyncMock(return_value="xoxb-test"))

    fake_post = AsyncMock(return_value={"ts": "1.0"})

    class FakeWebClient:
        def __init__(self, *_a, **_kw) -> None:
            pass

        chat_postMessage = fake_post  # noqa: N815 Slack SDK API contract

        async def files_upload_v2(self, **kw):
            return {"ok": True}

    monkeypatch.setattr(poster_mod, "AsyncWebClient", FakeWebClient)

    blocks = [{"type": "section", "text": {"type": "mrkdwn", "text": "x"}}]
    await poster_mod.post_reply(
        "tenant-1",
        OutboundReply(text="t", blocks=blocks, channel_id="ch", thread_ts="thr"),
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

        chat_postMessage = fake_post  # noqa: N815 Slack SDK API contract

        async def files_upload_v2(self, **kw):
            return {"ok": True}

    monkeypatch.setattr(poster_mod, "AsyncWebClient", FakeWebClient)

    await poster_mod.post_reply(
        "tenant-1",
        OutboundReply(channel_id="ch", thread_ts="thr"),
    )
    assert fake_post.call_args.kwargs["text"] == " "


@pytest.mark.asyncio
async def test_post_reply_top_level_when_thread_ts_none(monkeypatch) -> None:
    """When thread_ts is None we MUST pass thread_ts=None to chat_postMessage
    so the bot's reply is a fresh top-level message, not a thread reply on
    the user's previous message. This is what fixes the DM UX bug where
    every bot reply was nested under the user's first DM.
    """
    monkeypatch.setattr(poster_mod, "_bot_token_for", AsyncMock(return_value="xoxb-test"))

    fake_post = AsyncMock(return_value={"ts": "1.0"})

    class FakeWebClient:
        def __init__(self, *_a, **_kw) -> None:
            pass

        chat_postMessage = fake_post  # noqa: N815 Slack SDK API contract

        async def files_upload_v2(self, **kw):
            return {"ok": True}

    monkeypatch.setattr(poster_mod, "AsyncWebClient", FakeWebClient)

    await poster_mod.post_reply(
        "tenant-1",
        OutboundReply(text="hi", channel_id="D-dm"),
    )
    assert fake_post.call_args.kwargs["thread_ts"] is None
    assert fake_post.call_args.kwargs["channel"] == "D-dm"


# ---------------------------------------------------------------------------
# Idempotency / retry-dedup tests
# ---------------------------------------------------------------------------


def test_compute_content_hash_distinguishes_blocks() -> None:
    """Two replies with identical fallback `text` but different `blocks` must
    hash differently. This is the property that prevents distinct approval
    cards (which share `Plan ready for approval (goal: ...)` boilerplate)
    from collapsing into the same dedup slot.
    """
    a = OutboundReply(
        text="Plan ready for approval",
        blocks=[{"type": "section", "text": {"type": "mrkdwn", "text": "Plan A"}}],
        channel_id="C",
    )
    b = OutboundReply(
        text="Plan ready for approval",
        blocks=[{"type": "section", "text": {"type": "mrkdwn", "text": "Plan B"}}],
        channel_id="C",
    )
    assert poster_mod._compute_content_hash(a) != poster_mod._compute_content_hash(b)


def test_compute_content_hash_stable_under_dict_reordering() -> None:
    """Same payload, different key insertion order, must still hash equal --
    otherwise dict-ordering quirks would defeat the dedup on retry.
    """
    a = OutboundReply(
        text="hi",
        blocks=[{"type": "section", "text": {"type": "mrkdwn", "text": "x"}}],
        channel_id="C",
    )
    b = OutboundReply(
        blocks=[{"text": {"text": "x", "type": "mrkdwn"}, "type": "section"}],
        text="hi",
        channel_id="C",
    )
    assert poster_mod._compute_content_hash(a) == poster_mod._compute_content_hash(b)


@pytest.mark.asyncio
async def test_post_reply_skips_when_dedup_already_claimed(monkeypatch) -> None:
    """Core retry-storm guard: if `_claim_dedup_slot` reports the key was
    already claimed by an earlier attempt, post_reply MUST NOT call the
    Slack API. Returning "" is acceptable because all current callers
    discard the return value.
    """
    monkeypatch.setattr(
        poster_mod,
        "_claim_dedup_slot",
        AsyncMock(return_value=False),
    )
    fake_token = AsyncMock(return_value="xoxb-test")
    fake_post = AsyncMock(return_value={"ts": "1.0"})
    fake_upload = AsyncMock(return_value={"ok": True})
    monkeypatch.setattr(poster_mod, "_bot_token_for", fake_token)

    class FakeWebClient:
        def __init__(self, *_a, **_kw) -> None:
            pass

        chat_postMessage = fake_post  # noqa: N815 Slack SDK API contract
        files_upload_v2 = fake_upload

    monkeypatch.setattr(poster_mod, "AsyncWebClient", FakeWebClient)

    result = await poster_mod.post_reply(
        "tenant-1",
        OutboundReply(text="duplicate", channel_id="C", thread_ts="T"),
    )

    assert result == ""
    fake_post.assert_not_awaited()
    fake_upload.assert_not_awaited()
    fake_token.assert_not_awaited()


@pytest.mark.asyncio
async def test_post_reply_dedup_key_is_per_thread_and_per_content(monkeypatch) -> None:
    """The first call posts; an immediately-following call with identical
    (tenant, channel, thread, content) is suppressed; a call to a different
    thread is NOT suppressed. We exercise this by tracking claim keys in a
    fake Redis-style set inside a stand-in for `_claim_dedup_slot`.
    """
    claimed: set[str] = set()

    async def _fake_claim(key: str) -> bool:
        if key in claimed:
            return False
        claimed.add(key)
        return True

    monkeypatch.setattr(poster_mod, "_claim_dedup_slot", _fake_claim)
    monkeypatch.setattr(poster_mod, "_bot_token_for", AsyncMock(return_value="xoxb"))

    fake_post = AsyncMock(return_value={"ts": "1.0"})

    class FakeWebClient:
        def __init__(self, *_a, **_kw) -> None:
            pass

        chat_postMessage = fake_post  # noqa: N815 Slack SDK API contract

        async def files_upload_v2(self, **kw):
            return {"ok": True}

    monkeypatch.setattr(poster_mod, "AsyncWebClient", FakeWebClient)

    payload = OutboundReply(text="hi", channel_id="C", thread_ts="T1")
    other_thread = OutboundReply(text="hi", channel_id="C", thread_ts="T2")

    await poster_mod.post_reply("tenant", payload)
    await poster_mod.post_reply("tenant", payload)  # duplicate -> skipped
    await poster_mod.post_reply("tenant", other_thread)

    # Two posts: original on T1, and the new one on T2. The duplicate
    # second call to T1 must not have hit Slack.
    assert fake_post.await_count == 2
    threads = [c.kwargs["thread_ts"] for c in fake_post.await_args_list]
    assert threads == ["T1", "T2"]


@pytest.mark.real_dedup
@pytest.mark.asyncio
async def test_claim_dedup_slot_fails_open_on_redis_error(monkeypatch) -> None:
    """If the arq pool raises (Redis unreachable, auth, etc.) we MUST allow
    the post to proceed. Blocking a user reply on a dedup-cache outage is
    worse than the rare duplicate that would slip through.

    Note: `@pytest.mark.real_dedup` opts this test out of the autouse
    `_allow_dedup` fixture. Without the opt-out, the autouse AsyncMock
    would return True before our `_broken_pool` was ever consulted, so
    the test would pass for the wrong reason and silently mask a future
    regression of the fail-open semantics.
    """
    # Patch the lazy import path that `_claim_dedup_slot` uses.
    import sys
    import types

    fake_module = types.ModuleType("lyra_core.worker.queue")

    async def _broken_pool() -> None:
        raise RuntimeError("redis offline")

    fake_module._get_pool = _broken_pool  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "lyra_core.worker.queue", fake_module)

    result = await poster_mod._claim_dedup_slot("arlo:posted:test")
    assert result is True


@pytest.mark.real_dedup
@pytest.mark.asyncio
async def test_claim_dedup_slot_uses_set_nx_ex(monkeypatch) -> None:
    """The dedup primitive must be Redis SET NX EX -- not GET-then-SET, not
    plain SET. NX makes it atomic across concurrent retries; EX bounds the
    window. A regression here (e.g. someone "simplifying" to plain SET)
    would silently break dedup on concurrent attempts.

    Note: `@pytest.mark.real_dedup` opts this test out of the autouse
    `_allow_dedup` fixture so we can drive the REAL `_claim_dedup_slot`
    against a `FakePool`. Without the opt-out, the autouse AsyncMock
    short-circuited the real call and `captured` stayed empty, producing
    a `KeyError: 'key'` when this test asserted on the captured pool args.
    """
    captured: dict[str, object] = {}

    class FakePool:
        async def set(self, key, value, *, nx=False, ex=None):
            captured["key"] = key
            captured["value"] = value
            captured["nx"] = nx
            captured["ex"] = ex
            return True

    import sys
    import types

    fake_module = types.ModuleType("lyra_core.worker.queue")

    async def _fake_get_pool() -> FakePool:
        return FakePool()

    fake_module._get_pool = _fake_get_pool  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "lyra_core.worker.queue", fake_module)

    ok = await poster_mod._claim_dedup_slot("arlo:posted:tenant:C:T:abc")
    assert ok is True
    assert captured["key"] == "arlo:posted:tenant:C:T:abc"
    assert captured["nx"] is True
    assert captured["ex"] == poster_mod._DEDUP_TTL_SECONDS


@pytest.mark.asyncio
async def test_post_reply_dedup_key_includes_tenant_channel_thread_content(
    monkeypatch,
) -> None:
    """Source-shape guard: the key passed to `_claim_dedup_slot` must combine
    all four discriminators. If a future refactor drops one (e.g. forgets
    thread_ts), distinct conversations would dedup against each other.
    """
    captured_keys: list[str] = []

    async def _capturing_claim(key: str) -> bool:
        captured_keys.append(key)
        return True

    monkeypatch.setattr(poster_mod, "_claim_dedup_slot", _capturing_claim)
    monkeypatch.setattr(poster_mod, "_bot_token_for", AsyncMock(return_value="xoxb"))

    class FakeWebClient:
        def __init__(self, *_a, **_kw) -> None:
            pass

        chat_postMessage = AsyncMock(return_value={"ts": "1.0"})  # noqa: N815

        async def files_upload_v2(self, **kw):
            return {"ok": True}

    monkeypatch.setattr(poster_mod, "AsyncWebClient", FakeWebClient)

    await poster_mod.post_reply(
        "tenant-X",
        OutboundReply(text="hello", channel_id="C-Y", thread_ts="T-Z"),
    )

    assert len(captured_keys) == 1
    key = captured_keys[0]
    assert key.startswith(poster_mod._DEDUP_KEY_PREFIX)
    assert "tenant-X" in key
    assert "C-Y" in key
    assert "T-Z" in key
    # Last segment is the content hash; verify it's the 16-char prefix we expect.
    h = poster_mod._compute_content_hash(
        OutboundReply(text="hello", channel_id="C-Y", thread_ts="T-Z")
    )
    assert key.endswith(h)
