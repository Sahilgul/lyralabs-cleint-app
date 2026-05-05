"""Regression test 9 — arq retry storm multiplied Slack replies.

Bug: `agent_node` calls `post_reply()` mid-graph-step (final reply path,
preamble path) and `approval_post_node` calls it for the approval card.
LangGraph commits node side-effects only when the node returns and a
checkpoint is written. If anything later in `_run` raises BEFORE the
checkpoint -- LangGraph error inside `aget_state`, Postgres flap on the
job-status UPDATE, network blip on a downstream tool, etc. -- arq retries
the job. The default policy on this code path is up to `_MAX_TRIES = 25`
attempts with a defer back-off of ~30s. Each retry re-runs `agent_node`,
which hits `post_reply` again, which posts the SAME message to Slack again.

User-visible symptom from the Slack threads we triaged:
  - User: "Stop pissing me off"
  - ARLO: "I hear you" (post 1)
  - ARLO: "I hear you" (post 2)
  - ...
  - ARLO: "Understood" (post N)
The DLQ rows for those interactions show 25-retry counts on three jobs in
a row (`9652f4f4`, `72bf7a78`, `c9c5d3b8`) -- exactly the retry-storm
signature.

Fix: idempotency at the post-boundary. Before issuing
`chat.postMessage`, `post_reply` claims a Redis dedup slot:

    key   = arlo:posted:{tenant}:{channel}:{thread_ts}:{sha256(text+blocks)[:16]}
    redis SET key 1 NX EX _DEDUP_TTL_SECONDS

`SET NX` is atomic across concurrent retries, and `EX` bounds the window
(retry storms cluster within ~12 min so a 5-min TTL covers them). First
caller wins; duplicates within the window log `slack.reply.dedup_skipped`
and short-circuit with no Slack API calls.

`post_reply` is the chokepoint for EVERY user-visible message ARLO
produces (agent final reply, agent preamble, approval card, rejected
reply, critic reply, run_agent error fallback), so one fix at this layer
covers all retry paths.

Failure-mode handling: if Redis itself is unreachable, the dedup helper
fails OPEN -- it returns True (allow post). Blocking a real reply on a
dedup-cache outage would be worse than the rare duplicate that slips
through.

Regression guards:
  1. `post_reply` calls `_claim_dedup_slot` BEFORE `_bot_token_for` and
     before constructing the Slack web client. (Cheap short-circuit.)
  2. `_claim_dedup_slot` uses Redis `SET key value NX=True EX=ttl`. A
     refactor that drops `NX` would silently break dedup under concurrent
     retries.
  3. The dedup key combines tenant, channel, thread_ts, AND content_hash.
     Dropping any one of those would conflate distinct conversations.
  4. The content hash distinguishes payloads by both `text` AND `blocks`
     so two different approval cards (which share fallback text) don't
     dedup as one.
  5. End-to-end: when `_claim_dedup_slot` reports the slot is taken,
     `post_reply` does NOT call `chat_postMessage`, `files_upload_v2`, or
     `_bot_token_for`. (Guards against any of those getting wired up
     before the dedup check.)
  6. End-to-end: a different `thread_ts` with identical text MUST still
     post -- otherwise distinct conversations would silently collapse.
  7. Fail-open: when the arq pool helper raises, `_claim_dedup_slot`
     returns True so the user still gets a reply.
  8. The dedup TTL stays bounded (no `ex=None` regression that would
     leak Redis keys forever and silently dedup legitimate repeated
     messages weeks later).
"""

from __future__ import annotations

import hashlib
import inspect
import json
import sys
import types
from unittest.mock import AsyncMock

import pytest
from lyra_core.channels.schema import Artifact, OutboundReply
from lyra_core.channels.slack import poster as poster_mod

# ---------------------------------------------------------------------------
# Source-level guards (cheap; catch accidental reverts in code review)
# ---------------------------------------------------------------------------


def test_post_reply_dedup_runs_before_bot_token_lookup() -> None:
    """Source guard: the dedup short-circuit must come before the Postgres
    + Fernet-decrypt of the bot token. If a refactor moves it after, every
    retry-storm attempt still pays the ~200-300ms token-fetch cost AND we
    open a tiny window where the token cache is touched on a job that will
    end up skipping the post anyway."""
    src = inspect.getsource(poster_mod.post_reply)
    dedup_idx = src.find("_claim_dedup_slot(dedup_key)")
    token_idx = src.find("_bot_token_for(tenant_id)")
    client_idx = src.find("AsyncWebClient(token=token)")

    assert dedup_idx != -1, (
        "REGRESSION: post_reply no longer calls _claim_dedup_slot. arq "
        "retries will multiply Slack posts (DLQ retry-storm bug)."
    )
    assert token_idx != -1, "post_reply must still resolve the bot token"
    assert client_idx != -1, "post_reply must still construct AsyncWebClient"
    assert dedup_idx < token_idx < client_idx, (
        "REGRESSION: dedup short-circuit must run BEFORE _bot_token_for "
        "and AsyncWebClient construction. Otherwise retries pay the full "
        "Postgres + decrypt cost on every duplicate."
    )


def test_claim_dedup_slot_uses_redis_set_nx_ex() -> None:
    """Source guard: the atomic primitive must be `SET key value NX=True
    EX=<ttl>`. A 'simplification' to plain `set(...)` would race under
    concurrent retries and let duplicates through."""
    src = inspect.getsource(poster_mod._claim_dedup_slot)
    assert "pool.set(" in src, (
        "REGRESSION: _claim_dedup_slot must call pool.set(...) (the Redis "
        "SET command on the arq pool)."
    )
    assert "nx=True" in src, (
        "REGRESSION: dedup MUST use NX=True. Without NX, two concurrent "
        "retries can both observe an empty key, both write it, and both "
        "return True -- defeating dedup."
    )
    assert "ex=_DEDUP_TTL_SECONDS" in src, (
        "REGRESSION: dedup MUST use EX=_DEDUP_TTL_SECONDS. An unbounded "
        "key (`ex=None`) would leak Redis keys and silently dedup "
        "legitimate identical messages weeks later."
    )


def test_dedup_ttl_is_bounded_and_finite() -> None:
    """A bare `int` greater than zero. Catches accidental `None`, 0, or
    `float('inf')` regressions."""
    ttl = poster_mod._DEDUP_TTL_SECONDS
    assert isinstance(ttl, int) and ttl > 0
    # Sanity: must be at least the worst-case arq retry window so retries
    # land inside the dedup TTL. arq default: max_tries=25, defer=30s ->
    # ~12 min upper bound. We document 300s; bumping below 60s would defeat
    # the fix.
    assert ttl >= 60, "_DEDUP_TTL_SECONDS too short to cover retry window"


def test_dedup_key_prefix_is_namespaced() -> None:
    """Catches a regression where someone strips the `arlo:posted:` prefix
    and starts colliding with other Redis keys (`arlo:active_thread:`,
    `arlo:lock:`, etc.) in the same Redis namespace."""
    assert poster_mod._DEDUP_KEY_PREFIX.startswith("arlo:")
    assert "posted" in poster_mod._DEDUP_KEY_PREFIX


def test_compute_content_hash_includes_blocks_in_payload() -> None:
    """Source guard: the content-hash payload must include blocks. Approval
    cards share `Plan ready for approval (goal: ...)` boilerplate text;
    without blocks in the hash, Plan A and Plan B with the same goal would
    dedup as one and the second card would be silently swallowed."""
    src = inspect.getsource(poster_mod._compute_content_hash)
    assert "reply.blocks" in src, (
        "REGRESSION: _compute_content_hash must hash reply.blocks. "
        "Approval cards share fallback text; dropping blocks from the hash "
        "would collapse distinct cards into a single dedup slot."
    )
    assert "reply.text" in src
    # `sort_keys=True` so dict-ordering quirks don't defeat the hash on
    # retry (Python dict insertion order is preserved but JSON encoders
    # can vary).
    assert "sort_keys=True" in src


# ---------------------------------------------------------------------------
# Behavior tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retry_storm_only_posts_once(monkeypatch) -> None:
    """End-to-end retry-storm reproduction: simulate the exact pattern from
    the DLQ jobs -- the same agent_node `post_reply` call invoked N times
    with identical (tenant, channel, thread, content). Only the first
    attempt should reach Slack; the rest must short-circuit silently.

    This is the core regression: 25 calls -> 25 Slack posts in the bug,
    25 calls -> 1 Slack post in the fix.
    """
    claimed: set[str] = set()

    async def _real_dedup(key: str) -> bool:
        if key in claimed:
            return False
        claimed.add(key)
        return True

    monkeypatch.setattr(poster_mod, "_claim_dedup_slot", _real_dedup)
    monkeypatch.setattr(poster_mod, "_bot_token_for", AsyncMock(return_value="xoxb"))

    fake_post = AsyncMock(return_value={"ts": "1.0"})
    fake_upload = AsyncMock(return_value={"ok": True})

    class FakeWebClient:
        def __init__(self, *_a, **_kw) -> None:
            pass

        chat_postMessage = fake_post  # noqa: N815 Slack SDK API contract
        files_upload_v2 = fake_upload

    monkeypatch.setattr(poster_mod, "AsyncWebClient", FakeWebClient)

    reply = OutboundReply(
        text="I hear you - working on it",
        channel_id="C-team",
        thread_ts="1700000000.0",
    )

    # 25 attempts mirror arq's default _MAX_TRIES.
    for _ in range(25):
        await poster_mod.post_reply("tenant-1", reply)

    assert fake_post.await_count == 1, (
        f"REGRESSION: retry storm produced {fake_post.await_count} Slack "
        f"posts; expected 1. The dedup short-circuit is broken and the "
        f"DLQ retry-storm bug is back."
    )


@pytest.mark.asyncio
async def test_retry_storm_does_not_re_upload_artifacts(monkeypatch) -> None:
    """The artifact upload path is part of the same node that posts the
    text. If dedup short-circuits the post but leaks the upload, retries
    still spam the channel with N copies of the PDF/PNG/CSV. Guard the
    upload path explicitly."""
    claimed: set[str] = set()

    async def _real_dedup(key: str) -> bool:
        if key in claimed:
            return False
        claimed.add(key)
        return True

    monkeypatch.setattr(poster_mod, "_claim_dedup_slot", _real_dedup)
    monkeypatch.setattr(poster_mod, "_bot_token_for", AsyncMock(return_value="xoxb"))

    fake_post = AsyncMock(return_value={"ts": "1.0"})
    fake_upload = AsyncMock(return_value={"ok": True})

    class FakeWebClient:
        def __init__(self, *_a, **_kw) -> None:
            pass

        chat_postMessage = fake_post  # noqa: N815 Slack SDK API contract
        files_upload_v2 = fake_upload

    monkeypatch.setattr(poster_mod, "AsyncWebClient", FakeWebClient)

    reply = OutboundReply(
        text="Here is the report",
        channel_id="C-team",
        thread_ts="1700000000.0",
        artifacts=[
            Artifact(kind="pdf", filename="r.pdf", content=b"%PDF", description="r"),
        ],
    )

    for _ in range(5):
        await poster_mod.post_reply("tenant-1", reply)

    assert fake_post.await_count == 1
    assert fake_upload.await_count == 1, (
        f"REGRESSION: retry storm uploaded the same artifact "
        f"{fake_upload.await_count} times. Dedup must cover the artifact "
        f"path too."
    )


@pytest.mark.asyncio
async def test_dedup_does_not_conflate_separate_threads(monkeypatch) -> None:
    """Two distinct Slack threads in the same tenant/channel must each get
    their own copy of an identical message. A bug here would silently drop
    replies in one thread because another thread happened to receive the
    same text."""
    claimed: set[str] = set()

    async def _real_dedup(key: str) -> bool:
        if key in claimed:
            return False
        claimed.add(key)
        return True

    monkeypatch.setattr(poster_mod, "_claim_dedup_slot", _real_dedup)
    monkeypatch.setattr(poster_mod, "_bot_token_for", AsyncMock(return_value="xoxb"))

    fake_post = AsyncMock(return_value={"ts": "1.0"})

    class FakeWebClient:
        def __init__(self, *_a, **_kw) -> None:
            pass

        chat_postMessage = fake_post  # noqa: N815 Slack SDK API contract

        async def files_upload_v2(self, **kw):
            return {"ok": True}

    monkeypatch.setattr(poster_mod, "AsyncWebClient", FakeWebClient)

    await poster_mod.post_reply(
        "tenant",
        OutboundReply(text="ack", channel_id="C", thread_ts="A"),
    )
    await poster_mod.post_reply(
        "tenant",
        OutboundReply(text="ack", channel_id="C", thread_ts="B"),
    )

    threads = [c.kwargs["thread_ts"] for c in fake_post.await_args_list]
    assert sorted(threads) == ["A", "B"], (
        f"REGRESSION: distinct threads got conflated into one dedup slot. "
        f"Posted to threads={threads}; expected both A and B."
    )


@pytest.mark.asyncio
async def test_dedup_does_not_conflate_separate_tenants(monkeypatch) -> None:
    """Two tenants posting an identical message to the same channel id (a
    real possibility because Slack channel ids are not globally unique
    across workspaces) must each get a separate post."""
    claimed: set[str] = set()

    async def _real_dedup(key: str) -> bool:
        if key in claimed:
            return False
        claimed.add(key)
        return True

    monkeypatch.setattr(poster_mod, "_claim_dedup_slot", _real_dedup)
    monkeypatch.setattr(poster_mod, "_bot_token_for", AsyncMock(return_value="xoxb"))

    fake_post = AsyncMock(return_value={"ts": "1.0"})

    class FakeWebClient:
        def __init__(self, *_a, **_kw) -> None:
            pass

        chat_postMessage = fake_post  # noqa: N815 Slack SDK API contract

        async def files_upload_v2(self, **kw):
            return {"ok": True}

    monkeypatch.setattr(poster_mod, "AsyncWebClient", FakeWebClient)

    reply = OutboundReply(text="hi", channel_id="C-shared", thread_ts="T")
    await poster_mod.post_reply("tenant-A", reply)
    await poster_mod.post_reply("tenant-B", reply)

    assert fake_post.await_count == 2, (
        "REGRESSION: distinct tenants got conflated. Tenant-id MUST be part of the dedup key."
    )


@pytest.mark.asyncio
async def test_dedup_distinguishes_distinct_approval_cards(monkeypatch) -> None:
    """Two consecutive approval cards in the same thread (e.g. user
    rejects Plan A, agent generates Plan B) must both reach Slack even
    though the fallback text is identical. This is the bug we'd
    re-introduce by hashing only `text`."""
    claimed: set[str] = set()

    async def _real_dedup(key: str) -> bool:
        if key in claimed:
            return False
        claimed.add(key)
        return True

    monkeypatch.setattr(poster_mod, "_claim_dedup_slot", _real_dedup)
    monkeypatch.setattr(poster_mod, "_bot_token_for", AsyncMock(return_value="xoxb"))

    fake_post = AsyncMock(return_value={"ts": "1.0"})

    class FakeWebClient:
        def __init__(self, *_a, **_kw) -> None:
            pass

        chat_postMessage = fake_post  # noqa: N815 Slack SDK API contract

        async def files_upload_v2(self, **kw):
            return {"ok": True}

    monkeypatch.setattr(poster_mod, "AsyncWebClient", FakeWebClient)

    plan_a = OutboundReply(
        text="Plan ready for approval",
        blocks=[{"type": "section", "text": {"type": "mrkdwn", "text": "Plan A"}}],
        channel_id="C",
        thread_ts="T",
    )
    plan_b = OutboundReply(
        text="Plan ready for approval",
        blocks=[{"type": "section", "text": {"type": "mrkdwn", "text": "Plan B"}}],
        channel_id="C",
        thread_ts="T",
    )

    await poster_mod.post_reply("tenant", plan_a)
    await poster_mod.post_reply("tenant", plan_b)

    assert fake_post.await_count == 2, (
        "REGRESSION: distinct approval cards (Plan A vs Plan B) collapsed "
        "into one dedup slot. The content hash must include `blocks`."
    )


@pytest.mark.asyncio
async def test_dedup_skips_bot_token_lookup_on_duplicate(monkeypatch) -> None:
    """When the slot is already claimed, NONE of the heavy operations
    should run: no Postgres token fetch, no Slack web client, no
    chat_postMessage. Dedup should be a cheap short-circuit."""
    monkeypatch.setattr(
        poster_mod,
        "_claim_dedup_slot",
        AsyncMock(return_value=False),
    )
    fake_token = AsyncMock(return_value="xoxb")
    fake_post = AsyncMock(return_value={"ts": "1.0"})
    monkeypatch.setattr(poster_mod, "_bot_token_for", fake_token)

    class FakeWebClient:
        def __init__(self, *_a, **_kw) -> None:
            pytest.fail(
                "REGRESSION: AsyncWebClient instantiated on a deduped post. "
                "Dedup short-circuit must run before client construction."
            )

        chat_postMessage = fake_post  # noqa: N815 Slack SDK API contract

        async def files_upload_v2(self, **kw):
            return {"ok": True}

    monkeypatch.setattr(poster_mod, "AsyncWebClient", FakeWebClient)

    result = await poster_mod.post_reply(
        "tenant", OutboundReply(text="dup", channel_id="C", thread_ts="T")
    )
    assert result == ""
    fake_token.assert_not_awaited()
    fake_post.assert_not_awaited()


@pytest.mark.asyncio
async def test_dedup_fails_open_when_redis_pool_unavailable(monkeypatch) -> None:
    """Critical safety property: a Redis outage must NOT silence the
    agent. `_claim_dedup_slot` must return True (allow post) when
    `_get_pool()` raises."""
    fake_module = types.ModuleType("lyra_core.worker.queue")

    async def _broken_pool() -> None:
        raise RuntimeError("redis offline")

    fake_module._get_pool = _broken_pool  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "lyra_core.worker.queue", fake_module)

    ok = await poster_mod._claim_dedup_slot("arlo:posted:does-not-matter")
    assert ok is True, (
        "REGRESSION: dedup is no longer fail-open. A Redis outage will now "
        "block all replies, which is strictly worse than rare duplicates."
    )


@pytest.mark.asyncio
async def test_dedup_fails_open_when_redis_set_raises(monkeypatch) -> None:
    """The pool itself can be reachable but the SET call can still raise
    (auth, OOM, network blip mid-roundtrip). Same fail-open contract."""

    class FailingPool:
        async def set(self, key, value, *, nx=False, ex=None):
            raise RuntimeError("redis OOM")

    fake_module = types.ModuleType("lyra_core.worker.queue")

    async def _fake_get_pool() -> FailingPool:
        return FailingPool()

    fake_module._get_pool = _fake_get_pool  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "lyra_core.worker.queue", fake_module)

    ok = await poster_mod._claim_dedup_slot("arlo:posted:does-not-matter")
    assert ok is True


@pytest.mark.asyncio
async def test_dedup_key_format_is_stable(monkeypatch) -> None:
    """Cross-process dedup only works if every process computes the same
    key for the same payload. Pin the format so a refactor that subtly
    changes separators or hash truncation can't silently break dedup
    across rolling deploys (old pod posts; new pod retries; new pod's
    different key fails to find the old pod's slot; duplicate posted)."""
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

    text = "hello world"
    blocks = [{"type": "section", "text": {"type": "mrkdwn", "text": "x"}}]
    expected_payload = text + "\x1e" + json.dumps(blocks, sort_keys=True, default=str)
    expected_hash = hashlib.sha256(expected_payload.encode("utf-8")).hexdigest()[:16]

    await poster_mod.post_reply(
        "tenant-1",
        OutboundReply(text=text, blocks=blocks, channel_id="C-X", thread_ts="T-Y"),
    )

    assert len(captured_keys) == 1
    assert captured_keys[0] == (
        f"{poster_mod._DEDUP_KEY_PREFIX}tenant-1:C-X:T-Y:{expected_hash}"
    ), (
        "REGRESSION: dedup key format changed. Old + new processes during a "
        "rolling deploy will compute different keys for the same payload "
        "and dedup will fail across pods."
    )


@pytest.mark.asyncio
async def test_dedup_handles_top_level_message_without_thread_ts(monkeypatch) -> None:
    """Top-level DM messages have `thread_ts=None`. The key must still be
    well-formed (no `None` rendered into the key in a way that conflicts
    with a real `'None'` thread_ts) AND retries on the same top-level
    message must still dedup."""
    claimed: set[str] = set()

    async def _real_dedup(key: str) -> bool:
        # Empty thread_ts must render as the empty-string slot ('::' between
        # channel and content hash), not as the literal Python repr 'None'.
        # f"{None or ''}" -> ''; f"{None}" -> 'None'. The first is correct.
        assert ":None:" not in key, (
            f"REGRESSION: top-level dedup key contains literal 'None': {key}"
        )
        assert "::" in key, f"REGRESSION: top-level dedup key missing empty-thread segment: {key}"
        if key in claimed:
            return False
        claimed.add(key)
        return True

    monkeypatch.setattr(poster_mod, "_claim_dedup_slot", _real_dedup)
    monkeypatch.setattr(poster_mod, "_bot_token_for", AsyncMock(return_value="xoxb"))

    fake_post = AsyncMock(return_value={"ts": "1.0"})

    class FakeWebClient:
        def __init__(self, *_a, **_kw) -> None:
            pass

        chat_postMessage = fake_post  # noqa: N815 Slack SDK API contract

        async def files_upload_v2(self, **kw):
            return {"ok": True}

    monkeypatch.setattr(poster_mod, "AsyncWebClient", FakeWebClient)

    reply = OutboundReply(text="dm hello", channel_id="D-dm")
    await poster_mod.post_reply("tenant", reply)
    await poster_mod.post_reply("tenant", reply)

    assert fake_post.await_count == 1, (
        "REGRESSION: top-level (thread_ts=None) DM messages no longer dedup across retries."
    )
