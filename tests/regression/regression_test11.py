"""Regression test 11 — plan-path preamble silently swallowed.

Bug: When the LLM emitted text content (a 'speak before you work' pre-flight
note like "On it — building the template now, ~10s.") on the SAME turn as
a `submit_plan_for_approval` tool call, that text was dropped on the floor.

Why: `agent_node`'s router was structured as

    Case 1: plan -> validate, set pending_plan, return early
    Case 2: tool_calls -> post preamble, return
    Case 3: direct reply -> post the content, set final_summary

The plan branch (Case 1) returned BEFORE Case 2's preamble-post logic ran.
So when the model wrote text alongside `submit_plan_for_approval`, the
text never reached Slack. The user's only window into the agent's intent
for that turn was the Block Kit approval card subsequently posted by
`approval_post_node`. If the card failed to render (Slack DM quirks,
mobile UI scrolling, notification settings off, network blip on the
`chat.postMessage` for the card), the user saw NOTHING of the agent's
intent for that turn.

This UX hole compounded with the "card is above" hallucination loop on
Tehreem's thread (DLQ jobs 9652f4f4 / 72bf7a78 / c9c5d3b8): the user
typed "create the email template", ARLO's preamble "On it — building
the template now" was silently dropped, the card may have failed to
render too, and the next user turn ("where is the card?") landed against
a model that had no recent assistant message in history showing it had
intended to do work — driving it to confidently fabricate "the card is
above."

Fix: hoist the preamble-post block to a single point ABOVE the
plan-extraction branch, gated on `tool_calls and preamble`. A single
post per turn covers BOTH the plan path AND the read-tool path. Direct-
reply turns (no `tool_calls`) skip the hoisted block entirely; they
post via Case 3 as before, with no double-post risk.

Regression guards:
  1. The preamble post block in `agent_node` is positioned BEFORE the
     `_extract_submit_plan_call` call. (Source-level guard against a
     well-meaning refactor that inverts the order.)
  2. The hoisted block is gated on `tool_calls and preamble` — NOT on
     `tool_calls` alone (would post empty messages) and NOT on
     `preamble` alone (would double-post on direct-reply turns where
     Case 3 also handles the content).
  3. The duplicate preamble block was REMOVED from Case 2 — leaving it
     in place would double-post on read-tool turns now that the hoisted
     block also covers them.
  4. Behaviour: plan submission with content -> preamble posted once,
     `pending_plan` set, no extra Slack posts.
  5. Behaviour: plan submission WITHOUT content -> no Slack post,
     `pending_plan` set. Guards against the "post empty preamble"
     regression that would otherwise spam the channel.
  6. Behaviour: read-tool call with content -> preamble posted once
     (existing behaviour preserved post-hoist).
  7. Behaviour: read-tool call WITHOUT content -> no Slack post.
  8. Behaviour: direct reply (no tool_calls) -> single post via Case 3
     ONLY; the hoisted block must NOT fire.
  9. Behaviour: plan with content + malformed-args validation failure
     -> preamble STILL posted (acceptable trade-off; matches read-tool
     behaviour where preamble fires before the tool runs and may error).
 10. Behaviour: when the agent emits multiple tool calls (plan + read in
     the same turn -- rare but possible per OpenAI spec), the preamble
     fires exactly once.
"""

from __future__ import annotations

import inspect
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from lyra_core.agent.nodes import agent as agent_mod
from lyra_core.agent.nodes.agent import SUBMIT_PLAN_TOOL_NAME, agent_node

# ---------------------------------------------------------------------------
# Test scaffolding (mirrors test_node_agent.py so behaviour stays consistent)
# ---------------------------------------------------------------------------


def _resp(
    content: str | None = None,
    tool_calls: list[dict] | None = None,
    cost: float = 0.0,
):
    """Build a fake litellm completion response."""
    msg = MagicMock()
    msg.content = content
    if tool_calls:
        wrapped = []
        for tc in tool_calls:
            mtc = MagicMock()
            mtc.id = tc["id"]
            mtc.function.name = tc["function"]["name"]
            mtc.function.arguments = tc["function"]["arguments"]
            wrapped.append(mtc)
        msg.tool_calls = wrapped
    else:
        msg.tool_calls = None
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    resp._hidden_params = {"response_cost": cost}
    return resp


def _state(**overrides):
    base = {
        "tenant_id": "tenant-1",
        "job_id": "job-1",
        "channel_id": "C1",
        "thread_id": "thr",
        "user_id": "U1",
        "user_request": "hello",
        "messages": [],
        "total_cost_usd": 0.0,
        "reply_thread_ts": "1700000000.0",
    }
    base.update(overrides)
    return base


def _valid_plan_tool_call() -> dict:
    """A plan tool_call whose args pass `_validate_plan_step_args`. Uses
    `google.docs.create` because that's the canonical happy-path tool the
    existing unit suite (test_node_agent.py) uses for the same purpose."""
    plan_args = {
        "goal": "Create the doc",
        "steps": [
            {
                "id": "step_1",
                "tool_name": "google.docs.create",
                "args": {"title": "Test", "body_text": "hello"},
                "rationale": "create the doc",
                "requires_approval": True,
                "depends_on": [],
            }
        ],
        "needs_clarification": False,
        "clarification_question": None,
    }
    return {
        "id": "tc-plan",
        "function": {
            "name": SUBMIT_PLAN_TOOL_NAME,
            "arguments": json.dumps(plan_args),
        },
    }


def _malformed_plan_tool_call() -> dict:
    """A plan tool_call whose args fail step-level validation
    (`_validate_plan_step_args`). Empty step args -> the tool's Input
    schema rejects it. Used to verify preamble fires even when the plan
    will route through the validation-error sub-branch."""
    plan_args = {
        "goal": "do the thing",
        "steps": [
            {
                "id": "step_1",
                "tool_name": "google.docs.create",
                # `google.docs.create` requires `title` AND `body_text`.
                # Empty args -> validate_args returns an error string ->
                # plan branch returns the arg_error tool message.
                "args": {},
                "rationale": "create",
                "requires_approval": True,
                "depends_on": [],
            }
        ],
        "needs_clarification": False,
        "clarification_question": None,
    }
    return {
        "id": "tc-plan-bad",
        "function": {
            "name": SUBMIT_PLAN_TOOL_NAME,
            "arguments": json.dumps(plan_args),
        },
    }


# ---------------------------------------------------------------------------
# Source-level guards (cheap; catch accidental order inversions in review)
# ---------------------------------------------------------------------------


def test_preamble_post_runs_before_plan_extraction() -> None:
    """The hoisted preamble block must sit BEFORE the plan-extraction call.
    If a refactor moves it back below `_extract_submit_plan_call(tool_calls)`,
    the plan branch (which returns early) once again silently drops the
    user-visible 'On it...' note."""
    src = inspect.getsource(agent_mod.agent_node)
    preamble_marker = src.find('async with phase("agent.post_preamble"')
    plan_extract_marker = src.find("_extract_submit_plan_call(tool_calls)")

    assert preamble_marker != -1, (
        "REGRESSION: agent_node no longer contains the agent.post_preamble "
        "phase. The 'speak before you work' UX hook was removed entirely."
    )
    assert plan_extract_marker != -1, (
        "Source guard precondition failed: cannot locate _extract_submit_plan_call in agent_node."
    )
    assert preamble_marker < plan_extract_marker, (
        "REGRESSION: preamble post is no longer hoisted above the plan-"
        "extraction branch. When the LLM emits 'On it — building the "
        "template now' alongside submit_plan_for_approval, that text "
        "will be silently dropped (Tehreem-thread bug rebound)."
    )


def test_preamble_post_gated_on_tool_calls_and_preamble() -> None:
    """The hoisted block must be gated on BOTH `tool_calls` and `preamble`.

    - Without `tool_calls`: would double-post on direct-reply turns where
      Case 3 also posts the content (Fix 2's dedup catches it but it's
      cleaner to never issue it).
    - Without `preamble`: would post empty messages every time the LLM
      makes a silent tool call.
    """
    src = inspect.getsource(agent_mod.agent_node)
    assert "if tool_calls and preamble:" in src, (
        "REGRESSION: hoisted preamble block is no longer gated on "
        "(tool_calls and preamble). Either direct-reply turns will "
        "double-post or every silent tool call will spam an empty "
        "message."
    )


def test_preamble_block_removed_from_case2_read_tool_branch() -> None:
    """The Case 2 (read-tool) branch must NOT contain its own preamble-
    post block. The hoist moved this responsibility upstream; leaving
    a duplicate in Case 2 would double-post on every read-tool turn
    that has a preamble."""
    src = inspect.getsource(agent_mod.agent_node)
    case2_marker = src.find("# Case 2: agent called a read tool")
    case3_marker = src.find("# Case 3: direct reply.")
    assert case2_marker != -1 and case3_marker != -1, (
        "Source guard precondition failed: cannot find Case 2 or Case 3 "
        "comment markers in agent_node. The router structure was "
        "reorganised; update this guard."
    )
    case2_body = src[case2_marker:case3_marker]
    assert 'phase("agent.post_preamble"' not in case2_body, (
        "REGRESSION: agent.post_preamble block still exists inside Case 2 "
        "after the hoist. With the same block also above Case 1, every "
        "read-tool turn with a preamble will post twice (Fix 2 dedups it "
        "but the duplicate work is wasteful and confuses logs)."
    )
    assert "post_reply" not in case2_body, (
        "REGRESSION: Case 2 read-tool branch is calling post_reply. After "
        "the hoist it should only return base_update; preamble posting "
        "is owned by the hoisted block."
    )


# ---------------------------------------------------------------------------
# Behavior tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_plan_submission_with_preamble_posts_to_slack(monkeypatch) -> None:
    """The headline regression: model emits 'On it — building the template
    now, ~10s.' alongside `submit_plan_for_approval`. The preamble MUST
    reach Slack before the (subsequent, separate) approval card is rendered
    by approval_post_node. Without this, the user sees nothing of the
    agent's intent until/unless the card renders."""
    monkeypatch.setattr(
        agent_mod,
        "chat",
        AsyncMock(
            return_value=_resp(
                content="On it — building the template now, ~10s.",
                tool_calls=[_valid_plan_tool_call()],
            )
        ),
    )
    monkeypatch.setattr(agent_mod, "get_workspace_facts", AsyncMock(return_value={}))
    posted = AsyncMock()
    monkeypatch.setattr(agent_mod, "post_reply", posted)

    out = await agent_node(_state(user_request="create the email template"))

    posted.assert_awaited_once()
    reply = posted.await_args.args[1]
    assert reply.text == "On it — building the template now, ~10s.", (
        "REGRESSION: preamble was posted but with the wrong text. "
        "Expected the model's content verbatim; the assertion failure "
        "above shows what actually got sent."
    )
    assert reply.channel_id == "C1"
    assert reply.thread_ts == "1700000000.0", (
        "REGRESSION: preamble post is missing the reply_thread_ts. "
        "Top-level DM posts would land outside the thread the user is in."
    )
    # Plan still flows through normally -> pending_plan set, no final_summary
    # (the run hands off to approval_post_node).
    assert out["pending_plan"]["goal"] == "Create the doc"
    assert "final_summary" not in out


@pytest.mark.asyncio
async def test_plan_submission_without_preamble_does_not_post(monkeypatch) -> None:
    """Regression-safety: when the model submits a plan with NO content
    (the existing happy path), no Slack post should fire. Without this
    guard, a future change that drops the `preamble` truthiness check
    would post empty messages on every plan submission."""
    monkeypatch.setattr(
        agent_mod,
        "chat",
        AsyncMock(return_value=_resp(content=None, tool_calls=[_valid_plan_tool_call()])),
    )
    monkeypatch.setattr(agent_mod, "get_workspace_facts", AsyncMock(return_value={}))
    posted = AsyncMock()
    monkeypatch.setattr(agent_mod, "post_reply", posted)

    out = await agent_node(_state())

    posted.assert_not_called()
    assert out["pending_plan"]["goal"] == "Create the doc"


@pytest.mark.asyncio
async def test_plan_submission_with_whitespace_only_content_does_not_post(
    monkeypatch,
) -> None:
    """Whitespace-only `content` (e.g. `"   "` or `"\\n\\n"`) must be treated
    the same as no content. `(content or "").strip()` produces an empty
    string which evaluates falsy in the `if tool_calls and preamble:`
    gate."""
    monkeypatch.setattr(
        agent_mod,
        "chat",
        AsyncMock(return_value=_resp(content="   \n  ", tool_calls=[_valid_plan_tool_call()])),
    )
    monkeypatch.setattr(agent_mod, "get_workspace_facts", AsyncMock(return_value={}))
    posted = AsyncMock()
    monkeypatch.setattr(agent_mod, "post_reply", posted)

    await agent_node(_state())

    posted.assert_not_called()


@pytest.mark.asyncio
async def test_read_tool_with_preamble_still_posts_once(monkeypatch) -> None:
    """The hoist must not break the existing read-tool preamble path.
    `test_node_agent.py::test_agent_node_posts_preamble_text_alongside_tool_calls`
    covers this in the unit suite; we keep an equivalent here for the
    regression suite so a regression of this pre-existing behaviour
    surfaces alongside the plan-path coverage."""
    read_tool_calls = [
        {
            "id": "tc-read",
            "function": {
                "name": "ghl.contacts.search",
                "arguments": json.dumps({"query": "stuck deals"}),
            },
        }
    ]
    monkeypatch.setattr(
        agent_mod,
        "chat",
        AsyncMock(
            return_value=_resp(
                content="On it — pulling those stuck deals now, ~10s.",
                tool_calls=read_tool_calls,
            )
        ),
    )
    monkeypatch.setattr(agent_mod, "get_workspace_facts", AsyncMock(return_value={}))
    posted = AsyncMock()
    monkeypatch.setattr(agent_mod, "post_reply", posted)

    out = await agent_node(_state(user_request="pull the stuck deals"))

    assert posted.await_count == 1, (
        f"REGRESSION: read-tool preamble posted {posted.await_count} times. "
        f"After the hoist the duplicate Case 2 block should be gone, so "
        f"the count should be exactly 1."
    )
    assert out.get("pending_plan") is None
    assert "final_summary" not in out


@pytest.mark.asyncio
async def test_direct_reply_does_not_double_post(monkeypatch) -> None:
    """When the LLM produces content but NO tool_calls (smalltalk, simple
    factual answers), Case 3 posts the content as the final reply. The
    hoisted block must NOT also fire — that would issue two posts for the
    same text. Fix 2's Redis dedup catches the duplicate, but the cleanest
    contract is to never issue it."""
    monkeypatch.setattr(
        agent_mod,
        "chat",
        AsyncMock(return_value=_resp(content="Anytime.", tool_calls=None)),
    )
    monkeypatch.setattr(agent_mod, "get_workspace_facts", AsyncMock(return_value={}))
    posted = AsyncMock()
    monkeypatch.setattr(agent_mod, "post_reply", posted)

    out = await agent_node(_state(user_request="thanks"))

    assert posted.await_count == 1, (
        f"REGRESSION: direct-reply turn posted {posted.await_count} times. "
        f"Hoisted block fired in addition to Case 3 — the `tool_calls and "
        f"preamble` gate is broken."
    )
    # Case 3's post carries assistant_status_thread_ts so Slack clears the
    # 'thinking' indicator; the preamble post does NOT. If the count is 1
    # AND the post is the Case 3 path, the call should include the status
    # field.
    reply = posted.await_args.args[1]
    assert reply.text == "Anytime."
    assert out["final_summary"] == "Anytime."


@pytest.mark.asyncio
async def test_plan_with_preamble_and_malformed_args_still_posts_preamble(
    monkeypatch,
) -> None:
    """When the model emits a preamble + a plan that fails arg-validation,
    the preamble STILL fires. This is the same trade-off the read-tool
    branch already accepts: 'On it...' lands BEFORE we know whether the
    downstream work will succeed.

    Documenting this behaviour: the alternative (post preamble only after
    validation passes) was rejected because:
    - It complicates the hoist (two-phase post).
    - Read-tool turns ALREADY post preambles before tool execution; same
      surface area should behave the same way.
    - On retry, the corrected plan submission may or may not include a
      preamble. If it does, Fix 2's Redis dedup collapses identical
      preambles within the dedup window.
    """
    monkeypatch.setattr(
        agent_mod,
        "chat",
        AsyncMock(
            return_value=_resp(
                content="On it — drafting the doc now.",
                tool_calls=[_malformed_plan_tool_call()],
            )
        ),
    )
    monkeypatch.setattr(agent_mod, "get_workspace_facts", AsyncMock(return_value={}))
    posted = AsyncMock()
    monkeypatch.setattr(agent_mod, "post_reply", posted)

    out = await agent_node(_state(user_request="draft the doc"))

    posted.assert_awaited_once()
    assert posted.await_args.args[1].text == "On it — drafting the doc now."
    # And the validation-error tool message is fed back so the LLM can self-correct.
    assert out["messages"][-1]["role"] == "tool"
    assert "Plan rejected" in (out["messages"][-1].get("content") or "") or "invalid args" in (
        out["messages"][-1].get("content") or ""
    )
    # No pending_plan set because validation failed.
    assert out.get("pending_plan") is None


@pytest.mark.asyncio
async def test_preamble_fires_exactly_once_with_multiple_tool_calls(
    monkeypatch,
) -> None:
    """OpenAI's spec allows multiple tool_calls in a single assistant
    message. The hoist must fire EXACTLY ONCE per turn regardless of how
    many tool calls are present, otherwise the user gets N copies of the
    same 'On it...' preamble."""
    multi_calls = [
        {
            "id": "tc-1",
            "function": {
                "name": "ghl.contacts.search",
                "arguments": json.dumps({"query": "alice"}),
            },
        },
        {
            "id": "tc-2",
            "function": {
                "name": "ghl.opportunities.search",
                "arguments": json.dumps({"query": "alice"}),
            },
        },
        {
            "id": "tc-3",
            "function": {
                "name": "google.calendar.list_events",
                "arguments": json.dumps({"calendar_id": "primary"}),
            },
        },
    ]
    monkeypatch.setattr(
        agent_mod,
        "chat",
        AsyncMock(
            return_value=_resp(content="Pulling everything in parallel.", tool_calls=multi_calls)
        ),
    )
    monkeypatch.setattr(agent_mod, "get_workspace_facts", AsyncMock(return_value={}))
    posted = AsyncMock()
    monkeypatch.setattr(agent_mod, "post_reply", posted)

    await agent_node(_state(user_request="pull everything for alice"))

    assert posted.await_count == 1, (
        f"REGRESSION: hoist posted preamble {posted.await_count} times for "
        f"a single turn with 3 tool calls. The hoisted block should fire "
        f"exactly once per agent_node turn."
    )


@pytest.mark.asyncio
async def test_preamble_post_does_not_set_assistant_status_thread_ts(
    monkeypatch,
) -> None:
    """Subtle invariant: the preamble post must NOT carry the
    `assistant_status_thread_ts` field — that's reserved for the FINAL
    reply (Case 3), where Slack clears the 'thinking' indicator after.
    Clearing the indicator on a preamble would make Slack stop showing
    'ARLO is typing...' even though work is still happening downstream."""
    monkeypatch.setattr(
        agent_mod,
        "chat",
        AsyncMock(
            return_value=_resp(
                content="On it.",
                tool_calls=[_valid_plan_tool_call()],
            )
        ),
    )
    monkeypatch.setattr(agent_mod, "get_workspace_facts", AsyncMock(return_value={}))
    posted = AsyncMock()
    monkeypatch.setattr(agent_mod, "post_reply", posted)

    await agent_node(
        _state(
            user_request="x",
            assistant_status_thread_ts="user-msg-ts",
        )
    )

    posted.assert_awaited_once()
    reply = posted.await_args.args[1]
    assert reply.assistant_status_thread_ts is None, (
        "REGRESSION: preamble post is now carrying assistant_status_thread_ts. "
        "That field clears Slack's 'thinking' indicator -- which must stay "
        "active until the final reply (the approval card or executor "
        "summary) lands. Clearing it on a preamble misleads the user "
        "into thinking the agent is done."
    )
