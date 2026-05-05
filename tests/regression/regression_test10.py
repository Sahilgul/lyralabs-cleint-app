"""Regression test 10 — synthetic plan tool message poisoned conversation history.

Bug: When the agent called `submit_plan_for_approval`, `agent_node` injected
a synthetic `role: "tool"` message into `state.messages` to close the
OpenAI tool-call loop:

    {"role": "tool", "tool_call_id": <id>, "content": "Plan submitted for approval."}

The closure itself was necessary -- without it, DeepSeek (and other
providers) returns 400 "tool_calls without matching tool messages" on the
next LLM call. But the WORDING was a lie that survived in history forever.
On a follow-up turn the LLM read:

    [user]      "create email template"
    [assistant] (tool_calls=[submit_plan_for_approval(...)])
    [tool]      "Plan submitted for approval."          <-- THIS LIE
    [user]      "where is the card?"

...and confidently concluded "I already submitted; the card must be above;
the user just can't see it" -- producing the user-reported "the card is
above / I hear you / Understood / stop pissing me off" hallucination loop
on Tehreem's thread (DLQ jobs 9652f4f4, 72bf7a78, c9c5d3b8 -- 25 retries
each, three jobs back-to-back).

The same wording also survived rejection / auto-cancellation paths:
`rejected_reply_node` only APPENDED a `[plan rejected by user]` assistant
note, leaving the original "Plan submitted for approval." in place. The
LLM had to reconcile two contradictory signals (handoff vs. rejection) on
the next turn, which it sometimes did and sometimes didn't.

Fix: three coordinated changes.

  1. The synthetic tool-message content is now `PLAN_HANDOFF_TOOL_MESSAGE`,
     a control-flow ack that explicitly tells the model:
       - "this does NOT mean the user has seen the approval card"
       - "you cannot verify rendering from inside this turn"
       - "if a later user message indicates they cannot see the card,
          paste the plan inline -- do NOT claim the card is 'above'."
     The wording is what kills the "card is above" hallucination at the
     prompt-engineering level.

  2. When the plan resolves to rejected (explicit Reject button) or
     auto-cancelled (Fix 1's user_followup path), `rejected_reply_node`
     calls `_rewrite_synthetic_plan_tool_message(...)` to REPLACE the
     synthetic tool message in history with the resolution-specific
     marker (`PLAN_REJECTED_TOOL_MESSAGE` or `PLAN_AUTOCANCELLED_TOOL_MESSAGE`).
     Subsequent turns then read state-accurate history with no stale
     "handed off" claim sitting around.

  3. The system prompt now contains an explicit "After you submit a plan"
     section teaching the model to apologise + paste the plan inline when
     the user reports they can't see the card, instead of insisting it is
     above.

Regression guards:
  1. The synthetic tool message in `agent_node` references
     `PLAN_HANDOFF_TOOL_MESSAGE`, NOT a hardcoded literal that drifts.
  2. `PLAN_HANDOFF_TOOL_MESSAGE` contains the critical anti-hallucination
     phrasing: "does NOT mean", "cannot verify", "do NOT claim", "paste
     the plan steps inline".
  3. `PLAN_REJECTED_TOOL_MESSAGE` / `PLAN_AUTOCANCELLED_TOOL_MESSAGE`
     contain "no approval card is currently active" / "no longer active"
     so the model does not inherit a stale "card above" mental model.
  4. `rejected_reply_node` calls `_rewrite_synthetic_plan_tool_message`
     in both branches (explicit reject + user_followup auto-cancel).
  5. `_rewrite_synthetic_plan_tool_message` rewrites ONLY the synthetic
     plan message and leaves intermixed real tool results untouched. It
     also catches the legacy "Plan submitted for approval." sentinel so a
     deploy that lands mid-conversation does not orphan old threads.
  6. The OpenAI tool-call protocol invariant is preserved end-to-end:
     after rewrite, `tool_call_id` pairing still holds (rewrite mutates
     `content` only, never the ids or roles).
  7. `find_pending_plan_tool_call_id` returns the most recent
     `submit_plan_for_approval` call id (so multi-plan threads rewrite
     the right one), and returns None defensively when no such call
     exists.
  8. The system prompt (`SYSTEM_TEMPLATE`) contains the explicit
     "do NOT tell the user 'I posted the card' / 'the card is above' ...
     paste the plan steps as plain text" guardrail. This is the
     prompt-level last line of defence even if all other layers slip.
"""

from __future__ import annotations

import inspect
from typing import Any
from unittest.mock import AsyncMock

import pytest
from lyra_core.agent.nodes import agent as agent_mod
from lyra_core.agent.nodes import approval as approval_mod
from lyra_core.agent.nodes.agent import (
    PLAN_AUTOCANCELLED_TOOL_MESSAGE,
    PLAN_HANDOFF_TOOL_MESSAGE,
    PLAN_REJECTED_TOOL_MESSAGE,
    SUBMIT_PLAN_TOOL_NAME,
    SYSTEM_TEMPLATE,
    _rewrite_synthetic_plan_tool_message,
    find_pending_plan_tool_call_id,
)
from lyra_core.agent.nodes.approval import rejected_reply_node

# ---------------------------------------------------------------------------
# Source-level guards
# ---------------------------------------------------------------------------


def test_agent_node_uses_handoff_constant_not_literal() -> None:
    """Source guard: agent_node must reference `PLAN_HANDOFF_TOOL_MESSAGE`
    when injecting the synthetic tool message. A literal string would drift
    out of sync with the constant the rejection-rewriter looks for, and
    would silently re-introduce the original 'Plan submitted for approval.'
    lie if someone copied the literal back."""
    src = inspect.getsource(agent_mod.agent_node)
    assert "PLAN_HANDOFF_TOOL_MESSAGE" in src, (
        "REGRESSION: agent_node no longer injects PLAN_HANDOFF_TOOL_MESSAGE. "
        "Either the synthetic tool-message wiring was removed (DeepSeek 400 "
        "regression), or someone reintroduced a hardcoded literal that the "
        "rejection-rewriter cannot find."
    )
    assert '"Plan submitted for approval."' not in src, (
        "REGRESSION: agent_node injected the legacy literal "
        "'Plan submitted for approval.' -- this is exactly the wording "
        "that produced the 'card is above' hallucination loop on the "
        "Tehreem thread."
    )


def test_handoff_constant_does_not_lie_about_card_visibility() -> None:
    """The constant's purpose is to *prevent* the LLM from concluding 'the
    card is above'. Pin the key defensive phrases so a future 'tightening'
    edit cannot strip them silently."""
    msg = PLAN_HANDOFF_TOOL_MESSAGE
    assert "does NOT mean" in msg, (
        "REGRESSION: PLAN_HANDOFF_TOOL_MESSAGE no longer disclaims that "
        "the message is proof-of-render. Re-opens the 'card is above' "
        "hallucination loop."
    )
    assert "cannot verify" in msg, (
        "REGRESSION: handoff message no longer states the LLM cannot "
        "verify rendering. Drift would re-let the LLM assume the card "
        "is live and visible."
    )
    assert "do NOT claim" in msg, (
        "REGRESSION: handoff message no longer instructs the LLM not to claim the card is 'above'."
    )
    assert "paste the plan" in msg, (
        "REGRESSION: handoff message no longer tells the LLM what to do "
        "when the user reports they can't see the card (paste inline). "
        "Without that fallback, the LLM defaults to insisting the card "
        "is above."
    )


def test_rejected_constant_signals_no_active_card() -> None:
    """When the plan is rejected, the rewritten tool message must tell the
    LLM that NO card is currently active. Otherwise a follow-up turn
    inherits the previous "handed off" framing and re-proposes the same
    plan or claims the (rejected) card is "above"."""
    msg = PLAN_REJECTED_TOOL_MESSAGE
    msg_lower = msg.lower()
    assert "rejected" in msg_lower
    assert "no approval card is currently active" in msg_lower, (
        "REGRESSION: PLAN_REJECTED_TOOL_MESSAGE no longer states the "
        "card is no longer active. Subsequent turns may treat the "
        "rejected plan as still pending."
    )
    assert "do not claim a card is 'above'" in msg_lower or "is no longer active" in msg_lower


def test_autocancelled_constant_redirects_to_new_request() -> None:
    """When `_run` auto-cancels because the user typed a follow-up, the
    rewritten tool message must redirect the LLM to the new user_request.
    Otherwise the LLM keeps the old plan top-of-mind and ignores the new
    request (re-introducing the Tehreem-thread bug at a different layer)."""
    msg = PLAN_AUTOCANCELLED_TOOL_MESSAGE
    assert "auto-cancelled" in msg.lower()
    assert "no longer active" in msg
    assert "current user_request" in msg or "new message" in msg, (
        "REGRESSION: auto-cancelled marker no longer tells the LLM to "
        "treat the new user_request as the active task. The model may "
        "ignore the follow-up and re-propose the cancelled plan."
    )


def test_rejected_reply_node_rewrites_synthetic_message_in_both_branches() -> None:
    """Source guard: rejected_reply_node must call
    _rewrite_synthetic_plan_tool_message on BOTH the user_followup branch
    AND the explicit-reject branch. If either is missed, history poison
    leaks back."""
    src = inspect.getsource(rejected_reply_node)
    rewrite_calls = src.count("_rewrite_synthetic_plan_tool_message")
    assert rewrite_calls >= 2, (
        f"REGRESSION: rejected_reply_node calls "
        f"_rewrite_synthetic_plan_tool_message {rewrite_calls} times; "
        f"expected at least 2 (one per resolution branch). Missing the "
        f"rewrite leaves a stale 'handed off to gate' message in history "
        f"on subsequent agent turns."
    )
    assert "PLAN_AUTOCANCELLED_TOOL_MESSAGE" in src, (
        "REGRESSION: user_followup branch no longer rewrites with PLAN_AUTOCANCELLED_TOOL_MESSAGE."
    )
    assert "PLAN_REJECTED_TOOL_MESSAGE" in src, (
        "REGRESSION: explicit-reject branch no longer rewrites with PLAN_REJECTED_TOOL_MESSAGE."
    )


def test_system_prompt_contains_no_card_above_guardrail() -> None:
    """Source guard: the system prompt must contain the explicit
    "after you submit a plan" guidance. This is the LAST line of defence
    against the hallucination loop -- it works even when history rewrites
    don't (e.g. mid-stream user complaints before the gate resolves)."""
    assert "After you submit a plan" in SYSTEM_TEMPLATE, (
        "REGRESSION: system prompt no longer contains the 'After you "
        "submit a plan' section. Without it, the LLM lacks explicit "
        "anti-hallucination guidance and may insist 'the card is above' "
        "based on training-data instincts."
    )
    assert "the card is above" in SYSTEM_TEMPLATE, (
        "REGRESSION: system prompt no longer explicitly forbids 'the "
        "card is above'. This was the exact hallucination from the "
        "Tehreem thread."
    )
    assert "paste the plan" in SYSTEM_TEMPLATE, (
        "REGRESSION: system prompt no longer instructs the LLM to paste "
        "the plan inline when the user reports not seeing the card. "
        "Without this fallback the LLM has no recovery action."
    )


# ---------------------------------------------------------------------------
# Behavior tests — _rewrite_synthetic_plan_tool_message
# ---------------------------------------------------------------------------


def _plan_history(
    *,
    extra_tool: dict[str, Any] | None = None,
    plan_call_id: str = "call_plan_1",
) -> list[dict[str, Any]]:
    """Build a representative `messages` list with a submit_plan_for_approval
    tool_call closed by the synthetic handoff message. Optionally splices
    in an additional unrelated tool result to verify the rewriter only
    touches the plan message."""
    history: list[dict[str, Any]] = [
        {"role": "user", "content": "create email template"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": plan_call_id,
                    "type": "function",
                    "function": {
                        "name": SUBMIT_PLAN_TOOL_NAME,
                        "arguments": '{"goal":"x","steps":[]}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": plan_call_id,
            "content": PLAN_HANDOFF_TOOL_MESSAGE,
        },
    ]
    if extra_tool is not None:
        history.append(extra_tool)
    return history


def test_rewrite_replaces_handoff_with_new_content() -> None:
    """Core invariant: rewriting the synthetic message replaces ONLY the
    `content` field. The role, tool_call_id, and surrounding messages must
    be unchanged so OpenAI's tool-call pairing still validates."""
    history = _plan_history()
    out = _rewrite_synthetic_plan_tool_message(history, "call_plan_1", PLAN_REJECTED_TOOL_MESSAGE)
    assert len(out) == len(history), "rewriter must not add/drop messages"
    # User and assistant messages untouched
    assert out[0] == history[0]
    assert out[1] == history[1]
    # Synthetic tool message rewritten in place
    assert out[2]["role"] == "tool"
    assert out[2]["tool_call_id"] == "call_plan_1"
    assert out[2]["content"] == PLAN_REJECTED_TOOL_MESSAGE


def test_rewrite_preserves_tool_call_pairing() -> None:
    """OpenAI protocol invariant: every assistant tool_calls[].id MUST have
    a matching `role: "tool"` message with the same `tool_call_id`. Rewrite
    must never break this; otherwise the next LLM call returns 400."""
    history = _plan_history()
    out = _rewrite_synthetic_plan_tool_message(history, "call_plan_1", PLAN_REJECTED_TOOL_MESSAGE)

    assistant_call_ids = {
        tc["id"] for m in out if m.get("role") == "assistant" for tc in (m.get("tool_calls") or [])
    }
    tool_response_ids = {m["tool_call_id"] for m in out if m.get("role") == "tool"}
    assert assistant_call_ids.issubset(tool_response_ids), (
        f"REGRESSION: rewrite broke tool_call pairing. assistant ids="
        f"{assistant_call_ids}, tool response ids={tool_response_ids}. "
        f"This will produce DeepSeek 400 on the next LLM call."
    )


def test_rewrite_does_not_touch_unrelated_tool_results() -> None:
    """If a thread interleaves a real read-tool result (`crm.contacts.search`
    or similar) with the plan submission, the rewriter MUST leave that real
    result alone. Clobbering real tool data would silently drop CRM facts
    from the LLM's context."""
    real_tool_msg = {
        "role": "tool",
        "tool_call_id": "call_search_1",
        "content": '{"contacts":[{"name":"Hernandez","id":"123"}]}',
    }
    history = _plan_history(extra_tool=real_tool_msg)
    out = _rewrite_synthetic_plan_tool_message(history, "call_plan_1", PLAN_REJECTED_TOOL_MESSAGE)
    # The real read-tool result is untouched.
    real_after = next(m for m in out if m.get("tool_call_id") == "call_search_1")
    assert real_after == real_tool_msg, (
        f"REGRESSION: rewriter clobbered an unrelated real tool result. "
        f"Before: {real_tool_msg}, After: {real_after}"
    )


def test_rewrite_matches_legacy_plan_submitted_string() -> None:
    """Rolling-deploy safety: an in-flight thread checkpointed BEFORE this
    fix carries the legacy `"Plan submitted for approval."` string. After
    the deploy, when that thread resolves, the rewriter must catch the
    legacy sentinel and replace it -- otherwise the user keeps seeing the
    old wording bleed through into the LLM's next turn."""
    legacy_history = [
        {"role": "user", "content": "x"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_plan_legacy",
                    "type": "function",
                    "function": {
                        "name": SUBMIT_PLAN_TOOL_NAME,
                        "arguments": "{}",
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_plan_legacy",
            "content": "Plan submitted for approval.",
        },
    ]
    out = _rewrite_synthetic_plan_tool_message(
        legacy_history, "call_plan_legacy", PLAN_REJECTED_TOOL_MESSAGE
    )
    assert out[2]["content"] == PLAN_REJECTED_TOOL_MESSAGE, (
        "REGRESSION: rewriter no longer catches the legacy "
        "'Plan submitted for approval.' sentinel. Threads that span the "
        "deploy will keep the lie in history."
    )


def test_rewrite_is_no_op_when_call_id_not_found() -> None:
    """Defensive: if the plan_call_id is missing from history, the rewriter
    must NOT crash and MUST NOT mutate anything. Otherwise a malformed
    state would propagate as an exception, killing the whole graph turn."""
    history = _plan_history(plan_call_id="call_plan_1")
    out = _rewrite_synthetic_plan_tool_message(
        history, "call_does_not_exist", PLAN_REJECTED_TOOL_MESSAGE
    )
    assert out == history, "rewriter must be a no-op when id not found"


def test_rewrite_is_no_op_for_non_synthetic_content() -> None:
    """A genuine custom tool result with a content string we did NOT inject
    must NEVER be rewritten -- only the known synthetic markers are
    touched. Without this guard, a future tool that happens to produce
    similar prose could be silently replaced."""
    history = [
        {"role": "user", "content": "x"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_plan_1",
                    "type": "function",
                    "function": {
                        "name": SUBMIT_PLAN_TOOL_NAME,
                        "arguments": "{}",
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_plan_1",
            "content": "Custom non-synthetic content from elsewhere",
        },
    ]
    out = _rewrite_synthetic_plan_tool_message(history, "call_plan_1", PLAN_REJECTED_TOOL_MESSAGE)
    assert out == history, (
        "REGRESSION: rewriter touched non-synthetic content. The whitelist "
        "of known markers must be respected exactly."
    )


# ---------------------------------------------------------------------------
# Behavior tests — find_pending_plan_tool_call_id
# ---------------------------------------------------------------------------


def test_find_pending_plan_returns_most_recent_call_id() -> None:
    """In a multi-plan thread (rejected once, retried), the rewriter must
    target the MOST RECENT submit_plan_for_approval -- not the oldest.
    Otherwise a re-submission's resolution would rewrite the wrong slot."""
    history = [
        {"role": "user", "content": "first"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_old",
                    "type": "function",
                    "function": {"name": SUBMIT_PLAN_TOOL_NAME, "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_old", "content": PLAN_REJECTED_TOOL_MESSAGE},
        {"role": "user", "content": "try again"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_new",
                    "type": "function",
                    "function": {"name": SUBMIT_PLAN_TOOL_NAME, "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_new", "content": PLAN_HANDOFF_TOOL_MESSAGE},
    ]
    assert find_pending_plan_tool_call_id(history) == "call_new", (
        "REGRESSION: find_pending_plan_tool_call_id returned an older "
        "call id. Multi-plan threads will rewrite the wrong slot."
    )


def test_find_pending_plan_returns_none_when_absent() -> None:
    """Defensive: empty / no-plan history must return None (not crash, not
    return a wrong id)."""
    assert find_pending_plan_tool_call_id([]) is None
    assert find_pending_plan_tool_call_id([{"role": "user", "content": "hi"}]) is None


# ---------------------------------------------------------------------------
# End-to-end behaviour: rejected_reply_node integrates the rewrite correctly
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_user_followup_rewrites_handoff_to_autocancelled(monkeypatch) -> None:
    """The integration scenario: plan submitted, user typed a follow-up
    (Fix 1's auto-cancel path). After rejected_reply_node runs, the
    synthetic handoff message must be REPLACED with the auto-cancelled
    marker so the next agent_node turn sees state-accurate history.

    Without the rewrite, the next turn reads 'handed off to gate' and the
    LLM may insist 'the card is above' even though it was auto-cancelled.
    """
    history = _plan_history()
    monkeypatch.setattr(approval_mod, "post_reply", AsyncMock())

    out = await rejected_reply_node(  # type: ignore[arg-type]
        {
            "thread_id": "t",
            "channel_id": "c",
            "tenant_id": "ten",
            "approval_rejection_reason": "user_followup",
            "messages": history,
        }
    )

    msgs: list[dict[str, Any]] = out["messages"]
    plan_tool = next(m for m in msgs if m.get("tool_call_id") == "call_plan_1")
    assert plan_tool["content"] == PLAN_AUTOCANCELLED_TOOL_MESSAGE, (
        "REGRESSION: user_followup auto-cancel did not rewrite the "
        "synthetic handoff message in history. The LLM will read 'plan "
        "handed off to gate' on the next turn even though the plan was "
        "cancelled, re-introducing the 'card is above' loop."
    )
    # And the ASSISTANT marker note (existing Fix 1 behaviour) is still appended.
    assert any(
        m.get("role") == "assistant" and "auto-cancelled" in (m.get("content") or "") for m in msgs
    )


@pytest.mark.asyncio
async def test_explicit_reject_rewrites_handoff_to_rejected(monkeypatch) -> None:
    """The integration scenario for explicit Reject button click: same
    rewrite invariant, with the rejected marker."""
    history = _plan_history()
    monkeypatch.setattr(approval_mod, "post_reply", AsyncMock())

    out = await rejected_reply_node(  # type: ignore[arg-type]
        {
            "thread_id": "t",
            "channel_id": "c",
            "tenant_id": "ten",
            "messages": history,
            # No approval_rejection_reason -> explicit reject branch
        }
    )

    msgs: list[dict[str, Any]] = out["messages"]
    plan_tool = next(m for m in msgs if m.get("tool_call_id") == "call_plan_1")
    assert plan_tool["content"] == PLAN_REJECTED_TOOL_MESSAGE, (
        "REGRESSION: explicit-reject branch did not rewrite the synthetic "
        "handoff message. Subsequent turns will see the stale 'handed off' "
        "claim alongside a '[plan rejected by user]' note -- contradictory "
        "context confuses the LLM."
    )
    # Assistant rejection note still appended after the rewritten tool msg.
    assert msgs[-1]["role"] == "assistant"
    assert "[plan rejected by user]" in (msgs[-1].get("content") or "")


@pytest.mark.asyncio
async def test_rejected_reply_does_not_crash_on_history_without_plan_call(
    monkeypatch,
) -> None:
    """If for any reason the history doesn't contain a plan tool_call (test
    fixtures, malformed state, etc.) rejected_reply_node MUST still return
    a valid update and not raise. Pre-Fix-3 behaviour is preserved here."""
    monkeypatch.setattr(approval_mod, "post_reply", AsyncMock())
    out = await rejected_reply_node(  # type: ignore[arg-type]
        {
            "thread_id": "t",
            "channel_id": "c",
            "tenant_id": "ten",
            "messages": [{"role": "user", "content": "hi"}],
        }
    )
    assert out["final_summary"] == "rejected_by_user"
    # No KeyError, no rewrite, just the appended assistant note
    assert any(
        m.get("role") == "assistant" and "[plan rejected by user]" in (m.get("content") or "")
        for m in out["messages"]
    )
