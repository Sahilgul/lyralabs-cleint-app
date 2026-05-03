"""Regression test 1 — orphaned tool_calls after plan rejection.

Bug: When a user rejected an approval card, the next message to ARLO crashed
with DeepSeek 400 "An assistant message with 'tool_calls' must be followed by
tool messages responding to each 'tool_call_id'."

Root cause: agent_node saved the assistant message containing
`tool_calls: [submit_plan_for_approval]` to history but never appended the
corresponding `{"role": "tool", ...}` response. On the next turn, the
checkpointer replayed that orphaned tool_call → DeepSeek rejected the request.

Fix: agent_node now appends a synthetic tool response immediately when a valid
plan is routed to approval, so history is always well-formed.

Regression guard: history must always contain a tool message whose
tool_call_id matches every tool_call id in the preceding assistant message.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from lyra_core.agent.nodes.agent import SUBMIT_PLAN_TOOL_NAME, agent_node

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _submit_plan_resp(tool_call_id: str = "call_plan_001") -> MagicMock:
    """Fake LLM response that calls submit_plan_for_approval with a valid plan."""
    plan_payload = {
        "goal": "Send Alex Danner a message",
        "steps": [
            {
                "id": "step_1",
                "tool_name": "conversations_send_a_new_message",
                "args": {"conversation_id": "conv-abc", "message": "We'll call tomorrow at 10am"},
                "rationale": "Send follow-up to Alex Danner",
                "requires_approval": True,
            }
        ],
    }
    tc = MagicMock()
    tc.id = tool_call_id
    tc.function.name = SUBMIT_PLAN_TOOL_NAME
    tc.function.arguments = json.dumps(plan_payload)

    msg = MagicMock()
    msg.content = None
    msg.tool_calls = [tc]
    msg.reasoning_content = None

    choice = MagicMock()
    choice.message = msg

    resp = MagicMock()
    resp.choices = [choice]
    resp._hidden_params = {"response_cost": 0.0}
    return resp


def _base_state(**overrides) -> dict:
    base = {
        "tenant_id": "tenant-1",
        "job_id": "job-1",
        "channel_id": "C1",
        "thread_id": "thr",
        "user_id": "U1",
        "user_request": "Send Alex Danner a message saying we'll call tomorrow",
        "messages": [],
        "total_cost_usd": 0.0,
        "client_id": None,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Regression test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_plan_submission_closes_tool_call_loop(monkeypatch):
    """
    REGRESSION: after agent_node emits a submit_plan_for_approval tool_call,
    the saved message history must contain a matching tool response.

    Without the fix, history would be:
        [assistant(tool_calls=[{id: X}])]
    With the fix, history must be:
        [assistant(tool_calls=[{id: X}]), tool(tool_call_id=X)]

    The orphaned tool_call caused DeepSeek 400 on the next turn.
    """
    tool_call_id = "call_plan_001"

    # Patch chat() to return a plan submission response.
    mock_chat = AsyncMock(return_value=_submit_plan_resp(tool_call_id))
    monkeypatch.setattr("lyra_core.agent.nodes.agent.chat", mock_chat)

    # Patch workspace_facts so we don't need DB.
    monkeypatch.setattr(
        "lyra_core.agent.nodes.agent.get_workspace_facts",
        AsyncMock(return_value=[]),
    )

    state = _base_state()
    result = await agent_node(state)

    messages: list[dict] = result["messages"]

    # Locate the assistant message with the plan tool_call.
    assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
    assert assistant_msgs, "No assistant message saved to history"
    plan_assistant = next(
        (
            m
            for m in assistant_msgs
            if any(
                tc.get("function", {}).get("name") == SUBMIT_PLAN_TOOL_NAME
                for tc in (m.get("tool_calls") or [])
            )
        ),
        None,
    )
    assert plan_assistant is not None, "Assistant message with submit_plan_for_approval not found"

    # Every tool_call in the assistant message must have a matching tool response.
    assistant_call_ids = {tc["id"] for tc in (plan_assistant.get("tool_calls") or [])}
    tool_response_ids = {
        m["tool_call_id"] for m in messages if m.get("role") == "tool" and "tool_call_id" in m
    }

    orphaned = assistant_call_ids - tool_response_ids
    assert not orphaned, (
        f"Orphaned tool_call ids found in history (no matching tool response): {orphaned}. "
        "This would cause DeepSeek 400 on the next turn after plan rejection."
    )


@pytest.mark.asyncio
async def test_plan_rejection_followed_by_new_message_has_valid_history(monkeypatch):
    """
    REGRESSION: simulates the full reject → next-message flow.

    After a plan is rejected, the user sends a new message. The history passed
    to the LLM must satisfy: every assistant tool_call has a tool response.
    This test builds the history the way agent_node would see it on the second
    turn and verifies it would be accepted by the LLM (no orphans).
    """
    tool_call_id = "call_plan_002"

    # --- Turn 1: build the history that agent_node saves after plan submission ---
    mock_chat_turn1 = AsyncMock(return_value=_submit_plan_resp(tool_call_id))
    monkeypatch.setattr("lyra_core.agent.nodes.agent.chat", mock_chat_turn1)
    monkeypatch.setattr(
        "lyra_core.agent.nodes.agent.get_workspace_facts",
        AsyncMock(return_value=[]),
    )

    state_turn1 = _base_state()
    result_turn1 = await agent_node(state_turn1)
    history_after_plan = result_turn1["messages"]

    # Simulate: user rejects → rejection stored as a user message in history
    # (this mirrors what the adapter does when the user says "reject").
    history_after_reject = [
        *history_after_plan,
        {"role": "user", "content": "reject"},
    ]

    # --- Turn 2: new user message arrives ---
    # The new user message is appended; verify the full history has no orphans.
    history_turn2 = [
        *history_after_reject,
        {"role": "user", "content": "Actually just tell me what conversations are unread"},
    ]

    def _check_no_orphaned_tool_calls(messages: list[dict]) -> None:
        for i, msg in enumerate(messages):
            if msg.get("role") != "assistant":
                continue
            call_ids = {tc["id"] for tc in (msg.get("tool_calls") or [])}
            if not call_ids:
                continue
            # All subsequent messages until the next assistant message.
            subsequent_tool_ids = {
                m["tool_call_id"]
                for m in messages[i + 1 :]
                if m.get("role") == "tool" and "tool_call_id" in m
            }
            orphaned = call_ids - subsequent_tool_ids
            assert not orphaned, (
                f"Orphaned tool_call ids at position {i}: {orphaned}. "
                "DeepSeek would reject this history with a 400 error."
            )

    _check_no_orphaned_tool_calls(history_turn2)
