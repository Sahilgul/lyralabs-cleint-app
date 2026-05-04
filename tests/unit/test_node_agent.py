"""lyra_core.agent.nodes.agent — the unified tool-using agent node."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from lyra_core.agent.nodes import agent as agent_mod
from lyra_core.agent.nodes.agent import (
    SUBMIT_PLAN_TOOL_NAME,
    agent_node,
    route_after_agent,
)


def _resp(content: str | None = None, tool_calls: list[dict] | None = None, cost: float = 0.0):
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
    }
    base.update(overrides)
    return base


@pytest.mark.asyncio
async def test_agent_node_direct_reply_posts_to_slack(monkeypatch) -> None:
    """No tool_calls -> reply text is posted to Slack and final_summary is set."""
    monkeypatch.setattr(agent_mod, "chat", AsyncMock(return_value=_resp(content="Hi there!")))
    monkeypatch.setattr(agent_mod, "get_workspace_facts", AsyncMock(return_value={}))
    posted = AsyncMock()
    monkeypatch.setattr(agent_mod, "post_reply", posted)

    out = await agent_node(_state(user_request="hi"))

    assert out["final_summary"] == "Hi there!"
    assert posted.await_count == 1
    # Conversation history captures both the user msg and the assistant reply.
    msgs = out["messages"]
    assert msgs[0]["role"] == "user"
    assert msgs[-1]["role"] == "assistant"
    assert msgs[-1]["content"] == "Hi there!"
    assert "tool_calls" not in msgs[-1]
    assert out.get("pending_plan") is None


@pytest.mark.asyncio
async def test_agent_node_read_tool_call_routes_to_tool_node(monkeypatch) -> None:
    """A read-tool call returns updated messages, no pending_plan, no slack post."""
    tool_calls = [
        {
            "id": "tc-1",
            "function": {
                "name": "ghl.contacts.search",
                "arguments": json.dumps({"query": "alice"}),
            },
        }
    ]
    monkeypatch.setattr(agent_mod, "chat", AsyncMock(return_value=_resp(tool_calls=tool_calls)))
    monkeypatch.setattr(agent_mod, "get_workspace_facts", AsyncMock(return_value={}))
    posted = AsyncMock()
    monkeypatch.setattr(agent_mod, "post_reply", posted)

    out = await agent_node(_state(user_request="find alice"))

    posted.assert_not_called()
    assert out.get("pending_plan") is None
    assert out["messages"][-1]["role"] == "assistant"
    assert out["messages"][-1]["tool_calls"][0]["function"]["name"] == "ghl.contacts.search"
    assert "final_summary" not in out


@pytest.mark.asyncio
async def test_agent_node_posts_preamble_text_alongside_tool_calls(monkeypatch) -> None:
    """When the LLM produces both `content` (a progress note) and `tool_calls`
    on the same turn, the content must be posted to Slack as a pre-flight
    note BEFORE the tools run -- this is what makes 'speak, don't react'
    work for long tasks. Without this hook, the content was swallowed and
    the user only saw the final reply N seconds later."""
    tool_calls = [
        {
            "id": "tc-1",
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
                tool_calls=tool_calls,
            )
        ),
    )
    monkeypatch.setattr(agent_mod, "get_workspace_facts", AsyncMock(return_value={}))
    posted = AsyncMock()
    monkeypatch.setattr(agent_mod, "post_reply", posted)

    out = await agent_node(_state(user_request="pull the stuck deals"))

    posted.assert_awaited_once()
    reply_arg = posted.await_args.args[1]
    assert reply_arg.text == "On it — pulling those stuck deals now, ~10s."
    assert reply_arg.channel_id == "C1"
    # Still routes to tool_node -- preamble doesn't end the turn.
    assert out.get("pending_plan") is None
    assert "final_summary" not in out
    assert out["messages"][-1]["tool_calls"][0]["function"]["name"] == "ghl.contacts.search"


@pytest.mark.asyncio
async def test_agent_node_skips_empty_preamble(monkeypatch) -> None:
    """No preamble text + tool_calls -> no Slack post. Avoids posting empty
    placeholder messages when the LLM doesn't volunteer a progress note."""
    tool_calls = [
        {
            "id": "tc-1",
            "function": {
                "name": "ghl.contacts.search",
                "arguments": json.dumps({"query": "x"}),
            },
        }
    ]
    monkeypatch.setattr(
        agent_mod, "chat", AsyncMock(return_value=_resp(content="   ", tool_calls=tool_calls))
    )
    monkeypatch.setattr(agent_mod, "get_workspace_facts", AsyncMock(return_value={}))
    posted = AsyncMock()
    monkeypatch.setattr(agent_mod, "post_reply", posted)

    await agent_node(_state(user_request="x"))

    posted.assert_not_called()


@pytest.mark.asyncio
async def test_agent_node_submit_plan_sets_pending_plan(monkeypatch) -> None:
    """submit_plan_for_approval populates state.pending_plan -> routes to approval."""
    plan_args = {
        "goal": "Create the doc",
        "steps": [
            {
                "id": "step_1",
                "tool_name": "google.docs.create",
                # Both title and body_text are required by the tool's Input
                # schema. agent_node now validates step args against that
                # schema before accepting the plan, so partial args are
                # rejected (see regression_test7).
                "args": {"title": "Test", "body_text": "hello"},
                "rationale": "create the doc",
                "requires_approval": True,
                "depends_on": [],
            }
        ],
        "needs_clarification": False,
        "clarification_question": None,
    }
    tool_calls = [
        {
            "id": "tc-1",
            "function": {
                "name": SUBMIT_PLAN_TOOL_NAME,
                "arguments": json.dumps(plan_args),
            },
        }
    ]
    monkeypatch.setattr(agent_mod, "chat", AsyncMock(return_value=_resp(tool_calls=tool_calls)))
    monkeypatch.setattr(agent_mod, "get_workspace_facts", AsyncMock(return_value={}))
    posted = AsyncMock()
    monkeypatch.setattr(agent_mod, "post_reply", posted)

    out = await agent_node(_state(user_request="create a doc"))

    posted.assert_not_called()
    assert out["pending_plan"]["goal"] == "Create the doc"
    assert len(out["pending_plan"]["steps"]) == 1
    assert out["pending_plan"]["steps"][0]["tool_name"] == "google.docs.create"


@pytest.mark.asyncio
async def test_agent_node_invalid_plan_pushes_error_back_to_history(monkeypatch) -> None:
    """If the agent submits a malformed plan, we feed the error back as a tool
    message so it can retry on the next loop -- and we do NOT route to approval."""
    bad_plan_args = {"goal": "x"}  # missing required `steps` field
    tool_calls = [
        {
            "id": "tc-1",
            "function": {
                "name": SUBMIT_PLAN_TOOL_NAME,
                "arguments": json.dumps(bad_plan_args),
            },
        }
    ]
    monkeypatch.setattr(agent_mod, "chat", AsyncMock(return_value=_resp(tool_calls=tool_calls)))
    monkeypatch.setattr(agent_mod, "get_workspace_facts", AsyncMock(return_value={}))

    out = await agent_node(_state(user_request="do something"))

    assert out.get("pending_plan") is None
    last = out["messages"][-1]
    assert last["role"] == "tool"
    assert last["tool_call_id"] == "tc-1"
    assert "rejected" in last["content"].lower()


@pytest.mark.asyncio
async def test_agent_node_continues_existing_history_without_reseeding(monkeypatch) -> None:
    """If `messages` already has content, agent_node must NOT re-seed user_request."""
    history = [
        {"role": "user", "content": "search alice"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "tc-1",
                    "type": "function",
                    "function": {"name": "ghl.contacts.search", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "tc-1", "content": "[]"},
    ]
    captured_messages = {}

    async def fake_chat(**kwargs):
        captured_messages["messages"] = kwargs["messages"]
        return _resp(content="No matches.")

    monkeypatch.setattr(agent_mod, "chat", fake_chat)
    monkeypatch.setattr(agent_mod, "get_workspace_facts", AsyncMock(return_value={}))
    monkeypatch.setattr(agent_mod, "post_reply", AsyncMock())

    out = await agent_node(_state(user_request="search alice", messages=history))

    # The system msg + the 3 history msgs = 4 messages sent to the LLM.
    sent = captured_messages["messages"]
    assert sent[0]["role"] == "system"
    assert sent[1]["role"] == "user"  # original user msg, not re-seeded
    assert len([m for m in sent if m["role"] == "user"]) == 1  # not duplicated
    assert out["final_summary"] == "No matches."


def test_route_after_agent_pending_plan_first() -> None:
    state = {
        "pending_plan": {"goal": "x", "steps": []},
        "messages": [{"role": "assistant", "tool_calls": [{"id": "1"}]}],
    }
    # pending_plan wins over a tool_calls-only message.
    assert route_after_agent(state) == "approval"


def test_route_after_agent_tool_calls() -> None:
    state = {
        "messages": [
            {"role": "user", "content": "x"},
            {"role": "assistant", "tool_calls": [{"id": "1"}]},
        ],
    }
    assert route_after_agent(state) == "tool_node"


def test_route_after_agent_direct_reply_ends() -> None:
    from langgraph.constants import END

    state = {
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ],
    }
    assert route_after_agent(state) == END
