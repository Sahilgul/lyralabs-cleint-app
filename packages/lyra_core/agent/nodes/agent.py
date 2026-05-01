"""Unified tool-using agent node.

A single LLM call replaces the legacy classifier->planner->smalltalk graph.
The model decides per turn whether to:

  1. Reply directly  -- no tool calls. We post the text to Slack and end.
  2. Call a read-only tool  -- routed to `tool_node`, which executes and
     loops back so the model can incorporate the result.
  3. Call `submit_plan_for_approval`  -- a meta-tool that packages a
     structured Plan; sets `state["pending_plan"]` so the graph routes
     to the approval gate. After the user clicks Approve, the existing
     executor + critic nodes run unchanged.

Why a meta-tool instead of per-write-tool approval: today's UX shows ONE
approval card listing every planned write, and the user clicks Approve
once. Surfacing every write tool to the model directly would mean N
separate approval clicks per multi-step job -- both worse UX and a
security regression vs. legacy.

The hard guardrail lives in `tool_node`: any write tool (requires_approval=True)
called outside the plan path is rejected with a ToolError. The system
prompt also instructs the model to never call write tools directly.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from langgraph.constants import END

from ...channels.schema import OutboundReply
from ...channels.slack.poster import post_reply
from ...common.llm import ModelTier, chat, estimate_cost
from ...common.logging import get_logger
from ...tools.registry import default_registry
from ..memory import get_workspace_facts
from ..state import AgentState, Plan

log = get_logger(__name__)

SUBMIT_PLAN_TOOL_NAME = "submit_plan_for_approval"

SYSTEM_TEMPLATE = """You are ARLO, an autonomous AI coworker for the user's team. You operate inside Slack DMs and channels.

You have access to two kinds of tools:

  - READ tools (search/list/fetch): you may call these freely as you reason.
  - WRITE tools (create/update/send): you must NEVER call these directly.

Whenever the user's request requires writes, call `{submit_plan_tool}` with a
structured Plan listing every write step. The user sees ONE approval card
covering the entire plan and clicks Approve to authorize all writes at once.
Calling a write tool directly will fail with a ToolError -- the user will
not be informed and the work will not be done. Always go through
`{submit_plan_tool}` for writes.

WRITE TOOLS (require submit_plan_for_approval):
{write_tools}

Artifact tools (`artifact.pdf.from_markdown`, `artifact.chart.line`,
`artifact.chart.bar`) generate downloadable files but DO NOT mutate
external state, so they are safe to call directly OR include in a plan.

For smalltalk ("hi", "thanks") and simple questions, reply directly --
no tool calls.

For research that only reads data, call read tools, see the results,
and synthesize an answer. Don't ask the user permission to read.

Workspace facts (durable info you should remember about this team):
{workspace_facts}

Keep replies concise and friendly. Your response will be posted into Slack."""


def _split_tools() -> tuple[list[Any], list[Any]]:
    """Return (read_tools, write_tools) from the global registry."""
    reads, writes = [], []
    for tool in default_registry.all():
        (writes if tool.requires_approval else reads).append(tool)
    return reads, writes


def _submit_plan_tool_schema() -> dict[str, Any]:
    """OpenAI-format spec for the submit_plan_for_approval meta-tool."""
    return {
        "type": "function",
        "function": {
            "name": SUBMIT_PLAN_TOOL_NAME,
            "description": (
                "Propose a multi-step plan that includes write actions. The user will see "
                "an approval card listing every step and click Approve to run them all, or "
                "Reject to cancel. Use this for ANY task involving writes (creating docs, "
                "sending messages, booking meetings, creating contacts, etc). Steps run in "
                "order; reference earlier steps with {{ step_1.field }} placeholders."
            ),
            "parameters": Plan.model_json_schema(),
        },
    }


def _format_write_tools(write_tools: list[Any]) -> str:
    if not write_tools:
        return "(no write tools registered)"
    return "\n".join(f"  - {t.name}: {t.description}" for t in write_tools)


def _format_facts(facts: dict[str, Any]) -> str:
    if not facts:
        return "(none yet)"
    return "\n".join(f"  - {k}: {v}" for k, v in facts.items())


def _build_tool_param_list(read_tools: list[Any]) -> list[dict[str, Any]]:
    schemas = [t.to_openai_schema() for t in read_tools]
    schemas.append(_submit_plan_tool_schema())
    return schemas


def _serialize_assistant_message(msg: Any) -> dict[str, Any]:
    """Convert a LiteLLM assistant message into a plain dict for state.messages."""
    out: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
    tool_calls = getattr(msg, "tool_calls", None) or []
    if tool_calls:
        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in tool_calls
        ]
    return out


def _extract_submit_plan_call(
    tool_calls: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, str | None]:
    """If any tool call is submit_plan_for_approval, return (plan_dict, tool_call_id)."""
    for tc in tool_calls:
        fn = tc.get("function") or {}
        if fn.get("name") == SUBMIT_PLAN_TOOL_NAME:
            args_raw = fn.get("arguments") or "{}"
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            except json.JSONDecodeError:
                args = {}
            return args, tc.get("id")
    return None, None


async def agent_node(state: AgentState) -> dict[str, Any]:
    """Single tool-using LLM step. Routes via state mutations only."""
    tenant_id = state["tenant_id"]
    read_tools, write_tools = _split_tools()

    facts = await get_workspace_facts(tenant_id)
    system = SYSTEM_TEMPLATE.format(
        submit_plan_tool=SUBMIT_PLAN_TOOL_NAME,
        write_tools=_format_write_tools(write_tools),
        workspace_facts=_format_facts(facts),
    )

    # On the first turn `messages` is empty -- seed with the user's request.
    # On subsequent turns (after tool_node ran) we already have history.
    history = list(state.get("messages") or [])
    if not history:
        history.append({"role": "user", "content": state["user_request"]})
    messages = [{"role": "system", "content": system}, *history]

    resp = await chat(
        tier=ModelTier.PRIMARY,
        messages=messages,
        tools=_build_tool_param_list(read_tools),
        max_tokens=2000,
        temperature=0.3,
    )
    cost = estimate_cost(resp)
    assistant_msg = resp.choices[0].message
    serialized = _serialize_assistant_message(assistant_msg)
    new_history = [*history, serialized]

    base_update: dict[str, Any] = {
        "messages": new_history,
        "total_cost_usd": state.get("total_cost_usd", 0.0) + cost,
    }

    tool_calls = serialized.get("tool_calls") or []

    # Case 1: agent submitted a plan -> route to approval.
    plan_args, _ = _extract_submit_plan_call(tool_calls)
    if plan_args is not None:
        try:
            plan = Plan.model_validate(plan_args)
        except Exception as exc:
            # Malformed plan -- ask the agent to retry by feeding back the error.
            log.warning("agent.plan_validation_failed", error=str(exc))
            return {
                **base_update,
                "messages": [
                    *new_history,
                    {
                        "role": "tool",
                        "tool_call_id": tool_calls[0]["id"],
                        "content": f"Plan rejected: {exc}. Fix and resubmit.",
                    },
                ],
            }
        return {**base_update, "pending_plan": plan.model_dump()}

    # Case 2: agent called a read tool -> route to tool_node loop.
    if tool_calls:
        return base_update

    # Case 3: direct reply.
    text = assistant_msg.content or ""
    if text.strip():
        await post_reply(
            tenant_id,
            OutboundReply(
                text=text,
                channel_id=state["channel_id"],
                thread_ts=state.get("reply_thread_ts"),
            ),
        )
    return {**base_update, "final_summary": text}


def route_after_agent(state: AgentState) -> Literal["tool_node", "approval", "__end__"]:
    """Branch based on what the agent emitted on the last turn."""
    if state.get("pending_plan"):
        return "approval"
    msgs = state.get("messages") or []
    if msgs and msgs[-1].get("tool_calls"):
        return "tool_node"
    return END
