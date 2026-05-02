"""Unified tool-using agent node.

A single LLM call drives every turn. The model decides whether to:

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
called outside the plan path is rejected with a ToolError, preventing
the model from sneaking writes past the human approval gate. The system
prompt also instructs the model to never call write tools directly.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from langgraph.constants import END

from ...channels.schema import OutboundReply
from ...channels.slack.poster import post_reply
from ...common.llm import ModelTier, chat, estimate_cost
from ...common.logging import get_logger, phase
from ...tools.registry import default_registry
from ..living_artifact import format_artifact_for_prompt
from ..memory import get_workspace_facts
from ..state import AgentState, Plan

log = get_logger(__name__)

SUBMIT_PLAN_TOOL_NAME = "submit_plan_for_approval"

# Cap the conversation history we feed back into the LLM each turn. The
# checkpointer keeps EVERY turn for audit, but we only re-inject the most
# recent N user/assistant/tool messages into the next prompt. Without this
# cap a long-running DM thread would slowly grow the prompt to thousands
# of tokens, blowing both latency and cost. 20 messages comfortably covers
# a few back-and-forths plus their tool-call results.
MAX_HISTORY_MESSAGES = 20

SYSTEM_TEMPLATE = """You are ARLO, an autonomous AI coworker for the user's team. You operate inside Slack DMs and channels.

## Tools

You have access to two kinds of tools:

  - READ tools (search/list/fetch): call these freely to gather information.
  - WRITE tools (create/update/send/book): NEVER call these directly.

### Finding tools
Call `discover_tools(intent="what you want to do")` before any non-trivial task
to find the right tools and their exact argument schemas. Do NOT guess tool names.

### Writes require approval
For any task involving writes, call `{submit_plan_tool}` with a structured Plan
listing every write step. The user sees ONE approval card for the entire plan and
clicks Approve to authorize all writes at once. Calling a write tool directly will
fail — the write will not happen. Always go through `{submit_plan_tool}`.

WRITE TOOLS (must go through submit_plan_for_approval):
{write_tools}

Artifact tools (`artifact.pdf.from_markdown`, `artifact.chart.line`, `artifact.chart.bar`)
generate downloadable files without mutating external state — safe to call directly or include in a plan.

### Common flow
1. Call `discover_tools(intent="...")` to find relevant tools.
2. Call READ tools freely to gather information.
3. When writes are needed, call `{submit_plan_tool}` with all write steps in one plan.

Slack-native tools (ground yourself in real workspace context):
  - `slack.conversations.history` — recent messages in a channel/DM
  - `slack.conversations.replies` — all messages in a thread
  - `slack.users.info` — resolve U... user id to name/email
  - `slack.users.list` — resolve a name to a user id
  - `slack.search.messages` — workspace-wide search
  - `slack.canvas.create` — WRITE: create a canvas (use via plan)

For smalltalk and simple questions, reply directly without tool calls.

## Workspace context

Workspace facts (stable team info):
{workspace_facts}

Conversation artifact (durable facts learned in this thread):
{artifact}

Learned workflow shortcuts:
{skills}

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


def _format_skills(skills: list[dict[str, Any]]) -> str:
    if not skills:
        return "(none yet — skills are learned from repeated workflows)"
    lines = []
    for s in skills:
        desc = s.get("description") or f"{len(s.get('tool_sequence', []))} steps"
        lines.append(f"  - `{s['slug']}`: {s['name']} ({desc})")
    return "\n".join(lines)


def _build_tool_param_list(read_tools: list[Any]) -> list[dict[str, Any]]:
    schemas = [t.to_openai_schema() for t in read_tools]
    schemas.append(_submit_plan_tool_schema())
    return schemas


def _trim_history(history: list[dict[str, Any]], max_msgs: int) -> list[dict[str, Any]]:
    """Keep the most recent `max_msgs` history entries without orphaning tool results.

    OpenAI / LiteLLM tool-calling protocol: every `role: "tool"` message
    MUST be preceded (somewhere earlier in the list) by a `role: "assistant"`
    message whose `tool_calls[].id` matches the tool message's
    `tool_call_id`. A naive `history[-N:]` slice can chop off the
    assistant message and leave the tool result orphaned, which makes
    the model error or hallucinate. So if our cut point would land
    on a tool message, we walk backwards past the corresponding
    assistant message instead.
    """
    if len(history) <= max_msgs:
        return history
    cut = len(history) - max_msgs
    while cut < len(history) and history[cut].get("role") == "tool":
        cut -= 1
    cut = max(cut, 0)
    return history[cut:]


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

    async with phase("agent.workspace_facts_fetch"):
        facts = await get_workspace_facts(tenant_id)
    artifact_body = state.get("living_artifact") or {}
    active_skills = state.get("active_skills") or []
    system = SYSTEM_TEMPLATE.format(
        submit_plan_tool=SUBMIT_PLAN_TOOL_NAME,
        write_tools=_format_write_tools(write_tools),
        workspace_facts=_format_facts(facts),
        artifact=format_artifact_for_prompt(artifact_body),
        skills=_format_skills(active_skills),
    )

    # On the first turn of a fresh checkpointer thread `messages` is empty.
    # On subsequent turns (either same task after tool_node ran, or a NEW
    # user message landing in an existing DM/thread) we already have
    # history persisted by the LangGraph checkpointer. Append the new
    # user_request only when it isn't already the last user message --
    # otherwise on a follow-up DM we'd duplicate it.
    history = list(state.get("messages") or [])
    user_request = state["user_request"]
    last_user_text = next(
        (m.get("content") for m in reversed(history) if m.get("role") == "user"),
        None,
    )
    if last_user_text != user_request:
        history.append({"role": "user", "content": user_request})

    # Trim oldest messages but keep the prompt valid: never split a
    # tool_call from its matching tool result (the LLM rejects orphans).
    trimmed = _trim_history(history, MAX_HISTORY_MESSAGES)
    messages = [{"role": "system", "content": system}, *trimmed]

    tool_param_list = _build_tool_param_list(read_tools)
    async with phase(
        "agent.llm_call",
        n_messages=len(messages),
        n_tools=len(tool_param_list),
        history_len=len(history),
    ):
        resp = await chat(
            tier=ModelTier.PRIMARY,
            messages=messages,
            tools=tool_param_list,
            max_tokens=2000,
            temperature=0.3,
        )
    cost = estimate_cost(resp)
    usage = getattr(resp, "usage", None)
    log.info(
        "agent.llm_response",
        prompt_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
        completion_tokens=getattr(usage, "completion_tokens", None) if usage else None,
        cost_usd=cost,
    )
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
        async with phase("agent.post_reply", text_len=len(text)):
            await post_reply(
                tenant_id,
                OutboundReply(
                    text=text,
                    channel_id=state["channel_id"],
                    thread_ts=state.get("reply_thread_ts"),
                    assistant_status_thread_ts=state.get("assistant_status_thread_ts"),
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
