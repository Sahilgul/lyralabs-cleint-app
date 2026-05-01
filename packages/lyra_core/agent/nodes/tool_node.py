"""Tool execution node for the unified agent.

Reads the most recent assistant message from `state.messages`, executes
each tool_call (rejecting any that target a write tool), and appends the
results as `tool` role messages so the agent can see them on the next
turn.

The write-tool guard here is the load-bearing security check. The system
prompt instructs the model to call `submit_plan_for_approval` for writes,
but the prompt is advisory; this node enforces it. If the model calls a
write tool directly (whether from prompt drift, jailbreaks, or future
fine-tunes), we surface a ToolError back into the conversation rather
than executing the write. The user's approval gate is preserved.
"""

from __future__ import annotations

import json
from typing import Any

from ...common.audit import record_event
from ...common.logging import get_logger
from ...db.session import async_session
from ...tools.base import ToolContext
from ...tools.credentials import get_credentials
from ...tools.registry import default_registry
from ..state import AgentState
from .agent import SUBMIT_PLAN_TOOL_NAME

log = get_logger(__name__)


def _make_ctx(tenant_id: str, job_id: str | None, user_id: str | None) -> ToolContext:
    async def creds_lookup(provider: str):
        return await get_credentials(tenant_id, provider)

    return ToolContext(
        tenant_id=tenant_id,
        job_id=job_id,
        user_id=user_id,
        creds_lookup=creds_lookup,
    )


async def _execute_one(
    ctx: ToolContext,
    tc: dict[str, Any],
    tenant_id: str,
    job_id: str | None,
    user_id: str | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run one tool_call. Returns (tool_message_for_history, artifacts_to_lift)."""
    fn = tc.get("function") or {}
    name = fn.get("name", "")
    tool_call_id = tc.get("id", "")
    args_raw = fn.get("arguments") or "{}"

    def _err(msg: str) -> dict[str, Any]:
        return {"role": "tool", "tool_call_id": tool_call_id, "content": msg}

    # The submit_plan_for_approval call is a routing signal, not a
    # tool. agent_node already handled it; we should not get here for it.
    if name == SUBMIT_PLAN_TOOL_NAME:
        return _err("submit_plan_for_approval is handled by the approval gate."), []

    try:
        tool = default_registry.get(name)
    except KeyError:
        return _err(f"Unknown tool: {name}"), []

    # The guard. Write tools must go through the plan path.
    if tool.requires_approval:
        log.warning(
            "agent.write_tool_blocked",
            tool=name,
            tenant_id=tenant_id,
            reason="called_outside_plan",
        )
        return _err(
            f"'{name}' is a WRITE tool and cannot be called directly. "
            f"Call {SUBMIT_PLAN_TOOL_NAME} with a Plan that includes this step instead."
        ), []

    try:
        args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
    except json.JSONDecodeError as exc:
        return _err(f"Invalid JSON args: {exc}"), []

    try:
        args_obj = tool.Input(**args)
    except Exception as exc:
        return _err(f"Argument validation failed: {exc}"), []

    try:
        result = await tool.safe_run(ctx, args_obj)
    except Exception as exc:
        log.exception("tool_node.crash", tool=name)
        return _err(f"Tool crashed: {exc}"), []

    # Persist an audit event so read calls show up alongside writes.
    async with async_session() as s:
        await record_event(
            s,
            tenant_id=tenant_id,
            actor_user_id=user_id,
            job_id=job_id,
            event_type="tool_call",
            tool_name=name,
            args=args,
            result_status="ok" if result.ok else "error",
            cost_usd=result.cost_usd,
            extra={"error": result.error} if result.error else {},
        )
        await s.commit()

    if not result.ok:
        return _err(f"Tool error: {result.error or 'unknown'}"), []

    payload = result.data.model_dump() if result.data else {}
    return (
        {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": json.dumps(payload, default=str),
        },
        list(ctx.extra.get("artifacts", [])),
    )


async def tool_node(state: AgentState) -> dict[str, Any]:
    """Execute the assistant's pending tool_calls and append results."""
    msgs = list(state.get("messages") or [])
    if not msgs:
        return {}
    last = msgs[-1]
    tool_calls = last.get("tool_calls") or []
    if not tool_calls:
        return {}

    tenant_id = state["tenant_id"]
    job_id = state.get("job_id")
    user_id = state.get("user_id")
    ctx = _make_ctx(tenant_id, job_id, user_id)

    artifacts = list(state.get("artifacts", []))
    new_msgs = list(msgs)

    for tc in tool_calls:
        tool_msg, new_artifacts = await _execute_one(ctx, tc, tenant_id, job_id, user_id)
        new_msgs.append(tool_msg)
        artifacts.extend(new_artifacts)
        # Reset ctx.extra.artifacts between calls so we don't double-count.
        ctx.extra["artifacts"] = []

    return {"messages": new_msgs, "artifacts": artifacts}
