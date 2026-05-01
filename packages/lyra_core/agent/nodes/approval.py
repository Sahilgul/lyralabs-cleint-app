"""Approval gate.

If any step in the plan has requires_approval=True we render a Block Kit
preview card to Slack and call LangGraph's `interrupt()`. The graph's
state is persisted to the Postgres checkpointer; when the user clicks
Approve/Reject, the worker resumes the same thread_id with the decision.
"""

from __future__ import annotations

from typing import Any, Literal

from langgraph.types import interrupt

from ...channels.schema import OutboundReply
from ...channels.slack.poster import post_reply
from ..state import AgentState, Plan


def _plan_preview_blocks(plan: Plan, job_id: str) -> list[dict[str, Any]]:
    """Build a Block Kit preview with Approve/Reject buttons."""
    step_lines = [
        f"*{i + 1}. `{s.tool_name}`*\n   {s.rationale}"
        + (" _(write)_" if s.requires_approval else "")
        for i, s in enumerate(plan.steps)
    ]
    body = "\n\n".join(step_lines) or "_(no steps)_"

    return [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": "I'm about to do this. Approve?"},
        },
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Goal:* {plan.goal}\n\n{body}"},
        },
        {
            "type": "actions",
            "block_id": "approval",
            "elements": [
                {
                    "type": "button",
                    "style": "primary",
                    "text": {"type": "plain_text", "text": "Approve"},
                    "value": f"approve:{job_id}",
                    "action_id": "approval",
                },
                {
                    "type": "button",
                    "style": "danger",
                    "text": {"type": "plain_text", "text": "Reject"},
                    "value": f"reject:{job_id}",
                    "action_id": "approval",
                },
            ],
        },
    ]


def route_after_plan(state: AgentState) -> Literal["approval", "executor", "smalltalk_reply"]:
    plan_dict = state.get("plan")
    if not plan_dict:
        return "smalltalk_reply"
    plan = Plan.model_validate(plan_dict)
    if plan.needs_clarification:
        return "smalltalk_reply"
    if any(step.requires_approval for step in plan.steps):
        return "approval"
    return "executor"


async def approval_node(state: AgentState) -> dict[str, Any]:
    """Post a preview to Slack, then interrupt the graph.

    Reads from `pending_plan` (unified mode, set by `submit_plan_for_approval`)
    or `plan` (legacy mode, set by planner_node). On approval the pending
    plan is promoted to `plan` so the executor reads a uniform field
    regardless of which graph produced it.
    """
    plan_dict = state.get("pending_plan") or state.get("plan")
    plan = Plan.model_validate(plan_dict)
    blocks = _plan_preview_blocks(plan, state["job_id"])

    reply = OutboundReply(
        text=f"Plan ready for approval (goal: {plan.goal})",
        blocks=blocks,
        channel_id=state["channel_id"],
        thread_ts=state.get("reply_thread_ts"),
        requires_approval=True,
    )
    await post_reply(state["tenant_id"], reply)

    # interrupt() raises GraphInterrupt; the graph saves state and exits.
    # When resumed via Command(resume="approved"|"rejected") this returns it.
    decision = interrupt({"prompt": "approve_or_reject", "job_id": state["job_id"]})

    if isinstance(decision, dict):
        decision = decision.get("decision", "rejected")
    if decision not in {"approved", "rejected"}:
        decision = "rejected"

    return {
        "approval_decision": decision,
        # Promote pending_plan -> plan so the executor sees the same
        # `state["plan"]` shape as the legacy graph.
        "plan": plan_dict,
        "pending_plan": None,
    }


def route_after_approval(state: AgentState) -> Literal["executor", "rejected_reply"]:
    return "executor" if state.get("approval_decision") == "approved" else "rejected_reply"


async def rejected_reply_node(state: AgentState) -> dict[str, Any]:
    reply = OutboundReply(
        text="Got it - rejected. I won't do anything. Tell me what to change.",
        channel_id=state["channel_id"],
        thread_ts=state.get("reply_thread_ts"),
    )
    await post_reply(state["tenant_id"], reply)
    return {"final_summary": "rejected_by_user"}
