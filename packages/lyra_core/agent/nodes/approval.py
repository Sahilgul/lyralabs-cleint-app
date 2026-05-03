"""Approval gate with trust gradient and rehearsal engine.

Trust tiers:
  LOW    — all steps are reads; auto-approve, no interrupt.
  MEDIUM — writes with limited blast radius; Approve/Reject button (current UX).
  HIGH   — irreversible or bulk writes; user must type "I confirm".

The rehearsal engine concurrently calls `tool.simulate()` for every
non-LOW step (1.5 s hard timeout per step) and injects the previews into
the Block Kit card so the user sees exactly what will happen.
"""

from __future__ import annotations

import asyncio
from typing import Any, Literal

from langgraph.types import interrupt

from ...channels.schema import OutboundReply
from ...channels.slack.poster import post_reply
from ...tools.base import RiskProfile, TrustTier
from ..state import AgentState, Plan
from ..trust import classify_step, overall_plan_tier


async def _run_rehearsal(
    plan: Plan,
    profiles: list[RiskProfile],
    tenant_id: str,
    job_id: str | None,
    user_id: str | None,
    client_id: str | None,
) -> dict[str, str]:
    """Concurrently simulate each non-LOW step. Hard 1.5 s timeout per step."""
    from ...tools.base import ToolContext
    from ...tools.credentials import get_credentials
    from ...tools.registry import default_registry

    async def creds_lookup(provider: str) -> Any:
        return await get_credentials(tenant_id, provider, client_id)

    sim_ctx = ToolContext(
        tenant_id=tenant_id,
        job_id=job_id,
        user_id=user_id,
        client_id=client_id,
        simulation_mode=True,
        creds_lookup=creds_lookup,
    )

    async def _sim_one(step: Any, profile: RiskProfile) -> tuple[str, str]:
        if profile.tier == TrustTier.LOW:
            return step.id, ""
        try:
            tool = default_registry.get(step.tool_name)
            args_obj = tool.Input(**step.args)
            text = await asyncio.wait_for(tool.simulate(sim_ctx, args_obj), timeout=1.5)
        except Exception:
            text = f"Will call `{step.tool_name}`."
        return step.id, text

    results = await asyncio.gather(
        *[_sim_one(s, p) for s, p in zip(plan.steps, profiles, strict=True)]
    )
    return {sid: txt for sid, txt in results if txt}


def _plan_preview_blocks(
    plan: Plan,
    job_id: str,
    profiles: list[RiskProfile] | None = None,
    previews: dict[str, str] | None = None,
    overall_tier: TrustTier = TrustTier.MEDIUM,
) -> list[dict[str, Any]]:
    """Build a Block Kit preview card."""
    profiles = profiles or []
    previews = previews or {}

    tier_emoji = {TrustTier.LOW: "✅", TrustTier.MEDIUM: "⚠️", TrustTier.HIGH: "🚨"}

    step_lines = []
    for i, step in enumerate(plan.steps):
        profile = profiles[i] if i < len(profiles) else None
        emoji = tier_emoji.get(profile.tier, "•") if profile else "•"
        write_tag = " _(write)_" if step.requires_approval else ""
        preview = previews.get(step.id, "")
        line = f"{emoji} *{i + 1}. `{step.tool_name}`*\n   {step.rationale}{write_tag}"
        if preview:
            line += f"\n   _{preview}_"
        step_lines.append(line)

    body = "\n\n".join(step_lines) or "_(no steps)_"
    header_text = (
        "🚨 High-impact actions — please confirm"
        if overall_tier == TrustTier.HIGH
        else "I'm about to do this. Approve?"
    )
    instruction = (
        '\n\n*Reply "I confirm" in Slack to proceed, or "reject" to cancel.*'
        if overall_tier == TrustTier.HIGH
        else ""
    )
    blocks: list[dict[str, Any]] = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": header_text},
        },
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Goal:* {plan.goal}\n\n{body}{instruction}"},
        },
    ]
    if overall_tier != TrustTier.HIGH:
        blocks.append(
            {
                "type": "actions",
                "block_id": "approval",
                "elements": [
                    {
                        "type": "button",
                        "style": "primary",
                        "text": {"type": "plain_text", "text": "Approve"},
                        "value": f"approved:{job_id}",
                        "action_id": "approval_approve",
                    },
                    {
                        "type": "button",
                        "style": "danger",
                        "text": {"type": "plain_text", "text": "Reject"},
                        "value": f"rejected:{job_id}",
                        "action_id": "approval_reject",
                    },
                ],
            }
        )
    return blocks


async def approval_post_node(state: AgentState) -> dict[str, Any]:
    """Phase 1: classify plan, run rehearsal, post the preview card.

    This node completes and is checkpointed before the interrupt fires.
    LOW-only plans are auto-approved here with no card and no interrupt.
    MEDIUM/HIGH plans post a card and set needs_approval_wait=True so the
    next node (approval_wait_node) knows to interrupt.
    """
    from ...tools.registry import default_registry

    plan_dict = state.get("pending_plan") or state.get("plan")
    plan = Plan.model_validate(plan_dict)

    profiles: list[RiskProfile] = []
    for step in plan.steps:
        try:
            tool = default_registry.get(step.tool_name)
            profile = classify_step(step, tool)
        except KeyError:
            profile = RiskProfile(
                tier=TrustTier.MEDIUM, reversibility="reversible", blast_radius="single"
            )
        profiles.append(profile)
        step.trust_tier = profile.tier.value

    overall = overall_plan_tier(profiles)
    risk_profiles_dicts = [
        {
            "tier": p.tier.value,
            "reversibility": p.reversibility,
            "blast_radius": p.blast_radius,
        }
        for p in profiles
    ]

    # All-LOW plans execute immediately — no interrupt needed.
    if overall == TrustTier.LOW:
        return {
            "approval_decision": "approved",
            "needs_approval_wait": False,
            "plan": plan.model_dump(),
            "pending_plan": None,
            "risk_profiles": risk_profiles_dicts,
        }

    previews = await _run_rehearsal(
        plan=plan,
        profiles=profiles,
        tenant_id=state["tenant_id"],
        job_id=state.get("job_id"),
        user_id=state.get("user_id"),
        client_id=state.get("client_id"),
    )
    for step in plan.steps:
        if step.id in previews:
            step.simulation_preview = previews[step.id]

    blocks = _plan_preview_blocks(plan, state["job_id"], profiles, previews, overall)
    reply = OutboundReply(
        text=f"Plan ready for approval (goal: {plan.goal})",
        blocks=blocks,
        channel_id=state["channel_id"],
        thread_ts=state.get("reply_thread_ts"),
        assistant_status_thread_ts=state.get("assistant_status_thread_ts"),
        requires_approval=True,
    )
    await post_reply(state["tenant_id"], reply)

    return {
        "needs_approval_wait": True,
        "plan": plan.model_dump(),
        "pending_plan": None,
        "risk_profiles": risk_profiles_dicts,
    }


async def approval_wait_node(state: AgentState) -> dict[str, Any]:
    """Phase 2: block until the user clicks Approve or Reject.

    This is the ONLY node that calls interrupt(). Because approval_post_node
    already completed and was checkpointed, LangGraph resumes HERE on
    button click — the card is never re-posted.
    """
    decision = interrupt({"prompt": "approve_or_reject", "job_id": state["job_id"]})
    if isinstance(decision, dict):
        decision = decision.get("decision", "rejected")
    if decision not in {"approved", "rejected"}:
        decision = "rejected"

    return {
        "approval_decision": decision,
        "needs_approval_wait": False,
    }


def route_after_approval_post(state: AgentState) -> Literal["approval_wait", "executor"]:
    """After posting the card, wait for input unless LOW-tier (already approved)."""
    return "approval_wait" if state.get("needs_approval_wait") else "executor"


def route_after_approval(state: AgentState) -> Literal["executor", "rejected_reply"]:
    return "executor" if state.get("approval_decision") == "approved" else "rejected_reply"


async def rejected_reply_node(state: AgentState) -> dict[str, Any]:
    text = "Got it - rejected. I won't do anything. Tell me what to change."
    reply = OutboundReply(
        text=text,
        channel_id=state["channel_id"],
        thread_ts=state.get("reply_thread_ts"),
        assistant_status_thread_ts=state.get("assistant_status_thread_ts"),
    )
    await post_reply(state["tenant_id"], reply)
    # Record the rejection in message history so a follow-up turn knows the
    # plan was rejected (not still pending) and the model can reason about
    # what to change instead of reposting the same approval card.
    history = list(state.get("messages") or [])
    new_messages = [
        *history,
        {"role": "assistant", "content": f"[plan rejected by user]\n\n{text}"},
    ]
    return {"final_summary": "rejected_by_user", "messages": new_messages}
