"""Smalltalk + clarification reply node."""

from __future__ import annotations

from typing import Any

from ...channels.schema import OutboundReply
from ...channels.slack.poster import post_reply
from ...common.llm import ModelTier, chat
from ..state import AgentState, Plan

SYSTEM = (
    "You are a friendly AI coworker. Reply in 1-2 sentences. "
    "If the user is making smalltalk, reply warmly. If you need more info to act, ask."
)


async def smalltalk_reply_node(state: AgentState) -> dict[str, Any]:
    plan_dict = state.get("plan")
    if plan_dict:
        plan = Plan.model_validate(plan_dict)
        if plan.needs_clarification and plan.clarification_question:
            text = plan.clarification_question
        else:
            text = await _quick_reply(state["user_request"])
    else:
        text = await _quick_reply(state["user_request"])

    await post_reply(
        state["tenant_id"],
        OutboundReply(
            text=text,
            thread_id=state["thread_id"],
            channel_id=state["channel_id"],
        ),
    )
    return {"final_summary": text}


async def _quick_reply(user_request: str) -> str:
    resp = await chat(
        tier=ModelTier.CHEAP,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_request},
        ],
        max_tokens=200,
        temperature=0.5,
    )
    return resp.choices[0].message.content or "Hi!"
