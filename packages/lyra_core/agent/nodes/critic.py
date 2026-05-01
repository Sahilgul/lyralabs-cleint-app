"""Critic node: validate the executor's outputs against the original goal.

Produces a final natural-language summary, posts it to Slack, and decides
whether the result is good enough to ship or whether to retry.
"""

from __future__ import annotations

import base64
import json
from typing import Any, Literal

from ...channels.schema import Artifact, OutboundReply
from ...channels.slack.poster import post_reply
from ...common.llm import ModelTier, chat, estimate_cost
from ...common.logging import get_logger
from ..state import AgentState, Plan

log = get_logger(__name__)

SYSTEM = """You are the critic + summarizer for an AI coworker.

You see:
  - The original user request.
  - The plan that was executed.
  - The result of each tool call (ok/error + data).

Produce a JSON object:
  {
    "verdict": "ok" | "retry" | "give_up",
    "summary_for_user": "<friendly markdown reply to post in Slack>"
  }

- verdict='ok' if the executor's outputs satisfy the user's request.
- verdict='retry' if a transient error happened (rate limit, network) and rerunning would help.
- verdict='give_up' if the request can't be fulfilled (missing integration, permission denied).

Keep the summary tight: one paragraph, optionally followed by bullet highlights or a code block."""


async def critic_node(state: AgentState) -> dict[str, Any]:
    plan = Plan.model_validate(state["plan"]) if state.get("plan") else None
    results = state.get("step_results", [])

    payload = {
        "user_request": state["user_request"],
        "plan": plan.model_dump() if plan else None,
        "results": results,
    }

    resp = await chat(
        tier=ModelTier.PRIMARY,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps(payload)},
        ],
        response_format={"type": "json_object"},
        max_tokens=1000,
        temperature=0.2,
    )
    cost = estimate_cost(resp)

    raw = resp.choices[0].message.content or "{}"
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {"verdict": "give_up", "summary_for_user": "Sorry, something went wrong."}

    summary = parsed.get("summary_for_user", "Done.")
    verdict = parsed.get("verdict", "ok")

    artifacts = [
        Artifact(
            kind=a["kind"],
            filename=a["filename"],
            content=base64.b64decode(a["content_b64"]),
            description=a.get("description"),
        )
        for a in state.get("artifacts", [])
    ]

    reply = OutboundReply(
        text=summary,
        channel_id=state["channel_id"],
        thread_ts=state.get("reply_thread_ts"),
        assistant_status_thread_ts=state.get("assistant_status_thread_ts"),
        artifacts=artifacts,
    )
    await post_reply(state["tenant_id"], reply)

    return {
        "final_summary": summary,
        "total_cost_usd": state.get("total_cost_usd", 0.0) + cost,
        "_critic_verdict": verdict,
    }


def route_after_critic(state: AgentState) -> Literal["artifact", "end"]:
    """If any step produced bytes-bearing data, hand off to the artifact node.

    For MVP we always end after the critic posts the summary; the artifact
    node (PDF/chart) is wired in via planner steps that explicitly call
    artifact-generating tools.
    """
    return "end"
