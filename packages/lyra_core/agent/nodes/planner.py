"""Planner node.

Calls the primary model (Claude Sonnet) with the catalog of available
tools and asks for a structured Plan.
"""

from __future__ import annotations

import json
from typing import Any

from ...common.llm import ModelTier, chat, estimate_cost
from ...common.logging import get_logger
from ...tools.registry import default_registry
from ..state import AgentState, Plan, PlanStep

log = get_logger(__name__)

SYSTEM_TEMPLATE = """You are the planner for an autonomous AI coworker.

Given a user request, produce a JSON Plan that decomposes the work into
discrete steps. Each step calls exactly ONE tool from the catalog.

CATALOG:
{tool_catalog}

RULES:
- Use only tool names from the catalog.
- Steps run in order. Use `depends_on` to reference earlier step ids ('step_1', 'step_2', ...) when later steps need their output.
- Set `requires_approval=true` for any step that mutates external state (creates docs, sends messages, books meetings).
- If you genuinely lack info to plan (e.g. user said 'send the follow-up' but you don't know which contact), set `needs_clarification=true` with a `clarification_question`.
- Be concise. Prefer 1-5 steps. Bigger plans are usually wrong.
- If the user just wants to read/summarize data, no approval is needed.

OUTPUT FORMAT (strict JSON, nothing else):
{{
  "goal": "<1 sentence>",
  "steps": [
    {{"id":"step_1","tool_name":"...","args":{{...}},"rationale":"...","requires_approval":false,"depends_on":[]}}
  ],
  "needs_clarification": false,
  "clarification_question": null
}}"""


def _tool_catalog() -> str:
    rows: list[str] = []
    for t in default_registry.all():
        schema = t.Input.model_json_schema()
        props = schema.get("properties", {})
        required = schema.get("required", [])
        sig = ", ".join(
            f"{name}{'?' if name not in required else ''}: {prop.get('type', 'any')}"
            for name, prop in props.items()
        )
        suffix = " [requires approval]" if t.requires_approval else ""
        rows.append(f"- {t.name}({sig}): {t.description}{suffix}")
    return "\n".join(rows)


async def planner_node(state: AgentState) -> dict[str, Any]:
    catalog = _tool_catalog()
    system = SYSTEM_TEMPLATE.format(tool_catalog=catalog)

    resp = await chat(
        tier=ModelTier.PRIMARY,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": state["user_request"]},
        ],
        response_format={"type": "json_object"},
        max_tokens=2000,
        temperature=0.1,
    )

    raw = resp.choices[0].message.content or "{}"
    cost = estimate_cost(resp)
    try:
        parsed = json.loads(raw)
        plan = Plan(
            goal=parsed.get("goal", state["user_request"]),
            steps=[PlanStep(**s) for s in parsed.get("steps", [])],
            needs_clarification=parsed.get("needs_clarification", False),
            clarification_question=parsed.get("clarification_question"),
        )
    except (json.JSONDecodeError, ValueError) as exc:
        log.error("planner.parse_error", error=str(exc), raw=raw[:500])
        plan = Plan(
            goal=state["user_request"],
            steps=[],
            needs_clarification=True,
            clarification_question="I couldn't form a plan. Can you rephrase?",
        )

    return {
        "plan": plan.model_dump(),
        "total_cost_usd": state.get("total_cost_usd", 0.0) + cost,
    }
