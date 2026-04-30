"""Classifier node: decide if the user wants smalltalk, a real task, or a clarification.

Uses the cheap-tier model. Sets `state['classification']`.
"""

from __future__ import annotations

import json
from typing import Literal

from ...common.llm import ModelTier, chat
from ..state import AgentState

SYSTEM = """You are a routing classifier for an AI coworker.

Classify the user's message into exactly one of:
- "smalltalk":  greetings, thanks, casual chat, no real task
- "task":       the user wants the assistant to DO something (pull data, send msg, generate report)
- "clarification": the user is replying to a previous question with new info

Return ONLY a JSON object: {"label": "<one>"}"""


async def classifier_node(state: AgentState) -> dict:
    user_request = state["user_request"]

    resp = await chat(
        tier=ModelTier.CHEAP,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_request},
        ],
        response_format={"type": "json_object"},
        max_tokens=100,
        temperature=0,
    )

    raw = resp.choices[0].message.content or "{}"
    try:
        parsed = json.loads(raw)
        label = parsed.get("label", "task")
    except json.JSONDecodeError:
        label = "task"

    if label not in {"smalltalk", "task", "clarification"}:
        label = "task"

    return {"classification": label}


def route_after_classifier(state: AgentState) -> Literal["smalltalk_reply", "planner"]:
    return "smalltalk_reply" if state.get("classification") == "smalltalk" else "planner"
