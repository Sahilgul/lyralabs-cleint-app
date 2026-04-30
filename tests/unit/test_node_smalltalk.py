"""lyra_core.agent.nodes.smalltalk."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from lyra_core.agent.nodes import smalltalk as st_mod
from lyra_core.agent.nodes.smalltalk import smalltalk_reply_node
from lyra_core.agent.state import Plan


def _state(**overrides):
    base = {
        "user_request": "hi there",
        "thread_id": "thr",
        "channel_id": "ch",
        "tenant_id": "ten",
    }
    base.update(overrides)
    return base


@pytest.mark.asyncio
async def test_uses_clarification_question_when_present(monkeypatch) -> None:
    monkeypatch.setattr(st_mod, "post_reply", AsyncMock())

    plan = Plan(
        goal="x",
        steps=[],
        needs_clarification=True,
        clarification_question="Which pipeline?",
    )
    out = await smalltalk_reply_node(_state(plan=plan.model_dump()))  # type: ignore[arg-type]
    assert out["final_summary"] == "Which pipeline?"


@pytest.mark.asyncio
async def test_falls_back_to_quick_reply(monkeypatch, mock_litellm_response) -> None:
    monkeypatch.setattr(st_mod, "post_reply", AsyncMock())
    monkeypatch.setattr(
        st_mod, "chat", AsyncMock(return_value=mock_litellm_response("Hey there!"))
    )

    out = await smalltalk_reply_node(_state())  # type: ignore[arg-type]
    assert out["final_summary"] == "Hey there!"


@pytest.mark.asyncio
async def test_quick_reply_falls_back_to_hi_when_blank(
    monkeypatch, mock_litellm_response
) -> None:
    resp = mock_litellm_response("")
    resp.choices[0].message.content = None
    monkeypatch.setattr(st_mod, "post_reply", AsyncMock())
    monkeypatch.setattr(st_mod, "chat", AsyncMock(return_value=resp))

    out = await smalltalk_reply_node(_state())  # type: ignore[arg-type]
    assert out["final_summary"] == "Hi!"


@pytest.mark.asyncio
async def test_plan_without_clarification_uses_quick_reply(
    monkeypatch, mock_litellm_response
) -> None:
    monkeypatch.setattr(st_mod, "post_reply", AsyncMock())
    monkeypatch.setattr(
        st_mod, "chat", AsyncMock(return_value=mock_litellm_response("hi reply"))
    )

    plan = Plan(goal="g", steps=[])  # not needs_clarification
    out = await smalltalk_reply_node(_state(plan=plan.model_dump()))  # type: ignore[arg-type]
    assert out["final_summary"] == "hi reply"
