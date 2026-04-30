"""lyra_core.agent.nodes.planner."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from lyra_core.agent.nodes import planner as planner_mod
from lyra_core.agent.nodes.planner import _tool_catalog, planner_node


def test_tool_catalog_includes_tool_names_and_approval_marker() -> None:
    cat = _tool_catalog()
    assert "ghl.contacts.search" in cat
    assert "ghl.contacts.create" in cat
    assert "[requires approval]" in cat
    assert "google.drive.search" in cat


def test_tool_catalog_marks_only_write_tools_for_approval() -> None:
    cat = _tool_catalog()
    # Each tool line is on a separate line; pull lines and inspect.
    write_tools = {
        "ghl.contacts.create",
        "ghl.conversations.send_message",
        "ghl.calendars.book_appointment",
        "google.docs.create",
        "google.sheets.append",
        "google.calendar.create_event",
    }
    read_tools = {"google.drive.search", "google.sheets.read", "ghl.contacts.search"}
    for line in cat.splitlines():
        for w in write_tools:
            if line.startswith(f"- {w}("):
                assert "[requires approval]" in line, w
        for r in read_tools:
            if line.startswith(f"- {r}("):
                assert "[requires approval]" not in line, r


@pytest.mark.asyncio
async def test_planner_returns_parsed_plan(monkeypatch, mock_litellm_response) -> None:
    plan = {
        "goal": "find leads in pipeline",
        "steps": [
            {
                "id": "step_1",
                "tool_name": "ghl.pipelines.opportunities",
                "args": {},
                "rationale": "fetch pipeline",
                "requires_approval": False,
                "depends_on": [],
            }
        ],
        "needs_clarification": False,
        "clarification_question": None,
    }
    resp = mock_litellm_response(json.dumps(plan), cost=0.005)
    monkeypatch.setattr(planner_mod, "chat", AsyncMock(return_value=resp))

    out = await planner_node({"user_request": "find leads", "total_cost_usd": 0.0})  # type: ignore[arg-type]
    assert out["plan"]["goal"] == "find leads in pipeline"
    assert out["plan"]["steps"][0]["tool_name"] == "ghl.pipelines.opportunities"
    assert out["total_cost_usd"] == pytest.approx(0.005)


@pytest.mark.asyncio
async def test_planner_accumulates_cost(monkeypatch, mock_litellm_response) -> None:
    resp = mock_litellm_response('{"goal":"x","steps":[],"needs_clarification":false}', cost=0.01)
    monkeypatch.setattr(planner_mod, "chat", AsyncMock(return_value=resp))

    out = await planner_node({"user_request": "x", "total_cost_usd": 0.04})  # type: ignore[arg-type]
    assert out["total_cost_usd"] == pytest.approx(0.05)


@pytest.mark.asyncio
async def test_planner_falls_back_on_invalid_json(monkeypatch, mock_litellm_response) -> None:
    resp = mock_litellm_response("definitely-not-json")
    monkeypatch.setattr(planner_mod, "chat", AsyncMock(return_value=resp))

    out = await planner_node({"user_request": "x", "total_cost_usd": 0.0})  # type: ignore[arg-type]
    assert out["plan"]["needs_clarification"] is True
    assert "rephrase" in out["plan"]["clarification_question"].lower()
    assert out["plan"]["steps"] == []


@pytest.mark.asyncio
async def test_planner_falls_back_on_invalid_step_schema(
    monkeypatch, mock_litellm_response
) -> None:
    """A step missing required field -> Pydantic ValidationError -> fallback."""
    bogus = {
        "goal": "x",
        "steps": [{"id": "step_1"}],  # missing tool_name + rationale
        "needs_clarification": False,
    }
    resp = mock_litellm_response(json.dumps(bogus))
    monkeypatch.setattr(planner_mod, "chat", AsyncMock(return_value=resp))

    out = await planner_node({"user_request": "x", "total_cost_usd": 0.0})  # type: ignore[arg-type]
    assert out["plan"]["needs_clarification"] is True


@pytest.mark.asyncio
async def test_planner_handles_clarification_field(monkeypatch, mock_litellm_response) -> None:
    plan = {
        "goal": "x",
        "steps": [],
        "needs_clarification": True,
        "clarification_question": "Which pipeline do you mean?",
    }
    resp = mock_litellm_response(json.dumps(plan))
    monkeypatch.setattr(planner_mod, "chat", AsyncMock(return_value=resp))

    out = await planner_node({"user_request": "find stuck deals", "total_cost_usd": 0.0})  # type: ignore[arg-type]
    assert out["plan"]["needs_clarification"] is True
    assert out["plan"]["clarification_question"] == "Which pipeline do you mean?"
