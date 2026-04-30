"""lyra_core.agent.nodes.classifier."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from lyra_core.agent.nodes import classifier as classifier_mod
from lyra_core.agent.nodes.classifier import classifier_node, route_after_classifier


@pytest.mark.asyncio
async def test_classifier_returns_smalltalk(monkeypatch, mock_litellm_response) -> None:
    monkeypatch.setattr(
        classifier_mod, "chat",
        AsyncMock(return_value=mock_litellm_response('{"label":"smalltalk"}')),
    )
    out = await classifier_node({"user_request": "hi!"})  # type: ignore[arg-type]
    assert out == {"classification": "smalltalk"}


@pytest.mark.asyncio
async def test_classifier_returns_task(monkeypatch, mock_litellm_response) -> None:
    monkeypatch.setattr(
        classifier_mod, "chat",
        AsyncMock(return_value=mock_litellm_response('{"label":"task"}')),
    )
    out = await classifier_node({"user_request": "search leads"})  # type: ignore[arg-type]
    assert out == {"classification": "task"}


@pytest.mark.asyncio
async def test_classifier_returns_clarification(monkeypatch, mock_litellm_response) -> None:
    monkeypatch.setattr(
        classifier_mod, "chat",
        AsyncMock(return_value=mock_litellm_response('{"label":"clarification"}')),
    )
    out = await classifier_node({"user_request": "the second one"})  # type: ignore[arg-type]
    assert out == {"classification": "clarification"}


@pytest.mark.asyncio
async def test_classifier_falls_back_to_task_on_invalid_json(
    monkeypatch, mock_litellm_response
) -> None:
    monkeypatch.setattr(
        classifier_mod, "chat",
        AsyncMock(return_value=mock_litellm_response("not-json")),
    )
    out = await classifier_node({"user_request": "x"})  # type: ignore[arg-type]
    assert out == {"classification": "task"}


@pytest.mark.asyncio
async def test_classifier_falls_back_to_task_on_unknown_label(
    monkeypatch, mock_litellm_response
) -> None:
    monkeypatch.setattr(
        classifier_mod, "chat",
        AsyncMock(return_value=mock_litellm_response('{"label":"weird"}')),
    )
    out = await classifier_node({"user_request": "x"})  # type: ignore[arg-type]
    assert out == {"classification": "task"}


@pytest.mark.asyncio
async def test_classifier_handles_empty_content(
    monkeypatch, mock_litellm_response
) -> None:
    """When LLM returns None content, defaults to task."""
    resp = mock_litellm_response("")
    resp.choices[0].message.content = None
    monkeypatch.setattr(classifier_mod, "chat", AsyncMock(return_value=resp))
    out = await classifier_node({"user_request": "x"})  # type: ignore[arg-type]
    assert out == {"classification": "task"}


def test_route_after_classifier_smalltalk() -> None:
    assert route_after_classifier({"classification": "smalltalk"}) == "smalltalk_reply"


def test_route_after_classifier_task() -> None:
    assert route_after_classifier({"classification": "task"}) == "planner"


def test_route_after_classifier_clarification_goes_to_planner() -> None:
    assert (
        route_after_classifier({"classification": "clarification"}) == "planner"
    )


def test_route_after_classifier_missing_defaults_to_planner() -> None:
    assert route_after_classifier({}) == "planner"  # type: ignore[arg-type]
