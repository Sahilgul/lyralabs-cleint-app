"""lyra_core.agent.nodes.executor."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from lyra_core.agent.nodes import executor as executor_mod
from lyra_core.agent.nodes.executor import _resolve_args, executor_node
from lyra_core.agent.state import Plan, PlanStep
from lyra_core.tools.base import Tool, ToolContext
from lyra_core.tools.registry import default_registry
from pydantic import BaseModel


class _In(BaseModel):
    value: Any | None = None
    name: str | None = None


class _Out(BaseModel):
    echoed: str | None = None
    nested: dict | None = None


class _EchoTool(Tool[_In, _Out]):
    name = "test.echo"
    description = "echo"
    provider = "test"
    Input = _In
    Output = _Out

    async def run(self, ctx: ToolContext, args: _In) -> _Out:
        return _Out(echoed=str(args.value or args.name), nested={"id": "abc-123"})


class _BadTool(Tool[_In, _Out]):
    name = "test.bad"
    description = "raises"
    provider = "test"
    Input = _In
    Output = _Out

    async def run(self, ctx: ToolContext, args: _In) -> _Out:
        raise RuntimeError("boom")


class _StrictIn(BaseModel):
    query: str  # required — no default


class _StrictTool(Tool[_StrictIn, _Out]):
    """Tool with a required arg; used to test arg-validation failure paths."""

    name = "test.strict"
    description = "strict"
    provider = "test"
    Input = _StrictIn
    Output = _Out

    async def run(self, ctx: ToolContext, args: _StrictIn) -> _Out:
        return _Out(echoed=args.query)


@pytest.fixture(autouse=True)
def _register_test_tools():
    if "test.echo" not in {t.name for t in default_registry.all()}:
        default_registry.register(_EchoTool())
    if "test.bad" not in {t.name for t in default_registry.all()}:
        default_registry.register(_BadTool())
    if "test.strict" not in {t.name for t in default_registry.all()}:
        default_registry.register(_StrictTool())
    yield


# -----------------------------------------------------------------------------
# _resolve_args
# -----------------------------------------------------------------------------


class TestResolveArgs:
    def test_no_placeholders_passes_through(self) -> None:
        out = _resolve_args({"x": 1, "y": "two"}, prior={})
        assert out == {"x": 1, "y": "two"}

    def test_substitutes_simple_value(self) -> None:
        prior = {"step_1": {"id": "abc"}}
        out = _resolve_args({"x": "{{ step_1.id }}"}, prior=prior)
        assert out == {"x": "abc"}

    def test_substitutes_in_nested_dict(self) -> None:
        prior = {"step_1": {"name": "Alice"}}
        out = _resolve_args({"obj": {"who": "Hi {{ step_1.name }}"}}, prior=prior)
        assert out == {"obj": {"who": "Hi Alice"}}

    def test_substitutes_in_list(self) -> None:
        prior = {"step_1": {"id": "x"}}
        out = _resolve_args({"items": ["a", "{{ step_1.id }}", "z"]}, prior=prior)
        assert out == {"items": ["a", "x", "z"]}

    def test_unknown_step_left_unsubstituted(self) -> None:
        out = _resolve_args({"k": "{{ step_99.id }}"}, prior={})
        assert out == {"k": "{{ step_99.id }}"}

    def test_missing_path_returns_string_none(self) -> None:
        prior = {"step_1": {"id": "a"}}
        out = _resolve_args({"k": "{{ step_1.missing }}"}, prior=prior)
        assert out == {"k": "None"}

    def test_list_index_path_supported(self) -> None:
        prior = {"step_1": {"items": ["a", "b", "c"]}}
        out = _resolve_args({"k": "{{ step_1.items.1 }}"}, prior=prior)
        assert out == {"k": "b"}


# -----------------------------------------------------------------------------
# executor_node
# -----------------------------------------------------------------------------


def _state_with_plan(plan: Plan) -> dict:
    return {
        "tenant_id": "tenant-1",
        "job_id": "job-1",
        "user_id": "user-1",
        "user_request": "x",
        "plan": plan.model_dump(),
        "step_results": [],
        "artifacts": [],
        "total_cost_usd": 0.0,
    }


@pytest.mark.asyncio
async def test_executor_runs_single_step(monkeypatch, mock_session) -> None:
    monkeypatch.setattr(executor_mod, "get_credentials", AsyncMock(return_value=None))

    class _CM:
        async def __aenter__(self):
            return mock_session

        async def __aexit__(self, *_):
            return False

    monkeypatch.setattr(executor_mod, "async_session", lambda: _CM())

    plan = Plan(
        goal="g",
        steps=[PlanStep(id="step_1", tool_name="test.echo", args={"value": "hi"}, rationale="r")],
    )
    out = await executor_node(_state_with_plan(plan))  # type: ignore[arg-type]
    assert len(out["step_results"]) == 1
    res = out["step_results"][0]
    assert res["ok"] is True
    assert res["data"]["echoed"] == "hi"
    mock_session.commit.assert_awaited()


@pytest.mark.asyncio
async def test_executor_resolves_step_dependencies(monkeypatch, mock_session) -> None:
    monkeypatch.setattr(executor_mod, "get_credentials", AsyncMock(return_value=None))

    class _CM:
        async def __aenter__(self):
            return mock_session

        async def __aexit__(self, *_):
            return False

    monkeypatch.setattr(executor_mod, "async_session", lambda: _CM())

    plan = Plan(
        goal="g",
        steps=[
            PlanStep(id="step_1", tool_name="test.echo", args={"name": "Alice"}, rationale="r"),
            PlanStep(
                id="step_2",
                tool_name="test.echo",
                args={"value": "id={{ step_1.nested.id }}"},
                rationale="r2",
                depends_on=["step_1"],
            ),
        ],
    )
    out = await executor_node(_state_with_plan(plan))  # type: ignore[arg-type]
    assert out["step_results"][1]["data"]["echoed"] == "id=abc-123"


@pytest.mark.asyncio
async def test_executor_unknown_tool_records_error(monkeypatch, mock_session) -> None:
    monkeypatch.setattr(executor_mod, "get_credentials", AsyncMock(return_value=None))

    class _CM:
        async def __aenter__(self):
            return mock_session

        async def __aexit__(self, *_):
            return False

    monkeypatch.setattr(executor_mod, "async_session", lambda: _CM())

    plan = Plan(
        goal="g",
        steps=[PlanStep(id="step_1", tool_name="does.not.exist", rationale="r")],
    )
    out = await executor_node(_state_with_plan(plan))  # type: ignore[arg-type]
    assert out["step_results"][0]["ok"] is False
    assert "unknown tool" in out["step_results"][0]["error"]


@pytest.mark.asyncio
async def test_executor_arg_validation_failure_records_error(monkeypatch, mock_session) -> None:
    """Pass an arg type a Pydantic model can't validate."""
    monkeypatch.setattr(executor_mod, "get_credentials", AsyncMock(return_value=None))

    class _CM:
        async def __aenter__(self):
            return mock_session

        async def __aexit__(self, *_):
            return False

    monkeypatch.setattr(executor_mod, "async_session", lambda: _CM())

    # test.strict.Input requires 'query: str' — passing empty args must fail validation
    plan = Plan(
        goal="g",
        steps=[PlanStep(id="step_1", tool_name="test.strict", rationale="r", args={})],
    )
    out = await executor_node(_state_with_plan(plan))  # type: ignore[arg-type]
    assert out["step_results"][0]["ok"] is False
    assert "arg validation" in out["step_results"][0]["error"]


@pytest.mark.asyncio
async def test_executor_breaks_on_first_failure(monkeypatch, mock_session) -> None:
    monkeypatch.setattr(executor_mod, "get_credentials", AsyncMock(return_value=None))

    class _CM:
        async def __aenter__(self):
            return mock_session

        async def __aexit__(self, *_):
            return False

    monkeypatch.setattr(executor_mod, "async_session", lambda: _CM())

    plan = Plan(
        goal="g",
        steps=[
            PlanStep(id="step_1", tool_name="test.bad", rationale="r"),
            PlanStep(id="step_2", tool_name="test.echo", args={"value": "v"}, rationale="r"),
        ],
    )
    out = await executor_node(_state_with_plan(plan))  # type: ignore[arg-type]
    # First step recorded as failed, second never ran
    assert len(out["step_results"]) == 1
    assert out["step_results"][0]["ok"] is False


@pytest.mark.asyncio
async def test_executor_lifts_artifacts_from_ctx(monkeypatch, mock_session) -> None:
    """Tools that append to ctx.extra['artifacts'] flow into state.artifacts."""
    monkeypatch.setattr(executor_mod, "get_credentials", AsyncMock(return_value=None))

    class _CM:
        async def __aenter__(self):
            return mock_session

        async def __aexit__(self, *_):
            return False

    monkeypatch.setattr(executor_mod, "async_session", lambda: _CM())

    class _ArtifactTool(Tool[_In, _Out]):
        name = "test.artifact"
        description = "appends artifact"
        provider = "test"
        Input = _In
        Output = _Out

        async def run(self, ctx: ToolContext, args: _In) -> _Out:
            ctx.extra.setdefault("artifacts", []).append(
                {"kind": "pdf", "filename": "x.pdf", "content_b64": "AAA", "description": "d"}
            )
            return _Out(echoed="ok")

    if "test.artifact" not in {t.name for t in default_registry.all()}:
        default_registry.register(_ArtifactTool())

    plan = Plan(
        goal="g",
        steps=[PlanStep(id="step_1", tool_name="test.artifact", rationale="r")],
    )
    out = await executor_node(_state_with_plan(plan))  # type: ignore[arg-type]
    assert len(out["artifacts"]) == 1
    assert out["artifacts"][0]["filename"] == "x.pdf"
