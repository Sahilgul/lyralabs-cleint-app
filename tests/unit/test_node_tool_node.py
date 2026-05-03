"""lyra_core.agent.nodes.tool_node — read-tool executor with write-tool guard."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from lyra_core.agent.nodes import tool_node as tool_mod
from lyra_core.agent.nodes.agent import SUBMIT_PLAN_TOOL_NAME
from lyra_core.agent.nodes.tool_node import tool_node


def _state(tool_calls: list[dict] | None = None, **overrides):
    """Build a minimal state with one assistant msg holding tool_calls."""
    msgs = [{"role": "user", "content": "x"}]
    if tool_calls is not None:
        msgs.append({"role": "assistant", "content": "", "tool_calls": tool_calls})
    base = {
        "tenant_id": "tenant-1",
        "job_id": "job-1",
        "user_id": "U1",
        "messages": msgs,
        "artifacts": [],
    }
    base.update(overrides)
    return base


class _FakeTool:
    name: str = "fake.read"
    description: str = "Fake read tool"
    requires_approval: bool = False

    class Input(dict):
        def __init__(self, **kw):
            super().__init__(kw)

    def __init__(self, output_data=None, ok: bool = True, error: str | None = None):
        self._data = output_data or {}
        self._ok = ok
        self._error = error

    async def safe_run(self, ctx, args):
        result = MagicMock()
        result.ok = self._ok
        result.error = self._error
        result.cost_usd = 0.0
        if self._ok:
            data = MagicMock()
            data.model_dump.return_value = self._data
            result.data = data
        else:
            result.data = None
        return result


def _patch_audit(monkeypatch):
    """Stub out the audit DB write so tests don't need Postgres."""
    monkeypatch.setattr(tool_mod, "record_event", AsyncMock())

    class _CM:
        async def __aenter__(self):
            s = AsyncMock()
            s.commit = AsyncMock()
            return s

        async def __aexit__(self, *_):
            return False

    monkeypatch.setattr(tool_mod, "async_session", lambda: _CM())


@pytest.mark.asyncio
async def test_tool_node_executes_read_tool(monkeypatch) -> None:
    fake = _FakeTool(output_data={"hits": [{"id": "c-1", "name": "Alice"}]})
    fake.Input = lambda **kw: {**kw}  # Pydantic stub
    fake_registry = MagicMock()
    fake_registry.get = MagicMock(return_value=fake)
    monkeypatch.setattr(tool_mod, "default_registry", fake_registry)
    _patch_audit(monkeypatch)

    state = _state(
        tool_calls=[
            {
                "id": "tc-1",
                "function": {
                    "name": "fake.read",
                    "arguments": json.dumps({"query": "alice"}),
                },
            }
        ]
    )
    out = await tool_node(state)

    msgs = out["messages"]
    assert msgs[-1]["role"] == "tool"
    assert msgs[-1]["tool_call_id"] == "tc-1"
    payload = json.loads(msgs[-1]["content"])
    assert payload["hits"][0]["name"] == "Alice"


@pytest.mark.asyncio
async def test_tool_node_blocks_write_tool_with_guard(monkeypatch) -> None:
    """The hard guard: any write tool called outside the plan path is rejected.
    This is the security check that prevents the LLM from circumventing
    the human approval gate by ignoring its system prompt."""
    write_tool = _FakeTool()
    write_tool.requires_approval = True
    write_tool.name = "google.docs.create"
    fake_registry = MagicMock()
    fake_registry.get = MagicMock(return_value=write_tool)
    monkeypatch.setattr(tool_mod, "default_registry", fake_registry)
    _patch_audit(monkeypatch)

    state = _state(
        tool_calls=[
            {
                "id": "tc-1",
                "function": {
                    "name": "google.docs.create",
                    "arguments": json.dumps({"title": "Sneaky"}),
                },
            }
        ]
    )
    out = await tool_node(state)

    msgs = out["messages"]
    assert msgs[-1]["role"] == "tool"
    assert "WRITE tool" in msgs[-1]["content"]
    assert SUBMIT_PLAN_TOOL_NAME in msgs[-1]["content"]


@pytest.mark.asyncio
async def test_tool_node_unknown_tool_returns_error(monkeypatch) -> None:
    fake_registry = MagicMock()
    fake_registry.get = MagicMock(side_effect=KeyError("unknown"))
    monkeypatch.setattr(tool_mod, "default_registry", fake_registry)
    _patch_audit(monkeypatch)

    state = _state(
        tool_calls=[
            {
                "id": "tc-1",
                "function": {"name": "does.not.exist", "arguments": "{}"},
            }
        ]
    )
    out = await tool_node(state)

    assert "not found" in out["messages"][-1]["content"]


@pytest.mark.asyncio
async def test_tool_node_invalid_args_returns_error(monkeypatch) -> None:
    fake = _FakeTool()

    def _bad_input(**kw):
        raise ValueError("missing required field 'query'")

    fake.Input = _bad_input
    fake_registry = MagicMock()
    fake_registry.get = MagicMock(return_value=fake)
    monkeypatch.setattr(tool_mod, "default_registry", fake_registry)
    _patch_audit(monkeypatch)

    state = _state(
        tool_calls=[{"id": "tc-1", "function": {"name": "fake.read", "arguments": "{}"}}]
    )
    out = await tool_node(state)

    assert "Argument validation failed" in out["messages"][-1]["content"]


@pytest.mark.asyncio
async def test_tool_node_tool_error_propagated(monkeypatch) -> None:
    fake = _FakeTool(ok=False, error="rate limited by upstream")
    fake.Input = lambda **kw: {**kw}
    fake_registry = MagicMock()
    fake_registry.get = MagicMock(return_value=fake)
    monkeypatch.setattr(tool_mod, "default_registry", fake_registry)
    _patch_audit(monkeypatch)

    state = _state(
        tool_calls=[{"id": "tc-1", "function": {"name": "fake.read", "arguments": "{}"}}]
    )
    out = await tool_node(state)

    assert "rate limited" in out["messages"][-1]["content"]


@pytest.mark.asyncio
async def test_tool_node_skips_when_no_tool_calls(monkeypatch) -> None:
    out = await tool_node(_state(tool_calls=None))
    assert out == {}
