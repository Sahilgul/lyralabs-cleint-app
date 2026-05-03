"""lyra_core.tools.base."""

from __future__ import annotations

import pytest
from lyra_core.tools.base import (
    ApprovalRequired,
    Tool,
    ToolContext,
    ToolError,
    ToolResult,
)
from pydantic import BaseModel


class _InOk(BaseModel):
    n: int


class _OutOk(BaseModel):
    doubled: int


class _OkTool(Tool[_InOk, _OutOk]):
    name = "test.ok"
    description = "doubles a number"
    provider = ""
    Input = _InOk
    Output = _OutOk

    async def run(self, ctx: ToolContext, args: _InOk) -> _OutOk:
        return _OutOk(doubled=args.n * 2)


class _RaisesTool(Tool[_InOk, _OutOk]):
    name = "test.raises"
    description = "raises ToolError"
    provider = ""
    Input = _InOk
    Output = _OutOk

    async def run(self, ctx: ToolContext, args: _InOk) -> _OutOk:
        raise ToolError("boom")


class _CrashesTool(Tool[_InOk, _OutOk]):
    name = "test.crash"
    description = "raises generic exception"
    provider = ""
    Input = _InOk
    Output = _OutOk

    async def run(self, ctx: ToolContext, args: _InOk) -> _OutOk:
        raise RuntimeError("kaboom")


class _ApprovalTool(Tool[_InOk, _OutOk]):
    name = "test.approval"
    description = "requires approval"
    requires_approval = True
    Input = _InOk
    Output = _OutOk

    async def run(self, ctx: ToolContext, args: _InOk) -> _OutOk:
        raise ApprovalRequired(preview={"n": args.n})


class TestToolContext:
    def test_default_dry_run_false(self) -> None:
        c = ToolContext(tenant_id="t-1")
        assert c.dry_run is False

    def test_extra_default_empty(self) -> None:
        c = ToolContext(tenant_id="t-1")
        assert c.extra == {}

    def test_arbitrary_creds_lookup_allowed(self) -> None:
        c = ToolContext(tenant_id="t-1", creds_lookup=lambda p: None)
        assert c.creds_lookup is not None


class TestToOpenaiSchema:
    def test_includes_function_name_and_description(self) -> None:
        s = _OkTool().to_openai_schema()
        assert s["type"] == "function"
        assert s["function"]["name"] == "test.ok"
        assert "doubles" in s["function"]["description"]
        assert s["function"]["parameters"]["type"] == "object"
        assert "n" in s["function"]["parameters"]["properties"]


class TestSafeRun:
    @pytest.mark.asyncio
    async def test_safe_run_success(self) -> None:
        result = await _OkTool().safe_run(ToolContext(tenant_id="t-1"), _InOk(n=3))
        assert result.ok is True
        assert result.error is None
        assert result.data is not None
        assert result.data.doubled == 6

    @pytest.mark.asyncio
    async def test_safe_run_tool_error(self) -> None:
        result = await _RaisesTool().safe_run(ToolContext(tenant_id="t-1"), _InOk(n=1))
        assert result.ok is False
        assert result.data is None
        assert "boom" in (result.error or "")

    @pytest.mark.asyncio
    async def test_safe_run_unexpected_exception_caught(self) -> None:
        result = await _CrashesTool().safe_run(ToolContext(tenant_id="t-1"), _InOk(n=1))
        assert result.ok is False
        assert "RuntimeError" in (result.error or "")
        assert "kaboom" in (result.error or "")

    @pytest.mark.asyncio
    async def test_safe_run_propagates_approval_required(self) -> None:
        with pytest.raises(ApprovalRequired) as exc_info:
            await _ApprovalTool().safe_run(ToolContext(tenant_id="t-1"), _InOk(n=42))
        assert exc_info.value.preview == {"n": 42}


class TestToolResult:
    def test_ok_with_data(self) -> None:
        r = ToolResult[_OutOk](ok=True, data=_OutOk(doubled=2))
        assert r.ok and r.data and r.data.doubled == 2

    def test_failure_with_error(self) -> None:
        r = ToolResult[_OutOk](ok=False, error="x")
        assert r.ok is False
        assert r.cost_usd == 0.0
