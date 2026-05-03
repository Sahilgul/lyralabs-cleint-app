"""Rehearsal engine: concurrent simulate() with 1.5 s timeout."""

from __future__ import annotations

import asyncio

import pytest
from lyra_core.agent.nodes.approval import _run_rehearsal
from lyra_core.agent.state import Plan, PlanStep
from lyra_core.tools.base import RiskProfile, Tool, ToolContext, TrustTier
from lyra_core.tools.registry import default_registry
from pydantic import BaseModel


class _In(BaseModel):
    msg: str = "hi"


class _Out(BaseModel):
    pass


class _SlowTool(Tool[_In, _Out]):
    name = "_test_slow_tool"
    description = "slow"
    requires_approval = True
    trust_tier = TrustTier.MEDIUM
    Input = _In
    Output = _Out

    async def run(self, ctx: ToolContext, args: _In) -> _Out:
        return _Out()

    async def simulate(self, ctx: ToolContext, args: _In) -> str:
        await asyncio.sleep(10)  # will be cancelled by timeout
        return "should not reach"


class _FastTool(Tool[_In, _Out]):
    name = "_test_fast_tool"
    description = "fast"
    requires_approval = True
    trust_tier = TrustTier.MEDIUM
    Input = _In
    Output = _Out

    async def run(self, ctx: ToolContext, args: _In) -> _Out:
        return _Out()

    async def simulate(self, ctx: ToolContext, args: _In) -> str:
        return f"Will call with msg={args.msg}"


@pytest.fixture(autouse=True)
def _register_tools():
    for tool in [_SlowTool(), _FastTool()]:
        try:
            default_registry.register(tool)
        except ValueError:
            pass  # already registered


@pytest.mark.asyncio
async def test_slow_tool_falls_back_gracefully() -> None:
    plan = Plan(
        goal="g",
        steps=[PlanStep(id="s1", tool_name="_test_slow_tool", rationale="r", args={"msg": "x"})],
    )
    profiles = [
        RiskProfile(tier=TrustTier.MEDIUM, reversibility="reversible", blast_radius="single")
    ]
    previews = await _run_rehearsal(
        plan=plan,
        profiles=profiles,
        tenant_id="t",
        job_id=None,
        user_id=None,
        client_id=None,
    )
    assert "s1" in previews
    assert "_test_slow_tool" in previews["s1"]


@pytest.mark.asyncio
async def test_fast_tool_returns_preview() -> None:
    plan = Plan(
        goal="g",
        steps=[
            PlanStep(id="s2", tool_name="_test_fast_tool", rationale="r", args={"msg": "hello"})
        ],
    )
    profiles = [
        RiskProfile(tier=TrustTier.MEDIUM, reversibility="reversible", blast_radius="single")
    ]
    previews = await _run_rehearsal(
        plan=plan,
        profiles=profiles,
        tenant_id="t",
        job_id=None,
        user_id=None,
        client_id=None,
    )
    assert previews.get("s2") == "Will call with msg=hello"


@pytest.mark.asyncio
async def test_low_steps_skipped_in_rehearsal() -> None:
    plan = Plan(
        goal="g",
        steps=[PlanStep(id="s3", tool_name="_test_fast_tool", rationale="r")],
    )
    profiles = [RiskProfile(tier=TrustTier.LOW, reversibility="reversible", blast_radius="single")]
    previews = await _run_rehearsal(
        plan=plan,
        profiles=profiles,
        tenant_id="t",
        job_id=None,
        user_id=None,
        client_id=None,
    )
    # LOW steps return empty string → filtered out
    assert "s3" not in previews
