"""lyra_core.agent.trust — trust tier classification."""

from __future__ import annotations

from lyra_core.agent.state import PlanStep
from lyra_core.agent.trust import classify_step, overall_plan_tier
from lyra_core.tools.base import RiskProfile, Tool, ToolContext, TrustTier
from pydantic import BaseModel


class _In(BaseModel):
    pass


class _Out(BaseModel):
    pass


class _ReadTool(Tool[_In, _Out]):
    name = "test.read"
    description = "reads data"
    requires_approval = False
    trust_tier = TrustTier.LOW
    blast_radius = "single"
    Input = _In
    Output = _Out

    async def run(self, ctx: ToolContext, args: _In) -> _Out:
        return _Out()


class _WriteTool(Tool[_In, _Out]):
    name = "test.write"
    description = "creates something"
    requires_approval = True
    trust_tier = TrustTier.MEDIUM
    blast_radius = "single"
    Input = _In
    Output = _Out

    async def run(self, ctx: ToolContext, args: _In) -> _Out:
        return _Out()


class _BulkTool(Tool[_In, _Out]):
    name = "test.bulk"
    description = "sends SMS to many"
    requires_approval = True
    trust_tier = TrustTier.HIGH
    blast_radius = "bulk"
    Input = _In
    Output = _Out

    async def run(self, ctx: ToolContext, args: _In) -> _Out:
        return _Out()


def _step(tool_name: str) -> PlanStep:
    return PlanStep(id="s1", tool_name=tool_name, rationale="r")


class TestClassifyStep:
    def test_read_tool_is_low(self) -> None:
        profile = classify_step(_step("test.read"), _ReadTool())
        assert profile.tier == TrustTier.LOW
        assert profile.reversibility == "reversible"

    def test_write_tool_is_medium(self) -> None:
        profile = classify_step(_step("test.write"), _WriteTool())
        assert profile.tier == TrustTier.MEDIUM

    def test_bulk_tool_is_high(self) -> None:
        profile = classify_step(_step("test.bulk"), _BulkTool())
        assert profile.tier == TrustTier.HIGH
        assert profile.blast_radius == "bulk"

    def test_requires_approval_false_always_low(self) -> None:
        """Even if trust_tier=HIGH is declared, requires_approval=False wins."""

        class _Weird(Tool[_In, _Out]):
            name = "test.weird"
            description = "weird"
            requires_approval = False
            trust_tier = TrustTier.HIGH
            blast_radius = "bulk"
            Input = _In
            Output = _Out

            async def run(self, ctx, args):
                return _Out()

        profile = classify_step(_step("test.weird"), _Weird())
        assert profile.tier == TrustTier.LOW


class TestOverallPlanTier:
    def test_empty_is_low(self) -> None:
        assert overall_plan_tier([]) == TrustTier.LOW

    def test_all_low(self) -> None:
        profiles = [
            RiskProfile(tier=TrustTier.LOW, reversibility="reversible", blast_radius="single"),
            RiskProfile(tier=TrustTier.LOW, reversibility="reversible", blast_radius="single"),
        ]
        assert overall_plan_tier(profiles) == TrustTier.LOW

    def test_one_high_wins(self) -> None:
        profiles = [
            RiskProfile(tier=TrustTier.LOW, reversibility="reversible", blast_radius="single"),
            RiskProfile(tier=TrustTier.HIGH, reversibility="irreversible", blast_radius="bulk"),
        ]
        assert overall_plan_tier(profiles) == TrustTier.HIGH

    def test_medium_beats_low(self) -> None:
        profiles = [
            RiskProfile(tier=TrustTier.LOW, reversibility="reversible", blast_radius="single"),
            RiskProfile(tier=TrustTier.MEDIUM, reversibility="reversible", blast_radius="single"),
        ]
        assert overall_plan_tier(profiles) == TrustTier.MEDIUM
