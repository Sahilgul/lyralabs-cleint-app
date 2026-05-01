"""lyra_core.agent.state."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from lyra_core.agent.state import AgentState, Plan, PlanStep, StepResult


class TestPlanStep:
    def test_minimal(self) -> None:
        s = PlanStep(id="step_1", tool_name="x", rationale="why")
        assert s.requires_approval is False
        assert s.depends_on == []
        assert s.args == {}

    def test_required_fields(self) -> None:
        with pytest.raises(ValidationError):
            PlanStep()  # type: ignore[call-arg]

    def test_serialize_deserialize(self) -> None:
        s = PlanStep(
            id="step_1",
            tool_name="ghl.contacts.search",
            args={"q": "alice"},
            rationale="search before create",
            requires_approval=True,
            depends_on=["step_0"],
        )
        d = s.model_dump()
        assert d["id"] == "step_1"
        assert PlanStep.model_validate(d) == s


class TestPlan:
    def test_basic(self) -> None:
        p = Plan(goal="x", steps=[PlanStep(id="s1", tool_name="t", rationale="r")])
        assert p.needs_clarification is False
        assert p.clarification_question is None
        assert len(p.steps) == 1

    def test_clarification(self) -> None:
        p = Plan(
            goal="x",
            steps=[],
            needs_clarification=True,
            clarification_question="which pipeline?",
        )
        assert p.needs_clarification is True
        assert p.clarification_question == "which pipeline?"

    def test_round_trip(self) -> None:
        p = Plan(
            goal="g",
            steps=[
                PlanStep(id="s1", tool_name="a", rationale="r"),
                PlanStep(id="s2", tool_name="b", rationale="r"),
            ],
        )
        assert Plan.model_validate(p.model_dump()) == p


class TestStepResult:
    def test_ok_with_data(self) -> None:
        r = StepResult(
            step_id="s1", tool_name="t", ok=True, data={"x": 1}, cost_usd=0.001
        )
        assert r.ok is True
        assert r.data == {"x": 1}
        assert r.cost_usd == 0.001

    def test_failure(self) -> None:
        r = StepResult(step_id="s1", tool_name="t", ok=False, error="boom")
        assert r.error == "boom"
        assert r.data is None

    def test_default_cost_zero(self) -> None:
        r = StepResult(step_id="s", tool_name="t", ok=True)
        assert r.cost_usd == 0.0


class TestAgentState:
    def test_typeddict_allows_partial_dict(self) -> None:
        # AgentState uses total=False so any subset is allowed at runtime
        s: AgentState = {"tenant_id": "t-1", "user_request": "hi"}
        assert s["tenant_id"] == "t-1"

    def test_can_carry_full_state(self) -> None:
        s: AgentState = {
            "tenant_id": "t",
            "job_id": "j",
            "channel_id": "c",
            "thread_id": "thr",
            "user_id": "u",
            "user_request": "do x",
            "plan": {"goal": "g", "steps": []},
            "pending_plan": None,
            "step_results": [],
            "approval_decision": "pending",
            "final_summary": None,
            "artifacts": [],
            "error": None,
            "total_cost_usd": 0.0,
            "messages": [],
        }
        assert s["plan"]["goal"] == "g"
