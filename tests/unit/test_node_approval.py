"""lyra_core.agent.nodes.approval."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from lyra_core.agent.nodes import approval as approval_mod
from lyra_core.agent.nodes.approval import (
    _plan_preview_blocks,
    approval_post_node,
    approval_wait_node,
    rejected_reply_node,
    route_after_approval,
    route_after_approval_post,
)
from lyra_core.agent.state import Plan, PlanStep
from lyra_core.tools.base import RiskProfile, TrustTier


def _plan(*, has_write: bool = False, needs_clarification: bool = False) -> Plan:
    return Plan(
        goal="g",
        steps=(
            [
                PlanStep(id="s1", tool_name="ghl.contacts.search", rationale="r"),
                PlanStep(
                    id="s2",
                    tool_name="ghl.contacts.create",
                    rationale="r2",
                    requires_approval=has_write,
                ),
            ]
            if has_write
            else [PlanStep(id="s1", tool_name="t", rationale="r")]
        ),
        needs_clarification=needs_clarification,
    )


def _medium_profiles(plan: Plan) -> list[RiskProfile]:
    return [
        RiskProfile(tier=TrustTier.MEDIUM, reversibility="reversible", blast_radius="single")
        for _ in plan.steps
    ]


class TestRouteAfterApproval:
    def test_approved_goes_to_executor(self) -> None:
        assert route_after_approval({"approval_decision": "approved"}) == "executor"  # type: ignore[arg-type]

    def test_rejected_goes_to_rejected_reply(self) -> None:
        assert (
            route_after_approval({"approval_decision": "rejected"}) == "rejected_reply"
        )

    def test_default_pending_goes_to_rejected_reply(self) -> None:
        assert route_after_approval({}) == "rejected_reply"  # type: ignore[arg-type]


class TestRouteAfterApprovalPost:
    def test_needs_wait_routes_to_approval_wait(self) -> None:
        assert route_after_approval_post({"needs_approval_wait": True}) == "approval_wait"  # type: ignore[arg-type]

    def test_no_wait_routes_to_executor(self) -> None:
        assert route_after_approval_post({"needs_approval_wait": False}) == "executor"  # type: ignore[arg-type]

    def test_missing_flag_routes_to_executor(self) -> None:
        assert route_after_approval_post({}) == "executor"  # type: ignore[arg-type]


class TestPreviewBlocks:
    def test_includes_goal_steps_and_buttons(self) -> None:
        plan = _plan(has_write=True)
        profiles = _medium_profiles(plan)
        blocks = _plan_preview_blocks(plan, "job-1", profiles=profiles)
        assert blocks[0]["type"] == "header"
        assert "Approve" in blocks[0]["text"]["text"]
        assert "g" in blocks[1]["text"]["text"]
        # actions block exists for MEDIUM tier
        actions = blocks[2]
        assert actions["type"] == "actions"
        values = {b["value"] for b in actions["elements"]}
        assert values == {"approved:job-1", "rejected:job-1"}
        # write tag still present
        assert "_(write)_" in blocks[1]["text"]["text"]

    def test_empty_steps_renders_placeholder(self) -> None:
        plan = Plan(goal="g", steps=[])
        blocks = _plan_preview_blocks(plan, "j")
        assert "_(no steps)_" in blocks[1]["text"]["text"]

    def test_high_tier_omits_buttons_adds_confirm_instruction(self) -> None:
        plan = _plan(has_write=True)
        high_profiles = [
            RiskProfile(tier=TrustTier.HIGH, reversibility="irreversible", blast_radius="bulk")
            for _ in plan.steps
        ]
        blocks = _plan_preview_blocks(
            plan, "j", profiles=high_profiles, overall_tier=TrustTier.HIGH
        )
        # No actions block for HIGH tier
        assert all(b["type"] != "actions" for b in blocks)
        assert "confirm" in blocks[1]["text"]["text"].lower()

    def test_simulation_previews_appear_in_card(self) -> None:
        plan = _plan(has_write=True)
        profiles = _medium_profiles(plan)
        previews = {"s2": "Will create contact with email: test@example.com"}
        blocks = _plan_preview_blocks(plan, "j", profiles=profiles, previews=previews)
        assert "test@example.com" in blocks[1]["text"]["text"]


@pytest.mark.asyncio
async def test_approval_post_node_posts_preview_and_signals_wait(monkeypatch) -> None:
    """approval_post_node posts the card and returns needs_approval_wait=True."""
    plan = _plan(has_write=True)
    plan_dict = plan.model_dump()
    state = {
        "plan": plan_dict,
        "job_id": "j-1",
        "thread_id": "thr",
        "channel_id": "ch",
        "tenant_id": "tenant-1",
    }
    post_mock = AsyncMock()
    monkeypatch.setattr(approval_mod, "post_reply", post_mock)

    out = await approval_post_node(state)  # type: ignore[arg-type]
    assert out["needs_approval_wait"] is True
    assert out["plan"]["goal"] == "g"
    assert out["pending_plan"] is None
    assert "risk_profiles" in out
    post_mock.assert_called_once()


@pytest.mark.asyncio
async def test_approval_wait_node_returns_approved(monkeypatch) -> None:
    state = {"job_id": "j"}
    monkeypatch.setattr(approval_mod, "interrupt", lambda payload: "approved")
    out = await approval_wait_node(state)  # type: ignore[arg-type]
    assert out["approval_decision"] == "approved"
    assert out["needs_approval_wait"] is False


@pytest.mark.asyncio
async def test_approval_wait_node_handles_dict_decision(monkeypatch) -> None:
    state = {"job_id": "j"}
    monkeypatch.setattr(approval_mod, "interrupt", lambda payload: {"decision": "rejected"})
    out = await approval_wait_node(state)  # type: ignore[arg-type]
    assert out["approval_decision"] == "rejected"


@pytest.mark.asyncio
async def test_approval_wait_node_unknown_decision_defaults_rejected(monkeypatch) -> None:
    state = {"job_id": "j"}
    monkeypatch.setattr(approval_mod, "interrupt", lambda payload: "garbage")
    out = await approval_wait_node(state)  # type: ignore[arg-type]
    assert out["approval_decision"] == "rejected"


@pytest.mark.asyncio
async def test_approval_post_node_reads_pending_plan(monkeypatch) -> None:
    plan = _plan(has_write=True)
    plan_dict = plan.model_dump()
    state = {
        "pending_plan": plan_dict,
        "job_id": "j-1",
        "thread_id": "thr",
        "channel_id": "ch",
        "tenant_id": "tenant-1",
    }
    monkeypatch.setattr(approval_mod, "post_reply", AsyncMock())

    out = await approval_post_node(state)  # type: ignore[arg-type]
    assert out["plan"]["goal"] == plan_dict["goal"]
    assert out["pending_plan"] is None


@pytest.mark.asyncio
async def test_approval_post_node_auto_approves_low_only_plan(monkeypatch) -> None:
    """A plan whose only step is a read tool auto-approves without posting a card."""
    from lyra_core.tools.base import Tool, ToolContext, TrustTier
    from lyra_core.tools.registry import default_registry
    from pydantic import BaseModel

    class _In(BaseModel):
        q: str = ""

    class _Out(BaseModel):
        data: str = ""

    class _ReadTool(Tool[_In, _Out]):
        name = "_test_low_tool"
        description = "reads something"
        requires_approval = False
        trust_tier = TrustTier.LOW
        Input = _In
        Output = _Out

        async def run(self, ctx: ToolContext, args: _In) -> _Out:
            return _Out()

    tool = _ReadTool()
    try:
        default_registry.register(tool)
    except ValueError:
        pass  # already registered from a previous test run

    plan = Plan(goal="g", steps=[PlanStep(id="s1", tool_name="_test_low_tool", rationale="r")])
    state = {
        "plan": plan.model_dump(),
        "job_id": "j",
        "thread_id": "t",
        "channel_id": "c",
        "tenant_id": "x",
    }
    post_mock = AsyncMock()
    monkeypatch.setattr(approval_mod, "post_reply", post_mock)

    out = await approval_post_node(state)  # type: ignore[arg-type]
    assert out["approval_decision"] == "approved"
    assert out["needs_approval_wait"] is False
    post_mock.assert_not_called()


@pytest.mark.asyncio
async def test_rejected_reply_node_posts_message(monkeypatch) -> None:
    posted = []

    async def fake_post(tenant, reply):
        posted.append((tenant, reply))

    monkeypatch.setattr(approval_mod, "post_reply", fake_post)

    out = await rejected_reply_node(  # type: ignore[arg-type]
        {"thread_id": "thr", "channel_id": "ch", "tenant_id": "ten"}
    )
    assert out == {"final_summary": "rejected_by_user"}
    assert len(posted) == 1
    assert "rejected" in posted[0][1].text.lower()
