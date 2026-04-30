"""lyra_core.agent.nodes.approval."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from lyra_core.agent.nodes import approval as approval_mod
from lyra_core.agent.nodes.approval import (
    _plan_preview_blocks,
    approval_node,
    rejected_reply_node,
    route_after_approval,
    route_after_plan,
)
from lyra_core.agent.state import Plan, PlanStep


def _plan(*, has_write: bool = False, needs_clarification: bool = False) -> Plan:
    return Plan(
        goal="g",
        steps=[
            PlanStep(id="s1", tool_name="ghl.contacts.search", rationale="r"),
            PlanStep(
                id="s2",
                tool_name="ghl.contacts.create",
                rationale="r2",
                requires_approval=has_write,
            ),
        ] if has_write else [PlanStep(id="s1", tool_name="t", rationale="r")],
        needs_clarification=needs_clarification,
    )


class TestRouteAfterPlan:
    def test_no_plan_goes_to_smalltalk(self) -> None:
        assert route_after_plan({}) == "smalltalk_reply"  # type: ignore[arg-type]

    def test_clarification_goes_to_smalltalk(self) -> None:
        p = _plan(needs_clarification=True)
        assert route_after_plan({"plan": p.model_dump()}) == "smalltalk_reply"  # type: ignore[arg-type]

    def test_write_steps_route_to_approval(self) -> None:
        p = _plan(has_write=True)
        assert route_after_plan({"plan": p.model_dump()}) == "approval"  # type: ignore[arg-type]

    def test_read_only_routes_to_executor(self) -> None:
        p = _plan(has_write=False)
        assert route_after_plan({"plan": p.model_dump()}) == "executor"  # type: ignore[arg-type]


class TestRouteAfterApproval:
    def test_approved_goes_to_executor(self) -> None:
        assert route_after_approval({"approval_decision": "approved"}) == "executor"  # type: ignore[arg-type]

    def test_rejected_goes_to_rejected_reply(self) -> None:
        assert (
            route_after_approval({"approval_decision": "rejected"}) == "rejected_reply"
        )

    def test_default_pending_goes_to_rejected_reply(self) -> None:
        assert route_after_approval({}) == "rejected_reply"  # type: ignore[arg-type]


class TestPreviewBlocks:
    def test_includes_goal_steps_and_buttons(self) -> None:
        plan = _plan(has_write=True)
        blocks = _plan_preview_blocks(plan, "job-1")
        assert blocks[0]["type"] == "header"
        assert "Approve" in blocks[0]["text"]["text"]
        assert "g" in blocks[1]["text"]["text"]
        # actions block
        actions = blocks[2]
        assert actions["type"] == "actions"
        values = {b["value"] for b in actions["elements"]}
        assert values == {"approve:job-1", "reject:job-1"}
        # writes are tagged
        assert "_(write)_" in blocks[1]["text"]["text"]

    def test_empty_steps_renders_placeholder(self) -> None:
        plan = Plan(goal="g", steps=[])
        blocks = _plan_preview_blocks(plan, "j")
        assert "_(no steps)_" in blocks[1]["text"]["text"]


@pytest.mark.asyncio
async def test_approval_node_posts_preview_then_returns_decision(monkeypatch) -> None:
    """interrupt() returns the decision when resumed; we mock it to skip the actual interruption."""
    plan = _plan(has_write=True)
    state = {
        "plan": plan.model_dump(),
        "job_id": "j-1",
        "thread_id": "thr",
        "channel_id": "ch",
        "tenant_id": "tenant-1",
    }
    monkeypatch.setattr(approval_mod, "post_reply", AsyncMock())
    monkeypatch.setattr(approval_mod, "interrupt", lambda payload: "approved")

    out = await approval_node(state)  # type: ignore[arg-type]
    assert out == {"approval_decision": "approved"}


@pytest.mark.asyncio
async def test_approval_node_handles_dict_decision(monkeypatch) -> None:
    plan = _plan(has_write=True)
    state = {
        "plan": plan.model_dump(),
        "job_id": "j",
        "thread_id": "t",
        "channel_id": "c",
        "tenant_id": "x",
    }
    monkeypatch.setattr(approval_mod, "post_reply", AsyncMock())
    monkeypatch.setattr(approval_mod, "interrupt", lambda payload: {"decision": "rejected"})
    out = await approval_node(state)  # type: ignore[arg-type]
    assert out == {"approval_decision": "rejected"}


@pytest.mark.asyncio
async def test_approval_node_unknown_decision_defaults_rejected(monkeypatch) -> None:
    plan = _plan(has_write=True)
    state = {
        "plan": plan.model_dump(),
        "job_id": "j",
        "thread_id": "t",
        "channel_id": "c",
        "tenant_id": "x",
    }
    monkeypatch.setattr(approval_mod, "post_reply", AsyncMock())
    monkeypatch.setattr(approval_mod, "interrupt", lambda payload: "garbage")
    out = await approval_node(state)  # type: ignore[arg-type]
    assert out == {"approval_decision": "rejected"}


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
