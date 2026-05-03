"""Regression test 3 — duplicate action_id in approval card blocks.

Bug: Both the Approve and Reject buttons in the approval card had
action_id="approval". Slack rejects messages with duplicate action_ids
within the same block with:
    invalid_blocks: `action_id` "approval" already exists

This caused the approval_node to crash before writing closed_history,
which then poisoned the checkpoint and triggered the infinite retry loop.

Fix: Buttons now use action_id="approval_approve" and action_id="approval_reject".
"""

from __future__ import annotations

import pytest

from lyra_core.agent.nodes.approval import TrustTier, _plan_preview_blocks
from lyra_core.agent.state import Plan, PlanStep


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simple_plan() -> Plan:
    return Plan(
        goal="Create contact Jane Test",
        steps=[
            PlanStep(
                id="step_1",
                tool_name="contacts_create_contact",
                args={"firstName": "Jane", "email": "jane@test.com"},
                rationale="Create the requested contact",
                requires_approval=True,
            )
        ],
    )


def _extract_action_ids(blocks: list[dict]) -> list[str]:
    """Collect every action_id from all button elements in all blocks."""
    ids = []
    for block in blocks:
        for element in block.get("elements", []):
            action_id = element.get("action_id")
            if action_id:
                ids.append(action_id)
    return ids


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_approval_card_buttons_have_unique_action_ids():
    """
    REGRESSION: Approve and Reject buttons must have different action_ids.
    Slack rejects the message with 'invalid_blocks' if two elements in the
    same message share the same action_id.
    """
    plan = _simple_plan()
    blocks = _plan_preview_blocks(plan, job_id="job-123", overall_tier=TrustTier.MEDIUM)

    action_ids = _extract_action_ids(blocks)

    assert len(action_ids) == len(set(action_ids)), (
        f"Duplicate action_ids found in approval card: {action_ids}. "
        "Slack will reject the message with 'invalid_blocks'."
    )


def test_approval_card_has_approve_and_reject_buttons():
    """Both Approve and Reject buttons must be present in a MEDIUM-tier plan."""
    plan = _simple_plan()
    blocks = _plan_preview_blocks(plan, job_id="job-123", overall_tier=TrustTier.MEDIUM)

    action_ids = _extract_action_ids(blocks)
    values = [
        el.get("value", "")
        for block in blocks
        for el in block.get("elements", [])
    ]

    assert any(v.startswith("approve:") for v in values), "Approve button missing"
    assert any(v.startswith("reject:") for v in values), "Reject button missing"


def test_high_tier_plan_has_no_buttons():
    """
    HIGH-tier plans use text confirmation ('I confirm'), not buttons.
    No action elements should be present.
    """
    plan = _simple_plan()
    blocks = _plan_preview_blocks(plan, job_id="job-123", overall_tier=TrustTier.HIGH)

    action_ids = _extract_action_ids(blocks)
    assert not action_ids, (
        f"HIGH-tier plan should have no buttons, found action_ids: {action_ids}"
    )


def test_approval_card_job_id_embedded_in_button_values():
    """
    The job_id must be embedded in both button values so the adapter
    can route the decision back to the correct arq job.
    """
    job_id = "abc-123-def"
    plan = _simple_plan()
    blocks = _plan_preview_blocks(plan, job_id=job_id, overall_tier=TrustTier.MEDIUM)

    values = [
        el.get("value", "")
        for block in blocks
        for el in block.get("elements", [])
    ]

    assert f"approve:{job_id}" in values, f"approve:{job_id} not found in button values"
    assert f"reject:{job_id}" in values, f"reject:{job_id} not found in button values"
