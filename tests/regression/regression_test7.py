"""Regression test 7 — empty args propagate to the executor (GHL 422).

Bug: The model emitted `submit_plan_for_approval` with steps whose `args` was
`{}`. The plan looked structurally valid, was approved by the user, and the
executor called the GHL contacts_create-contact endpoint with no fields,
producing:

    422 Contacts without email, phone, firstName and lastName are not allowed.

ARLO then told the user "I wasn't able to create Jane Test… even though I
provided all four fields." — a confidence-destroying outcome where the plan
description claimed args that the actual tool call did not contain.

The system-prompt fix ("each step must include full args") helped on average
but couldn't guarantee non-empty args from a non-deterministic LLM, so the
underlying failure mode persisted.

Fix: in agent_node, after Plan.model_validate succeeds, run every step's
args through the tool's Input Pydantic schema. If any step would fail
validation at execution time (missing required fields, wrong types), reject
the plan synchronously with a tool-response error message — the same turn,
before the user ever sees an approval card.

Regression guards:
  1. A plan whose write step has empty args is rejected before reaching
     `pending_plan` / the approval gate.
  2. The rejection surfaces back to the LLM as a `role:tool` message so the
     model can retry within the same turn.
  3. A plan with valid args still reaches `pending_plan` (no false rejects).
"""

from __future__ import annotations

import pytest
from lyra_core.agent.nodes.agent import _validate_plan_step_args
from lyra_core.agent.state import Plan, PlanStep
from lyra_core.tools.base import Tool, ToolContext, TrustTier
from lyra_core.tools.registry import default_registry
from pydantic import BaseModel, Field


class _ContactCreateInput(BaseModel):
    email: str = Field(..., description="contact email")
    firstName: str = Field(..., description="first name")  # noqa: N815 GHL API contract
    lastName: str = Field(..., description="last name")  # noqa: N815 GHL API contract
    phone: str | None = Field(default=None)


class _ContactCreateOutput(BaseModel):
    id: str = ""


class _FakeContactCreateTool(Tool[_ContactCreateInput, _ContactCreateOutput]):
    name = "_regression7_create_contact"
    description = "create a contact (test fixture)"
    requires_approval = True
    trust_tier = TrustTier.MEDIUM
    Input = _ContactCreateInput
    Output = _ContactCreateOutput

    async def run(self, ctx: ToolContext, args):  # pragma: no cover
        return _ContactCreateOutput(id="x")


@pytest.fixture(autouse=True)
def _register_test_tool():
    tool = _FakeContactCreateTool()
    try:
        default_registry.register(tool)
    except ValueError:
        pass  # already registered from a previous run
    yield


def test_validate_rejects_empty_args_for_required_fields():
    """The exact bug from the user's logs: write step with args={}."""
    plan = Plan(
        goal="create Jane Test",
        steps=[
            PlanStep(
                id="step_1",
                tool_name="_regression7_create_contact",
                args={},
                rationale="create the contact",
                requires_approval=True,
            )
        ],
    )
    err = _validate_plan_step_args(plan)
    assert err is not None, (
        "REGRESSION: empty args for a write step with required fields "
        "must be rejected at plan submission, not propagated to the executor."
    )
    assert "step_1" in err
    assert "_regression7_create_contact" in err
    # Error must point the model at how to fix it.
    assert "submit_plan_for_approval" in err


def test_validate_accepts_complete_args():
    """A correctly-populated step must pass validation."""
    plan = Plan(
        goal="create Jane Test",
        steps=[
            PlanStep(
                id="step_1",
                tool_name="_regression7_create_contact",
                args={
                    "email": "jane@example.com",
                    "firstName": "Jane",
                    "lastName": "Test",
                    "phone": "+1234567890",
                },
                rationale="create the contact",
                requires_approval=True,
            )
        ],
    )
    err = _validate_plan_step_args(plan)
    assert err is None, f"valid plan was rejected: {err}"


def test_validate_skips_unknown_tool_gracefully():
    """Unknown tools (per-tenant discovery may be incomplete here) must not
    fail plan submission — they fail later at the executor with a clearer
    error. Validating against a missing schema would be impossible anyway."""
    plan = Plan(
        goal="x",
        steps=[
            PlanStep(
                id="step_1",
                tool_name="_regression7_definitely_not_registered",
                args={"anything": True},
                rationale="r",
                requires_approval=True,
            )
        ],
    )
    err = _validate_plan_step_args(plan)
    assert err is None, (
        "validator must not reject plans referencing tools that aren't in "
        "the registry — discovery is per-tenant and may run later."
    )


def test_validate_rejects_wrong_type():
    """Pydantic-coercible types still pass; truly-wrong types fail."""
    plan = Plan(
        goal="x",
        steps=[
            PlanStep(
                id="step_1",
                tool_name="_regression7_create_contact",
                # firstName should be a string; pass a list to force a real
                # type mismatch Pydantic can't coerce.
                args={
                    "email": "jane@example.com",
                    "firstName": ["Jane"],
                    "lastName": "Test",
                },
                rationale="r",
                requires_approval=True,
            )
        ],
    )
    err = _validate_plan_step_args(plan)
    assert err is not None
    assert "firstName" in err or "step_1" in err


def test_mcp_tool_uses_per_tool_args_schema_as_input():
    """Critical: every MCP-discovered tool (every GHL tool, every Slack MCP
    tool) must use the per-tool args_schema as its `Input` ClassVar — NOT
    the generic `McpInput` envelope.

    History: McpInput defined `arguments: dict[str, Any]` as its only field.
    On Pydantic v2, `extra='ignore'` is the default, so when the executor
    did `tool.Input(**step.args)` with a populated step.args like
    {'email': 'j@x.com', 'firstName': 'Jane'}, Pydantic SILENTLY DROPPED
    every field and produced `McpInput(arguments={})`. Then `tool.run`
    sent `{}` to the upstream MCP server and got back the GHL 422
    'Contacts without email, phone, firstName and lastName are not allowed'
    — even though the LLM had populated all four.

    This is the actual root cause of the user-reported bug. Validating
    step.args was insufficient because the args never made it through
    execution either way.

    Fix: McpToolAdapter.Input is now the lc_tool.args_schema, so:
      - validate_args (base implementation) checks the real fields
      - executor's tool.Input(**step.args) carries them through
      - run() passes args.model_dump() to the MCP server
    """
    from unittest.mock import MagicMock

    from lyra_core.tools.base import RiskProfile, TrustTier
    from lyra_core.tools.mcp_adapter import _make_mcp_tool_adapter

    class _ContactsCreateSchema(BaseModel):
        email: str = Field(...)
        firstName: str = Field(...)  # noqa: N815 GHL API contract
        lastName: str = Field(...)  # noqa: N815 GHL API contract
        phone: str | None = Field(default=None)

    fake_lc_tool = MagicMock()
    fake_lc_tool.name = "contacts_create-contact"
    fake_lc_tool.description = "create a contact"
    fake_lc_tool.args_schema = _ContactsCreateSchema

    adapter = _make_mcp_tool_adapter(
        lc_tool=fake_lc_tool,
        server_key="ghl",
        mcp_tool_name="contacts_create-contact",
        profile=RiskProfile(
            tier=TrustTier.MEDIUM, reversibility="reversible", blast_radius="single"
        ),
        provider="ghl",
    )

    # The adapter's Input must be the per-tool schema, not McpInput.
    assert adapter.Input is _ContactsCreateSchema, (
        "REGRESSION: McpToolAdapter.Input is not the per-tool args_schema. "
        "If Input stays generic, every MCP write call drops its args and "
        "fails upstream with the same 422 the user reported."
    )

    # Empty args → validate_args must reject (required fields missing).
    err_empty = adapter.validate_args({})
    assert err_empty is not None, "validator must reject empty args"

    # Complete args → validate_args passes AND tool.Input(**args) preserves
    # every field (the executor's exact call path).
    populated = {"email": "j@x.com", "firstName": "Jane", "lastName": "Test"}
    assert adapter.validate_args(populated) is None
    args_obj = adapter.Input(**populated)
    dumped = args_obj.model_dump(exclude_unset=True, exclude_none=True)
    assert dumped == populated, (
        "REGRESSION: Input(**step.args).model_dump() dropped fields. "
        "On Pydantic v2 the previous McpInput envelope did this silently — "
        "it's exactly how the user's args_schema-aware plan still produced "
        "an empty MCP request. "
        f"got: {dumped} expected: {populated}"
    )


def test_mcp_tool_falls_back_to_mcp_input_when_no_args_schema():
    """If LangChain's adapter didn't supply an args_schema (older versions,
    or a server that returned no inputSchema), the adapter should use
    McpInput as a permissive fallback — but with `extra='allow'` so any
    fields the LLM provides aren't silently dropped."""
    from unittest.mock import MagicMock

    from lyra_core.tools.base import RiskProfile, TrustTier
    from lyra_core.tools.mcp_adapter import McpInput, _make_mcp_tool_adapter

    fake_lc_tool = MagicMock()
    fake_lc_tool.name = "_regression7_no_schema"
    fake_lc_tool.description = "schemaless tool"
    fake_lc_tool.args_schema = None

    adapter = _make_mcp_tool_adapter(
        lc_tool=fake_lc_tool,
        server_key="ghl",
        mcp_tool_name="_regression7_no_schema",
        profile=RiskProfile(tier=TrustTier.LOW, reversibility="reversible", blast_radius="single"),
        provider="ghl",
    )
    assert adapter.Input is McpInput

    # Even with the fallback, fields supplied by the LLM must survive
    # round-trip. (Bug was Pydantic's default `extra='ignore'`.)
    instance = McpInput(email="j@x.com", firstName="Jane")
    dumped = instance.model_dump(exclude_unset=True, exclude_none=True)
    assert "email" in dumped and dumped["email"] == "j@x.com", (
        "REGRESSION: McpInput silently dropped extras. Set "
        "model_config = {'extra': 'allow'} so populated args survive."
    )


def test_validator_runs_inside_agent_node_before_pending_plan_set():
    """Source-level guard: agent_node calls _validate_plan_step_args
    BEFORE setting `pending_plan` and routing to approval. Without this
    ordering, the user could see an approval card for an unexecutable plan.
    """
    import inspect

    from lyra_core.agent.nodes import agent as agent_mod

    src = inspect.getsource(agent_mod.agent_node)
    val_idx = src.find("_validate_plan_step_args")
    pending_idx = src.find('"pending_plan": plan.model_dump()')
    assert val_idx != -1, "agent_node must call _validate_plan_step_args"
    assert pending_idx != -1, "agent_node must set pending_plan on success"
    assert val_idx < pending_idx, (
        "REGRESSION: _validate_plan_step_args must run BEFORE pending_plan "
        "is set. Otherwise a malformed plan reaches the approval gate."
    )
