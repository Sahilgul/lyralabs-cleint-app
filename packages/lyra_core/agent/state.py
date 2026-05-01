"""LangGraph state + plan/step Pydantic models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class PlanStep(BaseModel):
    """One step in a plan. Maps to one tool call."""

    id: str = Field(description="stable id like 'step_1', used to reference results")
    tool_name: str = Field(description="exact name from the tool registry")
    args: dict[str, Any] = Field(default_factory=dict)
    rationale: str = Field(description="one-sentence why")
    requires_approval: bool = False
    depends_on: list[str] = Field(
        default_factory=list, description="ids of earlier steps whose output this one needs"
    )


class Plan(BaseModel):
    """Structured plan emitted by the planner."""

    goal: str
    steps: list[PlanStep]
    needs_clarification: bool = False
    clarification_question: str | None = None


class StepResult(BaseModel):
    step_id: str
    tool_name: str
    ok: bool
    data: dict[str, Any] | None = None
    error: str | None = None
    cost_usd: float = 0.0


class AgentState(TypedDict, total=False):
    """LangGraph state. Persisted via the Postgres checkpointer.

    NOTE: TypedDict (not BaseModel) is required by LangGraph's checkpointer.
    """

    # --- inputs (set once at graph entry) ---
    tenant_id: str
    job_id: str
    channel_id: str
    thread_id: str
    user_id: str
    user_request: str
    # The Slack `thread_ts` to post replies into. None = post replies as
    # new top-level messages in the channel/DM. Computed by the channel
    # adapter based on whether the user threaded their message and whether
    # the surface is a DM. See lyra_core.channels.slack.adapter.
    reply_thread_ts: str | None

    # --- working state ---
    classification: Literal["smalltalk", "task", "clarification"]
    plan: dict[str, Any] | None  # serialized Plan
    step_results: list[dict[str, Any]]
    approval_decision: Literal["approved", "rejected", "pending"] | None
    final_summary: str | None
    artifacts: list[dict[str, Any]]  # [{kind, filename, b64_content, description}]
    error: str | None
    total_cost_usd: float

    # --- chat-style messages (used by some nodes for tool-calling LLM loops) ---
    # Note: not using langgraph's `add_messages` reducer for MVP; nodes return
    # new full lists. Switch to add_messages reducer if multi-agent fan-out is added.
    messages: list[Any]
