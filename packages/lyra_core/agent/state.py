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
    # Populated by approval_post_node after trust classification.
    trust_tier: str = "medium"
    # Human-readable preview of what this step would do; set by rehearsal engine.
    simulation_preview: str | None = None


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
    client_id: str | None  # which client this job is for; None = agency-internal
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
    # Slack assistant thread whose status/loading indicator was set when the
    # inbound event was accepted. Cleared after the reply is posted.
    assistant_status_thread_ts: str | None

    # --- working state ---
    # `plan` holds the approved/in-flight plan that the executor walks.
    # `agent_node` sets it via the `submit_plan_for_approval` meta-tool,
    # then the approval gate (approval_post → approval_wait) gates it
    # before `executor_node` runs it.
    plan: dict[str, Any] | None  # serialized Plan
    # The plan the agent wants to run, awaiting approval. `approval_post_node`
    # copies `pending_plan` -> `plan` after running rehearsal.
    pending_plan: dict[str, Any] | None
    step_results: list[dict[str, Any]]
    approval_decision: Literal["approved", "rejected", "pending"] | None
    # Why the plan was rejected. "user_followup" = auto-cancelled because the
    # user sent a new message instead of clicking Approve/Reject; in that case
    # `rejected_reply_node` suppresses the canned reject post so the user only
    # sees the response to their new request. Any other value (or None) is
    # treated as an explicit user reject and gets the canned post.
    approval_rejection_reason: str | None
    final_summary: str | None
    artifacts: list[dict[str, Any]]  # [{kind, filename, b64_content, description}]
    error: str | None
    total_cost_usd: float

    # Set by approval_post_node to signal that approval_wait_node should interrupt.
    # False for LOW-tier plans that auto-approve without user interaction.
    needs_approval_wait: bool

    # Trust gradient: per-step RiskProfile dicts populated by approval_post_node.
    risk_profiles: list[dict[str, Any]]
    # Living Artifact: durable per-thread workspace facts, distilled after each job.
    living_artifact: dict[str, Any]
    # Skill shortcuts promoted by the Skill Crystallizer for this (tenant, client).
    active_skills: list[dict[str, Any]]

    # --- chat-style messages (used by some nodes for tool-calling LLM loops) ---
    # Note: not using langgraph's `add_messages` reducer for MVP; nodes return
    # new full lists. Switch to add_messages reducer if multi-agent fan-out is added.
    messages: list[Any]
