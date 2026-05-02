"""Internal channel-agnostic message schema.

The agent runtime ONLY consumes these types. Slack and Teams adapters
translate platform events into InboundMessage and translate OutboundReply
back into platform-specific Block Kit / Adaptive Card payloads.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


class Surface(StrEnum):
    SLACK = "slack"
    TEAMS = "teams"


class InboundMessage(BaseModel):
    """Normalized inbound user message."""

    surface: Surface
    tenant_external_id: str = Field(description="Slack team_id or Teams tenant id")
    channel_id: str
    thread_id: str = Field(
        description=(
            "Slack-side conversation key (thread_ts when the user is in a thread, "
            "else the message ts). Used for audit/debugging and to derive "
            "`assistant_status_thread_ts`. NOT the agent's checkpointer key -- "
            "see `agent_thread_id` for that."
        )
    )
    agent_thread_id: str = Field(
        description=(
            "Stable LangGraph checkpointer key for this conversation. "
            "DM:    'slack:dm:{team}:{channel}:{user}'   -- one continuous agent "
            "       memory per DM partner; survives across top-level messages "
            "       (the user's natural mental model of a DM with a coworker). "
            "Thread:'slack:ch:{team}:{channel}:{thread_ts}' -- one agent memory "
            "       per Slack thread; new threads are independent conversations. "
            "Top-level channel @-mention: same form as Thread, keyed on the "
            "       message ts (the bot will reply threaded under it). "
            "This is independent of `reply_thread_ts` (the Slack threading the "
            "bot uses when posting), which stays a UX choice."
        )
    )
    user_id: str = Field(description="Platform user id")
    user_display_name: str | None = None
    text: str
    files: list[dict[str, Any]] = Field(default_factory=list)
    reply_thread_ts: str | None = Field(
        default=None,
        description=(
            "The Slack `thread_ts` to post the bot's reply into. None means "
            "post the reply as a new top-level message in the channel/DM. "
            "Computed by the channel adapter so reply UX is correct: top-level "
            "DMs stay linear, channel @-mentions get threaded."
        ),
    )
    is_dm: bool = False
    # Resolved by the channel adapter from primary_slack_channel_id → clients.id.
    # None when the channel is not mapped to a specific client (agency-internal job).
    client_id: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict, description="Original platform payload")


class Artifact(BaseModel):
    kind: Literal["pdf", "png", "csv", "xlsx", "pptx", "json", "md"]
    filename: str
    content: bytes
    description: str | None = None


class OutboundReply(BaseModel):
    """Normalized agent reply to be rendered into the channel."""

    text: str | None = None
    blocks: list[dict[str, Any]] | None = None
    channel_id: str
    thread_ts: str | None = Field(
        default=None,
        description=(
            "Slack thread_ts to reply into. None = post as a new top-level "
            "message in the channel/DM. Plumbed straight from "
            "InboundMessage.reply_thread_ts."
        ),
    )
    assistant_status_thread_ts: str | None = Field(
        default=None,
        description=(
            "Slack assistant thread whose loading/status indicator should be "
            "cleared after posting. May differ from thread_ts for top-level DMs."
        ),
    )
    artifacts: list[Artifact] = Field(default_factory=list)
    requires_approval: bool = False
    approval_payload: dict[str, Any] | None = None
