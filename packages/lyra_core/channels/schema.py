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
    thread_id: str = Field(description="Slack thread_ts or Teams conversation id")
    user_id: str = Field(description="Platform user id")
    user_display_name: str | None = None
    text: str
    files: list[dict[str, Any]] = Field(default_factory=list)
    parent_message_ts: str | None = None
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
    thread_id: str
    channel_id: str
    artifacts: list[Artifact] = Field(default_factory=list)
    requires_approval: bool = False
    approval_payload: dict[str, Any] | None = None
