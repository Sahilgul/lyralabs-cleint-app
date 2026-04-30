"""GHL Conversations: send a message (SMS / Email) to a contact."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ..base import Tool, ToolContext, ToolError
from ..registry import default_registry
from .client import GhlClient


class GhlSendMessageInput(BaseModel):
    contact_id: str
    type: Literal["SMS", "Email"] = "SMS"
    message: str | None = Field(
        default=None, description="Body for SMS or plain-text fallback for Email."
    )
    subject: str | None = Field(default=None, description="Required when type='Email'.")
    html: str | None = Field(default=None, description="Optional HTML for Email.")


class GhlSendMessageOutput(BaseModel):
    conversation_id: str | None = None
    message_id: str | None = None
    status: str = "sent"


class GhlSendMessage(Tool[GhlSendMessageInput, GhlSendMessageOutput]):
    name = "ghl.conversations.send_message"
    description = "Send an SMS or Email to a GHL contact via their conversations endpoint."
    provider = "ghl"
    requires_approval = True
    Input = GhlSendMessageInput
    Output = GhlSendMessageOutput

    async def run(self, ctx: ToolContext, args: GhlSendMessageInput) -> GhlSendMessageOutput:
        if args.type == "Email" and not args.subject:
            raise ToolError("subject is required for Email messages")
        if not args.message and not args.html:
            raise ToolError("message or html body required")

        creds = await ctx.creds_lookup("ghl")
        client = GhlClient(creds)

        if ctx.dry_run:
            return GhlSendMessageOutput(status="dry-run")

        body: dict = {
            "type": args.type,
            "contactId": args.contact_id,
            "message": args.message,
        }
        if args.type == "Email":
            body["subject"] = args.subject
            if args.html:
                body["html"] = args.html

        data = await client.request("POST", "/conversations/messages", json=body)
        return GhlSendMessageOutput(
            conversation_id=data.get("conversationId"),
            message_id=data.get("messageId"),
            status=data.get("status", "sent"),
        )


default_registry.register(GhlSendMessage())
