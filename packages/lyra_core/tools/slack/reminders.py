"""Slack reminders tool: queue a Slack-native reminder for a user.

Slack reminders are personal nudges — they DM the target user at the
chosen time and are visible only to them. Classified LOW per the
permissive policy: a reminder is private to one user, fully reversible
(`reminders.delete` would undo it), and a fundamental teammate behavior.

Note: as of 2024 Slack deprecated `reminders.add` for new apps in favor
of `reminders.add` via the assistant API surface for some clients. This
tool uses the classic Web API endpoint, which still works for installed
bot tokens with the `reminders:write` scope.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field
from slack_sdk.errors import SlackApiError
from slack_sdk.web.async_client import AsyncWebClient

from ..base import Tool, ToolContext, ToolError, TrustTier
from ..registry import default_registry
from ._client import SlackTokenMissing, _bot_token_for


class RemindersAddInput(BaseModel):
    text: str = Field(
        description="Reminder text (what the user will see in the DM).",
        min_length=1,
        max_length=1000,
    )
    time: str = Field(
        description=(
            "When to fire the reminder. Accepts a Unix epoch (seconds) as a "
            "string, OR a Slack natural-language phrase like "
            "'in 30 minutes', 'tomorrow at 9am', 'next Friday'."
        )
    )
    user_id: str | None = Field(
        default=None,
        description=(
            "Slack user id to remind. If omitted, Slack defaults to the "
            "user the bot token represents (rarely what you want — almost "
            "always pass the explicit user_id)."
        ),
    )


class RemindersAddOutput(BaseModel):
    reminder_id: str


class RemindersAdd(Tool[RemindersAddInput, RemindersAddOutput]):
    name = "slack.reminders.add"
    description = (
        "Queue a Slack reminder that DMs a user at a future time. Use for "
        "'remind me Friday to ship the PR' or 'remind @alice on Monday "
        "about the demo'. Private to the target user — only they see the "
        "reminder DM."
    )
    provider = "slack"
    requires_approval = False
    trust_tier = TrustTier.LOW
    Input = RemindersAddInput
    Output = RemindersAddOutput

    async def run(
        self, ctx: ToolContext, args: RemindersAddInput
    ) -> RemindersAddOutput:
        try:
            token = await _bot_token_for(ctx.tenant_id)
        except SlackTokenMissing as exc:
            raise ToolError(str(exc)) from exc

        client = AsyncWebClient(token=token)
        kwargs: dict[str, Any] = {"text": args.text, "time": args.time}
        if args.user_id is not None:
            kwargs["user"] = args.user_id

        try:
            resp = await client.reminders_add(**kwargs)
        except SlackApiError as exc:
            err = (exc.response.data or {}).get("error", str(exc))
            raise ToolError(f"slack.reminders.add failed: {err}") from exc

        assert isinstance(resp.data, dict)
        reminder = resp.data.get("reminder") or {}
        return RemindersAddOutput(reminder_id=reminder.get("id", ""))


default_registry.register(RemindersAdd())
