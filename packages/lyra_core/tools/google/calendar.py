"""Google Calendar — minimal: create event."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

from googleapiclient.errors import HttpError
from pydantic import BaseModel, Field

from ..base import Tool, ToolContext, ToolError
from ..registry import default_registry
from ._client import calendar_service


class CalendarCreateEventInput(BaseModel):
    calendar_id: str = Field(default="primary")
    summary: str
    description: str | None = None
    start: datetime
    end: datetime
    timezone: str = "UTC"
    attendees: list[str] = Field(default_factory=list, description="List of attendee emails.")
    location: str | None = None


class CalendarCreateEventOutput(BaseModel):
    event_id: str
    html_link: str
    start: str
    end: str


class CalendarCreateEvent(Tool[CalendarCreateEventInput, CalendarCreateEventOutput]):
    name = "google.calendar.create_event"
    description = "Create a Google Calendar event. Optional attendees + location."
    provider = "google"
    requires_approval = True
    Input = CalendarCreateEventInput
    Output = CalendarCreateEventOutput

    async def run(
        self, ctx: ToolContext, args: CalendarCreateEventInput
    ) -> CalendarCreateEventOutput:
        creds = await ctx.creds_lookup("google")
        svc = calendar_service(creds)

        body: dict[str, Any] = {
            "summary": args.summary,
            "description": args.description or "",
            "start": {"dateTime": args.start.isoformat(), "timeZone": args.timezone},
            "end": {"dateTime": args.end.isoformat(), "timeZone": args.timezone},
            "attendees": [{"email": e} for e in args.attendees],
        }
        if args.location:
            body["location"] = args.location

        if ctx.dry_run:
            return CalendarCreateEventOutput(
                event_id="dry-run",
                html_link="https://calendar.google.com/event/dry-run",
                start=args.start.isoformat(),
                end=args.end.isoformat(),
            )

        def _call() -> dict[str, Any]:
            return (
                svc.events()
                .insert(calendarId=args.calendar_id, body=body, sendUpdates="all")
                .execute()
            )

        try:
            event = await asyncio.to_thread(_call)
        except HttpError as exc:
            raise ToolError(f"Calendar create_event failed: {exc}") from exc

        return CalendarCreateEventOutput(
            event_id=event["id"],
            html_link=event.get("htmlLink", ""),
            start=event["start"].get("dateTime", ""),
            end=event["end"].get("dateTime", ""),
        )


default_registry.register(CalendarCreateEvent())
