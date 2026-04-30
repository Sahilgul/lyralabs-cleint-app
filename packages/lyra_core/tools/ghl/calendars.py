"""GHL Calendars: book an appointment with a contact."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel

from ..base import Tool, ToolContext
from ..registry import default_registry
from .client import GhlClient


class GhlBookAppointmentInput(BaseModel):
    calendar_id: str
    contact_id: str
    start: datetime
    end: datetime
    title: str | None = None
    notes: str | None = None
    timezone: str = "UTC"


class GhlBookAppointmentOutput(BaseModel):
    appointment_id: str
    calendar_id: str
    start: str
    end: str


class GhlBookAppointment(Tool[GhlBookAppointmentInput, GhlBookAppointmentOutput]):
    name = "ghl.calendars.book_appointment"
    description = "Book an appointment in a specific GHL calendar for a contact."
    provider = "ghl"
    requires_approval = True
    Input = GhlBookAppointmentInput
    Output = GhlBookAppointmentOutput

    async def run(
        self, ctx: ToolContext, args: GhlBookAppointmentInput
    ) -> GhlBookAppointmentOutput:
        creds = await ctx.creds_lookup("ghl")
        client = GhlClient(creds)

        if ctx.dry_run:
            return GhlBookAppointmentOutput(
                appointment_id="dry-run",
                calendar_id=args.calendar_id,
                start=args.start.isoformat(),
                end=args.end.isoformat(),
            )

        body: dict = {
            "calendarId": args.calendar_id,
            "locationId": client.location_id,
            "contactId": args.contact_id,
            "startTime": args.start.isoformat(),
            "endTime": args.end.isoformat(),
            "title": args.title or "Appointment",
            "appointmentStatus": "confirmed",
            "notes": args.notes or "",
        }

        data = await client.request("POST", "/calendars/events/appointments", json=body)
        appt = data.get("appointment", data)
        return GhlBookAppointmentOutput(
            appointment_id=appt.get("id", ""),
            calendar_id=args.calendar_id,
            start=appt.get("startTime", args.start.isoformat()),
            end=appt.get("endTime", args.end.isoformat()),
        )


default_registry.register(GhlBookAppointment())
