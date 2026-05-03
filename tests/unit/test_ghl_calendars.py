"""lyra_core.tools.ghl.calendars."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
import respx
from lyra_core.tools.credentials import ProviderCredentials
from lyra_core.tools.ghl.calendars import (
    GhlBookAppointment,
    GhlBookAppointmentInput,
)
from lyra_core.tools.ghl.client import GHL_BASE


def _creds():
    return ProviderCredentials(
        provider="ghl",
        access_token="t",
        refresh_token="rt",
        expires_at=datetime.now(UTC) + timedelta(hours=1),
        external_account_id="loc",
        scopes="calendars.write",
        metadata={"location_id": "loc"},
    )


def _input(**overrides):
    base = dict(
        calendar_id="cal-1",
        contact_id="c-1",
        start=datetime(2026, 5, 1, 10, 0, tzinfo=UTC),
        end=datetime(2026, 5, 1, 11, 0, tzinfo=UTC),
        title="Demo",
        notes="please be on time",
    )
    base.update(overrides)
    return GhlBookAppointmentInput(**base)


@pytest.mark.asyncio
async def test_dry_run(make_ctx) -> None:
    ctx = make_ctx(creds=_creds(), dry_run=True)
    out = await GhlBookAppointment().run(ctx, _input())
    assert out.appointment_id == "dry-run"
    assert out.calendar_id == "cal-1"
    assert out.start == "2026-05-01T10:00:00+00:00"


@pytest.mark.asyncio
async def test_books_appointment(make_ctx) -> None:
    ctx = make_ctx(creds=_creds())
    with respx.mock(base_url=GHL_BASE) as mock:
        route = mock.post("/calendars/events/appointments").respond(
            200,
            json={
                "appointment": {
                    "id": "ap-1",
                    "startTime": "2026-05-01T10:00:00Z",
                    "endTime": "2026-05-01T11:00:00Z",
                }
            },
        )
        out = await GhlBookAppointment().run(ctx, _input())

    body = route.calls[0].request.read().decode()
    assert '"calendarId":"cal-1"' in body or '"calendarId": "cal-1"' in body
    assert '"contactId":"c-1"' in body or '"contactId": "c-1"' in body
    assert '"appointmentStatus":"confirmed"' in body or '"appointmentStatus": "confirmed"' in body
    assert out.appointment_id == "ap-1"
    assert out.calendar_id == "cal-1"
    assert out.start == "2026-05-01T10:00:00Z"


@pytest.mark.asyncio
async def test_handles_response_without_appointment_wrapper(make_ctx) -> None:
    ctx = make_ctx(creds=_creds())
    with respx.mock(base_url=GHL_BASE) as mock:
        mock.post("/calendars/events/appointments").respond(200, json={"id": "ap-2"})
        out = await GhlBookAppointment().run(ctx, _input())
    assert out.appointment_id == "ap-2"


@pytest.mark.asyncio
async def test_default_title_when_omitted(make_ctx) -> None:
    ctx = make_ctx(creds=_creds())
    with respx.mock(base_url=GHL_BASE) as mock:
        route = mock.post("/calendars/events/appointments").respond(200, json={"id": "x"})
        await GhlBookAppointment().run(ctx, _input(title=None, notes=None))
    body = route.calls[0].request.read().decode()
    assert '"title":"Appointment"' in body or '"title": "Appointment"' in body


def test_book_appointment_requires_approval() -> None:
    assert GhlBookAppointment.requires_approval is True
