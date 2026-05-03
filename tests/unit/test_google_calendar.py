"""lyra_core.tools.google.calendar."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest
from googleapiclient.errors import HttpError
from lyra_core.tools.base import ToolError
from lyra_core.tools.google import calendar as cal_mod
from lyra_core.tools.google.calendar import (
    CalendarCreateEvent,
    CalendarCreateEventInput,
)


def _input(**overrides):
    base = dict(
        summary="Demo call",
        description="Quarterly sync",
        start=datetime(2026, 5, 1, 10, 0, tzinfo=UTC),
        end=datetime(2026, 5, 1, 11, 0, tzinfo=UTC),
        timezone="UTC",
        attendees=["a@x.com", "b@x.com"],
    )
    base.update(overrides)
    return CalendarCreateEventInput(**base)


@pytest.mark.asyncio
async def test_create_event_dry_run(monkeypatch, make_ctx) -> None:
    monkeypatch.setattr(cal_mod, "calendar_service", lambda c: MagicMock())
    out = await CalendarCreateEvent().run(make_ctx(dry_run=True), _input())
    assert out.event_id == "dry-run"
    assert out.html_link.endswith("dry-run")
    assert out.start == "2026-05-01T10:00:00+00:00"


@pytest.mark.asyncio
async def test_create_event_calls_insert_with_body(monkeypatch, make_ctx) -> None:
    svc = MagicMock()
    svc.events.return_value.insert.return_value.execute.return_value = {
        "id": "ev-1",
        "htmlLink": "https://cal/ev-1",
        "start": {"dateTime": "2026-05-01T10:00:00Z"},
        "end": {"dateTime": "2026-05-01T11:00:00Z"},
    }
    monkeypatch.setattr(cal_mod, "calendar_service", lambda c: svc)

    out = await CalendarCreateEvent().run(make_ctx(), _input(location="Zoom"))
    assert out.event_id == "ev-1"
    assert out.html_link == "https://cal/ev-1"
    assert out.start == "2026-05-01T10:00:00Z"

    kwargs = svc.events.return_value.insert.call_args.kwargs
    body = kwargs["body"]
    assert body["summary"] == "Demo call"
    assert body["start"]["dateTime"] == "2026-05-01T10:00:00+00:00"
    assert body["start"]["timeZone"] == "UTC"
    assert body["attendees"] == [{"email": "a@x.com"}, {"email": "b@x.com"}]
    assert body["location"] == "Zoom"
    assert kwargs["calendarId"] == "primary"
    assert kwargs["sendUpdates"] == "all"


@pytest.mark.asyncio
async def test_create_event_omits_location_when_absent(monkeypatch, make_ctx) -> None:
    svc = MagicMock()
    svc.events.return_value.insert.return_value.execute.return_value = {
        "id": "x",
        "htmlLink": "",
        "start": {"dateTime": ""},
        "end": {"dateTime": ""},
    }
    monkeypatch.setattr(cal_mod, "calendar_service", lambda c: svc)
    await CalendarCreateEvent().run(make_ctx(), _input())
    body = svc.events.return_value.insert.call_args.kwargs["body"]
    assert "location" not in body


@pytest.mark.asyncio
async def test_create_event_handles_no_description(monkeypatch, make_ctx) -> None:
    svc = MagicMock()
    svc.events.return_value.insert.return_value.execute.return_value = {
        "id": "x",
        "htmlLink": "",
        "start": {"dateTime": ""},
        "end": {"dateTime": ""},
    }
    monkeypatch.setattr(cal_mod, "calendar_service", lambda c: svc)
    await CalendarCreateEvent().run(make_ctx(), _input(description=None))
    body = svc.events.return_value.insert.call_args.kwargs["body"]
    assert body["description"] == ""


@pytest.mark.asyncio
async def test_create_event_wraps_http_error(monkeypatch, make_ctx) -> None:
    svc = MagicMock()
    svc.events.return_value.insert.return_value.execute.side_effect = HttpError(
        MagicMock(status=500), b""
    )
    monkeypatch.setattr(cal_mod, "calendar_service", lambda c: svc)

    with pytest.raises(ToolError, match="Calendar create_event failed"):
        await CalendarCreateEvent().run(make_ctx(), _input())


def test_create_event_requires_approval() -> None:
    assert CalendarCreateEvent.requires_approval is True
