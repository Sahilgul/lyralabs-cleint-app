"""lyra_core.tools.google.sheets."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from googleapiclient.errors import HttpError
from lyra_core.tools.base import ToolError
from lyra_core.tools.google import sheets as sheets_mod
from lyra_core.tools.google.sheets import (
    SheetsAppend,
    SheetsAppendInput,
    SheetsRead,
    SheetsReadInput,
)


@pytest.mark.asyncio
async def test_sheets_read_returns_rows_and_dimensions(monkeypatch, make_ctx) -> None:
    svc = MagicMock()
    svc.spreadsheets.return_value.values.return_value.get.return_value.execute.return_value = {
        "range": "Leads!A1:C2",
        "values": [["Name", "Email", "Phone"], ["Alice", "a@x.com", "555"]],
    }
    monkeypatch.setattr(sheets_mod, "sheets_service", lambda c: svc)

    out = await SheetsRead().run(
        make_ctx(),
        SheetsReadInput(spreadsheet_id="ssid", range_a1="Leads!A1:C2"),
    )
    assert out.spreadsheet_id == "ssid"
    assert out.range == "Leads!A1:C2"
    assert out.n_rows == 2
    assert out.n_cols == 3
    assert out.values[0] == ["Name", "Email", "Phone"]


@pytest.mark.asyncio
async def test_sheets_read_handles_empty_range(monkeypatch, make_ctx) -> None:
    svc = MagicMock()
    svc.spreadsheets.return_value.values.return_value.get.return_value.execute.return_value = {}
    monkeypatch.setattr(sheets_mod, "sheets_service", lambda c: svc)

    out = await SheetsRead().run(
        make_ctx(),
        SheetsReadInput(spreadsheet_id="ssid", range_a1="X"),
    )
    assert out.values == []
    assert out.n_rows == 0
    assert out.n_cols == 0


@pytest.mark.asyncio
async def test_sheets_read_wraps_http_error(monkeypatch, make_ctx) -> None:
    svc = MagicMock()
    svc.spreadsheets.return_value.values.return_value.get.return_value.execute.side_effect = (
        HttpError(MagicMock(status=403, reason="Forbidden"), b"")
    )
    monkeypatch.setattr(sheets_mod, "sheets_service", lambda c: svc)

    with pytest.raises(ToolError, match="Sheets read failed"):
        await SheetsRead().run(make_ctx(), SheetsReadInput(spreadsheet_id="x", range_a1="y"))


@pytest.mark.asyncio
async def test_sheets_append_dry_run(monkeypatch, make_ctx) -> None:
    monkeypatch.setattr(sheets_mod, "sheets_service", lambda c: MagicMock())

    out = await SheetsAppend().run(
        make_ctx(dry_run=True),
        SheetsAppendInput(spreadsheet_id="x", range_a1="A1", rows=[["a", "b"]]),
    )
    assert out.updated_rows == 1
    assert "dry-run" in out.updated_range


@pytest.mark.asyncio
async def test_sheets_append_passes_value_input_option(monkeypatch, make_ctx) -> None:
    svc = MagicMock()
    svc.spreadsheets.return_value.values.return_value.append.return_value.execute.return_value = {
        "updates": {"updatedRange": "Leads!A2:B2", "updatedRows": 1}
    }
    monkeypatch.setattr(sheets_mod, "sheets_service", lambda c: svc)

    out = await SheetsAppend().run(
        make_ctx(),
        SheetsAppendInput(
            spreadsheet_id="ssid",
            range_a1="Leads",
            rows=[["A", "B"]],
            value_input_option="RAW",
        ),
    )
    assert out.updated_range == "Leads!A2:B2"
    assert out.updated_rows == 1
    kwargs = svc.spreadsheets.return_value.values.return_value.append.call_args.kwargs
    assert kwargs["valueInputOption"] == "RAW"
    assert kwargs["body"] == {"values": [["A", "B"]]}


@pytest.mark.asyncio
async def test_sheets_append_handles_missing_updates(monkeypatch, make_ctx) -> None:
    svc = MagicMock()
    svc.spreadsheets.return_value.values.return_value.append.return_value.execute.return_value = {}
    monkeypatch.setattr(sheets_mod, "sheets_service", lambda c: svc)

    out = await SheetsAppend().run(
        make_ctx(),
        SheetsAppendInput(spreadsheet_id="x", range_a1="A", rows=[["a"]]),
    )
    assert out.updated_range == ""
    assert out.updated_rows == 0


@pytest.mark.asyncio
async def test_sheets_append_wraps_http_error(monkeypatch, make_ctx) -> None:
    svc = MagicMock()
    svc.spreadsheets.return_value.values.return_value.append.return_value.execute.side_effect = (
        HttpError(MagicMock(status=500, reason="x"), b"")
    )
    monkeypatch.setattr(sheets_mod, "sheets_service", lambda c: svc)

    with pytest.raises(ToolError, match="Sheets append failed"):
        await SheetsAppend().run(
            make_ctx(),
            SheetsAppendInput(spreadsheet_id="x", range_a1="A", rows=[["a"]]),
        )


def test_sheets_append_requires_approval() -> None:
    assert SheetsAppend.requires_approval is True


def test_sheets_read_does_not_require_approval() -> None:
    assert SheetsRead.requires_approval is False
