"""lyra_core.tools.google.drive."""

from __future__ import annotations

import io
from unittest.mock import MagicMock

import pytest
from googleapiclient.errors import HttpError
from lyra_core.tools.base import ToolError
from lyra_core.tools.google import drive as drive_mod
from lyra_core.tools.google.drive import (
    DriveRead,
    DriveReadInput,
    DriveSearch,
    DriveSearchInput,
)


def _make_drive_service(list_response: dict | None = None, **overrides):
    """Build a chained MagicMock that mimics the Drive service .files().list().execute()."""
    svc = MagicMock(name="drive_service")
    if list_response is not None:
        svc.files.return_value.list.return_value.execute.return_value = list_response
    for k, v in overrides.items():
        setattr(svc, k, v)
    return svc


@pytest.mark.asyncio
async def test_drive_search_returns_files(monkeypatch, make_ctx) -> None:
    svc = _make_drive_service(
        {
            "files": [
                {
                    "id": "file-1",
                    "name": "Doc1",
                    "mimeType": "application/vnd.google-apps.document",
                    "modifiedTime": "2026-04-30T12:00:00Z",
                    "webViewLink": "https://docs.google.com/document/d/file-1",
                },
                {
                    "id": "file-2",
                    "name": "Sheet1",
                    "mimeType": "application/vnd.google-apps.spreadsheet",
                },
            ]
        }
    )
    monkeypatch.setattr(drive_mod, "drive_service", lambda creds: svc)

    out = await DriveSearch().run(make_ctx(), DriveSearchInput(query="leads"))
    assert len(out.files) == 2
    assert out.files[0].id == "file-1"
    assert out.files[0].name == "Doc1"
    assert out.files[1].mime_type == "application/vnd.google-apps.spreadsheet"
    # web_view_link optional
    assert out.files[1].web_view_link is None


@pytest.mark.asyncio
async def test_drive_search_includes_mime_filter(monkeypatch, make_ctx) -> None:
    svc = _make_drive_service({"files": []})
    monkeypatch.setattr(drive_mod, "drive_service", lambda creds: svc)

    await DriveSearch().run(
        make_ctx(),
        DriveSearchInput(
            query="quarterly report", mime_type="application/vnd.google-apps.spreadsheet"
        ),
    )
    q = svc.files.return_value.list.call_args.kwargs["q"]
    assert "fullText contains 'quarterly report'" in q
    assert "mimeType = 'application/vnd.google-apps.spreadsheet'" in q
    assert "trashed = false" in q


@pytest.mark.asyncio
async def test_drive_search_strips_quotes_from_query(monkeypatch, make_ctx) -> None:
    svc = _make_drive_service({"files": []})
    monkeypatch.setattr(drive_mod, "drive_service", lambda creds: svc)

    await DriveSearch().run(make_ctx(), DriveSearchInput(query="bob's leads"))
    q = svc.files.return_value.list.call_args.kwargs["q"]
    assert "'" not in q.replace("'bobs leads'", "")  # only the wrapping quotes remain
    assert "bobs leads" in q


@pytest.mark.asyncio
async def test_drive_search_page_size_passthrough(monkeypatch, make_ctx) -> None:
    svc = _make_drive_service({"files": []})
    monkeypatch.setattr(drive_mod, "drive_service", lambda creds: svc)

    await DriveSearch().run(make_ctx(), DriveSearchInput(query="x", page_size=25))
    assert svc.files.return_value.list.call_args.kwargs["pageSize"] == 25


@pytest.mark.asyncio
async def test_drive_search_validation_caps_page_size() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        DriveSearchInput(query="x", page_size=200)


@pytest.mark.asyncio
async def test_drive_search_wraps_http_error_as_tool_error(monkeypatch, make_ctx) -> None:
    svc = MagicMock()
    err = HttpError(MagicMock(status=403, reason="Forbidden"), b"forbidden")
    svc.files.return_value.list.return_value.execute.side_effect = err
    monkeypatch.setattr(drive_mod, "drive_service", lambda creds: svc)

    with pytest.raises(ToolError, match="Drive search failed"):
        await DriveSearch().run(make_ctx(), DriveSearchInput(query="x"))


@pytest.mark.asyncio
async def test_drive_read_doc_exports_to_text(monkeypatch, make_ctx) -> None:
    svc = MagicMock()
    svc.files.return_value.get.return_value.execute.return_value = {
        "id": "file-1",
        "name": "MyDoc",
        "mimeType": "application/vnd.google-apps.document",
    }
    monkeypatch.setattr(drive_mod, "drive_service", lambda creds: svc)

    # Patch MediaIoBaseDownload to write some bytes
    class FakeDownloader:
        def __init__(self, buf: io.BytesIO, *_a, **_kw) -> None:
            self.buf = buf
            self.calls = 0

        def next_chunk(self):
            if self.calls == 0:
                self.buf.write(b"hello world")
                self.calls += 1
                return MagicMock(progress=lambda: 1.0), True
            return MagicMock(), True

    monkeypatch.setattr(drive_mod, "MediaIoBaseDownload", FakeDownloader)

    out = await DriveRead().run(make_ctx(), DriveReadInput(file_id="file-1"))
    assert out.file_id == "file-1"
    assert out.name == "MyDoc"
    assert out.content_text == "hello world"
    assert out.truncated is False
    # Doc -> default export to text/plain
    svc.files.return_value.export_media.assert_called_once_with(
        fileId="file-1", mimeType="text/plain"
    )


@pytest.mark.asyncio
async def test_drive_read_sheet_exports_to_csv_by_default(monkeypatch, make_ctx) -> None:
    svc = MagicMock()
    svc.files.return_value.get.return_value.execute.return_value = {
        "id": "f",
        "name": "S",
        "mimeType": "application/vnd.google-apps.spreadsheet",
    }
    monkeypatch.setattr(drive_mod, "drive_service", lambda creds: svc)

    class FakeDownloader:
        def __init__(self, buf, *_a, **_kw):
            self.buf = buf

        def next_chunk(self):
            self.buf.write(b"a,b\n1,2\n")
            return MagicMock(), True

    monkeypatch.setattr(drive_mod, "MediaIoBaseDownload", FakeDownloader)

    out = await DriveRead().run(make_ctx(), DriveReadInput(file_id="f"))
    assert out.content_text == "a,b\n1,2\n"
    assert out.mime_type == "application/vnd.google-apps.spreadsheet"
    svc.files.return_value.export_media.assert_called_once_with(fileId="f", mimeType="text/csv")


@pytest.mark.asyncio
async def test_drive_read_binary_uses_get_media(monkeypatch, make_ctx) -> None:
    svc = MagicMock()
    svc.files.return_value.get.return_value.execute.return_value = {
        "id": "f",
        "name": "n.pdf",
        "mimeType": "application/pdf",
    }
    monkeypatch.setattr(drive_mod, "drive_service", lambda creds: svc)

    class FakeDownloader:
        def __init__(self, buf, *_a, **_kw):
            self.buf = buf

        def next_chunk(self):
            self.buf.write(b"%PDF-binary")
            return MagicMock(), True

    monkeypatch.setattr(drive_mod, "MediaIoBaseDownload", FakeDownloader)

    await DriveRead().run(make_ctx(), DriveReadInput(file_id="f"))
    svc.files.return_value.get_media.assert_called_once_with(fileId="f")
    svc.files.return_value.export_media.assert_not_called()


@pytest.mark.asyncio
async def test_drive_read_truncates_when_exceeds_max(monkeypatch, make_ctx) -> None:
    svc = MagicMock()
    svc.files.return_value.get.return_value.execute.return_value = {
        "id": "f",
        "name": "big.txt",
        "mimeType": "text/plain",
    }
    monkeypatch.setattr(drive_mod, "drive_service", lambda creds: svc)

    class FakeDownloader:
        def __init__(self, buf, *_a, **_kw):
            self.buf = buf

        def next_chunk(self):
            self.buf.write(b"x" * (DriveRead._MAX_BYTES + 100))
            return MagicMock(), True

    monkeypatch.setattr(drive_mod, "MediaIoBaseDownload", FakeDownloader)

    out = await DriveRead().run(make_ctx(), DriveReadInput(file_id="f"))
    assert out.truncated is True
    assert len(out.content_text) == DriveRead._MAX_BYTES


@pytest.mark.asyncio
async def test_drive_read_wraps_http_error(monkeypatch, make_ctx) -> None:
    svc = MagicMock()
    svc.files.return_value.get.return_value.execute.return_value = {
        "id": "f",
        "name": "x",
        "mimeType": "text/plain",
    }
    monkeypatch.setattr(drive_mod, "drive_service", lambda creds: svc)

    class BadDownloader:
        def __init__(self, *_a, **_kw):
            raise HttpError(MagicMock(status=500, reason="Err"), b"boom")

    monkeypatch.setattr(drive_mod, "MediaIoBaseDownload", BadDownloader)
    with pytest.raises(ToolError, match="Drive read failed"):
        await DriveRead().run(make_ctx(), DriveReadInput(file_id="f"))
