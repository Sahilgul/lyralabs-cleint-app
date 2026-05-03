"""lyra_core.tools.google.docs."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from googleapiclient.errors import HttpError
from lyra_core.tools.base import ToolError
from lyra_core.tools.google import docs as docs_mod
from lyra_core.tools.google.docs import DocsCreate, DocsCreateInput


def _docs_svc(create_response=None):
    s = MagicMock()
    s.documents.return_value.create.return_value.execute.return_value = create_response or {
        "documentId": "doc-XYZ",
        "title": "T",
    }
    return s


def _drive_svc(parents=None):
    s = MagicMock()
    s.files.return_value.get.return_value.execute.return_value = {"parents": parents or ["root"]}
    return s


@pytest.mark.asyncio
async def test_docs_create_returns_dry_run(monkeypatch, make_ctx) -> None:
    monkeypatch.setattr(docs_mod, "docs_service", lambda c: MagicMock())
    monkeypatch.setattr(docs_mod, "drive_service", lambda c: MagicMock())

    out = await DocsCreate().run(
        make_ctx(dry_run=True),
        DocsCreateInput(title="DryRun Doc", body_text="hello"),
    )
    assert out.document_id == "dry-run"
    assert out.title == "DryRun Doc"
    assert out.web_view_link.endswith("dry-run/edit")


@pytest.mark.asyncio
async def test_docs_create_inserts_body_and_returns_link(monkeypatch, make_ctx) -> None:
    docs_svc = _docs_svc({"documentId": "abc", "title": "Hello"})
    drive_svc = _drive_svc()
    monkeypatch.setattr(docs_mod, "docs_service", lambda c: docs_svc)
    monkeypatch.setattr(docs_mod, "drive_service", lambda c: drive_svc)

    out = await DocsCreate().run(
        make_ctx(),
        DocsCreateInput(title="Hello", body_text="World!"),
    )

    assert out.document_id == "abc"
    assert out.title == "Hello"
    assert out.web_view_link == "https://docs.google.com/document/d/abc/edit"
    docs_svc.documents.return_value.batchUpdate.assert_called_once()
    body = docs_svc.documents.return_value.batchUpdate.call_args.kwargs["body"]
    assert body["requests"][0]["insertText"]["text"] == "World!"


@pytest.mark.asyncio
async def test_docs_create_moves_to_folder(monkeypatch, make_ctx) -> None:
    docs_svc = _docs_svc({"documentId": "abc", "title": "T"})
    drive_svc = _drive_svc(parents=["root", "old-folder"])
    monkeypatch.setattr(docs_mod, "docs_service", lambda c: docs_svc)
    monkeypatch.setattr(docs_mod, "drive_service", lambda c: drive_svc)

    await DocsCreate().run(
        make_ctx(),
        DocsCreateInput(title="T", body_text="x", folder_id="folder-1"),
    )

    update_call = drive_svc.files.return_value.update.call_args
    assert update_call.kwargs["addParents"] == "folder-1"
    assert update_call.kwargs["removeParents"] == "root,old-folder"
    assert update_call.kwargs["fileId"] == "abc"


@pytest.mark.asyncio
async def test_docs_create_skips_move_when_no_folder(monkeypatch, make_ctx) -> None:
    docs_svc = _docs_svc({"documentId": "abc", "title": "T"})
    drive_svc = _drive_svc()
    monkeypatch.setattr(docs_mod, "docs_service", lambda c: docs_svc)
    monkeypatch.setattr(docs_mod, "drive_service", lambda c: drive_svc)

    await DocsCreate().run(make_ctx(), DocsCreateInput(title="T", body_text="x"))
    drive_svc.files.return_value.update.assert_not_called()


@pytest.mark.asyncio
async def test_docs_create_wraps_http_error(monkeypatch, make_ctx) -> None:
    docs_svc = MagicMock()
    docs_svc.documents.return_value.create.return_value.execute.side_effect = HttpError(
        MagicMock(status=500, reason="x"), b""
    )
    monkeypatch.setattr(docs_mod, "docs_service", lambda c: docs_svc)
    monkeypatch.setattr(docs_mod, "drive_service", lambda c: MagicMock())

    with pytest.raises(ToolError, match="Docs create failed"):
        await DocsCreate().run(make_ctx(), DocsCreateInput(title="t", body_text="b"))


def test_docs_create_requires_approval() -> None:
    assert DocsCreate.requires_approval is True
