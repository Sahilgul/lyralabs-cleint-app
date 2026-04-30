"""lyra_core.tools.artifacts.pdf — focused on the markdown->HTML translator.

The actual WeasyPrint render is mocked because it requires native libraries
that are slow to install in CI; we trust WeasyPrint's own tests for the bytes.
"""

from __future__ import annotations

import base64
from unittest.mock import MagicMock

import pytest

from lyra_core.tools.artifacts import pdf as pdf_mod
from lyra_core.tools.artifacts.pdf import (
    PdfFromMarkdown,
    PdfFromMarkdownInput,
    _markdown_to_html,
)
from lyra_core.tools.base import ToolContext, ToolError


class TestMarkdownToHtml:
    def test_h1(self) -> None:
        assert "<h1>Title</h1>" in _markdown_to_html("# Title")

    def test_h2(self) -> None:
        assert "<h2>Sub</h2>" in _markdown_to_html("## Sub")

    def test_h3(self) -> None:
        assert "<h3>Deep</h3>" in _markdown_to_html("### Deep")

    def test_bullets(self) -> None:
        out = _markdown_to_html("- one\n- two")
        assert "<ul>" in out and "<li>one</li>" in out and "<li>two</li>" in out
        assert "</ul>" in out

    def test_star_bullets_recognized(self) -> None:
        out = _markdown_to_html("* a\n* b")
        assert "<li>a</li>" in out

    def test_paragraph_with_bold_italic_code(self) -> None:
        out = _markdown_to_html("hello **world** *italic* `code`")
        assert "<strong>world</strong>" in out
        assert "<em>italic</em>" in out
        assert "<code>code</code>" in out

    def test_code_fence(self) -> None:
        out = _markdown_to_html("```\nprint('x')\n```")
        assert "<pre><code>print('x')</code></pre>" in out

    def test_pipe_table(self) -> None:
        md = "| Name | Age |\n|------|-----|\n| Alice | 30 |\n| Bob | 25 |"
        out = _markdown_to_html(md)
        assert "<table>" in out
        assert "<th>Name</th>" in out
        assert "<th>Age</th>" in out
        assert "<td>Alice</td>" in out
        assert "<td>30</td>" in out

    def test_blank_line_breaks_list(self) -> None:
        out = _markdown_to_html("- a\n\nNot a list")
        assert "<ul>" in out
        assert "</ul>" in out
        assert "<p>Not a list</p>" in out

    def test_table_too_few_rows_skipped(self) -> None:
        # Single row table -> dropped silently
        out = _markdown_to_html("| only |")
        assert "<table>" not in out

    def test_empty_markdown_yields_no_html(self) -> None:
        assert _markdown_to_html("").strip() == ""


class TestPdfFromMarkdown:
    @pytest.mark.asyncio
    async def test_renders_and_appends_artifact(
        self, monkeypatch, make_ctx
    ) -> None:
        # Mock WeasyPrint HTML
        fake_html_cls = MagicMock()
        fake_instance = MagicMock()
        fake_instance.write_pdf.return_value = b"%PDF-MOCK-OUTPUT"
        fake_html_cls.return_value = fake_instance

        # WeasyPrint is imported inside the run method
        import sys
        import types

        fake_module = types.ModuleType("weasyprint")
        fake_module.HTML = fake_html_cls
        monkeypatch.setitem(sys.modules, "weasyprint", fake_module)

        ctx = make_ctx()
        out = await PdfFromMarkdown().run(
            ctx,
            PdfFromMarkdownInput(
                title="Q3 Report",
                markdown="# Hello\n\n- item",
                filename="q3.pdf",
                footer="© Acme",
            ),
        )
        assert out.filename == "q3.pdf"
        assert out.size_bytes == len(b"%PDF-MOCK-OUTPUT")
        assert base64.b64decode(out.content_b64) == b"%PDF-MOCK-OUTPUT"

        artifacts = ctx.extra["artifacts"]
        assert len(artifacts) == 1
        assert artifacts[0]["kind"] == "pdf"
        assert artifacts[0]["filename"] == "q3.pdf"
        assert artifacts[0]["description"] == "Q3 Report"

    @pytest.mark.asyncio
    async def test_renders_html_template_includes_title_and_footer(
        self, monkeypatch
    ) -> None:
        captured = {}

        class FakeHTML:
            def __init__(self, *, string):
                captured["html"] = string

            def write_pdf(self):
                return b"%PDF-mock"

        import sys
        import types

        fake_module = types.ModuleType("weasyprint")
        fake_module.HTML = FakeHTML
        monkeypatch.setitem(sys.modules, "weasyprint", fake_module)

        ctx = ToolContext(tenant_id="t-1")
        await PdfFromMarkdown().run(
            ctx,
            PdfFromMarkdownInput(title="Hi", markdown="# Hi", footer="bottom"),
        )
        assert "<title>Hi</title>" in captured["html"]
        assert "bottom" in captured["html"]
        assert "<h1>Hi</h1>" in captured["html"]

    @pytest.mark.asyncio
    async def test_render_failure_wraps_as_tool_error(self, monkeypatch) -> None:
        class BadHTML:
            def __init__(self, *, string):
                pass

            def write_pdf(self):
                raise RuntimeError("weasy boom")

        import sys
        import types

        fake_module = types.ModuleType("weasyprint")
        fake_module.HTML = BadHTML
        monkeypatch.setitem(sys.modules, "weasyprint", fake_module)

        ctx = ToolContext(tenant_id="t-1")
        with pytest.raises(ToolError, match="PDF render failed"):
            await PdfFromMarkdown().run(
                ctx, PdfFromMarkdownInput(title="x", markdown="# x")
            )
