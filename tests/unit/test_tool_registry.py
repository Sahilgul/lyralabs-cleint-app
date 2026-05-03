"""lyra_core.tools.registry."""

from __future__ import annotations

import pytest
from lyra_core.tools.base import Tool, ToolContext
from lyra_core.tools.registry import ToolRegistry, default_registry
from pydantic import BaseModel


class _In(BaseModel):
    pass


class _Out(BaseModel):
    pass


class _T1(Tool[_In, _Out]):
    name = "p.t1"
    description = "t1"
    provider = "p"
    Input = _In
    Output = _Out

    async def run(self, ctx: ToolContext, args: _In) -> _Out:
        return _Out()


class _T2(Tool[_In, _Out]):
    name = "p.t2"
    description = "t2"
    provider = "p"
    Input = _In
    Output = _Out

    async def run(self, ctx: ToolContext, args: _In) -> _Out:
        return _Out()


class _T3(Tool[_In, _Out]):
    name = "q.t3"
    description = "t3"
    provider = "q"
    Input = _In
    Output = _Out

    async def run(self, ctx: ToolContext, args: _In) -> _Out:
        return _Out()


class TestRegistry:
    def test_register_get(self) -> None:
        r = ToolRegistry()
        t = _T1()
        r.register(t)
        assert r.get("p.t1") is t

    def test_register_duplicate_raises(self) -> None:
        r = ToolRegistry()
        r.register(_T1())
        with pytest.raises(ValueError, match="duplicate"):
            r.register(_T1())

    def test_get_unknown_raises(self) -> None:
        r = ToolRegistry()
        with pytest.raises(KeyError):
            r.get("no-such-tool")

    def test_all_returns_list(self) -> None:
        r = ToolRegistry()
        r.register(_T1())
        r.register(_T2())
        names = [t.name for t in r.all()]
        assert sorted(names) == ["p.t1", "p.t2"]

    def test_by_provider_filters(self) -> None:
        r = ToolRegistry()
        r.register(_T1())
        r.register(_T2())
        r.register(_T3())
        p = sorted(t.name for t in r.by_provider("p"))
        q = sorted(t.name for t in r.by_provider("q"))
        assert p == ["p.t1", "p.t2"]
        assert q == ["q.t3"]

    def test_schemas_filter_by_names(self) -> None:
        r = ToolRegistry()
        r.register(_T1())
        r.register(_T2())
        ss = r.schemas(["p.t2"])
        assert len(ss) == 1
        assert ss[0]["function"]["name"] == "p.t2"

    def test_schemas_default_returns_all(self) -> None:
        r = ToolRegistry()
        r.register(_T1())
        r.register(_T2())
        assert len(r.schemas()) == 2


class TestDefaultRegistry:
    def test_google_registered_on_import(self) -> None:
        from lyra_core.tools import google as _g  # noqa: F401

        names = {t.name for t in default_registry.by_provider("google")}
        for n in (
            "google.drive.search",
            "google.drive.read",
            "google.docs.create",
            "google.sheets.read",
            "google.sheets.append",
            "google.calendar.create_event",
        ):
            assert n in names

    def test_ghl_registered_on_import(self) -> None:
        from lyra_core.tools import ghl as _g  # noqa: F401

        names = {t.name for t in default_registry.by_provider("ghl")}
        for n in (
            "ghl.contacts.search",
            "ghl.contacts.create",
            "ghl.pipelines.opportunities",
            "ghl.conversations.send_message",
            "ghl.calendars.book_appointment",
        ):
            assert n in names

    def test_artifacts_registered_on_import(self) -> None:
        from lyra_core.tools import artifacts as _a  # noqa: F401

        names = {t.name for t in default_registry.all()}
        for n in (
            "artifact.pdf.from_markdown",
            "artifact.chart.line",
            "artifact.chart.bar",
        ):
            assert n in names

    def test_write_tools_marked_for_approval(self) -> None:
        from lyra_core.tools import artifacts, ghl, google  # noqa: F401

        for n in [
            "google.docs.create",
            "google.sheets.append",
            "google.calendar.create_event",
            "ghl.contacts.create",
            "ghl.conversations.send_message",
            "ghl.calendars.book_appointment",
        ]:
            assert default_registry.get(n).requires_approval, n

    def test_read_tools_not_marked_for_approval(self) -> None:
        from lyra_core.tools import ghl, google  # noqa: F401

        for n in [
            "google.drive.search",
            "google.drive.read",
            "google.sheets.read",
            "ghl.contacts.search",
            "ghl.pipelines.opportunities",
        ]:
            assert default_registry.get(n).requires_approval is False, n
