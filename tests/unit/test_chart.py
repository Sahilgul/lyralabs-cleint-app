"""lyra_core.tools.artifacts.chart — line + bar charts.

We mock plotly to avoid spinning kaleido in CI. Validates chart construction
+ artifact lifting + error handling.
"""

from __future__ import annotations

import base64
import sys
import types

import pytest
from lyra_core.tools.artifacts.chart import (
    ChartBar,
    ChartBarInput,
    ChartLine,
    ChartLineInput,
)
from lyra_core.tools.base import ToolContext, ToolError


class _FakeFigure:
    """Captures Plotly calls into a dict for assertions."""

    def __init__(self, *args, **kwargs) -> None:
        self.constructor_args = args
        self.constructor_kwargs = kwargs
        self.layout: dict = {}
        self.traces: list = list(args)

    def add_trace(self, trace) -> None:
        self.traces.append(trace)

    def update_layout(self, **kw) -> None:
        self.layout.update(kw)

    def to_image(self, format: str, engine: str) -> bytes:
        assert format == "png"
        assert engine == "kaleido"
        return b"\x89PNG-MOCK"


def _install_fake_plotly(monkeypatch) -> dict:
    captured = {"scatter_calls": [], "bar_calls": []}

    def Scatter(**kw):  # noqa: N802 mocks Plotly's PascalCase class
        captured["scatter_calls"].append(kw)
        return ("scatter", kw)

    def Bar(**kw):  # noqa: N802 mocks Plotly's PascalCase class
        captured["bar_calls"].append(kw)
        return ("bar", kw)

    fake_go = types.SimpleNamespace(Figure=_FakeFigure, Scatter=Scatter, Bar=Bar)
    fake_pkg = types.ModuleType("plotly")
    fake_pkg.graph_objects = fake_go
    monkeypatch.setitem(sys.modules, "plotly", fake_pkg)
    monkeypatch.setitem(sys.modules, "plotly.graph_objects", fake_go)
    return captured


@pytest.mark.asyncio
async def test_line_chart_adds_one_trace_per_series(monkeypatch) -> None:
    captured = _install_fake_plotly(monkeypatch)

    # Bar's __init__ takes positional, but Scatter is called via go.Scatter(...).
    # Above fake stores call kwargs (we'll see if mode='lines+markers' is passed).

    ctx = ToolContext(tenant_id="t-1")
    out = await ChartLine().run(
        ctx,
        ChartLineInput(
            title="Revenue",
            x=["Jan", "Feb", "Mar"],
            series={"2025": [1.0, 2.0, 3.0], "2026": [4.0, 5.0, 6.0]},
            x_label="Month",
            y_label="USD",
            filename="rev.png",
        ),
    )
    assert out.filename == "rev.png"
    assert base64.b64decode(out.content_b64) == b"\x89PNG-MOCK"
    assert out.size_bytes == len(b"\x89PNG-MOCK")
    assert len(captured["scatter_calls"]) == 2
    assert all(c["mode"] == "lines+markers" for c in captured["scatter_calls"])
    assert {c["name"] for c in captured["scatter_calls"]} == {"2025", "2026"}

    arts = ctx.extra["artifacts"]
    assert len(arts) == 1
    assert arts[0]["kind"] == "png"
    assert arts[0]["description"] == "Revenue"


@pytest.mark.asyncio
async def test_bar_chart_uses_categories_and_values(monkeypatch) -> None:
    captured = _install_fake_plotly(monkeypatch)
    ctx = ToolContext(tenant_id="t-1")

    out = await ChartBar().run(
        ctx,
        ChartBarInput(
            title="By Region",
            categories=["NA", "EU", "APAC"],
            values=[10.0, 20.0, 30.0],
            x_label="Region",
            y_label="Sales",
            filename="b.png",
        ),
    )
    assert out.filename == "b.png"
    assert len(captured["bar_calls"]) == 1
    bar_kw = captured["bar_calls"][0]
    assert bar_kw["x"] == ["NA", "EU", "APAC"]
    assert bar_kw["y"] == [10.0, 20.0, 30.0]


@pytest.mark.asyncio
async def test_chart_render_failure_raises_tool_error(monkeypatch) -> None:
    class BadFigure:
        def __init__(self, *a, **kw):
            pass

        def add_trace(self, *a, **kw):
            pass

        def update_layout(self, **kw):
            pass

        def to_image(self, *, format, engine):
            raise RuntimeError("kaleido missing")

    fake_go = types.SimpleNamespace(
        Figure=BadFigure, Scatter=lambda **kw: None, Bar=lambda **kw: None
    )
    fake_pkg = types.ModuleType("plotly")
    fake_pkg.graph_objects = fake_go
    monkeypatch.setitem(sys.modules, "plotly", fake_pkg)
    monkeypatch.setitem(sys.modules, "plotly.graph_objects", fake_go)

    ctx = ToolContext(tenant_id="t-1")
    with pytest.raises(ToolError, match="chart render failed"):
        await ChartLine().run(ctx, ChartLineInput(title="t", x=["a"], series={"s": [1.0]}))


@pytest.mark.asyncio
async def test_chart_layout_includes_title_and_axis_labels(monkeypatch) -> None:
    """Verify the layout kwargs flow into Plotly."""
    layout_seen = {}

    class CapturingFigure:
        def __init__(self, *a, **kw):
            pass

        def add_trace(self, *a, **kw):
            pass

        def update_layout(self, **kw):
            layout_seen.update(kw)

        def to_image(self, *, format, engine):
            return b"x"

    fake_go = types.SimpleNamespace(
        Figure=CapturingFigure,
        Scatter=lambda **kw: None,
        Bar=lambda **kw: None,
    )
    fake_pkg = types.ModuleType("plotly")
    fake_pkg.graph_objects = fake_go
    monkeypatch.setitem(sys.modules, "plotly", fake_pkg)
    monkeypatch.setitem(sys.modules, "plotly.graph_objects", fake_go)

    ctx = ToolContext(tenant_id="t-1")
    await ChartBar().run(
        ctx,
        ChartBarInput(
            title="MyChart",
            categories=["a"],
            values=[1.0],
            x_label="X",
            y_label="Y",
        ),
    )
    assert layout_seen["title"] == "MyChart"
    assert layout_seen["xaxis_title"] == "X"
    assert layout_seen["yaxis_title"] == "Y"
    assert layout_seen["template"] == "simple_white"
