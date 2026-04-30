"""Plotly-based chart tools (line + bar). Output PNG bytes."""

from __future__ import annotations

import asyncio
import base64

from pydantic import BaseModel, Field

from ..base import Tool, ToolContext, ToolError
from ..registry import default_registry


class ChartLineInput(BaseModel):
    title: str
    x: list[str | float | int] = Field(description="X-axis values.")
    series: dict[str, list[float]] = Field(description="Map of series name -> y values.")
    x_label: str = ""
    y_label: str = ""
    filename: str = "chart.png"


class ChartLineOutput(BaseModel):
    filename: str
    size_bytes: int
    content_b64: str


class ChartLine(Tool[ChartLineInput, ChartLineOutput]):
    name = "artifact.chart.line"
    description = "Render a multi-series line chart as PNG. One key in `series` per line."
    provider = ""
    Input = ChartLineInput
    Output = ChartLineOutput

    async def run(self, ctx: ToolContext, args: ChartLineInput) -> ChartLineOutput:
        return await _render_chart(ctx, args, kind="line")


class ChartBarInput(BaseModel):
    title: str
    categories: list[str]
    values: list[float]
    x_label: str = ""
    y_label: str = ""
    filename: str = "chart.png"


class ChartBarOutput(BaseModel):
    filename: str
    size_bytes: int
    content_b64: str


class ChartBar(Tool[ChartBarInput, ChartBarOutput]):
    name = "artifact.chart.bar"
    description = "Render a single-series bar chart as PNG."
    provider = ""
    Input = ChartBarInput
    Output = ChartBarOutput

    async def run(self, ctx: ToolContext, args: ChartBarInput) -> ChartBarOutput:
        return await _render_chart(ctx, args, kind="bar")


async def _render_chart(ctx: ToolContext, args, *, kind: str):
    import plotly.graph_objects as go

    def _render() -> bytes:
        if kind == "line":
            fig = go.Figure()
            for name, ys in args.series.items():
                fig.add_trace(go.Scatter(x=args.x, y=ys, mode="lines+markers", name=name))
            fig.update_layout(
                title=args.title,
                xaxis_title=args.x_label,
                yaxis_title=args.y_label,
                template="simple_white",
                width=900,
                height=500,
            )
        else:
            fig = go.Figure(go.Bar(x=args.categories, y=args.values))
            fig.update_layout(
                title=args.title,
                xaxis_title=args.x_label,
                yaxis_title=args.y_label,
                template="simple_white",
                width=900,
                height=500,
            )
        return fig.to_image(format="png", engine="kaleido")

    try:
        png = await asyncio.to_thread(_render)
    except Exception as exc:  # noqa: BLE001
        raise ToolError(f"chart render failed: {exc}") from exc

    ctx.extra.setdefault("artifacts", []).append(
        {
            "kind": "png",
            "filename": args.filename,
            "content_b64": base64.b64encode(png).decode("ascii"),
            "description": args.title,
        }
    )

    out_cls = ChartLineOutput if kind == "line" else ChartBarOutput
    return out_cls(
        filename=args.filename, size_bytes=len(png), content_b64=base64.b64encode(png).decode("ascii")
    )


default_registry.register(ChartLine())
default_registry.register(ChartBar())
