"""Artifact-generating tools (PDFs, charts).

These tools differ from integration tools: they produce bytes that the
agent attaches to its Slack reply, rather than calling a third-party API.
"""

from .chart import ChartBar, ChartLine
from .pdf import PdfFromMarkdown

__all__ = ["ChartBar", "ChartLine", "PdfFromMarkdown"]
