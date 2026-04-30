"""Artifact post-processing node.

Walks step_results for any step that produced raw artifacts (e.g. PDFs)
and uploads them to Slack alongside the critic's summary message.
For MVP this is a placeholder; PDF/chart tools render artifacts inline.
"""

from __future__ import annotations

from typing import Any

from ..state import AgentState


async def artifact_node(state: AgentState) -> dict[str, Any]:
    return {"artifacts": state.get("artifacts", [])}
