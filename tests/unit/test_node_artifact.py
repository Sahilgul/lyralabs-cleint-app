"""lyra_core.agent.nodes.artifact (passthrough placeholder)."""

from __future__ import annotations

import pytest

from lyra_core.agent.nodes.artifact import artifact_node


@pytest.mark.asyncio
async def test_artifact_node_passes_through() -> None:
    state = {"artifacts": [{"kind": "pdf", "filename": "x.pdf"}]}
    out = await artifact_node(state)  # type: ignore[arg-type]
    assert out == {"artifacts": state["artifacts"]}


@pytest.mark.asyncio
async def test_artifact_node_empty_default() -> None:
    out = await artifact_node({})  # type: ignore[arg-type]
    assert out == {"artifacts": []}
