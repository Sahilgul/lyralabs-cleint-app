"""lyra_core.agent.graph wiring."""

from __future__ import annotations

import pytest


def test_build_agent_graph_compiles() -> None:
    """The graph must compile with the in-memory checkpointer."""
    from langgraph.checkpoint.memory import MemorySaver

    from lyra_core.agent import build_agent_graph

    g = build_agent_graph(MemorySaver())
    assert g is not None


def test_graph_has_expected_nodes() -> None:
    from langgraph.checkpoint.memory import MemorySaver

    from lyra_core.agent import build_agent_graph

    g = build_agent_graph(MemorySaver())
    # Compiled graph exposes its internal graph; nodes live on .nodes
    nodes = set(g.nodes.keys())
    expected = {
        "classifier",
        "planner",
        "approval",
        "rejected_reply",
        "executor",
        "critic",
        "smalltalk_reply",
    }
    # __start__ etc. are internal markers; check the user-defined are present.
    assert expected <= nodes
