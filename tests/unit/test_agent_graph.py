"""lyra_core.agent.graph wiring."""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from lyra_core.agent import build_agent_graph


def test_build_agent_graph_compiles() -> None:
    g = build_agent_graph(MemorySaver())
    assert g is not None


def test_graph_has_expected_nodes() -> None:
    """Unified graph: agent + tool_node + approval/executor/critic. No legacy nodes."""
    g = build_agent_graph(MemorySaver())
    nodes = set(g.nodes.keys())
    expected = {"agent", "tool_node", "approval_post", "approval_wait", "rejected_reply", "executor", "critic"}
    assert expected <= nodes
    for legacy in ("classifier", "planner", "smalltalk_reply"):
        assert legacy not in nodes
