"""lyra_core.agent.graph wiring."""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from lyra_core.agent import build_agent_graph


def test_build_agent_graph_compiles_legacy(monkeypatch) -> None:
    """The legacy graph must compile."""
    from lyra_core.common import config as cfg

    s = cfg.get_settings()
    monkeypatch.setattr(s, "agent_mode", "legacy", raising=False)

    g = build_agent_graph(MemorySaver())
    assert g is not None


def test_legacy_graph_has_expected_nodes(monkeypatch) -> None:
    from lyra_core.common import config as cfg

    s = cfg.get_settings()
    monkeypatch.setattr(s, "agent_mode", "legacy", raising=False)

    g = build_agent_graph(MemorySaver())
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
    assert expected <= nodes


def test_unified_graph_has_expected_nodes(monkeypatch) -> None:
    """Unified mode wires agent + tool_node, drops classifier/planner/smalltalk."""
    from lyra_core.common import config as cfg

    s = cfg.get_settings()
    monkeypatch.setattr(s, "agent_mode", "unified", raising=False)

    g = build_agent_graph(MemorySaver())
    nodes = set(g.nodes.keys())
    expected = {"agent", "tool_node", "approval", "rejected_reply", "executor", "critic"}
    assert expected <= nodes
    # The legacy nodes must NOT be in the unified graph.
    assert "classifier" not in nodes
    assert "planner" not in nodes
    assert "smalltalk_reply" not in nodes
