"""LangGraph wiring.

Two modes are exported via the `AGENT_MODE` env var (default `legacy`):

  legacy   classifier -> planner -> approval -> executor -> critic
  unified  agent_node <-> tool_node, with submit_plan_for_approval
           routing into approval -> executor -> critic

Both share the same approval gate, executor, and critic, so writes are
gated identically. The unified graph is the migration target -- once it
soaks in production, delete the legacy nodes.

```mermaid
flowchart TD
  start([entry]) --> agent
  agent -- direct reply --> finish([end])
  agent -- read tool --> tool_node --> agent
  agent -- submit_plan_for_approval --> approval
  approval -- approved --> executor --> critic --> finish
  approval -- rejected --> rejected_reply --> finish
```
"""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from ..common.config import get_settings
from .nodes.agent import agent_node, route_after_agent
from .nodes.approval import (
    approval_node,
    rejected_reply_node,
    route_after_approval,
    route_after_plan,
)
from .nodes.classifier import classifier_node, route_after_classifier
from .nodes.critic import critic_node
from .nodes.executor import executor_node
from .nodes.planner import planner_node
from .nodes.smalltalk import smalltalk_reply_node
from .nodes.tool_node import tool_node
from .state import AgentState


def _build_legacy_graph(checkpointer: BaseCheckpointSaver):
    g = StateGraph(AgentState)

    g.add_node("classifier", classifier_node)
    g.add_node("planner", planner_node)
    g.add_node("approval", approval_node)
    g.add_node("rejected_reply", rejected_reply_node)
    g.add_node("executor", executor_node)
    g.add_node("critic", critic_node)
    g.add_node("smalltalk_reply", smalltalk_reply_node)

    g.add_edge(START, "classifier")
    g.add_conditional_edges(
        "classifier",
        route_after_classifier,
        {"smalltalk_reply": "smalltalk_reply", "planner": "planner"},
    )
    g.add_conditional_edges(
        "planner",
        route_after_plan,
        {
            "smalltalk_reply": "smalltalk_reply",
            "approval": "approval",
            "executor": "executor",
        },
    )
    g.add_conditional_edges(
        "approval",
        route_after_approval,
        {"executor": "executor", "rejected_reply": "rejected_reply"},
    )
    g.add_edge("executor", "critic")

    g.add_edge("critic", END)
    g.add_edge("smalltalk_reply", END)
    g.add_edge("rejected_reply", END)

    return g.compile(checkpointer=checkpointer)


def _build_unified_graph(checkpointer: BaseCheckpointSaver):
    g = StateGraph(AgentState)

    g.add_node("agent", agent_node)
    g.add_node("tool_node", tool_node)
    g.add_node("approval", approval_node)
    g.add_node("rejected_reply", rejected_reply_node)
    g.add_node("executor", executor_node)
    g.add_node("critic", critic_node)

    g.add_edge(START, "agent")
    g.add_conditional_edges(
        "agent",
        route_after_agent,
        {"tool_node": "tool_node", "approval": "approval", END: END},
    )
    g.add_edge("tool_node", "agent")
    g.add_conditional_edges(
        "approval",
        route_after_approval,
        {"executor": "executor", "rejected_reply": "rejected_reply"},
    )
    g.add_edge("executor", "critic")
    g.add_edge("critic", END)
    g.add_edge("rejected_reply", END)

    return g.compile(checkpointer=checkpointer)


def build_agent_graph(checkpointer: BaseCheckpointSaver):
    mode = get_settings().agent_mode
    if mode == "unified":
        return _build_unified_graph(checkpointer)
    return _build_legacy_graph(checkpointer)
