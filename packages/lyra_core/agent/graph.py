"""LangGraph wiring.

```mermaid
flowchart TD
  start([entry]) --> classifier
  classifier -- smalltalk --> smalltalk_reply --> finish([end])
  classifier -- task ----> planner
  planner -- needs_clarification --> smalltalk_reply
  planner -- has write step ---> approval
  planner -- read-only ---------> executor
  approval -- approved --------> executor
  approval -- rejected --------> rejected_reply --> finish
  executor --> critic --> finish
```

The graph is checkpointed in Postgres so `approval` can `interrupt()`
and the same thread can be resumed after the user clicks Approve.
"""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

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
from .state import AgentState


def build_agent_graph(checkpointer: BaseCheckpointSaver):
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
