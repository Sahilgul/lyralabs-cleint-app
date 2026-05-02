"""LangGraph wiring (unified tool-using agent).

```mermaid
flowchart TD
  start([entry]) --> agent
  agent -- direct reply --> finish([end])
  agent -- read tool --> tool_node --> agent
  agent -- submit_plan_for_approval --> approval
  approval -- all-LOW auto-approved --> executor
  approval -- MEDIUM/HIGH approved --> executor
  approval -- rejected --> rejected_reply --> finish
  executor --> critic --> living_artifact --> finish
```
"""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from .nodes.agent import agent_node, route_after_agent
from .nodes.approval import (
    approval_node,
    rejected_reply_node,
    route_after_approval,
)
from .nodes.critic import critic_node
from .nodes.executor import executor_node
from .nodes.living_artifact import living_artifact_node
from .nodes.tool_node import tool_node
from .state import AgentState


def build_agent_graph(checkpointer: BaseCheckpointSaver):
    g = StateGraph(AgentState)

    g.add_node("agent", agent_node)
    g.add_node("tool_node", tool_node)
    g.add_node("approval", approval_node)
    g.add_node("rejected_reply", rejected_reply_node)
    g.add_node("executor", executor_node)
    g.add_node("critic", critic_node)
    g.add_node("living_artifact", living_artifact_node)

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
    g.add_edge("critic", "living_artifact")
    g.add_edge("living_artifact", END)
    g.add_edge("rejected_reply", END)

    return g.compile(checkpointer=checkpointer)
