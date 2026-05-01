"""Agent runtime. Heavy imports (langgraph, llm) are deferred."""

from .state import AgentState, Plan, PlanStep

__all__ = ["AgentState", "Plan", "PlanStep", "build_agent_graph"]


def build_agent_graph(checkpointer):
    """Lazy passthrough so importing lyra_core.agent doesn't pull in langgraph."""
    from .graph import build_agent_graph as _impl

    return _impl(checkpointer)
