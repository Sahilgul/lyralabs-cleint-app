"""Regression test 5 — duplicate plan card on approval resume.

Bug: After a user clicked Approve on a plan card, ARLO posted a SECOND identical
plan card with live buttons. The user could (and did) click the second one,
triggering a duplicate `resume_agent` job.

Root cause: `approval_node` posted the plan card before calling `interrupt()`.
LangGraph's interrupt mechanism re-runs the entire node body when resuming.
On resume the node ran from the top again, called `post_reply` a second time
(producing the duplicate card), then `interrupt()` returned the resume value
immediately because the resume Command was pending.

Fix: split `approval_node` into:
  - `approval_post_node` — posts the card, returns a flag, completes (and
    is checkpointed at the node boundary)
  - `approval_wait_node` — calls `interrupt()` only

LangGraph checkpoints between nodes. On resume from `approval_wait_node`'s
interrupt, only that node re-runs. `approval_post_node` is not re-entered,
so the card is never re-posted.

Regression guards:
  1. A run that triggers the approval gate posts the plan card exactly once.
  2. Resuming the same thread with `Command(resume="approved")` does NOT
     post the card again. Total `post_reply` calls for the approval card
     stays at 1 across the run + resume cycle.
  3. The split-node graph topology contains both `approval_post` and
     `approval_wait` nodes, with only `approval_wait` calling `interrupt()`.
"""

from __future__ import annotations

import inspect

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from lyra_core.agent import build_agent_graph
from lyra_core.agent.nodes import approval as approval_mod
from lyra_core.agent.nodes.approval import (
    approval_post_node,
    approval_wait_node,
    route_after_approval,
)
from lyra_core.agent.state import AgentState, Plan, PlanStep


def _build_approval_subgraph(saver):
    """A minimal graph wiring just the approval gate.

    Mirrors the production wiring (approval_post → approval_wait → END)
    without requiring agent_node, the executor, Postgres, or the LLM. This
    lets the regression exercise the EXACT bug — LangGraph re-running a
    node body across an interrupt resume — in isolation.
    """
    g = StateGraph(AgentState)
    g.add_node("approval_post", approval_post_node)
    g.add_node("approval_wait", approval_wait_node)

    async def _terminal(state):
        return {}

    g.add_node("done", _terminal)

    g.add_edge(START, "approval_post")
    # approval_post → approval_wait when needs_approval_wait, else done.
    g.add_conditional_edges(
        "approval_post",
        lambda s: "approval_wait" if s.get("needs_approval_wait") else "done",
        {"approval_wait": "approval_wait", "done": "done"},
    )
    g.add_conditional_edges(
        "approval_wait",
        route_after_approval,
        {"executor": "done", "rejected_reply": "done"},
    )
    g.add_edge("done", END)
    return g.compile(checkpointer=saver)


def _pending_plan_state(plan: Plan) -> dict:
    return {
        "tenant_id": "tenant-x",
        "client_id": None,
        "job_id": "job-abc",
        "channel_id": "C123",
        "thread_id": "thr-1",
        "user_id": "U1",
        "user_request": "do the thing",
        "pending_plan": plan.model_dump(),
        "step_results": [],
        "artifacts": [],
        "total_cost_usd": 0.0,
        "messages": [],
    }


def _medium_plan_with_unknown_tool() -> Plan:
    """A plan whose tool isn't in the registry — classify_step falls back to
    MEDIUM, which is the path that posts the approval card."""
    return Plan(
        goal="create a contact",
        steps=[
            PlanStep(
                id="step_1",
                tool_name="_regression5_unknown_write_tool",
                args={"name": "Jane"},
                rationale="create the contact",
                requires_approval=True,
            )
        ],
    )


@pytest.mark.asyncio
async def test_approval_card_posted_exactly_once_across_run_and_resume(
    monkeypatch,
):
    """End-to-end: run subgraph until interrupt, resume, assert ONE card post.

    This is the regression for the user-reported "got the same message again
    after I clicked approve" bug. Without the split-node fix, the second
    `ainvoke(Command(resume=...))` re-runs the full approval body and posts
    a second identical card.
    """
    posted_cards: list[dict] = []

    async def fake_post_reply(tenant_id, reply):
        if getattr(reply, "requires_approval", False):
            posted_cards.append({"tenant_id": tenant_id})

    monkeypatch.setattr(approval_mod, "post_reply", fake_post_reply)

    saver = MemorySaver()
    graph = _build_approval_subgraph(saver)

    plan = _medium_plan_with_unknown_tool()
    initial_state = _pending_plan_state(plan)
    config = {"configurable": {"thread_id": "thr-regression5"}}

    # First invocation: should suspend at approval_wait's interrupt().
    result = await graph.ainvoke(initial_state, config=config)
    assert result.get("__interrupt__"), (
        "graph must suspend at approval_wait's interrupt() — got "
        f"keys: {list(result.keys())}"
    )
    assert len(posted_cards) == 1, (
        f"approval card must post exactly once on first run, got "
        f"{len(posted_cards)}"
    )

    # Resume with approval. Without the split-node fix, approval_post
    # re-runs and posts a SECOND card. With the fix, only approval_wait
    # re-runs and the card count stays at 1.
    await graph.ainvoke(Command(resume="approved"), config=config)

    assert len(posted_cards) == 1, (
        "REGRESSION: approval card was re-posted on resume. The split-node "
        "fix in approval.py / graph.py must place the card-post in a "
        "separate node from the interrupt(). "
        f"posted_cards count: {len(posted_cards)}"
    )


@pytest.mark.asyncio
async def test_approval_resume_routes_correctly_on_rejected(monkeypatch):
    """Companion check: rejection path must also resume cleanly without
    re-posting the card. Same bug class, different decision branch."""
    posted_cards: list[dict] = []

    async def fake_post_reply(tenant_id, reply):
        if getattr(reply, "requires_approval", False):
            posted_cards.append({"tenant_id": tenant_id})

    monkeypatch.setattr(approval_mod, "post_reply", fake_post_reply)

    saver = MemorySaver()
    graph = _build_approval_subgraph(saver)

    initial_state = _pending_plan_state(_medium_plan_with_unknown_tool())
    config = {"configurable": {"thread_id": "thr-reg5-reject"}}

    await graph.ainvoke(initial_state, config=config)
    assert len(posted_cards) == 1

    final = await graph.ainvoke(Command(resume="rejected"), config=config)
    assert len(posted_cards) == 1, "card re-posted on reject resume"
    assert final.get("approval_decision") == "rejected"


def test_only_approval_wait_calls_interrupt():
    """Static guard against re-merging the two nodes.

    If a future engineer "simplifies" the split back into one node, the
    interrupt() call would migrate next to the post_reply() call and the
    bug returns. This test asserts only `approval_wait_node` mentions
    `interrupt(` in its source — so the split is structurally preserved.
    """
    post_src = inspect.getsource(approval_mod.approval_post_node)
    wait_src = inspect.getsource(approval_mod.approval_wait_node)
    assert "interrupt(" not in post_src, (
        "approval_post_node must NOT call interrupt() — it must complete "
        "and be checkpointed at the node boundary so the card is not "
        "re-posted on resume."
    )
    assert "interrupt(" in wait_src, (
        "approval_wait_node must call interrupt() — it is the suspension "
        "point for the approval gate."
    )


def test_graph_contains_both_split_nodes():
    """Static topology check: both approval_post and approval_wait exist."""
    g = build_agent_graph(MemorySaver())
    nodes = set(g.nodes.keys())
    assert "approval_post" in nodes, (
        "graph missing approval_post — the split-node fix has been reverted."
    )
    assert "approval_wait" in nodes, (
        "graph missing approval_wait — the split-node fix has been reverted."
    )


def test_approval_post_and_approval_wait_are_distinct_nodes():
    """The two nodes must not be aliased to the same callable."""
    assert (
        approval_mod.approval_post_node is not approval_mod.approval_wait_node
    ), "approval_post_node and approval_wait_node must be distinct functions"
