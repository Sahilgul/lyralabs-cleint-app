"""Regression test 8 — user follow-up message during pending approval.

Bug: When a plan was awaiting approval (graph paused at `approval_wait_node`'s
`interrupt()`) and the user typed a new message in Slack instead of clicking
Approve/Reject, `_run` called `graph.ainvoke(initial_state, config=config)`
on the paused thread. With LangGraph's interrupt protocol, calling `ainvoke`
with a non-Command input on an interrupted thread does not safely deliver
the new `user_request` to `agent_node`. Two failure modes followed:

  1. The pre-existing interrupt re-fired with no resume value, the graph's
     `approval_wait_node` fall-through coerced `decision` to `"rejected"`,
     and the canned `rejected_reply_node` post hit Slack — all without the
     user ever explicitly rejecting anything.
  2. Worse, on subsequent turns `agent_node` ran with `state.messages`
     already containing the synthetic `tool: "Plan submitted for approval."`
     message from the original plan turn, convincing the LLM that the card
     "is above" and producing the user-reported "stop / I hear you /
     Understood" hallucination loop.

Fix: `_run` now calls `graph.aget_state(config)` BEFORE the main `ainvoke`.
If `state.next` contains `"approval_wait"` (meaning the previous turn left
this thread paused at the approval gate), `_run`:

  a. atomically flips the previous `awaiting_approval` Job for this thread
     to `"rejected"` so a stale Approve-button click can't later resume a
     plan we've now superseded;
  b. calls `graph.ainvoke(Command(resume={"decision": "rejected",
     "reason": "user_followup"}), config=config)` to clear the interrupt
     cleanly;
  c. then proceeds with the normal `ainvoke(initial_state, config)` for
     the new request.

The `reason="user_followup"` field flows through `approval_wait_node` into
`state["approval_rejection_reason"]`, and `rejected_reply_node` checks
that field to suppress the canned 'Got it - rejected' Slack post — the new
invocation will produce the actual reply for the new user_request.

Regression guards:
  1. `_run` calls `graph.aget_state(config)` before the main `ainvoke`.
  2. When `aget_state` reports `next` contains `"approval_wait"`, `_run`
     issues a `Command(resume=...)` ainvoke with `decision="rejected"` and
     `reason="user_followup"` BEFORE the main `ainvoke`.
  3. When `aget_state` reports no pending interrupt, `_run` does NOT issue
     the cancel-resume (no false positives on healthy threads).
  4. `approval_wait_node` propagates the resume `reason` into
     `state["approval_rejection_reason"]`.
  5. `rejected_reply_node` does NOT call `post_reply` when
     `approval_rejection_reason == "user_followup"`, but DOES still record
     the implicit cancellation in `state.messages`.
"""

from __future__ import annotations

import inspect
from unittest.mock import AsyncMock, MagicMock

import pytest
from lyra_core.agent.nodes import approval as approval_mod
from lyra_core.agent.nodes.approval import (
    approval_wait_node,
    rejected_reply_node,
)
from lyra_core.channels.schema import InboundMessage, Surface
from lyra_core.db.models import Job, Tenant

from apps.worker.tasks import run_agent as task_mod
from apps.worker.tasks.run_agent import _run

# ---------------------------------------------------------------------------
# Test scaffolding (mirrors the patterns already used in test_run_agent_task)
# ---------------------------------------------------------------------------


def _make_ctx() -> dict:
    redis = AsyncMock()
    redis.set = AsyncMock(return_value=True)
    redis.delete = AsyncMock(return_value=1)
    return {"redis": redis, "job_try": 1}


def _make_msg(text: str = "follow-up message") -> str:
    return InboundMessage(
        surface=Surface.SLACK,
        tenant_external_id="T-XYZ",
        channel_id="C1",
        thread_id="thr-1",
        agent_thread_id="slack:ch:T-XYZ:C1:thr-1",
        user_id="U1",
        text=text,
        is_dm=False,
    ).model_dump_json()


def _tenant() -> Tenant:
    t = Tenant(external_team_id="T-XYZ", channel="slack", name="Acme")
    t.id = "tenant-1"
    t.status = "active"
    return t


class _FakeSession:
    """Minimal async-context-manager fake that records UPDATE statements.

    Reuses the shape of `_FakeSession` from test_run_agent_task.py but adds
    capture of the WHERE-clause column names so the regression test can
    assert that the cancel path hits `Job.thread_id` + `Job.status` and
    sets `Job.status = 'rejected'`.
    """

    def __init__(self, *, tenant=None, job=None, captured: dict | None = None):
        self.tenant = tenant
        self.job = job
        self.captured = captured if captured is not None else {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False

    async def execute(self, stmt):
        from sqlalchemy.sql import Update

        r = MagicMock()
        if isinstance(stmt, Update):
            self.captured.setdefault("updates", []).append(stmt)
            r.first.return_value = (self.job.id,) if self.job is not None else None
            return r
        target = self.job if self.job is not None else self.tenant
        r.scalar_one_or_none.return_value = target
        r.scalar_one.return_value = target
        return r

    def add(self, obj):
        self.captured.setdefault("added", []).append(obj)
        if isinstance(obj, Job):
            obj.id = "job-uuid-new"

    async def commit(self):
        self.captured["committed"] = True

    async def flush(self):
        return None

    async def refresh(self, obj):
        if not getattr(obj, "id", None):
            obj.id = "job-uuid-new"


def _patch_session_chain(monkeypatch, sessions: list) -> None:
    it = iter(sessions)
    monkeypatch.setattr(task_mod, "async_session", lambda: next(it))


def _snapshot(*, next_nodes: tuple[str, ...] = ()) -> MagicMock:
    """Mock LangGraph StateSnapshot with a configurable `next` field."""
    snap = MagicMock()
    snap.next = next_nodes
    return snap


def _fake_saver_factory():
    class _FakeSaver:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

    return _FakeSaver


# ---------------------------------------------------------------------------
# Source-level guards (cheap; catch accidental reverts)
# ---------------------------------------------------------------------------


def test_run_calls_aget_state_before_ainvoke() -> None:
    """Source guard: the cancel-pending-approval helper must run before the
    main ainvoke. If a future refactor moves the call after `final = await
    graph.ainvoke(initial_state, ...)` the gap reopens."""
    src = inspect.getsource(task_mod._run)
    pre_idx = src.find("_cancel_pending_approval_if_any(graph, config")
    main_idx = src.find("await graph.ainvoke(initial_state")
    assert pre_idx != -1, (
        "REGRESSION: _run no longer calls _cancel_pending_approval_if_any. "
        "A new user message arriving while the thread is paused at "
        "approval_wait will silently re-fire the interrupt and the "
        "user's follow-up is lost (Tehreem-thread bug)."
    )
    assert main_idx != -1, "_run must still call graph.ainvoke(initial_state, ...)"
    assert pre_idx < main_idx, (
        "REGRESSION: cancel-pending-approval must run BEFORE the main ainvoke. "
        "Inverting this order means the new user_request is delivered into a "
        "paused thread, which re-fires the interrupt without resolving it."
    )


def test_cancel_helper_uses_command_resume_with_user_followup_reason() -> None:
    """Source guard: the helper must use Command(resume=...) with both
    decision='rejected' and reason='user_followup'. Without `reason`,
    rejected_reply_node falls back to the canned post → spam regression."""
    src = inspect.getsource(task_mod._cancel_pending_approval_if_any)
    assert "Command(resume=" in src, (
        "REGRESSION: cancel helper must clear the interrupt with "
        "Command(resume=...). Plain ainvoke(state_dict, ...) re-fires the "
        "interrupt without resolving it."
    )
    assert '"decision": "rejected"' in src, (
        "REGRESSION: cancel helper must set decision='rejected' in the resume."
    )
    assert '"reason": "user_followup"' in src, (
        "REGRESSION: cancel helper must set reason='user_followup' so "
        "rejected_reply_node can suppress the canned reject post."
    )


def test_cancel_helper_atomic_db_update_on_thread_status() -> None:
    """Source guard: the helper must atomically flip the previous
    awaiting_approval Job for this thread to 'rejected'. Without this, a
    stale Approve-button click later would pass the resume_agent atomic
    claim (status still='awaiting_approval') and try to resume a plan
    we've already superseded."""
    src = inspect.getsource(task_mod._cancel_pending_approval_if_any)
    assert "update(Job)" in src
    assert "Job.thread_id == thread_id" in src
    assert 'Job.status == "awaiting_approval"' in src
    assert 'status="rejected"' in src


# ---------------------------------------------------------------------------
# Behavior — pre-flight check skips when the thread isn't interrupted
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_pending_interrupt_skips_cancel_resume(monkeypatch) -> None:
    """Healthy thread: aget_state shows no pending interrupt → the helper is
    a no-op, the main ainvoke runs unchanged. Guards against false positives
    that would degrade every fresh message with extra Command(resume=...)
    calls and DB writes."""
    job_row = Job(
        tenant_id="tenant-1", thread_id="t", user_id="u", user_request="x", status="running"
    )
    job_row.id = "job-uuid-new"

    sessions = [_FakeSession(tenant=_tenant()), _FakeSession(job=job_row)]
    _patch_session_chain(monkeypatch, sessions)

    monkeypatch.setattr(task_mod, "checkpointer", lambda: _fake_saver_factory()())

    fake_graph = MagicMock()
    fake_graph.aget_state = AsyncMock(return_value=_snapshot(next_nodes=()))
    fake_graph.ainvoke = AsyncMock(return_value={"final_summary": "ok", "total_cost_usd": 0.001})
    monkeypatch.setattr(task_mod, "build_agent_graph", lambda saver: fake_graph)

    out = await _run(_make_ctx(), _make_msg())
    assert out["status"] == "done"
    # Exactly ONE ainvoke (the main one). No cancel-resume.
    assert fake_graph.ainvoke.await_count == 1
    # And the single call carried a state dict, not a Command.
    arg = fake_graph.ainvoke.await_args.args[0]
    assert isinstance(arg, dict), "main ainvoke should be called with the initial_state dict"


# ---------------------------------------------------------------------------
# Behavior — pre-flight check fires the cancel-resume when paused
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pending_approval_triggers_cancel_resume_then_invokes(monkeypatch) -> None:
    """Tehreem-thread case: aget_state reports approval_wait pending. _run
    must (a) flip the prev awaiting_approval Job to rejected, (b) issue a
    Command(resume=...) ainvoke to clear the interrupt, (c) THEN run the
    main ainvoke for the new request."""
    from langgraph.types import Command

    new_job_row = Job(
        tenant_id="tenant-1",
        thread_id="t",
        user_id="u",
        user_request="follow-up",
        status="running",
    )
    new_job_row.id = "job-uuid-new"

    cancel_capture: dict = {}
    sessions = [
        # tenant lookup + new Job INSERT
        _FakeSession(tenant=_tenant()),
        # cancel-helper UPDATE on the previous awaiting_approval row
        _FakeSession(job=new_job_row, captured=cancel_capture),
        # _mark_job lookup at the end
        _FakeSession(job=new_job_row),
    ]
    _patch_session_chain(monkeypatch, sessions)

    monkeypatch.setattr(task_mod, "checkpointer", lambda: _fake_saver_factory()())

    fake_graph = MagicMock()
    fake_graph.aget_state = AsyncMock(return_value=_snapshot(next_nodes=("approval_wait",)))
    fake_graph.ainvoke = AsyncMock(
        return_value={"final_summary": "answered followup", "total_cost_usd": 0.002}
    )
    monkeypatch.setattr(task_mod, "build_agent_graph", lambda saver: fake_graph)

    out = await _run(_make_ctx(), _make_msg("preview link please"))
    assert out["status"] == "done"

    # Two ainvoke calls total: one cancel-resume (Command), one main (dict).
    assert fake_graph.ainvoke.await_count == 2, (
        "expected exactly 2 ainvoke calls (cancel-resume + main), got "
        f"{fake_graph.ainvoke.await_count}"
    )
    first_arg = fake_graph.ainvoke.await_args_list[0].args[0]
    second_arg = fake_graph.ainvoke.await_args_list[1].args[0]
    assert isinstance(first_arg, Command), (
        "REGRESSION: first ainvoke on a paused thread MUST be Command(resume=...). "
        f"Got: {type(first_arg).__name__}. Plain dict here re-fires the interrupt "
        "without resolving it (the original Tehreem-thread bug)."
    )
    # `Command(resume=...)`: extract the resume value and check the contract.
    resume_payload = getattr(first_arg, "resume", None)
    assert isinstance(resume_payload, dict), (
        f"resume payload must be a dict, got {resume_payload!r}"
    )
    assert resume_payload.get("decision") == "rejected"
    assert resume_payload.get("reason") == "user_followup", (
        "REGRESSION: reason must be 'user_followup' so rejected_reply_node "
        "suppresses the canned 'Got it - rejected' Slack post."
    )
    # Main invocation carries the new request.
    assert isinstance(second_arg, dict)
    assert second_arg.get("user_request") == "preview link please"

    # The DB UPDATE for retiring the prev awaiting_approval job must have run.
    assert cancel_capture.get("updates"), (
        "REGRESSION: cancel helper did not issue an UPDATE statement. A stale "
        "Approve-button click later could try to resume a plan we've already "
        "superseded with this new user message."
    )
    assert cancel_capture.get("committed") is True


# ---------------------------------------------------------------------------
# Behavior — approval_wait_node propagates the reason into state
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_approval_wait_node_propagates_user_followup_reason(monkeypatch) -> None:
    """approval_wait_node must capture the `reason` from the resume payload
    so rejected_reply_node can vary its behavior. Without propagation, the
    auto-cancel path is indistinguishable from an explicit reject."""
    state = {"job_id": "j"}
    monkeypatch.setattr(
        approval_mod,
        "interrupt",
        lambda payload: {"decision": "rejected", "reason": "user_followup"},
    )
    out = await approval_wait_node(state)  # type: ignore[arg-type]
    assert out["approval_decision"] == "rejected"
    assert out["approval_rejection_reason"] == "user_followup", (
        "REGRESSION: approval_wait_node dropped the resume `reason`. "
        "rejected_reply_node now can't tell auto-cancel from explicit reject "
        "and will spam the canned 'Got it - rejected' Slack post."
    )


@pytest.mark.asyncio
async def test_approval_wait_node_explicit_reject_has_no_reason(monkeypatch) -> None:
    """The button-click path passes `Command(resume={"decision": decision})`
    with no `reason`. approval_wait_node must leave reason=None so
    rejected_reply_node falls through to the canned post (the existing
    behavior, preserved)."""
    state = {"job_id": "j"}
    monkeypatch.setattr(approval_mod, "interrupt", lambda payload: {"decision": "rejected"})
    out = await approval_wait_node(state)  # type: ignore[arg-type]
    assert out["approval_decision"] == "rejected"
    assert out.get("approval_rejection_reason") is None


# ---------------------------------------------------------------------------
# Behavior — rejected_reply_node suppresses the canned post on user_followup
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rejected_reply_skips_post_on_user_followup(monkeypatch) -> None:
    """rejected_reply_node must NOT call post_reply when the rejection was
    auto-triggered by a user follow-up. Without this guard, every follow-up
    message produces a duplicate 'Got it - rejected' Slack message that
    shadows the actual response to the new request."""
    posted: list = []

    async def fake_post(tenant, reply):
        posted.append((tenant, reply))

    monkeypatch.setattr(approval_mod, "post_reply", fake_post)

    out = await rejected_reply_node(  # type: ignore[arg-type]
        {
            "thread_id": "thr",
            "channel_id": "ch",
            "tenant_id": "ten",
            "approval_rejection_reason": "user_followup",
            "messages": [{"role": "user", "content": "preview please"}],
        }
    )

    assert posted == [], (
        "REGRESSION: rejected_reply_node posted to Slack on a user_followup "
        "auto-cancel. The new run will produce the real reply; this canned "
        "post just spams the thread."
    )
    assert out["final_summary"] == "auto_cancelled_for_followup"

    # Implicit cancellation must still be recorded in messages so the next
    # agent_node turn knows the previous plan is no longer pending.
    msgs = out["messages"]
    assert msgs, "messages must not be empty"
    assert msgs[-1]["role"] == "assistant"
    assert "auto-cancelled" in msgs[-1]["content"].lower()


@pytest.mark.asyncio
async def test_rejected_reply_explicit_reject_still_posts(monkeypatch) -> None:
    """Companion: an explicit Reject-button click (no `reason`) must still
    fire the canned post. We're suppressing only the auto-cancel case."""
    posted: list = []

    async def fake_post(tenant, reply):
        posted.append((tenant, reply))

    monkeypatch.setattr(approval_mod, "post_reply", fake_post)

    out = await rejected_reply_node(  # type: ignore[arg-type]
        {
            "thread_id": "thr",
            "channel_id": "ch",
            "tenant_id": "ten",
            "messages": [],
        }
    )
    assert len(posted) == 1, "explicit reject must still post the canned message"
    assert "rejected" in posted[0][1].text.lower()
    assert out["final_summary"] == "rejected_by_user"
