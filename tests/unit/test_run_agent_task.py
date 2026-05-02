"""apps.worker.tasks.run_agent — covers _run, _resume, _mark_job."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from lyra_core.channels.schema import InboundMessage, Surface
from lyra_core.db.models import Job, Tenant

from apps.worker.tasks import run_agent as task_mod
from apps.worker.tasks.run_agent import (
    _mark_job,
    _resume,
    _run,
)


def _make_ctx() -> dict:
    """Minimal arq context dict with a mock Redis client (lock always acquired)."""
    redis = AsyncMock()
    redis.set = AsyncMock(return_value=True)   # lock acquired
    redis.delete = AsyncMock(return_value=1)
    return {"redis": redis, "job_try": 1}


def _make_msg(text: str = "do x", *, is_dm: bool = False) -> str:
    return InboundMessage(
        surface=Surface.SLACK,
        tenant_external_id="T-XYZ",
        channel_id="C1",
        thread_id="thr-1",
        agent_thread_id=("slack:dm:T-XYZ:C1:U1" if is_dm else "slack:ch:T-XYZ:C1:thr-1"),
        user_id="U1",
        text=text,
        is_dm=is_dm,
    ).model_dump_json()


def _tenant(status: str = "active") -> Tenant:
    t = Tenant(external_team_id="T-XYZ", channel="slack", name="Acme")
    t.id = "tenant-1"
    t.status = status
    return t


def _patch_session_chain(monkeypatch, sessions: list) -> None:
    """Provide a sequence of fake sessions, one per `async with async_session()` use."""
    it = iter(sessions)

    def factory():
        return next(it)

    monkeypatch.setattr(task_mod, "async_session", factory)


class _FakeSession:
    def __init__(self, tenant=None, job=None, captured: dict | None = None):
        self.tenant = tenant
        self.job = job
        self.captured = captured if captured is not None else {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False

    async def execute(self, _stmt):
        r = MagicMock()
        # Naive: prefer job over tenant if one is set on this session
        r.scalar_one_or_none.return_value = self.job if self.job is not None else self.tenant
        return r

    def add(self, obj):
        self.captured.setdefault("added", []).append(obj)
        if isinstance(obj, Job):
            obj.id = "job-uuid-1"
            obj.thread_id = obj.thread_id  # already set

    async def commit(self):
        self.captured["committed"] = True

    async def refresh(self, obj):
        if not getattr(obj, "id", None):
            obj.id = "job-uuid-1"

    async def flush(self):
        return None


# -----------------------------------------------------------------------------
# _run
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_no_tenant(monkeypatch) -> None:
    _patch_session_chain(monkeypatch, [_FakeSession(tenant=None)])
    out = await _run(_make_ctx(), _make_msg())
    assert out == {"status": "no_tenant"}


@pytest.mark.asyncio
async def test_run_inactive_tenant(monkeypatch) -> None:
    _patch_session_chain(monkeypatch, [_FakeSession(tenant=_tenant(status="cancelled"))])
    out = await _run(_make_ctx(), _make_msg())
    assert out == {"status": "tenant_inactive"}


@pytest.mark.asyncio
async def test_run_happy_path_invokes_graph(monkeypatch) -> None:
    """Tenant active -> Job created (same session) -> graph runs -> job marked done."""
    job_state = {}
    job_row = Job(
        tenant_id="tenant-1", thread_id="t", user_id="u", user_request="x", status="running"
    )
    job_row.id = "job-uuid-1"

    sessions = [
        # Single session now handles both the tenant SELECT and the Job INSERT.
        _FakeSession(tenant=_tenant(), captured=job_state),
        _FakeSession(job=job_row),  # _mark_job lookup
    ]
    _patch_session_chain(monkeypatch, sessions)

    # Patch checkpointer + graph
    class FakeSaver:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

    monkeypatch.setattr(task_mod, "checkpointer", lambda: FakeSaver())

    fake_graph = MagicMock()
    fake_graph.ainvoke = AsyncMock(
        return_value={"final_summary": "All done", "total_cost_usd": 0.012}
    )
    monkeypatch.setattr(task_mod, "build_agent_graph", lambda saver: fake_graph)

    out = await _run(_make_ctx(), _make_msg("hello"))
    assert out["status"] == "done"
    assert out["job_id"] == "job-uuid-1"
    assert job_row.status == "done"
    assert job_row.result_summary == "All done"
    assert job_row.cost_usd == pytest.approx(0.012)
    assert job_state.get("committed") is True
    initial_state = fake_graph.ainvoke.await_args.args[0]
    assert initial_state["assistant_status_thread_ts"] is None


@pytest.mark.asyncio
async def test_run_dm_preserves_assistant_status_thread(monkeypatch) -> None:
    job_row = Job(
        tenant_id="tenant-1", thread_id="t", user_id="u", user_request="x", status="running"
    )
    job_row.id = "job-uuid-1"
    sessions = [_FakeSession(tenant=_tenant()), _FakeSession(job=job_row)]
    _patch_session_chain(monkeypatch, sessions)

    class FakeSaver:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

    monkeypatch.setattr(task_mod, "checkpointer", lambda: FakeSaver())

    fake_graph = MagicMock()
    fake_graph.ainvoke = AsyncMock(return_value={"final_summary": "ok", "total_cost_usd": 0})
    monkeypatch.setattr(task_mod, "build_agent_graph", lambda saver: fake_graph)

    await _run(_make_ctx(), _make_msg("hello", is_dm=True))

    initial_state = fake_graph.ainvoke.await_args.args[0]
    assert initial_state["assistant_status_thread_ts"] == "thr-1"


@pytest.mark.asyncio
async def test_run_graph_crash_marks_job_failed(monkeypatch) -> None:
    job_row = Job(
        tenant_id="tenant-1", thread_id="t", user_id="u", user_request="x", status="running"
    )
    job_row.id = "job-uuid-1"

    sessions = [
        # Single session now handles both the tenant SELECT and the Job INSERT.
        _FakeSession(tenant=_tenant()),
        _FakeSession(job=job_row),
    ]
    _patch_session_chain(monkeypatch, sessions)

    class FakeSaver:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

    monkeypatch.setattr(task_mod, "checkpointer", lambda: FakeSaver())

    fake_graph = MagicMock()
    # ValueError is non-retryable — _should_retry returns False, so the handler
    # returns {"status": "failed"} immediately without raising Retry.
    fake_graph.ainvoke = AsyncMock(side_effect=ValueError("boom"))
    monkeypatch.setattr(task_mod, "build_agent_graph", lambda saver: fake_graph)

    out = await _run(_make_ctx(), _make_msg())
    assert out["status"] == "failed"
    assert out["error"] == "boom"
    assert job_row.status == "failed"
    assert job_row.error == "boom"


# -----------------------------------------------------------------------------
# _resume
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resume_no_job(monkeypatch) -> None:
    _patch_session_chain(monkeypatch, [_FakeSession(job=None)])
    out = await _resume(_make_ctx(), job_id="missing", decision="approved", user_id="u")
    assert out == {"status": "no_job"}


@pytest.mark.asyncio
async def test_resume_approved_marks_done(monkeypatch) -> None:
    job_row = Job(
        tenant_id="tenant-1",
        thread_id="t",
        user_id="u",
        user_request="x",
        status="awaiting_approval",
    )
    job_row.id = "job-1"

    sessions = [_FakeSession(job=job_row), _FakeSession(job=job_row)]
    _patch_session_chain(monkeypatch, sessions)

    class FakeSaver:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

    monkeypatch.setattr(task_mod, "checkpointer", lambda: FakeSaver())

    fake_graph = MagicMock()
    fake_graph.ainvoke = AsyncMock(
        return_value={"final_summary": "ok", "total_cost_usd": 0.005}
    )
    monkeypatch.setattr(task_mod, "build_agent_graph", lambda saver: fake_graph)

    out = await _resume(_make_ctx(), job_id="job-1", decision="approved", user_id="u")
    assert out == {"status": "done", "job_id": "job-1"}
    assert job_row.status == "done"
    assert job_row.cost_usd == pytest.approx(0.005)


@pytest.mark.asyncio
async def test_resume_rejected_marks_rejected(monkeypatch) -> None:
    job_row = Job(
        tenant_id="t", thread_id="t", user_id="u", user_request="x", status="awaiting_approval"
    )
    job_row.id = "job-1"

    sessions = [_FakeSession(job=job_row), _FakeSession(job=job_row)]
    _patch_session_chain(monkeypatch, sessions)

    class FakeSaver:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

    monkeypatch.setattr(task_mod, "checkpointer", lambda: FakeSaver())
    fake_graph = MagicMock()
    fake_graph.ainvoke = AsyncMock(
        return_value={"final_summary": "rejected_by_user", "total_cost_usd": 0}
    )
    monkeypatch.setattr(task_mod, "build_agent_graph", lambda saver: fake_graph)

    out = await _resume(_make_ctx(), job_id="job-1", decision="rejected", user_id="u")
    assert out == {"status": "rejected", "job_id": "job-1"}
    assert job_row.status == "rejected"


@pytest.mark.asyncio
async def test_resume_graph_crash_marks_failed(monkeypatch) -> None:
    job_row = Job(
        tenant_id="t", thread_id="t", user_id="u", user_request="x", status="awaiting_approval"
    )
    job_row.id = "job-1"

    sessions = [_FakeSession(job=job_row), _FakeSession(job=job_row)]
    _patch_session_chain(monkeypatch, sessions)

    class FakeSaver:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

    monkeypatch.setattr(task_mod, "checkpointer", lambda: FakeSaver())
    fake_graph = MagicMock()
    # ValueError is non-retryable — _should_retry returns False, so the handler
    # returns {"status": "failed"} immediately without raising Retry.
    fake_graph.ainvoke = AsyncMock(side_effect=ValueError("explode"))
    monkeypatch.setattr(task_mod, "build_agent_graph", lambda saver: fake_graph)

    out = await _resume(_make_ctx(), job_id="job-1", decision="approved", user_id="u")
    assert out["status"] == "failed"
    assert job_row.status == "failed"
    assert job_row.error == "explode"


# -----------------------------------------------------------------------------
# _mark_job
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mark_job_updates_fields(monkeypatch) -> None:
    job = Job(
        tenant_id="t", thread_id="t", user_id="u", user_request="x", status="running"
    )
    job.id = "j"
    _patch_session_chain(monkeypatch, [_FakeSession(job=job)])

    await _mark_job("j", status="done", result_summary="great", cost_usd=0.005)
    assert job.status == "done"
    assert job.result_summary == "great"
    assert job.cost_usd == 0.005


@pytest.mark.asyncio
async def test_mark_job_no_job_no_op(monkeypatch) -> None:
    _patch_session_chain(monkeypatch, [_FakeSession(job=None)])
    # Should not raise
    await _mark_job("missing", status="done")


def test_run_agent_is_async() -> None:
    """run_agent and resume_agent must be plain async functions (not sync wrappers)."""
    import inspect

    assert inspect.iscoroutinefunction(task_mod.run_agent)
    assert inspect.iscoroutinefunction(task_mod.resume_agent)
