"""Regression test 6 — TOCTOU race in resume_agent dedup.

Bug: A user could click Approve twice (intentional fast double-click, or a
single click on a duplicate plan card from regression #5) and trigger TWO
`resume_agent` arq jobs against the same Job row. Each job ran:

    job = SELECT * FROM jobs WHERE id = :job_id     # both see status='awaiting_approval'
    if job.status != 'awaiting_approval': bail       # both skip the bail
    ...                                               # both run the graph

The Redis thread-lock serialized graph execution, but the SECOND job still
ran the graph after the first completed — against a checkpoint that was
already past END. Best case: a wasted Slack post. Worst case: re-execution
of side effects.

Fix: replace the read-then-check with an atomic conditional UPDATE:

    UPDATE jobs SET status='resuming'
     WHERE id = :job_id AND status = 'awaiting_approval'
    RETURNING id

Only one of the two concurrent rows-flips wins. The loser sees rowcount=0
and returns `already_processed` immediately, before acquiring the lock or
re-running the graph.

Regression guards:
  1. _resume's first DB statement is an UPDATE on jobs WHERE status = 'awaiting_approval'.
  2. When the UPDATE returns no rows (loser of the race), _resume bails with
     status='already_processed' and never acquires the Redis lock or runs
     the graph.
  3. When the UPDATE returns a row (winner), _resume proceeds normally.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.mark.asyncio
async def test_resume_bails_when_atomic_claim_finds_no_awaiting_row(monkeypatch):
    """The race-loser path: another resume already flipped status; this one bails."""

    # Build a session whose execute() returns a result with .first() == None
    # — simulating UPDATE … WHERE status='awaiting_approval' affecting 0 rows
    # because the other concurrent resume already flipped it.
    update_result = MagicMock()
    update_result.first = MagicMock(return_value=None)

    # Also simulate the follow-up SELECT for logging the existing status.
    existing_job = MagicMock()
    existing_job.status = "resuming"  # the other resume claimed it

    select_result = MagicMock()
    select_result.scalar_one_or_none = MagicMock(return_value=existing_job)

    session = AsyncMock()
    session.execute = AsyncMock(side_effect=[update_result, select_result])
    session.commit = AsyncMock()

    class _CM:
        async def __aenter__(self):
            return session

        async def __aexit__(self, *exc):
            return False

    import apps.worker.tasks.run_agent as ra

    monkeypatch.setattr(ra, "async_session", lambda: _CM())

    # Redis must NOT be touched by a bailed-out resume. Make any access
    # raise so we can assert the bail happened before the lock attempt.
    redis_mock = MagicMock()
    redis_mock.set = AsyncMock(
        side_effect=AssertionError(
            "REGRESSION: race-loser resume reached the Redis lock — atomic "
            "dedup must bail BEFORE acquiring the thread lock."
        )
    )
    redis_mock.delete = AsyncMock()

    result = await ra._resume(
        ctx={"redis": redis_mock, "job_try": 1},
        job_id="job-xyz",
        decision="approved",
        user_id="U1",
    )

    assert result == {"status": "already_processed", "job_id": "job-xyz"}, (
        f"loser of dedup race must return already_processed, got {result}"
    )
    redis_mock.set.assert_not_called()


@pytest.mark.asyncio
async def test_resume_bails_when_job_does_not_exist(monkeypatch):
    """If the job_id is invalid the UPDATE matches nothing AND the SELECT
    returns None. _resume must surface this as `no_job`, not crash."""
    update_result = MagicMock()
    update_result.first = MagicMock(return_value=None)
    select_result = MagicMock()
    select_result.scalar_one_or_none = MagicMock(return_value=None)

    session = AsyncMock()
    session.execute = AsyncMock(side_effect=[update_result, select_result])

    class _CM:
        async def __aenter__(self):
            return session

        async def __aexit__(self, *exc):
            return False

    import apps.worker.tasks.run_agent as ra

    monkeypatch.setattr(ra, "async_session", lambda: _CM())

    result = await ra._resume(
        ctx={"redis": MagicMock(), "job_try": 1},
        job_id="job-does-not-exist",
        decision="approved",
        user_id="U1",
    )
    assert result == {"status": "no_job"}


def test_resume_uses_conditional_update_on_awaiting_approval():
    """Source-level guard: _resume must use an atomic UPDATE keyed on
    status='awaiting_approval', not a SELECT-then-check pattern.

    Without this guard, a future engineer could "simplify" back to:
        job = SELECT ...
        if job.status != 'awaiting_approval': bail
    which re-introduces the TOCTOU window the user's bug exploited.
    """
    import inspect

    import apps.worker.tasks.run_agent as ra

    src = inspect.getsource(ra._resume)
    assert "update(Job)" in src, (
        "REGRESSION: _resume must use update(Job) to atomically claim the "
        "resume. Found no update(Job) call — has the SELECT-then-check "
        "pattern been re-introduced?"
    )
    assert "awaiting_approval" in src, (
        "_resume must condition the atomic update on status='awaiting_approval'."
    )
    # Sanity: still has the bail-out path.
    assert "already_processed" in src
