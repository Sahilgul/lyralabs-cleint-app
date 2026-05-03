"""Regression test 4 — resume_agent enqueued with wrong kwargs format.

Bug: enqueue_resume_agent passed job_id/decision/user_id as:
    pool.enqueue_job("resume_agent", _kwargs={"job_id": ..., ...})

arq does not unpack _kwargs into the function call — it passes it as a
literal keyword argument named "_kwargs". The worker then crashed with:
    TypeError: resume_agent() got an unexpected keyword argument '_kwargs'

This meant clicking Approve or Reject on the approval card did nothing —
the button click was received by Cloud Run but the resume job always failed,
leaving the approval card stuck forever with both buttons still showing.

Fix: pass kwargs directly to enqueue_job:
    pool.enqueue_job("resume_agent", job_id=..., decision=..., user_id=...)
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_enqueue_resume_agent_passes_kwargs_directly(monkeypatch):
    """
    REGRESSION: enqueue_resume_agent must call pool.enqueue_job with
    job_id, decision, user_id as direct keyword arguments — NOT wrapped
    in a _kwargs dict.

    Using _kwargs={"job_id": ...} causes arq to pass a literal '_kwargs'
    keyword to the function, which raises TypeError at runtime.
    """
    mock_pool = AsyncMock()
    mock_pool.enqueue_job = AsyncMock()

    async def _fake_get_pool():
        return mock_pool

    monkeypatch.setattr("lyra_core.worker.queue._get_pool", _fake_get_pool)

    from lyra_core.worker.queue import enqueue_resume_agent

    await enqueue_resume_agent(
        job_id="job-abc-123",
        decision="approved",
        user_id="U0A94NWG16X",
    )

    mock_pool.enqueue_job.assert_called_once()
    call_args = mock_pool.enqueue_job.call_args

    # Must be called as: enqueue_job("resume_agent", job_id=..., decision=..., user_id=...)
    assert call_args.args[0] == "resume_agent", "First arg must be the function name"

    kwargs = call_args.kwargs
    assert "_kwargs" not in kwargs, (
        "enqueue_job must NOT receive '_kwargs' — arq passes it literally "
        "to the function, causing TypeError: got unexpected keyword argument '_kwargs'"
    )
    assert kwargs.get("job_id") == "job-abc-123"
    assert kwargs.get("decision") == "approved"
    assert kwargs.get("user_id") == "U0A94NWG16X"


@pytest.mark.asyncio
async def test_enqueue_resume_agent_approve_and_reject_both_work(monkeypatch):
    """Both 'approved' and 'rejected' decisions must be enqueueable without error."""
    mock_pool = AsyncMock()
    mock_pool.enqueue_job = AsyncMock()

    monkeypatch.setattr("lyra_core.worker.queue._get_pool", AsyncMock(return_value=mock_pool))

    from lyra_core.worker.queue import enqueue_resume_agent

    for decision in ("approved", "rejected"):
        mock_pool.enqueue_job.reset_mock()
        await enqueue_resume_agent(job_id="job-1", decision=decision, user_id="U1")

        kwargs = mock_pool.enqueue_job.call_args.kwargs
        assert kwargs["decision"] == decision
        assert "_kwargs" not in kwargs
