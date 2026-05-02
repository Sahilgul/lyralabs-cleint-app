"""apps.worker.arq_app — WorkerSettings smoke tests."""

from __future__ import annotations

from apps.worker.arq_app import WorkerSettings
from apps.worker.tasks.crystallize_skills import crystallize_skills
from apps.worker.tasks.run_agent import resume_agent, run_agent


def test_functions_registered() -> None:
    assert run_agent in WorkerSettings.functions
    assert resume_agent in WorkerSettings.functions
    assert crystallize_skills in WorkerSettings.functions


def test_cron_jobs_crystallize_at_3am() -> None:
    assert len(WorkerSettings.cron_jobs) == 1
    job = WorkerSettings.cron_jobs[0]
    assert job.coroutine is crystallize_skills
    assert job.hour == 3
    assert job.minute == 0


def test_job_timeout_is_five_minutes() -> None:
    assert WorkerSettings.job_timeout == 300


def test_max_jobs_conservative() -> None:
    assert WorkerSettings.max_jobs == 4


def test_max_tries_allows_lock_contention_budget() -> None:
    # 20 lock-contention retries + 5 real-failure retries = 25
    assert WorkerSettings.max_tries == 25


def test_redis_settings_present() -> None:
    from arq.connections import RedisSettings

    assert isinstance(WorkerSettings.redis_settings, RedisSettings)
