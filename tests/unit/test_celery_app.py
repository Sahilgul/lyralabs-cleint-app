"""apps.worker.celery_app."""

from __future__ import annotations

from apps.worker.celery_app import celery


def test_celery_app_name() -> None:
    """Celery app must have a non-empty main name (used in worker logs / inspection).
    Brand-agnostic so renames don't break this test."""
    assert celery.main and isinstance(celery.main, str)


def test_eager_mode_in_test_env() -> None:
    """APP_ENV=test must enable eager execution for sync tests."""
    assert celery.conf.task_always_eager is True
    assert celery.conf.task_eager_propagates is True


def test_serializer_is_json() -> None:
    assert celery.conf.task_serializer == "json"
    assert celery.conf.result_serializer == "json"
    assert celery.conf.accept_content == ["json"]


def test_timezone_is_utc() -> None:
    assert celery.conf.timezone == "UTC"
    assert celery.conf.enable_utc is True


def test_task_routes_use_agent_queue() -> None:
    routes = celery.conf.task_routes
    assert routes["apps.worker.tasks.run_agent.run_agent"] == {"queue": "agent"}
    assert routes["apps.worker.tasks.run_agent.resume_agent"] == {"queue": "agent"}


def test_default_queue_is_default() -> None:
    assert celery.conf.task_default_queue == "default"


def test_acks_late_and_reject_on_lost() -> None:
    assert celery.conf.task_acks_late is True
    assert celery.conf.task_reject_on_worker_lost is True


def test_worker_prefetch_one() -> None:
    assert celery.conf.worker_prefetch_multiplier == 1


def test_visibility_timeout_set() -> None:
    assert celery.conf.broker_transport_options.get("visibility_timeout") == 3600


def test_tasks_registered_after_import() -> None:
    import apps.worker.tasks.run_agent  # noqa: F401

    assert "apps.worker.tasks.run_agent.run_agent" in celery.tasks
    assert "apps.worker.tasks.run_agent.resume_agent" in celery.tasks
