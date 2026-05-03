"""lyra_core.common.audit."""

from __future__ import annotations

import hashlib
import json

import pytest
from lyra_core.common.audit import _hash_args, record_event


def test_hash_args_is_stable_for_same_dict() -> None:
    a = {"a": 1, "b": "x"}
    b = {"b": "x", "a": 1}
    assert _hash_args(a) == _hash_args(b)  # key order doesn't matter (sort_keys=True)


def test_hash_args_changes_when_value_changes() -> None:
    a = _hash_args({"k": "v1"})
    b = _hash_args({"k": "v2"})
    assert a != b


def test_hash_args_is_sha256_hex_64() -> None:
    h = _hash_args({"a": 1})
    assert len(h) == 64
    int(h, 16)  # confirms hex


def test_hash_args_handles_non_serializable_via_default_str() -> None:
    from datetime import datetime

    h = _hash_args({"when": datetime(2026, 5, 1)})
    assert (
        h
        == hashlib.sha256(
            json.dumps({"when": str(datetime(2026, 5, 1))}, sort_keys=True).encode()
        ).hexdigest()
    )


@pytest.mark.asyncio
async def test_record_event_persists_with_hashed_args(mock_session) -> None:
    event = await record_event(
        mock_session,
        tenant_id="t-1",
        actor_user_id="u-1",
        job_id="j-1",
        event_type="tool_call",
        tool_name="google.drive.search",
        args={"query": "leads"},
    )
    assert event.tenant_id == "t-1"
    assert event.actor_user_id == "u-1"
    assert event.event_type == "tool_call"
    assert event.tool_name == "google.drive.search"
    assert event.args_hash is not None and len(event.args_hash) == 64
    assert event.raw_args is None  # store_raw=False (default)
    mock_session.add.assert_called_once_with(event)
    mock_session.flush.assert_awaited_once()


@pytest.mark.asyncio
async def test_record_event_stores_raw_when_opted_in(mock_session) -> None:
    event = await record_event(
        mock_session,
        tenant_id="t-1",
        actor_user_id=None,
        job_id=None,
        event_type="tool_call",
        tool_name="x",
        args={"v": 1},
        store_raw_args=True,
    )
    assert event.raw_args == {"v": 1}


@pytest.mark.asyncio
async def test_record_event_no_args_means_no_hash(mock_session) -> None:
    event = await record_event(
        mock_session,
        tenant_id="t-1",
        actor_user_id=None,
        job_id=None,
        event_type="approval",
    )
    assert event.args_hash is None
    assert event.raw_args is None


@pytest.mark.asyncio
async def test_record_event_records_cost_and_model(mock_session) -> None:
    event = await record_event(
        mock_session,
        tenant_id="t-1",
        actor_user_id=None,
        job_id="j",
        event_type="llm_call",
        cost_usd=0.0123,
        model_used="anthropic/claude-sonnet-4-5",
    )
    assert event.cost_usd == pytest.approx(0.0123)
    assert event.model_used == "anthropic/claude-sonnet-4-5"


@pytest.mark.asyncio
async def test_record_event_extra_defaults_to_empty_dict(mock_session) -> None:
    event = await record_event(
        mock_session, tenant_id="t-1", actor_user_id=None, job_id=None, event_type="x"
    )
    assert event.extra == {}


@pytest.mark.asyncio
async def test_record_event_default_status_ok(mock_session) -> None:
    event = await record_event(
        mock_session, tenant_id="t-1", actor_user_id=None, job_id=None, event_type="x"
    )
    assert event.result_status == "ok"
