"""lyra_core.agent.nodes.critic."""

from __future__ import annotations

import base64
import json
from unittest.mock import AsyncMock

import pytest

from lyra_core.agent.nodes import critic as critic_mod
from lyra_core.agent.nodes.critic import critic_node, route_after_critic


def _state(**overrides):
    base = {
        "user_request": "do x",
        "plan": {"goal": "g", "steps": []},
        "step_results": [{"step_id": "s1", "tool_name": "x", "ok": True}],
        "thread_id": "thr",
        "channel_id": "ch",
        "tenant_id": "ten",
        "total_cost_usd": 0.0,
        "artifacts": [],
    }
    base.update(overrides)
    return base


@pytest.mark.asyncio
async def test_critic_returns_summary(monkeypatch, mock_litellm_response) -> None:
    posted = []

    async def fake_post(tenant, reply):
        posted.append((tenant, reply))

    monkeypatch.setattr(critic_mod, "post_reply", fake_post)
    monkeypatch.setattr(
        critic_mod, "chat",
        AsyncMock(
            return_value=mock_litellm_response(
                json.dumps({"verdict": "ok", "summary_for_user": "All done!"}), cost=0.001
            )
        ),
    )

    out = await critic_node(_state())  # type: ignore[arg-type]
    assert out["final_summary"] == "All done!"
    assert out["_critic_verdict"] == "ok"
    assert out["total_cost_usd"] == pytest.approx(0.001)
    assert len(posted) == 1
    assert posted[0][1].text == "All done!"


@pytest.mark.asyncio
async def test_critic_invalid_json_falls_back(monkeypatch, mock_litellm_response) -> None:
    monkeypatch.setattr(critic_mod, "post_reply", AsyncMock())
    monkeypatch.setattr(
        critic_mod, "chat",
        AsyncMock(return_value=mock_litellm_response("garbage")),
    )

    out = await critic_node(_state())  # type: ignore[arg-type]
    assert out["_critic_verdict"] == "give_up"
    assert "wrong" in out["final_summary"].lower()


@pytest.mark.asyncio
async def test_critic_attaches_artifacts_to_reply(monkeypatch, mock_litellm_response) -> None:
    posted = []

    async def fake_post(tenant, reply):
        posted.append(reply)

    monkeypatch.setattr(critic_mod, "post_reply", fake_post)
    monkeypatch.setattr(
        critic_mod, "chat",
        AsyncMock(
            return_value=mock_litellm_response(
                json.dumps({"verdict": "ok", "summary_for_user": "done"})
            )
        ),
    )

    artifact = {
        "kind": "pdf",
        "filename": "report.pdf",
        "content_b64": base64.b64encode(b"%PDF-1").decode(),
        "description": "Q3 report",
    }
    await critic_node(_state(artifacts=[artifact]))  # type: ignore[arg-type]

    assert len(posted) == 1
    assert len(posted[0].artifacts) == 1
    assert posted[0].artifacts[0].filename == "report.pdf"
    assert posted[0].artifacts[0].content == b"%PDF-1"


@pytest.mark.asyncio
async def test_critic_handles_missing_plan(monkeypatch, mock_litellm_response) -> None:
    monkeypatch.setattr(critic_mod, "post_reply", AsyncMock())
    monkeypatch.setattr(
        critic_mod, "chat",
        AsyncMock(
            return_value=mock_litellm_response(
                json.dumps({"verdict": "ok", "summary_for_user": "ok"})
            )
        ),
    )

    state = _state()
    state["plan"] = None
    out = await critic_node(state)  # type: ignore[arg-type]
    assert out["_critic_verdict"] == "ok"


def test_route_after_critic_always_ends() -> None:
    assert route_after_critic({}) == "end"  # type: ignore[arg-type]
    assert route_after_critic({"_critic_verdict": "retry"}) == "end"  # type: ignore[arg-type]
