"""lyra_core.agent.nodes.agent._route_read_tools — tool router unit tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from lyra_core.agent.nodes import agent as agent_mod
from lyra_core.agent.nodes.agent import _route_read_tools


def _tool(name: str, description: str = "") -> MagicMock:
    t = MagicMock()
    t.name = name
    t.description = description or f"Does {name}"
    return t


def _router_resp(text: str) -> MagicMock:
    msg = MagicMock()
    msg.content = text
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _many_tools(n: int = 10) -> list[MagicMock]:
    return [_tool(f"provider.tool_{i}", f"Tool {i} description") for i in range(n)]


# ---------------------------------------------------------------------------
# Bypass: ≤ 5 tools — no LLM call at all
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_skips_routing_when_five_or_fewer_tools(monkeypatch) -> None:
    chat_mock = AsyncMock()
    monkeypatch.setattr(agent_mod, "chat", chat_mock)

    tools = _many_tools(5)
    result = await _route_read_tools("hi", [], tools)

    chat_mock.assert_not_awaited()
    assert result is tools


@pytest.mark.asyncio
async def test_skips_routing_when_zero_tools(monkeypatch) -> None:
    chat_mock = AsyncMock()
    monkeypatch.setattr(agent_mod, "chat", chat_mock)

    result = await _route_read_tools("hi", [], [])

    chat_mock.assert_not_awaited()
    assert result == []


# ---------------------------------------------------------------------------
# Happy path: router returns a subset by name
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_returns_only_tools_named_by_router(monkeypatch) -> None:
    tools = [
        _tool("slack.conversations.history"),
        _tool("slack.users.info"),
        _tool("google.drive.search"),
        _tool("google.sheets.read"),
        _tool("ghl.pipelines.opportunities"),
        _tool("ghl.contacts.search"),
    ]
    monkeypatch.setattr(
        agent_mod,
        "chat",
        AsyncMock(return_value=_router_resp("slack.conversations.history\nghl.contacts.search")),
    )

    result = await _route_read_tools("pull the stuck deals", [], tools)

    assert [t.name for t in result] == ["slack.conversations.history", "ghl.contacts.search"]


@pytest.mark.asyncio
async def test_router_output_with_bullet_prefix_is_parsed(monkeypatch) -> None:
    """Router may reply with '- toolname' bullets — strip the prefix."""
    tools = _many_tools(8)
    target = tools[2]
    monkeypatch.setattr(
        agent_mod,
        "chat",
        AsyncMock(return_value=_router_resp(f"- {target.name}\n- {tools[5].name}")),
    )

    result = await _route_read_tools("something", [], tools)

    assert target in result
    assert tools[5] in result
    assert len(result) == 2


@pytest.mark.asyncio
async def test_router_output_ignores_blank_lines(monkeypatch) -> None:
    tools = _many_tools(8)
    target = tools[1]
    monkeypatch.setattr(
        agent_mod,
        "chat",
        AsyncMock(return_value=_router_resp(f"\n{target.name}\n\n")),
    )

    result = await _route_read_tools("query", [], tools)

    assert result == [target]


# ---------------------------------------------------------------------------
# Fallback cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_falls_back_to_all_tools_on_exception(monkeypatch) -> None:
    tools = _many_tools(10)
    monkeypatch.setattr(agent_mod, "chat", AsyncMock(side_effect=RuntimeError("network")))

    result = await _route_read_tools("anything", [], tools)

    assert result is tools


@pytest.mark.asyncio
async def test_falls_back_to_all_tools_when_router_returns_empty_string(monkeypatch) -> None:
    tools = _many_tools(10)
    monkeypatch.setattr(agent_mod, "chat", AsyncMock(return_value=_router_resp("")))

    result = await _route_read_tools("anything", [], tools)

    assert result is tools


@pytest.mark.asyncio
async def test_falls_back_to_all_tools_when_no_names_match(monkeypatch) -> None:
    """Router returns tool names that don't exist in the registry — return all."""
    tools = _many_tools(10)
    monkeypatch.setattr(
        agent_mod,
        "chat",
        AsyncMock(return_value=_router_resp("nonexistent.tool\nanother.fake")),
    )

    result = await _route_read_tools("query", [], tools)

    assert result is tools


@pytest.mark.asyncio
async def test_falls_back_when_chat_returns_none_content(monkeypatch) -> None:
    tools = _many_tools(8)
    resp = _router_resp(None)
    resp.choices[0].message.content = None
    monkeypatch.setattr(agent_mod, "chat", AsyncMock(return_value=resp))

    result = await _route_read_tools("query", [], tools)

    assert result is tools


# ---------------------------------------------------------------------------
# Router call content — what gets sent to the cheap model
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_router_prompt_contains_user_request(monkeypatch) -> None:
    tools = _many_tools(8)
    captured: dict = {}

    async def fake_chat(**kwargs):
        captured["messages"] = kwargs["messages"]
        return _router_resp(tools[0].name)

    monkeypatch.setattr(agent_mod, "chat", fake_chat)

    await _route_read_tools("find the Hernandez contact", [], tools)

    prompt = captured["messages"][0]["content"]
    assert "find the Hernandez contact" in prompt


@pytest.mark.asyncio
async def test_router_prompt_contains_all_tool_names(monkeypatch) -> None:
    tools = _many_tools(8)
    captured: dict = {}

    async def fake_chat(**kwargs):
        captured["messages"] = kwargs["messages"]
        return _router_resp(tools[0].name)

    monkeypatch.setattr(agent_mod, "chat", fake_chat)

    await _route_read_tools("anything", [], tools)

    prompt = captured["messages"][0]["content"]
    for t in tools:
        assert t.name in prompt


@pytest.mark.asyncio
async def test_router_prompt_includes_recent_history(monkeypatch) -> None:
    tools = _many_tools(8)
    captured: dict = {}
    history = [
        {"role": "user", "content": "earlier question about Hernandez"},
        {"role": "assistant", "content": "I found the Hernandez contact"},
    ]

    async def fake_chat(**kwargs):
        captured["messages"] = kwargs["messages"]
        return _router_resp(tools[0].name)

    monkeypatch.setattr(agent_mod, "chat", fake_chat)

    await _route_read_tools("follow up", history, tools)

    prompt = captured["messages"][0]["content"]
    assert "Hernandez" in prompt


@pytest.mark.asyncio
async def test_router_uses_cheap_model_tier(monkeypatch) -> None:
    from lyra_core.common.llm import ModelTier

    tools = _many_tools(8)
    captured: dict = {}

    async def fake_chat(**kwargs):
        captured["kwargs"] = kwargs
        return _router_resp(tools[0].name)

    monkeypatch.setattr(agent_mod, "chat", fake_chat)

    await _route_read_tools("anything", [], tools)

    assert captured["kwargs"]["tier"] == ModelTier.CHEAP


@pytest.mark.asyncio
async def test_router_uses_low_max_tokens(monkeypatch) -> None:
    """Router should request a small token budget — it only needs tool names."""
    tools = _many_tools(8)
    captured: dict = {}

    async def fake_chat(**kwargs):
        captured["kwargs"] = kwargs
        return _router_resp(tools[0].name)

    monkeypatch.setattr(agent_mod, "chat", fake_chat)

    await _route_read_tools("anything", [], tools)

    assert captured["kwargs"]["max_tokens"] <= 200


# ---------------------------------------------------------------------------
# Preservation of order
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_returned_tools_preserve_registry_order(monkeypatch) -> None:
    """Result order should follow the original tool list, not the router's output order."""
    tools = [
        _tool("a.first"),
        _tool("b.second"),
        _tool("c.third"),
        _tool("d.fourth"),
        _tool("e.fifth"),
        _tool("f.sixth"),
    ]
    # Router returns them in reverse order
    monkeypatch.setattr(
        agent_mod,
        "chat",
        AsyncMock(return_value=_router_resp("c.third\na.first")),
    )

    result = await _route_read_tools("query", [], tools)

    assert [t.name for t in result] == ["a.first", "c.third"]
