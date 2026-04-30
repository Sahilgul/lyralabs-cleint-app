"""Process-wide registry of all Tool instances."""

from __future__ import annotations

from .base import Tool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> Tool:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool name: {tool.name}")
        self._tools[tool.name] = tool
        return tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"unknown tool: {name}")
        return self._tools[name]

    def all(self) -> list[Tool]:
        return list(self._tools.values())

    def by_provider(self, provider: str) -> list[Tool]:
        return [t for t in self._tools.values() if t.provider == provider]

    def schemas(self, names: list[str] | None = None) -> list[dict]:
        tools = self.all() if names is None else [self.get(n) for n in names]
        return [t.to_openai_schema() for t in tools]


default_registry = ToolRegistry()
