"""lyra_core.agent.living_artifact — load/upsert/format helpers."""

from __future__ import annotations

from lyra_core.agent.living_artifact import format_artifact_for_prompt


class TestFormatArtifactForPrompt:
    def test_empty_body_returns_placeholder(self) -> None:
        result = format_artifact_for_prompt({})
        assert "no prior context" in result

    def test_facts_formatted_as_bullet_list(self) -> None:
        result = format_artifact_for_prompt({"client_name": "Acme", "pipeline": "Sales"})
        assert "client_name: Acme" in result
        assert "pipeline: Sales" in result
        # Each fact on its own line prefixed with "  - "
        lines = result.split("\n")
        assert all(line.startswith("  - ") for line in lines)

    def test_single_fact(self) -> None:
        result = format_artifact_for_prompt({"key": "value"})
        assert "key: value" in result
