"""lyra_core.agent.skill_crystallizer — pure helper functions."""

from __future__ import annotations

from lyra_core.agent.skill_crystallizer import (
    MINE_FREQUENCY_THRESHOLD,
    _arg_schema_shape,
    _sequence_hash,
)


class TestArgSchemaShape:
    def test_empty_args(self) -> None:
        assert _arg_schema_shape(None) == []
        assert _arg_schema_shape({}) == []

    def test_captures_structure_not_values(self) -> None:
        shape_a = _arg_schema_shape({"email": "alice@example.com", "name": "Alice"})
        shape_b = _arg_schema_shape({"email": "bob@example.com", "name": "Bob"})
        assert shape_a == shape_b  # same keys/types → same shape

    def test_sorted_by_key(self) -> None:
        shape = _arg_schema_shape({"z": 1, "a": "x"})
        keys = [k for k, _ in shape]
        assert keys == sorted(keys)

    def test_captures_type_names(self) -> None:
        shape = _arg_schema_shape({"count": 5, "name": "Alice", "active": True})
        type_map = dict(shape)
        assert type_map["count"] == "int"
        assert type_map["name"] == "str"
        assert type_map["active"] == "bool"


class TestSequenceHash:
    def test_same_sequence_same_hash(self) -> None:
        seq = [("tool_a", [("n", "int")]), ("tool_b", [])]
        assert _sequence_hash(seq) == _sequence_hash(seq)

    def test_different_sequence_different_hash(self) -> None:
        seq_a = [("tool_a", [("n", "int")])]
        seq_b = [("tool_b", [("n", "int")])]
        assert _sequence_hash(seq_a) != _sequence_hash(seq_b)

    def test_hash_is_16_chars(self) -> None:
        assert len(_sequence_hash([])) == 16


def test_mine_frequency_threshold_is_five() -> None:
    assert MINE_FREQUENCY_THRESHOLD == 5
