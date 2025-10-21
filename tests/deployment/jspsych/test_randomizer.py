"""Tests for randomizer code generator."""

from __future__ import annotations

from uuid import uuid4

import pytest

from sash.deployment.jspsych.randomizer import (
    _serialize_constraints,
    _serialize_metadata,
    _validate_constraints,
    _validate_property_path,
    generate_randomizer_function,
)
from sash.lists.constraints import OrderingConstraint


class TestGenerateRandomizerFunction:
    """Tests for generate_randomizer_function."""

    def test_generate_basic(self) -> None:
        """Test generating basic randomizer without constraints."""
        item_ids = [uuid4() for _ in range(5)]
        constraints: list[OrderingConstraint] = []
        metadata = {item_id: {"condition": "A"} for item_id in item_ids}

        js_code = generate_randomizer_function(item_ids, constraints, metadata)

        assert "function randomizeTrials" in js_code
        assert "const trialMetadata" in js_code

    def test_generate_with_precedence(self) -> None:
        """Test generating with precedence constraints."""
        item_ids = [uuid4() for _ in range(5)]
        constraint = OrderingConstraint(precedence_pairs=[(item_ids[0], item_ids[1])])
        metadata = {item_id: {"condition": "A"} for item_id in item_ids}

        js_code = generate_randomizer_function(item_ids, [constraint], metadata)

        assert "checkPrecedence" in js_code
        assert "precedence_pairs" in js_code

    def test_generate_with_no_adjacent(self) -> None:
        """Test generating with no-adjacent constraint."""
        item_ids = [uuid4() for _ in range(5)]
        constraint = OrderingConstraint(no_adjacent_property="condition")
        metadata = {
            item_ids[0]: {"condition": "A"},
            item_ids[1]: {"condition": "B"},
            item_ids[2]: {"condition": "A"},
            item_ids[3]: {"condition": "B"},
            item_ids[4]: {"condition": "A"},
        }

        js_code = generate_randomizer_function(item_ids, [constraint], metadata)

        assert "checkNoAdjacent" in js_code
        assert "no_adjacent_property" in js_code

    def test_generate_with_min_distance(self) -> None:
        """Test generating with min_distance constraint."""
        item_ids = [uuid4() for _ in range(5)]
        constraint = OrderingConstraint(
            no_adjacent_property="condition", min_distance=2
        )
        metadata = {
            item_ids[0]: {"condition": "A"},
            item_ids[1]: {"condition": "B"},
            item_ids[2]: {"condition": "A"},
            item_ids[3]: {"condition": "B"},
            item_ids[4]: {"condition": "A"},
        }

        js_code = generate_randomizer_function(item_ids, [constraint], metadata)

        assert "checkMinDistance" in js_code
        assert "min_distance" in js_code or "minDist" in js_code

    def test_generate_with_blocking(self) -> None:
        """Test generating with blocking constraint."""
        item_ids = [uuid4() for _ in range(5)]
        constraint = OrderingConstraint(
            block_by_property="block_type", randomize_within_blocks=True
        )
        metadata = {
            item_ids[0]: {"block_type": "A"},
            item_ids[1]: {"block_type": "B"},
            item_ids[2]: {"block_type": "A"},
            item_ids[3]: {"block_type": "B"},
            item_ids[4]: {"block_type": "A"},
        }

        js_code = generate_randomizer_function(item_ids, [constraint], metadata)

        assert "block_property" in js_code or "block_by_property" in js_code
        assert "blocks" in js_code

    def test_generate_with_practice_items(self) -> None:
        """Test generating with practice items constraint."""
        item_ids = [uuid4() for _ in range(5)]
        constraint = OrderingConstraint(practice_item_property="is_practice")
        metadata = {
            item_ids[0]: {"is_practice": True},
            item_ids[1]: {"is_practice": False},
            item_ids[2]: {"is_practice": False},
            item_ids[3]: {"is_practice": False},
            item_ids[4]: {"is_practice": True},
        }

        js_code = generate_randomizer_function(item_ids, [constraint], metadata)

        assert "practiceTrials" in js_code
        assert "practice_property" in js_code or "is_practice" in js_code

    def test_validation_error_on_missing_property(self) -> None:
        """Test validation error when property not in metadata."""
        item_ids = [uuid4() for _ in range(2)]
        constraint = OrderingConstraint(no_adjacent_property="missing_property")
        metadata = {item_id: {"condition": "A"} for item_id in item_ids}

        with pytest.raises(ValueError) as exc_info:
            generate_randomizer_function(item_ids, [constraint], metadata)

        assert "missing_property" in str(exc_info.value)
        assert "not found in metadata" in str(exc_info.value)

    def test_empty_metadata_skips_validation(self) -> None:
        """Test empty metadata doesn't cause validation error."""
        item_ids = [uuid4() for _ in range(2)]
        constraint = OrderingConstraint(no_adjacent_property="any_property")
        metadata = {}

        # Should not raise error with empty metadata
        js_code = generate_randomizer_function(item_ids, [constraint], metadata)
        assert "function randomizeTrials" in js_code


class TestValidateConstraints:
    """Tests for _validate_constraints."""

    def test_validate_no_adjacent_property(self) -> None:
        """Test validation of no_adjacent_property."""
        constraint = OrderingConstraint(no_adjacent_property="condition")
        metadata = {uuid4(): {"condition": "A"}}

        # Should not raise
        _validate_constraints([constraint], metadata)

    def test_validate_block_by_property(self) -> None:
        """Test validation of block_by_property."""
        constraint = OrderingConstraint(block_by_property="block_type")
        metadata = {uuid4(): {"block_type": "A"}}

        # Should not raise
        _validate_constraints([constraint], metadata)

    def test_validate_practice_item_property(self) -> None:
        """Test validation of practice_item_property."""
        constraint = OrderingConstraint(practice_item_property="is_practice")
        metadata = {uuid4(): {"is_practice": True}}

        # Should not raise
        _validate_constraints([constraint], metadata)

    def test_validate_nested_property(self) -> None:
        """Test validation of nested property path."""
        constraint = OrderingConstraint(no_adjacent_property="item_metadata.condition")
        metadata = {uuid4(): {"item_metadata": {"condition": "A"}}}

        # Should not raise
        _validate_constraints([constraint], metadata)

    def test_validate_missing_property_raises(self) -> None:
        """Test validation raises on missing property."""
        constraint = OrderingConstraint(no_adjacent_property="missing")
        metadata = {uuid4(): {"condition": "A"}}

        with pytest.raises(ValueError) as exc_info:
            _validate_constraints([constraint], metadata)

        assert "missing" in str(exc_info.value)

    def test_validate_empty_metadata_skips(self) -> None:
        """Test validation skips with empty metadata."""
        constraint = OrderingConstraint(no_adjacent_property="any_property")
        metadata = {}

        # Should not raise
        _validate_constraints([constraint], metadata)


class TestValidatePropertyPath:
    """Tests for _validate_property_path."""

    def test_validate_simple_property(self) -> None:
        """Test validation of simple property."""
        metadata = {"condition": "A"}
        # Should not raise
        _validate_property_path("condition", metadata, "test_field")

    def test_validate_nested_property(self) -> None:
        """Test validation of nested property."""
        metadata = {"item_metadata": {"condition": "A"}}
        # Should not raise
        _validate_property_path("item_metadata.condition", metadata, "test_field")

    def test_validate_deeply_nested_property(self) -> None:
        """Test validation of deeply nested property."""
        metadata = {"a": {"b": {"c": "value"}}}
        # Should not raise
        _validate_property_path("a.b.c", metadata, "test_field")

    def test_validate_missing_property_raises(self) -> None:
        """Test validation raises on missing property."""
        metadata = {"condition": "A"}

        with pytest.raises(ValueError) as exc_info:
            _validate_property_path("missing", metadata, "test_field")

        assert "missing" in str(exc_info.value)
        assert "test_field" in str(exc_info.value)

    def test_validate_missing_nested_property_raises(self) -> None:
        """Test validation raises on missing nested property."""
        metadata = {"item_metadata": {"condition": "A"}}

        with pytest.raises(ValueError) as exc_info:
            _validate_property_path("item_metadata.missing", metadata, "test_field")

        assert "item_metadata.missing" in str(exc_info.value)


class TestSerializeConstraints:
    """Tests for _serialize_constraints."""

    def test_serialize_empty_list(self) -> None:
        """Test serializing empty constraint list."""
        constraints: list[OrderingConstraint] = []
        serialized = _serialize_constraints(constraints)

        assert serialized == []

    def test_serialize_precedence_pairs(self) -> None:
        """Test serializing precedence pairs."""
        item_a, item_b = uuid4(), uuid4()
        constraint = OrderingConstraint(precedence_pairs=[(item_a, item_b)])
        serialized = _serialize_constraints([constraint])

        assert len(serialized) == 1
        assert serialized[0]["constraint_type"] == "ordering"
        assert len(serialized[0]["precedence_pairs"]) == 1
        assert serialized[0]["precedence_pairs"][0] == (str(item_a), str(item_b))

    def test_serialize_properties(self) -> None:
        """Test serializing property fields."""
        constraint = OrderingConstraint(
            no_adjacent_property="condition",
            block_by_property="block_type",
            practice_item_property="is_practice",
        )
        serialized = _serialize_constraints([constraint])

        assert serialized[0]["no_adjacent_property"] == "condition"
        assert serialized[0]["block_by_property"] == "block_type"
        assert serialized[0]["practice_item_property"] == "is_practice"

    def test_serialize_distances(self) -> None:
        """Test serializing distance fields."""
        constraint = OrderingConstraint(
            no_adjacent_property="condition",
            block_by_property="block_type",
            min_distance=2,
            max_distance=5,
        )
        serialized = _serialize_constraints([constraint])

        assert serialized[0]["min_distance"] == 2
        assert serialized[0]["max_distance"] == 5


class TestSerializeMetadata:
    """Tests for _serialize_metadata."""

    def test_serialize_empty_metadata(self) -> None:
        """Test serializing empty metadata."""
        metadata = {}
        serialized = _serialize_metadata(metadata)

        assert serialized == {}

    def test_serialize_uuid_keys_to_strings(self) -> None:
        """Test UUID keys converted to strings."""
        item_id = uuid4()
        metadata = {item_id: {"condition": "A"}}
        serialized = _serialize_metadata(metadata)

        assert str(item_id) in serialized
        assert serialized[str(item_id)] == {"condition": "A"}

    def test_serialize_multiple_items(self) -> None:
        """Test serializing multiple items."""
        item_ids = [uuid4() for _ in range(3)]
        metadata = {
            item_ids[0]: {"condition": "A"},
            item_ids[1]: {"condition": "B"},
            item_ids[2]: {"condition": "C"},
        }
        serialized = _serialize_metadata(metadata)

        assert len(serialized) == 3
        assert serialized[str(item_ids[0])]["condition"] == "A"
        assert serialized[str(item_ids[1])]["condition"] == "B"
        assert serialized[str(item_ids[2])]["condition"] == "C"

    def test_serialize_nested_metadata(self) -> None:
        """Test serializing nested metadata structures."""
        item_id = uuid4()
        metadata = {item_id: {"item_metadata": {"condition": "A", "value": 42}}}
        serialized = _serialize_metadata(metadata)

        assert serialized[str(item_id)]["item_metadata"]["condition"] == "A"
        assert serialized[str(item_id)]["item_metadata"]["value"] == 42
