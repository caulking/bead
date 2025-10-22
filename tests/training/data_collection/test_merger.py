"""Tests for data merger.

This module tests the DataMerger class including:
- Merging with all matches
- Merging with partial matches
- Merging with no matches
- PID extraction from different fields
- Custom merge keys
"""

from __future__ import annotations

import pytest

from sash.training.data_collection.merger import DataMerger


class TestDataMerger:
    """Test suite for DataMerger."""

    def test_initialization(self, data_merger: DataMerger) -> None:
        """Test merger initialization."""
        assert data_merger.merge_key == "PROLIFIC_PID"

    def test_initialization_custom_key(self) -> None:
        """Test merger initialization with custom key."""
        merger = DataMerger(merge_key="CUSTOM_PID")
        assert merger.merge_key == "CUSTOM_PID"

    def test_merge_all_matches(self, data_merger: DataMerger) -> None:
        """Test merging when all records match."""
        jatos_results = [
            {
                "result_id": 1,
                "data": {"PROLIFIC_PID": "abc123", "response": "A"},
                "metadata": {},
            },
            {
                "result_id": 2,
                "data": {"PROLIFIC_PID": "def456", "response": "B"},
                "metadata": {},
            },
        ]

        prolific_submissions = [
            {"participant_id": "abc123", "status": "APPROVED"},
            {"participant_id": "def456", "status": "APPROVED"},
        ]

        merged = data_merger.merge(jatos_results, prolific_submissions)

        # All should be merged
        assert len(merged) == 2
        assert all(record["merged"] for record in merged)

        # Verify data
        assert merged[0]["jatos_data"]["result_id"] == 1
        assert merged[0]["prolific_metadata"]["participant_id"] == "abc123"
        assert merged[1]["jatos_data"]["result_id"] == 2
        assert merged[1]["prolific_metadata"]["participant_id"] == "def456"

    def test_merge_partial_matches(self, data_merger: DataMerger) -> None:
        """Test merging when only some records match."""
        jatos_results = [
            {
                "result_id": 1,
                "data": {"PROLIFIC_PID": "abc123", "response": "A"},
                "metadata": {},
            },
            {
                "result_id": 2,
                "data": {"PROLIFIC_PID": "xyz999", "response": "B"},
                "metadata": {},
            },
        ]

        prolific_submissions = [
            {"participant_id": "abc123", "status": "APPROVED"},
        ]

        merged = data_merger.merge(jatos_results, prolific_submissions)

        # Should get both records
        assert len(merged) == 2

        # First should be merged
        assert merged[0]["merged"] is True
        assert merged[0]["prolific_metadata"]["participant_id"] == "abc123"

        # Second should not be merged
        assert merged[1]["merged"] is False
        assert merged[1]["prolific_metadata"] is None

    def test_merge_no_matches(self, data_merger: DataMerger) -> None:
        """Test merging when no records match."""
        jatos_results = [
            {
                "result_id": 1,
                "data": {"PROLIFIC_PID": "xyz999", "response": "A"},
                "metadata": {},
            },
        ]

        prolific_submissions = [
            {"participant_id": "abc123", "status": "APPROVED"},
        ]

        merged = data_merger.merge(jatos_results, prolific_submissions)

        # Should get record but not merged
        assert len(merged) == 1
        assert merged[0]["merged"] is False
        assert merged[0]["prolific_metadata"] is None
        assert merged[0]["jatos_data"]["result_id"] == 1

    def test_merge_empty_jatos(self, data_merger: DataMerger) -> None:
        """Test merging with empty JATOS results."""
        jatos_results = []
        prolific_submissions = [
            {"participant_id": "abc123", "status": "APPROVED"},
        ]

        merged = data_merger.merge(jatos_results, prolific_submissions)

        assert len(merged) == 0

    def test_merge_empty_prolific(self, data_merger: DataMerger) -> None:
        """Test merging with empty Prolific submissions."""
        jatos_results = [
            {
                "result_id": 1,
                "data": {"PROLIFIC_PID": "abc123", "response": "A"},
                "metadata": {},
            },
        ]
        prolific_submissions = []

        merged = data_merger.merge(jatos_results, prolific_submissions)

        # Should get record but not merged
        assert len(merged) == 1
        assert merged[0]["merged"] is False
        assert merged[0]["prolific_metadata"] is None

    def test_merge_both_empty(self, data_merger: DataMerger) -> None:
        """Test merging with both empty."""
        merged = data_merger.merge([], [])
        assert len(merged) == 0

    def test_extract_prolific_pid_from_data(
        self,
        data_merger: DataMerger,
    ) -> None:
        """Test extracting PID from data field."""
        result = {
            "data": {"PROLIFIC_PID": "abc123", "other": "value"},
            "metadata": {},
        }

        pid = data_merger._extract_prolific_pid(result)
        assert pid == "abc123"

    def test_extract_prolific_pid_from_metadata(
        self,
        data_merger: DataMerger,
    ) -> None:
        """Test extracting PID from metadata field."""
        result = {
            "data": {},
            "metadata": {"PROLIFIC_PID": "def456"},
        }

        pid = data_merger._extract_prolific_pid(result)
        assert pid == "def456"

    def test_extract_prolific_pid_data_priority(
        self,
        data_merger: DataMerger,
    ) -> None:
        """Test that data field takes priority over metadata."""
        result = {
            "data": {"PROLIFIC_PID": "abc123"},
            "metadata": {"PROLIFIC_PID": "def456"},
        }

        pid = data_merger._extract_prolific_pid(result)
        assert pid == "abc123"  # Data field wins

    def test_extract_prolific_pid_not_found(
        self,
        data_merger: DataMerger,
    ) -> None:
        """Test extracting PID when not found."""
        result = {
            "data": {"other": "value"},
            "metadata": {"other": "value"},
        }

        pid = data_merger._extract_prolific_pid(result)
        assert pid is None

    def test_extract_prolific_pid_missing_fields(
        self,
        data_merger: DataMerger,
    ) -> None:
        """Test extracting PID when fields are missing."""
        result = {}

        pid = data_merger._extract_prolific_pid(result)
        assert pid is None

    def test_extract_prolific_pid_non_dict_data(
        self,
        data_merger: DataMerger,
    ) -> None:
        """Test extracting PID when data is not a dict."""
        result = {
            "data": "not a dict",
            "metadata": {"PROLIFIC_PID": "abc123"},
        }

        pid = data_merger._extract_prolific_pid(result)
        assert pid == "abc123"  # Should fall back to metadata

    def test_custom_merge_key(self) -> None:
        """Test merging with custom merge key."""
        merger = DataMerger(merge_key="CUSTOM_ID")

        jatos_results = [
            {
                "result_id": 1,
                "data": {"CUSTOM_ID": "custom123", "response": "A"},
                "metadata": {},
            },
        ]

        prolific_submissions = [
            {"participant_id": "custom123", "status": "APPROVED"},
        ]

        merged = merger.merge(jatos_results, prolific_submissions)

        assert len(merged) == 1
        assert merged[0]["merged"] is True

    def test_merge_preserves_all_data(self, data_merger: DataMerger) -> None:
        """Test that merge preserves all original data."""
        jatos_results = [
            {
                "result_id": 1,
                "data": {
                    "PROLIFIC_PID": "abc123",
                    "response": "A",
                    "rt": 1234,
                },
                "metadata": {"componentId": 5},
            },
        ]

        prolific_submissions = [
            {
                "participant_id": "abc123",
                "status": "APPROVED",
                "age": 25,
                "sex": "M",
            },
        ]

        merged = data_merger.merge(jatos_results, prolific_submissions)

        # Verify all data preserved
        assert merged[0]["jatos_data"]["data"]["rt"] == 1234
        assert merged[0]["jatos_data"]["metadata"]["componentId"] == 5
        assert merged[0]["prolific_metadata"]["age"] == 25
        assert merged[0]["prolific_metadata"]["sex"] == "M"

    def test_merge_multiple_prolific_same_pid(
        self,
        data_merger: DataMerger,
    ) -> None:
        """Test merging when multiple Prolific submissions have same PID.

        In this case, the last submission should be used (dict behavior).
        """
        jatos_results = [
            {
                "result_id": 1,
                "data": {"PROLIFIC_PID": "abc123", "response": "A"},
                "metadata": {},
            },
        ]

        # Two submissions with same PID
        prolific_submissions = [
            {"participant_id": "abc123", "status": "APPROVED", "attempt": 1},
            {"participant_id": "abc123", "status": "APPROVED", "attempt": 2},
        ]

        merged = data_merger.merge(jatos_results, prolific_submissions)

        # Should use the last submission
        assert merged[0]["merged"] is True
        assert merged[0]["prolific_metadata"]["attempt"] == 2

    @pytest.mark.parametrize(
        ("merge_key", "jatos_key", "should_match"),
        [
            ("PROLIFIC_PID", "PROLIFIC_PID", True),
            ("CUSTOM_ID", "CUSTOM_ID", True),
            ("WRONG_KEY", "PROLIFIC_PID", False),
        ],
    )
    def test_merge_key_matching(
        self,
        merge_key: str,
        jatos_key: str,
        should_match: bool,
    ) -> None:
        """Test merging with different key configurations."""
        merger = DataMerger(merge_key=merge_key)

        jatos_results = [
            {
                "result_id": 1,
                "data": {jatos_key: "abc123"},
                "metadata": {},
            },
        ]

        prolific_submissions = [
            {"participant_id": "abc123", "status": "APPROVED"},
        ]

        merged = merger.merge(jatos_results, prolific_submissions)

        assert merged[0]["merged"] == should_match
