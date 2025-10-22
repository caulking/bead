"""Shared pytest fixtures for data_collection tests.

This module provides fixtures for testing JATOS and Prolific data collection.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from sash.training.data_collection.jatos import JATOSDataCollector
from sash.training.data_collection.merger import DataMerger
from sash.training.data_collection.prolific import ProlificDataCollector


@pytest.fixture
def jatos_collector() -> JATOSDataCollector:
    """JATOS collector instance."""
    return JATOSDataCollector(
        base_url="https://test.jatos.org",
        api_token="test_token",
        study_id=123,
    )


@pytest.fixture
def prolific_collector() -> ProlificDataCollector:
    """Prolific collector instance."""
    return ProlificDataCollector(
        api_key="test_api_key",
        study_id="test_study_id",
    )


@pytest.fixture
def data_merger() -> DataMerger:
    """Create a data merger instance."""
    return DataMerger(merge_key="PROLIFIC_PID")


@pytest.fixture
def mock_jatos_response() -> dict:
    """Mock JATOS API response for a single result."""
    return {
        "id": 123,
        "componentId": 1,
        "workerType": "Prolific",
        "workerId": 456,
        "studyId": 123,
    }


@pytest.fixture
def mock_jatos_result_data() -> dict:
    """Mock JATOS result data."""
    return {
        "PROLIFIC_PID": "abc123",
        "responses": [
            {"trial": 1, "response": "A"},
            {"trial": 2, "response": "B"},
        ],
    }


@pytest.fixture
def mock_prolific_response() -> dict:
    """Mock Prolific API response for a submission."""
    return {
        "participant_id": "abc123",
        "status": "APPROVED",
        "started_at": "2025-01-17T10:00:00Z",
        "completed_at": "2025-01-17T10:15:00Z",
        "time_taken": 900,
    }


@pytest.fixture
def mock_jatos_client(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Mock JATOSClient for testing JATOSDataCollector.

    This fixture mocks the JATOSClient import in the jatos module
    to avoid making real API calls.
    """
    mock_client_class = Mock()
    mock_client_instance = Mock()
    mock_client_class.return_value = mock_client_instance

    # Mock the JATOSClient import
    monkeypatch.setattr(
        "sash.training.data_collection.jatos.JATOSClient",
        mock_client_class,
    )

    return mock_client_instance
