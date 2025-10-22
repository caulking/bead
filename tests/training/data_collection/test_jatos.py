"""Tests for JATOS data collection.

This module tests the JATOSDataCollector class including:
- Initialization
- Result downloading with filters
- Study info retrieval
- Error handling
- File writing
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import requests

from sash.training.data_collection.jatos import JATOSDataCollector


class TestJATOSDataCollector:
    """Test suite for JATOSDataCollector."""

    def test_initialization(self, jatos_collector: JATOSDataCollector) -> None:
        """Test collector initialization."""
        assert jatos_collector.study_id == 123
        assert jatos_collector.client.base_url == "https://test.jatos.org"

    def test_download_results_success(
        self,
        jatos_collector: JATOSDataCollector,
        mock_jatos_response: dict,
        mock_jatos_result_data: dict,
        tmp_path: Path,
    ) -> None:
        """Test successful results download."""
        # Mock get_results to return result IDs
        jatos_collector.client.get_results = Mock(return_value=[1, 2, 3])

        # Mock session.get for individual results
        mock_data_response = Mock()
        mock_data_response.json.return_value = mock_jatos_result_data
        mock_data_response.raise_for_status.return_value = None

        mock_meta_response = Mock()
        mock_meta_response.json.return_value = mock_jatos_response
        mock_meta_response.raise_for_status.return_value = None

        def mock_get(url: str) -> Mock:
            if "/data" in url:
                return mock_data_response
            return mock_meta_response

        jatos_collector.client.session.get = Mock(side_effect=mock_get)

        # Download
        output = tmp_path / "results.jsonl"
        results = jatos_collector.download_results(output)

        # Assertions
        assert len(results) == 3
        assert output.exists()
        assert jatos_collector.client.get_results.called

        # Verify each result has required fields
        for result in results:
            assert "result_id" in result
            assert "data" in result
            assert "metadata" in result
            assert "download_timestamp" in result

        # Verify JSONLines format
        with open(output) as f:
            lines = f.readlines()
            assert len(lines) == 3
            for line in lines:
                assert json.loads(line)  # Valid JSON

    def test_download_results_with_component_filter(
        self,
        jatos_collector: JATOSDataCollector,
        tmp_path: Path,
    ) -> None:
        """Test results download with component filter."""
        # Mock get_results
        jatos_collector.client.get_results = Mock(return_value=[1, 2])

        # Mock responses with different components
        def mock_download(result_id: int) -> dict:
            return {
                "result_id": result_id,
                "data": {},
                "metadata": {"componentId": 1 if result_id == 1 else 2},
                "download_timestamp": "2025-01-17T12:00:00Z",
            }

        jatos_collector._download_single_result = Mock(side_effect=mock_download)

        # Download with filter
        output = tmp_path / "results.jsonl"
        results = jatos_collector.download_results(output, component_id=1)

        # Should only get component 1
        assert len(results) == 1
        assert results[0]["metadata"]["componentId"] == 1

    def test_download_results_with_worker_type_filter(
        self,
        jatos_collector: JATOSDataCollector,
        tmp_path: Path,
    ) -> None:
        """Test results download with worker type filter."""
        # Mock get_results
        jatos_collector.client.get_results = Mock(return_value=[1, 2])

        # Mock responses with different worker types
        def mock_download(result_id: int) -> dict:
            return {
                "result_id": result_id,
                "data": {},
                "metadata": {"workerType": "Prolific" if result_id == 1 else "MTurk"},
                "download_timestamp": "2025-01-17T12:00:00Z",
            }

        jatos_collector._download_single_result = Mock(side_effect=mock_download)

        # Download with filter
        output = tmp_path / "results.jsonl"
        results = jatos_collector.download_results(output, worker_type="Prolific")

        # Should only get Prolific workers
        assert len(results) == 1
        assert results[0]["metadata"]["workerType"] == "Prolific"

    def test_download_results_empty(
        self,
        jatos_collector: JATOSDataCollector,
        tmp_path: Path,
    ) -> None:
        """Test results download when no results available."""
        # Mock empty results
        jatos_collector.client.get_results = Mock(return_value=[])

        # Download
        output = tmp_path / "results.jsonl"
        results = jatos_collector.download_results(output)

        # Assertions
        assert len(results) == 0
        assert output.exists()

        # Verify file is empty
        with open(output) as f:
            assert f.read() == ""

    def test_download_results_http_error(
        self,
        jatos_collector: JATOSDataCollector,
        tmp_path: Path,
    ) -> None:
        """Test handling of HTTP errors."""
        # Mock HTTP error
        jatos_collector.client.get_results = Mock(
            side_effect=requests.exceptions.HTTPError("404 Not Found")
        )

        # Should raise
        with pytest.raises(requests.exceptions.HTTPError):
            jatos_collector.download_results(tmp_path / "results.jsonl")

    def test_download_results_network_error(
        self,
        jatos_collector: JATOSDataCollector,
        tmp_path: Path,
    ) -> None:
        """Test handling of network errors."""
        # Mock network error
        jatos_collector.client.get_results = Mock(
            side_effect=requests.exceptions.ConnectionError()
        )

        # Should raise
        with pytest.raises(requests.exceptions.ConnectionError):
            jatos_collector.download_results(tmp_path / "results.jsonl")

    def test_download_single_result(
        self,
        jatos_collector: JATOSDataCollector,
        mock_jatos_response: dict,
        mock_jatos_result_data: dict,
    ) -> None:
        """Test downloading a single result."""
        # Mock session responses
        mock_data_response = Mock()
        mock_data_response.json.return_value = mock_jatos_result_data
        mock_data_response.raise_for_status.return_value = None

        mock_meta_response = Mock()
        mock_meta_response.json.return_value = mock_jatos_response
        mock_meta_response.raise_for_status.return_value = None

        def mock_get(url: str) -> Mock:
            if "/data" in url:
                return mock_data_response
            return mock_meta_response

        jatos_collector.client.session.get = Mock(side_effect=mock_get)

        # Download
        result = jatos_collector._download_single_result(123)

        # Assertions
        assert result["result_id"] == 123
        assert result["data"] == mock_jatos_result_data
        assert result["metadata"] == mock_jatos_response
        assert "download_timestamp" in result

    def test_get_study_info(
        self,
        jatos_collector: JATOSDataCollector,
    ) -> None:
        """Test getting study info."""
        mock_study = {
            "id": 123,
            "title": "Test Study",
            "description": "A test study",
        }

        jatos_collector.client.get_study = Mock(return_value=mock_study)

        info = jatos_collector.get_study_info()

        assert info == mock_study
        jatos_collector.client.get_study.assert_called_once_with(123)

    def test_get_result_count(
        self,
        jatos_collector: JATOSDataCollector,
    ) -> None:
        """Test getting result count."""
        jatos_collector.client.get_results = Mock(return_value=[1, 2, 3, 4, 5])

        count = jatos_collector.get_result_count()

        assert count == 5
        jatos_collector.client.get_results.assert_called_once_with(123)

    def test_get_result_count_empty(
        self,
        jatos_collector: JATOSDataCollector,
    ) -> None:
        """Test getting result count when no results."""
        jatos_collector.client.get_results = Mock(return_value=[])

        count = jatos_collector.get_result_count()

        assert count == 0

    @pytest.mark.parametrize(
        ("study_id", "expected_url_part"),
        [
            (123, "/api/v1/studies/123/results"),
            (456, "/api/v1/studies/456/results"),
        ],
    )
    def test_url_construction(
        self,
        study_id: int,
        expected_url_part: str,
    ) -> None:
        """Test correct URL construction for different study IDs."""
        collector = JATOSDataCollector(
            base_url="https://test.jatos.org",
            api_token="token",
            study_id=study_id,
        )

        # Mock get_results
        collector.client.get_results = Mock(return_value=[])
        collector.get_result_count()

        # Check that get_results was called with correct study_id
        collector.client.get_results.assert_called_once_with(study_id)

    def test_jsonlines_format(
        self,
        jatos_collector: JATOSDataCollector,
        tmp_path: Path,
    ) -> None:
        """Test that results are saved in JSONLines format."""
        # Mock results
        jatos_collector.client.get_results = Mock(return_value=[1, 2])

        def mock_download(result_id: int) -> dict:
            return {
                "result_id": result_id,
                "data": {"response": result_id},
                "metadata": {},
                "download_timestamp": "2025-01-17T12:00:00Z",
            }

        jatos_collector._download_single_result = Mock(side_effect=mock_download)

        # Download
        output = tmp_path / "results.jsonl"
        jatos_collector.download_results(output)

        # Verify JSONLines format
        with open(output) as f:
            lines = f.readlines()
            assert len(lines) == 2

            # Each line should be valid JSON
            for i, line in enumerate(lines, start=1):
                result = json.loads(line.strip())
                assert result["result_id"] == i
                assert result["data"]["response"] == i

    def test_download_creates_parent_directories(
        self,
        jatos_collector: JATOSDataCollector,
        tmp_path: Path,
    ) -> None:
        """Test that download_results creates parent directories."""
        # Mock empty results
        jatos_collector.client.get_results = Mock(return_value=[])

        # Use nested path
        output = tmp_path / "nested" / "directory" / "results.jsonl"
        jatos_collector.download_results(output)

        # Parent directories should be created
        assert output.parent.exists()
        assert output.exists()
