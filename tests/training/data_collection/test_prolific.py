"""Tests for Prolific data collection.

This module tests the ProlificDataCollector class including:
- Initialization
- Submission downloading with pagination
- Filtering by status
- Study info retrieval
- Submission approval
- Error handling
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import requests

from sash.training.data_collection.prolific import ProlificDataCollector


class TestProlificDataCollector:
    """Test suite for ProlificDataCollector."""

    def test_initialization(
        self,
        prolific_collector: ProlificDataCollector,
    ) -> None:
        """Test collector initialization."""
        assert prolific_collector.api_key == "test_api_key"
        assert prolific_collector.study_id == "test_study_id"
        assert prolific_collector.base_url == "https://api.prolific.co/api/v1"
        auth_header = prolific_collector.session.headers["Authorization"]
        assert "Token test_api_key" in auth_header

    def test_download_submissions_without_pagination(
        self,
        prolific_collector: ProlificDataCollector,
        mock_prolific_response: dict,
        tmp_path: Path,
    ) -> None:
        """Test submission download without pagination."""
        # Mock single page response
        mock_response = Mock()
        mock_response.json.return_value = {
            "results": [mock_prolific_response],
        }
        mock_response.raise_for_status.return_value = None

        # First call returns data, second returns empty
        responses = [mock_response, Mock(json=Mock(return_value={"results": []}))]
        prolific_collector.session.get = Mock(side_effect=responses)

        # Download
        output = tmp_path / "submissions.json"
        submissions = prolific_collector.download_submissions(output)

        # Assertions
        assert len(submissions) == 1
        assert output.exists()
        assert submissions[0]["participant_id"] == "abc123"
        assert "download_timestamp" in submissions[0]

        # Verify JSON format
        with open(output) as f:
            data = json.load(f)
            assert len(data) == 1

    def test_download_submissions_with_pagination(
        self,
        prolific_collector: ProlificDataCollector,
        tmp_path: Path,
    ) -> None:
        """Test submission download with pagination."""

        # Mock multiple pages
        def mock_get(url: str, params: dict | None = None) -> Mock:
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None

            page = params.get("page", 1) if params else 1

            if page == 1:
                mock_response.json.return_value = {
                    "results": [
                        {"participant_id": "p1", "status": "APPROVED"},
                        {"participant_id": "p2", "status": "APPROVED"},
                    ]
                }
            elif page == 2:
                mock_response.json.return_value = {
                    "results": [
                        {"participant_id": "p3", "status": "APPROVED"},
                    ]
                }
            else:
                mock_response.json.return_value = {"results": []}

            return mock_response

        prolific_collector.session.get = Mock(side_effect=mock_get)

        # Download
        output = tmp_path / "submissions.json"
        submissions = prolific_collector.download_submissions(output)

        # Should get all 3 submissions across 2 pages
        assert len(submissions) == 3
        assert submissions[0]["participant_id"] == "p1"
        assert submissions[1]["participant_id"] == "p2"
        assert submissions[2]["participant_id"] == "p3"

        # All should have timestamps
        for sub in submissions:
            assert "download_timestamp" in sub

    def test_download_submissions_with_status_filter(
        self,
        prolific_collector: ProlificDataCollector,
        tmp_path: Path,
    ) -> None:
        """Test submission download with status filter."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "results": [
                {"participant_id": "p1", "status": "APPROVED"},
            ]
        }
        mock_response.raise_for_status.return_value = None

        empty_response = Mock()
        empty_response.json.return_value = {"results": []}

        prolific_collector.session.get = Mock(
            side_effect=[mock_response, empty_response]
        )

        # Download with status filter
        output = tmp_path / "submissions.json"
        submissions = prolific_collector.download_submissions(
            output,
            status="APPROVED",
        )

        # Verify status was passed to API
        call_args = prolific_collector.session.get.call_args_list[0]
        assert call_args[1]["params"]["status"] == "APPROVED"

        assert len(submissions) == 1
        assert submissions[0]["status"] == "APPROVED"

    def test_download_submissions_empty(
        self,
        prolific_collector: ProlificDataCollector,
        tmp_path: Path,
    ) -> None:
        """Test submission download when no submissions available."""
        # Mock empty response
        mock_response = Mock()
        mock_response.json.return_value = {"results": []}
        mock_response.raise_for_status.return_value = None

        prolific_collector.session.get = Mock(return_value=mock_response)

        # Download
        output = tmp_path / "submissions.json"
        submissions = prolific_collector.download_submissions(output)

        # Assertions
        assert len(submissions) == 0
        assert output.exists()

        # Verify file contains empty list
        with open(output) as f:
            data = json.load(f)
            assert data == []

    def test_download_submissions_http_error(
        self,
        prolific_collector: ProlificDataCollector,
        tmp_path: Path,
    ) -> None:
        """Test handling of HTTP errors."""
        # Mock HTTP error
        prolific_collector.session.get = Mock(
            side_effect=requests.exceptions.HTTPError("401 Unauthorized")
        )

        # Should raise
        with pytest.raises(requests.exceptions.HTTPError):
            prolific_collector.download_submissions(tmp_path / "submissions.json")

    def test_download_submissions_network_error(
        self,
        prolific_collector: ProlificDataCollector,
        tmp_path: Path,
    ) -> None:
        """Test handling of network errors."""
        # Mock network error
        prolific_collector.session.get = Mock(
            side_effect=requests.exceptions.ConnectionError()
        )

        # Should raise
        with pytest.raises(requests.exceptions.ConnectionError):
            prolific_collector.download_submissions(tmp_path / "submissions.json")

    def test_get_study_info(
        self,
        prolific_collector: ProlificDataCollector,
    ) -> None:
        """Test getting study info."""
        mock_study = {
            "id": "test_study_id",
            "name": "Test Study",
            "total_available_places": 100,
        }

        mock_response = Mock()
        mock_response.json.return_value = mock_study
        mock_response.raise_for_status.return_value = None

        prolific_collector.session.get = Mock(return_value=mock_response)

        info = prolific_collector.get_study_info()

        assert info == mock_study
        assert prolific_collector.session.get.called

    def test_approve_submissions_single(
        self,
        prolific_collector: ProlificDataCollector,
    ) -> None:
        """Test approving a single submission."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None

        prolific_collector.session.post = Mock(return_value=mock_response)

        # Approve
        prolific_collector.approve_submissions(["sub1"])

        # Verify post was called
        assert prolific_collector.session.post.call_count == 1

        call_args = prolific_collector.session.post.call_args
        assert "/submissions/sub1/transition/" in call_args[0][0]
        assert call_args[1]["json"]["action"] == "APPROVE"

    def test_approve_submissions_multiple(
        self,
        prolific_collector: ProlificDataCollector,
    ) -> None:
        """Test approving multiple submissions."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None

        prolific_collector.session.post = Mock(return_value=mock_response)

        # Approve multiple
        prolific_collector.approve_submissions(["sub1", "sub2", "sub3"])

        # Verify post was called 3 times
        assert prolific_collector.session.post.call_count == 3

    def test_approve_submissions_http_error(
        self,
        prolific_collector: ProlificDataCollector,
    ) -> None:
        """Test handling of HTTP errors during approval."""
        # Mock HTTP error
        prolific_collector.session.post = Mock(
            side_effect=requests.exceptions.HTTPError("400 Bad Request")
        )

        # Should raise
        with pytest.raises(requests.exceptions.HTTPError):
            prolific_collector.approve_submissions(["sub1"])

    def test_json_format(
        self,
        prolific_collector: ProlificDataCollector,
        tmp_path: Path,
    ) -> None:
        """Test that submissions are saved in JSON format."""
        # Mock submissions
        mock_response = Mock()
        mock_response.json.return_value = {
            "results": [
                {"participant_id": "p1", "status": "APPROVED"},
                {"participant_id": "p2", "status": "APPROVED"},
            ]
        }
        mock_response.raise_for_status.return_value = None

        empty_response = Mock()
        empty_response.json.return_value = {"results": []}

        prolific_collector.session.get = Mock(
            side_effect=[mock_response, empty_response]
        )

        # Download
        output = tmp_path / "submissions.json"
        prolific_collector.download_submissions(output)

        # Verify JSON format
        with open(output) as f:
            data = json.load(f)
            assert isinstance(data, list)
            assert len(data) == 2
            assert data[0]["participant_id"] == "p1"
            assert data[1]["participant_id"] == "p2"

    def test_download_creates_parent_directories(
        self,
        prolific_collector: ProlificDataCollector,
        tmp_path: Path,
    ) -> None:
        """Test that download_submissions creates parent directories."""
        # Mock empty response
        mock_response = Mock()
        mock_response.json.return_value = {"results": []}
        mock_response.raise_for_status.return_value = None

        prolific_collector.session.get = Mock(return_value=mock_response)

        # Use nested path
        output = tmp_path / "nested" / "directory" / "submissions.json"
        prolific_collector.download_submissions(output)

        # Parent directories should be created
        assert output.parent.exists()
        assert output.exists()

    def test_url_construction(
        self,
        prolific_collector: ProlificDataCollector,
    ) -> None:
        """Test correct URL construction."""
        mock_response = Mock()
        mock_response.json.return_value = {"results": []}
        mock_response.raise_for_status.return_value = None

        prolific_collector.session.get = Mock(return_value=mock_response)

        prolific_collector.get_study_info()

        # Verify URL
        call_args = prolific_collector.session.get.call_args
        expected_url = f"{prolific_collector.base_url}/studies/test_study_id/"
        assert call_args[0][0] == expected_url
