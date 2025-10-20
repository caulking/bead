"""Fixtures for template filling model tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.templates.adapters.adapter_helpers import MockMLMAdapter


@pytest.fixture
def mock_adapter() -> MockMLMAdapter:
    """Create mock adapter fixture.

    Returns
    -------
    MockMLMAdapter
        Mock adapter
    """
    return MockMLMAdapter()


@pytest.fixture
def loaded_mock_adapter() -> MockMLMAdapter:
    """Create loaded mock adapter fixture.

    Returns
    -------
    MockMLMAdapter
        Loaded mock adapter
    """
    adapter = MockMLMAdapter()
    adapter.load_model()
    return adapter


@pytest.fixture
def tmp_cache_dir(tmp_path: Path) -> Path:
    """Create temporary cache directory.

    Parameters
    ----------
    tmp_path : Path
        Pytest tmp_path fixture

    Returns
    -------
    Path
        Cache directory
    """
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return cache_dir
