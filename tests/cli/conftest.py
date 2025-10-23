"""Test fixtures for CLI tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner


@pytest.fixture
def cli_runner() -> CliRunner:
    """Provide Click CLI test runner.

    Returns
    -------
    CliRunner
        Click test runner.
    """
    return CliRunner()


@pytest.fixture
def temp_project_dir(tmp_path: Path) -> Path:
    """Provide temporary project directory.

    Parameters
    ----------
    tmp_path : Path
        Pytest temporary directory.

    Returns
    -------
    Path
        Temporary project directory.
    """
    return tmp_path / "test_project"


@pytest.fixture
def mock_config_file(tmp_path: Path) -> Path:
    """Create mock sash.yaml config file.

    Parameters
    ----------
    tmp_path : Path
        Pytest temporary directory.

    Returns
    -------
    Path
        Path to mock config file.
    """
    config_file = tmp_path / "sash.yaml"
    config_content = """
profile: test

logging:
  level: DEBUG
  format: "%(message)s"

paths:
  data_dir: .test_data

resources:
  auto_download: false
  cache_resources: true
  default_language: eng

templates:
  filling_strategy: random
  max_combinations: 100
  random_seed: 42

models:
  default_language_model: gpt2
  use_gpu: false
  cache_model_outputs: true
"""
    config_file.write_text(config_content)
    return config_file


@pytest.fixture
def mock_invalid_config_file(tmp_path: Path) -> Path:
    """Create invalid config file for error testing.

    Parameters
    ----------
    tmp_path : Path
        Pytest temporary directory.

    Returns
    -------
    Path
        Path to invalid config file.
    """
    config_file = tmp_path / "invalid.yaml"
    # Invalid YAML syntax
    config_content = """
profile: test
logging:
  level: DEBUG
  invalid_indentation
"""
    config_file.write_text(config_content)
    return config_file


@pytest.fixture
def mock_config_with_validation_errors(tmp_path: Path) -> Path:
    """Create config file with validation errors.

    Parameters
    ----------
    tmp_path : Path
        Pytest temporary directory.

    Returns
    -------
    Path
        Path to config file with validation errors.
    """
    config_file = tmp_path / "validation_errors.yaml"
    config_content = """
profile: test

templates:
  filling_strategy: mlm
  # Missing mlm_model_name - validation error
  max_combinations: -1  # Invalid negative value

lists:
  n_lists: -1  # Invalid negative value
"""
    config_file.write_text(config_content)
    return config_file
