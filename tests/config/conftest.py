"""Pytest fixtures for config module tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from sash.config.models import (
    DeploymentConfig,
    ItemConfig,
    ListConfig,
    LoggingConfig,
    ModelConfig,
    PathsConfig,
    ResourceConfig,
    SashConfig,
    TemplateConfig,
    TrainingConfig,
)


@pytest.fixture
def sample_paths_config() -> PathsConfig:
    """Provide sample PathsConfig for testing.

    Returns
    -------
    PathsConfig
        A sample paths configuration with common test values.

    Examples
    --------
    >>> def test_paths(sample_paths_config):
    ...     assert sample_paths_config.data_dir == Path("data")
    """
    return PathsConfig(
        data_dir=Path("data"),
        output_dir=Path("output"),
        cache_dir=Path(".cache"),
    )


@pytest.fixture
def sample_resource_config() -> ResourceConfig:
    """Provide sample ResourceConfig for testing.

    Returns
    -------
    ResourceConfig
        A sample resource configuration with common test values.
    """
    return ResourceConfig(
        lexicon_path=Path("lexicon.json"),
        templates_path=Path("templates.json"),
    )


@pytest.fixture
def sample_template_config() -> TemplateConfig:
    """Provide sample TemplateConfig for testing.

    Returns
    -------
    TemplateConfig
        A sample template configuration with common test values.
    """
    return TemplateConfig(
        filling_strategy="exhaustive",
        batch_size=100,
    )


@pytest.fixture
def sample_model_config() -> ModelConfig:
    """Provide sample ModelConfig for testing.

    Returns
    -------
    ModelConfig
        A sample model configuration with common test values.

    Examples
    --------
    >>> def test_model(sample_model_config):
    ...     assert sample_model_config.provider == "huggingface"
    """
    return ModelConfig(
        provider="huggingface",
        model_name="gpt2",
        batch_size=8,
        device="cpu",
    )


@pytest.fixture
def sample_item_config(sample_model_config: ModelConfig) -> ItemConfig:
    """Provide sample ItemConfig for testing.

    Parameters
    ----------
    sample_model_config : ModelConfig
        Sample model configuration fixture.

    Returns
    -------
    ItemConfig
        A sample item configuration with common test values.
    """
    return ItemConfig(
        model=sample_model_config,
        apply_constraints=True,
    )


@pytest.fixture
def sample_list_config() -> ListConfig:
    """Provide sample ListConfig for testing.

    Returns
    -------
    ListConfig
        A sample list configuration with common test values.
    """
    return ListConfig(
        partitioning_strategy="balanced",
        num_lists=2,
    )


@pytest.fixture
def sample_deployment_config() -> DeploymentConfig:
    """Provide sample DeploymentConfig for testing.

    Returns
    -------
    DeploymentConfig
        A sample deployment configuration with common test values.
    """
    return DeploymentConfig(
        platform="jspsych",
        jspsych_version="7.3.0",
    )


@pytest.fixture
def sample_training_config() -> TrainingConfig:
    """Provide sample TrainingConfig for testing.

    Returns
    -------
    TrainingConfig
        A sample training configuration with common test values.
    """
    return TrainingConfig(
        trainer="huggingface",
        epochs=3,
        batch_size=16,
    )


@pytest.fixture
def sample_logging_config() -> LoggingConfig:
    """Provide sample LoggingConfig for testing.

    Returns
    -------
    LoggingConfig
        A sample logging configuration with common test values.
    """
    return LoggingConfig(
        level="INFO",
        console=True,
    )


@pytest.fixture
def sample_full_config(
    sample_paths_config: PathsConfig,
    sample_resource_config: ResourceConfig,
    sample_template_config: TemplateConfig,
    sample_item_config: ItemConfig,
    sample_list_config: ListConfig,
    sample_deployment_config: DeploymentConfig,
    sample_training_config: TrainingConfig,
    sample_logging_config: LoggingConfig,
) -> SashConfig:
    """Provide complete sample configuration for testing.

    Parameters
    ----------
    sample_paths_config : PathsConfig
        Sample paths configuration fixture.
    sample_resource_config : ResourceConfig
        Sample resource configuration fixture.
    sample_template_config : TemplateConfig
        Sample template configuration fixture.
    sample_item_config : ItemConfig
        Sample item configuration fixture.
    sample_list_config : ListConfig
        Sample list configuration fixture.
    sample_deployment_config : DeploymentConfig
        Sample deployment configuration fixture.
    sample_training_config : TrainingConfig
        Sample training configuration fixture.
    sample_logging_config : LoggingConfig
        Sample logging configuration fixture.

    Returns
    -------
    SashConfig
        A complete sample configuration with all sections configured.

    Examples
    --------
    >>> def test_config(sample_full_config):
    ...     assert sample_full_config.profile == "test"
    ...     assert isinstance(sample_full_config.paths, PathsConfig)
    """
    return SashConfig(
        profile="test",
        paths=sample_paths_config,
        resources=sample_resource_config,
        templates=sample_template_config,
        items=sample_item_config,
        lists=sample_list_config,
        deployment=sample_deployment_config,
        training=sample_training_config,
        logging=sample_logging_config,
    )
