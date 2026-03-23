"""Configuration profiles for the bead package.

This module provides pre-configured profiles for different environments
(development, production, testing) with optimized settings for each use case.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import gettempdir

from bead.config.active_learning import (
    ActiveLearningConfig,
    ForcedChoiceModelConfig,
    TrainerConfig,
)
from bead.config.config import BeadConfig
from bead.config.deployment import DeploymentConfig
from bead.config.item import ItemConfig
from bead.config.list import ListConfig
from bead.config.logging import LoggingConfig
from bead.config.model import ModelConfig
from bead.config.paths import PathsConfig
from bead.config.resources import ResourceConfig
from bead.config.template import TemplateConfig

# development profile: verbose logging, small batches, relative paths
DEV_CONFIG = BeadConfig(
    profile="dev",
    paths=PathsConfig(
        data_dir=Path("data"),
        output_dir=Path("output"),
        cache_dir=Path(".cache"),
        temp_dir=Path(gettempdir()) / "bead_dev",
        create_dirs=True,
    ),
    resources=ResourceConfig(
        cache_external=False,  # disable caching for development
    ),
    templates=TemplateConfig(
        filling_strategy="exhaustive",
        batch_size=100,  # small batch size for quick iteration
        stream_mode=False,
    ),
    items=ItemConfig(
        model=ModelConfig(
            provider="huggingface",
            model_name="gpt2",
            batch_size=4,  # small batch for development
            device="cpu",
        ),
        parallel_processing=False,  # simpler debugging without parallelism
    ),
    lists=ListConfig(
        num_lists=1,
    ),
    deployment=DeploymentConfig(),
    active_learning=ActiveLearningConfig(
        forced_choice_model=ForcedChoiceModelConfig(
            num_epochs=1,  # quick training for testing
            batch_size=8,
            learning_rate=2e-5,
        ),
        trainer=TrainerConfig(epochs=1),
    ),
    logging=LoggingConfig(
        level="DEBUG",  # verbose logging for development
        console=True,
    ),
)
"""Development configuration profile.

Optimized for:
- Quick iteration and debugging
- Verbose logging (DEBUG level)
- Small batch sizes for fast feedback
- No caching for fresh data
- Simple single-threaded processing
- Temporary directories for easy cleanup

Examples
--------
>>> from bead.config.profiles import DEV_CONFIG
>>> DEV_CONFIG.logging.level
'DEBUG'
>>> DEV_CONFIG.templates.batch_size
100
"""

# production profile: optimized settings, large batches, absolute paths
PROD_CONFIG = BeadConfig(
    profile="prod",
    paths=PathsConfig(
        data_dir=Path("/var/bead/data").absolute(),
        output_dir=Path("/var/bead/output").absolute(),
        cache_dir=Path("/var/bead/cache").absolute(),
        temp_dir=Path("/var/bead/temp").absolute(),
        create_dirs=True,
    ),
    resources=ResourceConfig(
        cache_external=True,  # enable caching for performance
    ),
    templates=TemplateConfig(
        filling_strategy="exhaustive",
        batch_size=10000,  # large batches for efficiency
        stream_mode=True,  # handle large templates efficiently
    ),
    items=ItemConfig(
        model=ModelConfig(
            provider="huggingface",
            model_name="gpt2",
            batch_size=32,  # large batch for production
            device="cuda",  # use GPU if available
        ),
        parallel_processing=True,  # enable parallelism
        num_workers=8,  # multiple workers
    ),
    lists=ListConfig(
        num_lists=1,
    ),
    deployment=DeploymentConfig(
        apply_material_design=True,
        include_demographics=True,
        include_attention_checks=True,
    ),
    active_learning=ActiveLearningConfig(
        forced_choice_model=ForcedChoiceModelConfig(
            num_epochs=10,  # full training
            batch_size=32,
            learning_rate=2e-5,
        ),
        trainer=TrainerConfig(epochs=10, use_wandb=True),
    ),
    logging=LoggingConfig(
        level="WARNING",  # minimal logging for production
        console=False,  # log to file only
        file=Path("/var/log/bead/app.log"),
    ),
)
"""Production configuration profile.

Optimized for:
- Maximum performance and throughput
- Large batch sizes for efficiency
- GPU acceleration (when available)
- Parallel processing
- External caching enabled
- Minimal logging (WARNING level)
- Absolute paths to production directories
- Metrics tracking with W&B

Examples
--------
>>> from bead.config.profiles import PROD_CONFIG
>>> PROD_CONFIG.logging.level
'WARNING'
>>> PROD_CONFIG.templates.batch_size
10000
>>> PROD_CONFIG.items.parallel_processing
True
"""

# test profile: minimal logging, tiny batches, temp directories
TEST_CONFIG = BeadConfig(
    profile="test",
    paths=PathsConfig(
        data_dir=Path(gettempdir()) / "bead_test" / "data",
        output_dir=Path(gettempdir()) / "bead_test" / "output",
        cache_dir=Path(gettempdir()) / "bead_test" / "cache",
        temp_dir=Path(gettempdir()) / "bead_test" / "temp",
        create_dirs=True,
    ),
    resources=ResourceConfig(
        cache_external=False,  # no caching for tests
    ),
    templates=TemplateConfig(
        filling_strategy="exhaustive",
        batch_size=10,  # tiny batches for fast tests
        max_combinations=100,  # limit for tests
        random_seed=42,  # reproducible tests
    ),
    items=ItemConfig(
        model=ModelConfig(
            provider="huggingface",
            model_name="gpt2",
            batch_size=1,  # minimal batch for tests
            device="cpu",  # CPU only for tests
        ),
        parallel_processing=False,  # simpler test execution
        num_workers=1,
    ),
    lists=ListConfig(
        num_lists=1,
        random_seed=42,  # reproducible tests
    ),
    deployment=DeploymentConfig(
        apply_material_design=False,  # minimal for tests
        include_demographics=False,
        include_attention_checks=False,
    ),
    active_learning=ActiveLearningConfig(
        forced_choice_model=ForcedChoiceModelConfig(
            num_epochs=1,  # minimal training
            batch_size=2,
            learning_rate=2e-5,
        ),
        trainer=TrainerConfig(epochs=1, use_wandb=False),
    ),
    logging=LoggingConfig(
        level="CRITICAL",  # minimal logging for tests
        console=False,  # quiet tests
    ),
)
"""Test configuration profile.

Optimized for:
- Fast test execution
- Reproducibility (fixed random seeds)
- Minimal resource usage
- Tiny batch sizes
- Temporary directories for isolation
- Minimal logging (CRITICAL level)
- No external dependencies
- CPU-only execution

Examples
--------
>>> from bead.config.profiles import TEST_CONFIG
>>> TEST_CONFIG.logging.level
'CRITICAL'
>>> TEST_CONFIG.templates.batch_size
10
>>> TEST_CONFIG.templates.random_seed
42
"""

# profile registry
PROFILES: dict[str, BeadConfig] = {
    "default": BeadConfig(),  # default from models
    "dev": DEV_CONFIG,
    "prod": PROD_CONFIG,
    "test": TEST_CONFIG,
}
"""Registry of all available configuration profiles.

Maps profile names to their corresponding BeadConfig instances.

Examples
--------
>>> from bead.config.profiles import PROFILES
>>> list(PROFILES.keys())
['default', 'dev', 'prod', 'test']
>>> PROFILES["dev"].logging.level
'DEBUG'
"""


def get_profile(name: str) -> BeadConfig:
    """Get configuration profile by name.

    Parameters
    ----------
    name : str
        Profile name. Must be one of: 'default', 'dev', 'prod', 'test'.

    Returns
    -------
    BeadConfig
        Configuration for the specified profile.

    Raises
    ------
    ValueError
        If profile name is not found in the registry.

    Examples
    --------
    >>> from bead.config.profiles import get_profile
    >>> config = get_profile("dev")
    >>> config.profile
    'dev'
    >>> config.logging.level
    'DEBUG'

    >>> try:
    ...     get_profile("invalid")
    ... except ValueError as e:
    ...     print(str(e))
    Profile 'invalid' not found. Available profiles: default, dev, prod, test
    """
    if name not in PROFILES:
        available = ", ".join(sorted(PROFILES.keys()))
        msg = f"Profile {name!r} not found. Available profiles: {available}"
        raise ValueError(msg)

    return PROFILES[name].model_copy(deep=True)


def list_profiles() -> list[str]:
    """Return list of available profile names.

    Returns
    -------
    list[str]
        List of available profile names, sorted alphabetically.

    Examples
    --------
    >>> from bead.config.profiles import list_profiles
    >>> profiles = list_profiles()
    >>> "default" in profiles
    True
    >>> "dev" in profiles
    True
    >>> "prod" in profiles
    True
    >>> "test" in profiles
    True
    """
    return sorted(PROFILES.keys())
