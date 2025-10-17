"""Configuration profiles for the sash package.

This module provides pre-configured profiles for different environments
(development, production, testing) with optimized settings for each use case.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import gettempdir

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

# Development profile: verbose logging, small batches, relative paths
DEV_CONFIG = SashConfig(
    profile="dev",
    paths=PathsConfig(
        data_dir=Path("data"),
        output_dir=Path("output"),
        cache_dir=Path(".cache"),
        temp_dir=Path(gettempdir()) / "sash_dev",
        create_dirs=True,
    ),
    resources=ResourceConfig(
        cache_external=False,  # Disable caching for development
    ),
    templates=TemplateConfig(
        filling_strategy="exhaustive",
        batch_size=100,  # Small batch size for quick iteration
        stream_mode=False,
    ),
    items=ItemConfig(
        model=ModelConfig(
            provider="huggingface",
            model_name="gpt2",
            batch_size=4,  # Small batch for development
            device="cpu",
        ),
        parallel_processing=False,  # Simpler debugging without parallelism
    ),
    lists=ListConfig(
        num_lists=1,
    ),
    deployment=DeploymentConfig(),
    training=TrainingConfig(
        epochs=1,  # Quick training for testing
        batch_size=8,
        learning_rate=2e-5,
    ),
    logging=LoggingConfig(
        level="DEBUG",  # Verbose logging for development
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
>>> from sash.config.profiles import DEV_CONFIG
>>> DEV_CONFIG.logging.level
'DEBUG'
>>> DEV_CONFIG.templates.batch_size
100
"""

# Production profile: optimized settings, large batches, absolute paths
PROD_CONFIG = SashConfig(
    profile="prod",
    paths=PathsConfig(
        data_dir=Path("/var/sash/data").absolute(),
        output_dir=Path("/var/sash/output").absolute(),
        cache_dir=Path("/var/sash/cache").absolute(),
        temp_dir=Path("/var/sash/temp").absolute(),
        create_dirs=True,
    ),
    resources=ResourceConfig(
        cache_external=True,  # Enable caching for performance
    ),
    templates=TemplateConfig(
        filling_strategy="exhaustive",
        batch_size=10000,  # Large batches for efficiency
        stream_mode=True,  # Handle large templates efficiently
    ),
    items=ItemConfig(
        model=ModelConfig(
            provider="huggingface",
            model_name="gpt2",
            batch_size=32,  # Large batch for production
            device="cuda",  # Use GPU if available
        ),
        parallel_processing=True,  # Enable parallelism
        num_workers=8,  # Multiple workers
    ),
    lists=ListConfig(
        num_lists=1,
    ),
    deployment=DeploymentConfig(
        apply_material_design=True,
        include_demographics=True,
        include_attention_checks=True,
    ),
    training=TrainingConfig(
        epochs=10,  # Full training
        batch_size=32,
        learning_rate=2e-5,
        use_wandb=True,  # Track metrics
    ),
    logging=LoggingConfig(
        level="WARNING",  # Minimal logging for production
        console=False,  # Log to file only
        file=Path("/var/log/sash/app.log"),
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
>>> from sash.config.profiles import PROD_CONFIG
>>> PROD_CONFIG.logging.level
'WARNING'
>>> PROD_CONFIG.templates.batch_size
10000
>>> PROD_CONFIG.items.parallel_processing
True
"""

# Test profile: minimal logging, tiny batches, temp directories
TEST_CONFIG = SashConfig(
    profile="test",
    paths=PathsConfig(
        data_dir=Path(gettempdir()) / "sash_test" / "data",
        output_dir=Path(gettempdir()) / "sash_test" / "output",
        cache_dir=Path(gettempdir()) / "sash_test" / "cache",
        temp_dir=Path(gettempdir()) / "sash_test" / "temp",
        create_dirs=True,
    ),
    resources=ResourceConfig(
        cache_external=False,  # No caching for tests
    ),
    templates=TemplateConfig(
        filling_strategy="exhaustive",
        batch_size=10,  # Tiny batches for fast tests
        max_combinations=100,  # Limit for tests
        random_seed=42,  # Reproducible tests
    ),
    items=ItemConfig(
        model=ModelConfig(
            provider="huggingface",
            model_name="gpt2",
            batch_size=1,  # Minimal batch for tests
            device="cpu",  # CPU only for tests
        ),
        parallel_processing=False,  # Simpler test execution
        num_workers=1,
    ),
    lists=ListConfig(
        num_lists=1,
        random_seed=42,  # Reproducible tests
    ),
    deployment=DeploymentConfig(
        apply_material_design=False,  # Minimal for tests
        include_demographics=False,
        include_attention_checks=False,
    ),
    training=TrainingConfig(
        epochs=1,  # Minimal training
        batch_size=2,
        learning_rate=2e-5,
        use_wandb=False,  # No tracking for tests
    ),
    logging=LoggingConfig(
        level="CRITICAL",  # Minimal logging for tests
        console=False,  # Quiet tests
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
>>> from sash.config.profiles import TEST_CONFIG
>>> TEST_CONFIG.logging.level
'CRITICAL'
>>> TEST_CONFIG.templates.batch_size
10
>>> TEST_CONFIG.templates.random_seed
42
"""

# Profile registry
PROFILES: dict[str, SashConfig] = {
    "default": SashConfig(),  # Default from models
    "dev": DEV_CONFIG,
    "prod": PROD_CONFIG,
    "test": TEST_CONFIG,
}
"""Registry of all available configuration profiles.

Maps profile names to their corresponding SashConfig instances.

Examples
--------
>>> from sash.config.profiles import PROFILES
>>> list(PROFILES.keys())
['default', 'dev', 'prod', 'test']
>>> PROFILES["dev"].logging.level
'DEBUG'
"""


def get_profile(name: str) -> SashConfig:
    """Get configuration profile by name.

    Parameters
    ----------
    name : str
        Profile name. Must be one of: 'default', 'dev', 'prod', 'test'.

    Returns
    -------
    SashConfig
        Configuration for the specified profile.

    Raises
    ------
    ValueError
        If profile name is not found in the registry.

    Examples
    --------
    >>> from sash.config.profiles import get_profile
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
    >>> from sash.config.profiles import list_profiles
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
