"""Configuration system for sash package.

This module provides configuration models, default settings, and profiles
for all aspects of the sash pipeline.

Examples
--------
>>> from sash.config import SashConfig, get_default_config, get_profile
>>> # Use default configuration
>>> config = get_default_config()
>>> config.profile
'default'
>>> # Use a profile
>>> dev_config = get_profile("dev")
>>> dev_config.logging.level
'DEBUG'
>>> # Create custom configuration
>>> from sash.config import PathsConfig
>>> from pathlib import Path
>>> custom = SashConfig(paths=PathsConfig(data_dir=Path("my_data")))
"""

from __future__ import annotations

from sash.config.defaults import DEFAULT_CONFIG, get_default_config
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
from sash.config.profiles import (
    DEV_CONFIG,
    PROD_CONFIG,
    PROFILES,
    TEST_CONFIG,
    get_profile,
    list_profiles,
)

__all__ = [
    # Main config
    "SashConfig",
    # Config sections
    "PathsConfig",
    "ResourceConfig",
    "TemplateConfig",
    "ModelConfig",
    "ItemConfig",
    "ListConfig",
    "DeploymentConfig",
    "TrainingConfig",
    "LoggingConfig",
    # Defaults
    "DEFAULT_CONFIG",
    "get_default_config",
    # Profiles
    "DEV_CONFIG",
    "PROD_CONFIG",
    "TEST_CONFIG",
    "PROFILES",
    "get_profile",
    "list_profiles",
]
