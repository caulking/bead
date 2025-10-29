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
from sash.config.env import load_from_env
from sash.config.loader import load_config, load_yaml_file, merge_configs
from sash.config.models import (
    ActiveLearningConfig,
    DeploymentConfig,
    ItemConfig,
    ListConfig,
    LoggingConfig,
    ModelConfig,
    PathsConfig,
    ResourceConfig,
    SashConfig,
    TemplateConfig,
)
from sash.config.profiles import (
    DEV_CONFIG,
    PROD_CONFIG,
    PROFILES,
    TEST_CONFIG,
    get_profile,
    list_profiles,
)
from sash.config.serialization import save_yaml, to_yaml
from sash.config.validation import validate_config

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
    "ActiveLearningConfig",
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
    # Loading
    "load_config",
    "load_yaml_file",
    "merge_configs",
    # Environment
    "load_from_env",
    # Validation
    "validate_config",
    # Serialization
    "to_yaml",
    "save_yaml",
]
