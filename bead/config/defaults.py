"""Default configurations for the bead package.

Provides ``DEFAULT_CONFIG`` and helpers for retrieving default
configuration instances.
"""

from __future__ import annotations

import didactic.api as dx

from bead.config.active_learning import ActiveLearningConfig
from bead.config.config import BeadConfig
from bead.config.deployment import DeploymentConfig
from bead.config.item import ItemConfig
from bead.config.list import ListConfig
from bead.config.logging import LoggingConfig
from bead.config.paths import PathsConfig
from bead.config.resources import ResourceConfig
from bead.config.template import TemplateConfig

DEFAULT_CONFIG = BeadConfig(
    profile="default",
    paths=PathsConfig(),
    resources=ResourceConfig(),
    templates=TemplateConfig(),
    items=ItemConfig(),
    lists=ListConfig(),
    deployment=DeploymentConfig(),
    active_learning=ActiveLearningConfig(),
    logging=LoggingConfig(),
)
"""The default configuration instance."""


def get_default_config() -> BeadConfig:
    """Return the default ``BeadConfig``.

    didactic Models are frozen, so the returned instance is shared with
    ``DEFAULT_CONFIG``; callers may pass it to ``with_(...)`` to derive
    overrides without affecting other consumers.
    """
    return DEFAULT_CONFIG


def get_default_for_model[T: dx.Model](model_type: type[T]) -> T:
    """Return a default instance of the given didactic Model class."""
    return model_type()
