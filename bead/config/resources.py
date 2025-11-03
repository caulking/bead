"""Resource configuration models for the bead package."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class ResourceConfig(BaseModel):
    """Configuration for external resources.

    Parameters
    ----------
    lexicon_path : Path | None
        Path to lexicon file.
    templates_path : Path | None
        Path to templates file.
    constraints_path : Path | None
        Path to constraints file.
    external_adapters : list[str]
        List of external adapters to enable.
    cache_external : bool
        Whether to cache external resource lookups.

    Examples
    --------
    >>> config = ResourceConfig()
    >>> config.cache_external
    True
    >>> config.external_adapters
    []
    """

    lexicon_path: Path | None = Field(default=None, description="Path to lexicon file")
    templates_path: Path | None = Field(
        default=None, description="Path to templates file"
    )
    constraints_path: Path | None = Field(
        default=None, description="Path to constraints file"
    )
    external_adapters: list[str] = Field(
        default_factory=list, description="External adapters to enable"
    )
    cache_external: bool = Field(
        default=True, description="Cache external resource lookups"
    )
