"""Logging configuration models for the bead package."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class LoggingConfig(BaseModel):
    """Configuration for logging.

    Parameters
    ----------
    level : str
        Log level.
    format : str
        Log format string.
    file : Path | None
        Log file path.
    console : bool
        Whether to log to console.

    Examples
    --------
    >>> config = LoggingConfig()
    >>> config.level
    'INFO'
    >>> config.console
    True
    """

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Log level"
    )
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format",
    )
    file: Path | None = Field(default=None, description="Log file path")
    console: bool = Field(default=True, description="Log to console")
