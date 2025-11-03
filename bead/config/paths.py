"""Path configuration models for the bead package."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class PathsConfig(BaseModel):
    """Configuration for file system paths.

    Parameters
    ----------
    data_dir : Path
        Base directory for data files.
    output_dir : Path
        Base directory for outputs.
    cache_dir : Path
        Cache directory.
    temp_dir : Path | None
        Temporary directory. If None, uses system temp.
    create_dirs : bool
        Whether to create directories if they don't exist.

    Examples
    --------
    >>> config = PathsConfig()
    >>> config.data_dir
    PosixPath('data')
    >>> config = PathsConfig(data_dir=Path("/absolute/path"))
    >>> config.data_dir
    PosixPath('/absolute/path')
    """

    data_dir: Path = Field(
        default=Path("data"), description="Base directory for data files"
    )
    output_dir: Path = Field(
        default=Path("output"), description="Base directory for outputs"
    )
    cache_dir: Path = Field(default=Path(".cache"), description="Cache directory")
    temp_dir: Path | None = Field(default=None, description="Temporary directory")
    create_dirs: bool = Field(
        default=True, description="Create directories if they don't exist"
    )
