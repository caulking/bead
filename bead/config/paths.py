"""Path configuration models."""

from __future__ import annotations

import didactic.api as dx


class PathsConfig(dx.Model):
    """Configuration for filesystem paths.

    Attributes
    ----------
    data_dir : str
        Base directory for data files.
    output_dir : str
        Base directory for outputs.
    cache_dir : str
        Cache directory.
    temp_dir : str | None
        Temporary directory; ``None`` defers to the system default.
    create_dirs : bool
        Whether to create directories if they don't exist.
    """

    data_dir: str = "data"
    output_dir: str = "output"
    cache_dir: str = ".cache"
    temp_dir: str | None = None
    create_dirs: bool = True
