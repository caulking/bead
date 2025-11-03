"""Item configuration models for the bead package."""

from __future__ import annotations

from pydantic import BaseModel, Field

from bead.config.model import ModelConfig


class ItemConfig(BaseModel):
    """Configuration for item generation.

    Parameters
    ----------
    model : ModelConfig
        Model configuration.
    apply_constraints : bool
        Whether to apply model-based constraints.
    track_metadata : bool
        Whether to track item metadata.
    parallel_processing : bool
        Whether to use parallel processing.
    num_workers : int
        Number of workers for parallel processing.

    Examples
    --------
    >>> config = ItemConfig()
    >>> config.apply_constraints
    True
    >>> config.num_workers
    4
    """

    model: ModelConfig = Field(
        default_factory=ModelConfig, description="Model configuration"
    )
    apply_constraints: bool = Field(
        default=True, description="Apply model-based constraints"
    )
    track_metadata: bool = Field(default=True, description="Track item metadata")
    parallel_processing: bool = Field(
        default=False, description="Use parallel processing"
    )
    num_workers: int = Field(default=4, description="Number of workers", gt=0)
