"""Model configuration models for the bead package."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Configuration for language models.

    Parameters
    ----------
    provider : str
        Model provider name.
    model_name : str
        Model identifier.
    batch_size : int
        Inference batch size.
    device : str
        Device to use for computation.
    max_length : int
        Maximum sequence length.
    temperature : float
        Sampling temperature.
    cache_outputs : bool
        Whether to cache model outputs.

    Examples
    --------
    >>> config = ModelConfig()
    >>> config.provider
    'huggingface'
    >>> config.device
    'cpu'
    """

    provider: Literal["huggingface", "openai", "anthropic"] = Field(
        default="huggingface", description="Model provider"
    )
    model_name: str = Field(default="gpt2", description="Model identifier")
    batch_size: int = Field(default=8, description="Inference batch size", gt=0)
    device: Literal["cpu", "cuda", "mps"] = Field(
        default="cpu", description="Device to use"
    )
    max_length: int = Field(default=512, description="Max sequence length", gt=0)
    temperature: float = Field(default=1.0, description="Sampling temperature", ge=0)
    cache_outputs: bool = Field(default=True, description="Cache model outputs")
