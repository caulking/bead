"""Simulation configuration models for the bead package."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class NoiseModelConfig(BaseModel):
    """Configuration for noise model in simulated judgments.

    Attributes
    ----------
    noise_type : Literal["temperature", "systematic", "random", "none"]
        Type of noise to apply.
    temperature : float
        Temperature for scaling (higher = more random). Default: 1.0.
    bias_strength : float
        Strength of systematic biases (0.0-1.0). Default: 0.0.
    bias_type : str | None
        Type of bias ("length", "frequency", "position"). Default: None.
    random_noise_stddev : float
        Standard deviation for random noise. Default: 0.0.

    Examples
    --------
    >>> # Temperature-scaled decisions (more random)
    >>> config = NoiseModelConfig(noise_type="temperature", temperature=2.0)
    >>>
    >>> # Systematic length bias (prefer shorter)
    >>> config = NoiseModelConfig(
    ...     noise_type="systematic",
    ...     bias_strength=0.3,
    ...     bias_type="length"
    ... )
    >>>
    >>> # Random noise injection
    >>> config = NoiseModelConfig(
    ...     noise_type="random",
    ...     random_noise_stddev=0.1
    ... )
    """

    noise_type: Literal["temperature", "systematic", "random", "none"] = Field(
        default="temperature",
        description="Type of noise model",
    )
    temperature: float = Field(
        default=1.0,
        ge=0.01,
        le=10.0,
        description="Temperature for scaling decisions",
    )
    bias_strength: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Strength of systematic biases",
    )
    bias_type: str | None = Field(
        default=None,
        description="Type of systematic bias",
    )
    random_noise_stddev: float = Field(
        default=0.0,
        ge=0.0,
        description="Standard deviation for random noise",
    )


class SimulatedAnnotatorConfig(BaseModel):
    """Configuration for simulated annotator.

    Attributes
    ----------
    strategy : Literal["lm_score", "distance", "random", "oracle", "dsl"]
        Base strategy for generating judgments.
    noise_model : NoiseModelConfig
        Noise model configuration.
    dsl_expression : str | None
        Custom DSL expression for simulation logic.
    random_state : int | None
        Random seed for reproducibility.
    model_output_key : str
        Key to extract from Item.model_outputs. Default: "lm_score".
    fallback_to_random : bool
        Whether to fallback to random if model outputs missing. Default: True.

    Examples
    --------
    >>> # LM score-based with temperature
    >>> config = SimulatedAnnotatorConfig(
    ...     strategy="lm_score",
    ...     noise_model=NoiseModelConfig(noise_type="temperature", temperature=1.5),
    ...     random_state=42
    ... )
    >>>
    >>> # Distance-based with embeddings
    >>> config = SimulatedAnnotatorConfig(
    ...     strategy="distance",
    ...     model_output_key="embedding",
    ...     noise_model=NoiseModelConfig(noise_type="none")
    ... )
    >>>
    >>> # Custom DSL logic
    >>> config = SimulatedAnnotatorConfig(
    ...     strategy="dsl",
    ...     dsl_expression="sample_categorical(softmax(model_scores) / temperature)",
    ...     noise_model=NoiseModelConfig(noise_type="temperature", temperature=1.0)
    ... )
    """

    strategy: Literal["lm_score", "distance", "random", "oracle", "dsl"] = Field(
        default="lm_score",
        description="Base simulation strategy",
    )
    noise_model: NoiseModelConfig = Field(
        default_factory=NoiseModelConfig,
        description="Noise model configuration",
    )
    dsl_expression: str | None = Field(
        default=None,
        description="Custom DSL expression for simulation",
    )
    random_state: int | None = Field(
        default=None,
        description="Random seed for reproducibility",
    )
    model_output_key: str = Field(
        default="lm_score",
        description="Key to extract from model outputs",
    )
    fallback_to_random: bool = Field(
        default=True,
        description="Fallback to random if model outputs missing",
    )


class SimulationRunnerConfig(BaseModel):
    """Configuration for simulation runner.

    Attributes
    ----------
    annotator_configs : list[SimulatedAnnotatorConfig]
        List of annotator configurations (for multi-annotator simulation).
    n_annotators : int
        Number of simulated annotators. Default: 1.
    inter_annotator_correlation : float | None
        Desired correlation between annotators (0.0-1.0). Default: None (independent).
    output_format : Literal["dict", "dataframe", "jsonl"]
        Output format for simulation results. Default: "dict".
    save_path : Path | None
        Path to save simulation results. Default: None.

    Examples
    --------
    >>> # Single annotator
    >>> config = SimulationRunnerConfig(
    ...     annotator_configs=[SimulatedAnnotatorConfig(strategy="lm_score")],
    ...     n_annotators=1
    ... )
    >>>
    >>> # Multiple independent annotators
    >>> config = SimulationRunnerConfig(
    ...     annotator_configs=[
    ...         SimulatedAnnotatorConfig(strategy="lm_score", random_state=1),
    ...         SimulatedAnnotatorConfig(strategy="lm_score", random_state=2),
    ...         SimulatedAnnotatorConfig(strategy="lm_score", random_state=3)
    ...     ],
    ...     n_annotators=3
    ... )
    >>>
    >>> # Correlated annotators
    >>> config = SimulationRunnerConfig(
    ...     annotator_configs=[SimulatedAnnotatorConfig(strategy="lm_score")],
    ...     n_annotators=5,
    ...     inter_annotator_correlation=0.7  # 70% agreement
    ... )
    """

    annotator_configs: list[SimulatedAnnotatorConfig] = Field(
        default_factory=lambda: [SimulatedAnnotatorConfig()],
        description="Annotator configurations",
    )
    n_annotators: int = Field(
        default=1,
        ge=1,
        le=100,
        description="Number of simulated annotators",
    )
    inter_annotator_correlation: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Inter-annotator correlation",
    )
    output_format: Literal["dict", "dataframe", "jsonl"] = Field(
        default="dict",
        description="Output format",
    )
    save_path: Path | None = Field(
        default=None,
        description="Path to save results",
    )
