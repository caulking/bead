"""Configuration models for mixed effects active learning.

Separated from base.py to avoid circular imports.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

__all__ = [
    "VarianceComponents",
    "RandomEffectsSpec",
    "MixedEffectsConfig",
]


class VarianceComponents(BaseModel):
    """Variance-covariance structure for random effects (G matrix in GLMM theory).

    Tracks estimated variances for random effects, enabling:
    - Shrinkage estimation (groups with few samples → prior mean)
    - Model diagnostics (proportion of variance explained by random effects)
    - Uncertainty quantification

    In GLMM notation: u ~ N(0, G), where G is the variance-covariance matrix.
    For random intercepts, G is diagonal with entries σ²_u.
    For random slopes, G can be full (correlated) or diagonal (independent).

    Attributes
    ----------
    grouping_factor : str
        Name of grouping factor (e.g., "participant", "item", "lab").
    effect_type : Literal["intercept", "slope"]
        Type of random effect.
    variance : float
        Estimated variance σ² for this random effect.
        Higher values indicate more heterogeneity across groups.
    n_groups : int
        Number of groups (e.g., 50 participants).
    n_observations_per_group : dict[str, int]
        Number of observations per group.
        Used for adaptive regularization and shrinkage.

    Examples
    --------
    >>> vc = VarianceComponents(
    ...     grouping_factor="participant",
    ...     effect_type="intercept",
    ...     variance=0.25,
    ...     n_groups=50,
    ...     n_observations_per_group={"p1": 10, "p2": 15}
    ... )
    >>> vc.variance
    0.25
    """

    grouping_factor: str = Field(description="Grouping factor name")
    effect_type: Literal["intercept", "slope"] = Field(
        description="Type of random effect"
    )
    variance: float = Field(
        ge=0.0, description="Estimated variance for this random effect"
    )
    n_groups: int = Field(ge=1, description="Number of groups")
    n_observations_per_group: dict[str, int] = Field(
        description="Observations per group"
    )


class RandomEffectsSpec(BaseModel):
    """Specification of random effects structure.

    Inspired by lme4 formula notation: (expr | factor).

    Phase 5 (current): Supports single grouping factor (participant).
    Future phases: Multiple factors, crossed/nested structure.

    Attributes
    ----------
    grouping_factors : dict[str, Literal["intercept", "slope", "both"]]
        Mapping from grouping factor name to effect type.
        Phase 5: {"participant": "intercept"} or {"participant": "slope"}
        Future: {"participant": "intercept", "item": "intercept"}
    correlation_structure : Literal["independent", "correlated"]
        If "both" specified: whether intercept and slope are correlated.
        Independent: G is diagonal.
        Correlated: G has off-diagonal covariances.

    Examples
    --------
    >>> # Random intercepts for participants
    >>> spec = RandomEffectsSpec(
    ...     grouping_factors={"participant": "intercept"}
    ... )

    >>> # Random slopes for participants
    >>> spec = RandomEffectsSpec(
    ...     grouping_factors={"participant": "slope"}
    ... )

    >>> # Future: Multiple grouping factors
    >>> spec = RandomEffectsSpec(
    ...     grouping_factors={"participant": "intercept", "item": "intercept"}
    ... )
    """

    grouping_factors: dict[str, Literal["intercept", "slope", "both"]] = Field(
        description="Grouping factors and their random effect types"
    )
    correlation_structure: Literal["independent", "correlated"] = Field(
        default="independent",
        description="Correlation structure for 'both' case",
    )


class MixedEffectsConfig(BaseModel):
    """Configuration for mixed effects modeling in active learning.

    Based on GLMM theory: y = Xβ + Zu + ε

    Where:
    - Xβ: Fixed effects (population-level parameters, shared across all groups)
    - Zu: Random effects (group-specific parameters, e.g., per-participant)
    - u ~ N(0, G): Random effects with variance-covariance matrix G
    - ε ~ N(0, σ²): Residuals

    Attributes
    ----------
    mode : Literal['fixed', 'random_intercepts', 'random_slopes']
        Modeling mode:
        - 'fixed': Standard model, no group-specific parameters (Z = 0)
        - 'random_intercepts': Per-group biases (Z = I, u = bias vectors)
        - 'random_slopes': Per-group model parameters (Z = I, u = full model heads)
    prior_mean : float
        Mean μ₀ of Gaussian prior for random effects initialization.
        Random effects initialized from N(μ₀, σ²₀).
    prior_variance : float
        Variance σ²₀ of Gaussian prior for random effects initialization.
        Controls initial spread of random effects.
    estimate_variance_components : bool
        Whether to estimate variance components (G matrix) during training.
        If True, returns variance estimates in training metrics.
    variance_estimation_method : Literal["mle", "reml"]
        Method for variance component estimation:
        - 'mle': Maximum Likelihood Estimation
        - 'reml': Restricted Maximum Likelihood (adjusts for fixed effects)
    regularization_strength : float
        Strength λ of regularization pulling random effects toward prior.
        Loss: L_total = L_data + λ * ||u - μ₀||²
    adaptive_regularization : bool
        If True, use stronger regularization for groups with fewer samples.
        Weight: w_g = 1 / max(n_g, min_samples_for_random_effects)
    min_samples_for_random_effects : int
        Minimum samples before estimating group-specific random effects.
        Below threshold: use prior mean for predictions (shrinkage).
    random_effects_spec : RandomEffectsSpec | None
        Advanced: Specification for multiple grouping factors.
        If None: infer from mode (backward compatibility).
        Future: Enable item random effects, crossed effects, etc.

    Examples
    --------
    >>> # Fixed effects (standard model)
    >>> config = MixedEffectsConfig(mode='fixed')

    >>> # Random intercepts (participant biases)
    >>> config = MixedEffectsConfig(
    ...     mode='random_intercepts',
    ...     prior_mean=0.0,
    ...     prior_variance=1.0,
    ...     regularization_strength=0.01
    ... )

    >>> # Random slopes (participant-specific models)
    >>> config = MixedEffectsConfig(
    ...     mode='random_slopes',
    ...     prior_variance=0.1,
    ...     adaptive_regularization=True
    ... )
    """

    mode: Literal["fixed", "random_intercepts", "random_slopes"] = Field(
        default="fixed",
        description="Modeling mode: fixed, random_intercepts, or random_slopes",
    )

    # Prior hyperparameters (for initialization)
    prior_mean: float = Field(
        default=0.0, description="Mean of Gaussian prior for random effects"
    )
    prior_variance: float = Field(
        default=1.0,
        ge=0.0,
        description="Variance of Gaussian prior for random effects (must be >= 0)",
    )

    # Variance component estimation
    estimate_variance_components: bool = Field(
        default=True,
        description=(
            "Whether to estimate variance components (G matrix) during training"
        ),
    )
    variance_estimation_method: Literal["mle", "reml"] = Field(
        default="mle",
        description="Method for variance component estimation: mle or reml",
    )

    # Regularization (pulls random effects toward prior)
    regularization_strength: float = Field(
        default=0.01,
        ge=0.0,
        description="Strength of regularization toward prior (must be >= 0)",
    )
    adaptive_regularization: bool = Field(
        default=True,
        description="Use stronger regularization for groups with fewer samples",
    )
    min_samples_for_random_effects: int = Field(
        default=5,
        ge=1,
        description="Minimum samples before using random effects (must be >= 1)",
    )

    # Extensibility (Phase 6+)
    random_effects_spec: RandomEffectsSpec | None = Field(
        default=None,
        description="Advanced: specification for multiple grouping factors (Phase 6+)",
    )
