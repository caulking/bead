"""Manager for random effects in GLMM-based active learning.

Implements:
- Random effect storage and retrieval (intercepts and slopes)
- Variance component estimation (G matrix via MLE/REML)
- Empirical Bayes shrinkage for small groups
- Adaptive regularization based on sample counts
- Save/load with variance component history
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from bead.active_learning.config import MixedEffectsConfig, VarianceComponents

__all__ = ["RandomEffectsManager"]


class RandomEffectsManager:
    """Manages random effects following GLMM theory: u ~ N(0, G).

    Core responsibilities:
    1. Store random effect values: u_i for each participant i
    2. Estimate variance components: σ²_u (the G matrix)
    3. Implement shrinkage: u_shrunk_i = λ_i * u_i + (1-λ_i) * μ_0
    4. Compute prior loss: L_prior = λ * Σ_i w_i * ||u_i - μ_0||²
    5. Handle unknown participants: Use population mean (μ_0)

    Attributes
    ----------
    config : MixedEffectsConfig
        Configuration including mode, priors, regularization.
    intercepts : dict[str, torch.Tensor]
        Random intercepts per participant.
        Key: participant_id, Value: bias vector of shape (n_classes,)
    slopes : dict[str, nn.Module]
        Random slopes per participant.
        Key: participant_id, Value: model head (nn.Module)
    participant_sample_counts : dict[str, int]
        Training samples per participant (for adaptive regularization).
    variance_components : VarianceComponents | None
        Latest variance component estimates.
    variance_history : list[VarianceComponents]
        Variance components over training (for diagnostics).

    Examples
    --------
    >>> config = MixedEffectsConfig(mode='random_intercepts')
    >>> manager = RandomEffectsManager(config, n_classes=3)

    >>> # Register participants during training
    >>> manager.register_participant("alice", n_samples=10)
    >>> manager.register_participant("bob", n_samples=15)

    >>> # Get intercepts (creates if missing)
    >>> bias_alice = manager.get_intercepts("alice", n_classes=3)

    >>> # Estimate variance components after training
    >>> var_comp = manager.estimate_variance_components()
    >>> print(f"σ²_u = {var_comp.variance:.3f}")

    >>> # Compute prior loss for regularization
    >>> loss_prior = manager.compute_prior_loss()
    """

    def __init__(self, config: MixedEffectsConfig, **kwargs: Any) -> None:
        """Initialize random effects manager.

        Parameters
        ----------
        config : MixedEffectsConfig
            GLMM configuration.
        **kwargs : Any
            Additional arguments (e.g., n_classes, hidden_dim).
            Required arguments depend on mode.

        Raises
        ------
        ValueError
            If mode='random_slopes' but required kwargs missing.
        """
        self.config = config
        self.intercepts: dict[str, torch.Tensor] = {}
        self.slopes: dict[str, nn.Module] = {}
        self.participant_sample_counts: dict[str, int] = {}

        # NEW: Variance component tracking
        self.variance_components: VarianceComponents | None = None
        self.variance_history: list[VarianceComponents] = []

        # Store kwargs for creating new random effects
        self.creation_kwargs = kwargs

    def register_participant(self, participant_id: str, n_samples: int) -> None:
        """Register participant and track sample count.

        Used for:
        - Adaptive regularization (fewer samples → stronger regularization)
        - Shrinkage estimation (fewer samples → shrink toward mean)
        - Variance component estimation

        Parameters
        ----------
        participant_id : str
            Participant identifier.
        n_samples : int
            Number of samples for this participant.

        Raises
        ------
        ValueError
            If participant_id empty or n_samples not positive.

        Examples
        --------
        >>> manager.register_participant("alice", n_samples=10)
        >>> manager.register_participant("bob", n_samples=15)
        """
        if not participant_id:
            raise ValueError(
                "participant_id cannot be empty. "
                "Ensure all participants have valid string identifiers."
            )
        if n_samples <= 0:
            raise ValueError(
                f"n_samples must be positive, got {n_samples}. "
                f"Each participant must have at least 1 sample."
            )

        # Accumulate samples if participant seen before
        if participant_id in self.participant_sample_counts:
            self.participant_sample_counts[participant_id] += n_samples
        else:
            self.participant_sample_counts[participant_id] = n_samples

    def get_intercepts(
        self, participant_id: str, n_classes: int, create_if_missing: bool = True
    ) -> torch.Tensor:
        """Get random intercepts for participant.

        Behavior:
        - Known participant: Return learned intercepts
        - Unknown participant:
          - If create_if_missing=True: Initialize from prior N(μ_0, σ²_0)
          - If create_if_missing=False: Return zeros (prior mean)

        Parameters
        ----------
        participant_id : str
            Participant identifier.
        n_classes : int
            Number of classes (length of bias vector).
        create_if_missing : bool, default=True
            Whether to create new intercepts for unknown participants.
            True: Training (create new random effects)
            False: Prediction (use prior mean for unknown)

        Returns
        -------
        torch.Tensor
            Bias vector of shape (n_classes,).

        Raises
        ------
        ValueError
            If mode is not 'random_intercepts'.

        Examples
        --------
        >>> # Training: Create if missing
        >>> bias = manager.get_intercepts("alice", n_classes=3, create_if_missing=True)
        >>> bias.shape
        torch.Size([3])

        >>> # Prediction: Use prior for unknown
        >>> bias_new = manager.get_intercepts(
        ...     "unknown", n_classes=3, create_if_missing=False
        ... )
        >>> torch.allclose(bias_new, torch.zeros(3))
        True
        """
        if self.config.mode != "random_intercepts":
            raise ValueError(
                f"get_intercepts() called but mode is '{self.config.mode}', "
                f"expected 'random_intercepts'. "
                f"Use mode='random_intercepts' in MixedEffectsConfig."
            )

        # Known participant: return learned intercepts
        if participant_id in self.intercepts:
            return self.intercepts[participant_id]

        # Unknown participant: use prior mean
        if not create_if_missing:
            return torch.zeros(n_classes) + self.config.prior_mean

        # Create new intercepts from prior: u_i ~ N(μ_0, σ²_0)
        bias = (
            torch.randn(n_classes) * np.sqrt(self.config.prior_variance)
            + self.config.prior_mean
        )
        bias.requires_grad = True
        self.intercepts[participant_id] = bias
        return bias

    def get_intercepts_with_shrinkage(
        self, participant_id: str, n_classes: int
    ) -> torch.Tensor:
        """Get random intercepts with Empirical Bayes shrinkage.

        Implements shrinkage toward population mean:

            u_shrunk_i = λ_i * u_mle_i + (1 - λ_i) * μ_0

        where:
            λ_i = n_i / (n_i + k)
            k ≈ σ²_ε / σ²_u  (ratio of residual to random effect variance)

        For participants with few samples, shrink toward μ_0 (population mean).
        For participants with many samples, use their specific estimate.

        Parameters
        ----------
        participant_id : str
            Participant identifier.
        n_classes : int
            Number of classes.

        Returns
        -------
        torch.Tensor
            Shrunk bias vector of shape (n_classes,).

        Examples
        --------
        >>> # Participant with 2 samples → strong shrinkage
        >>> manager.register_participant("alice", n_samples=2)
        >>> bias_shrunk = manager.get_intercepts_with_shrinkage("alice", 3)

        >>> # Participant with 100 samples → little shrinkage
        >>> manager.register_participant("bob", n_samples=100)
        >>> bias_shrunk_bob = manager.get_intercepts_with_shrinkage("bob", 3)
        """
        if self.config.mode != "random_intercepts":
            raise ValueError(
                f"Shrinkage only for random_intercepts mode, got '{self.config.mode}'"
            )

        # Get MLE estimate (or prior if unknown)
        u_mle = self.get_intercepts(participant_id, n_classes, create_if_missing=False)

        # Unknown participant: return prior mean (no shrinkage needed)
        if participant_id not in self.intercepts:
            return u_mle

        # Compute shrinkage factor λ_i
        n_i = self.participant_sample_counts.get(participant_id, 1)

        # Estimate k from variance components if available
        if self.variance_components is not None:
            sigma2_u = self.variance_components.variance
            # Estimate σ²_ε from residuals (simplified: assume σ²_ε ≈ 1)
            sigma2_epsilon = 1.0
            k = sigma2_epsilon / max(sigma2_u, 1e-6)
        else:
            # Fallback: use min_samples as proxy for k
            k = self.config.min_samples_for_random_effects

        lambda_i = n_i / (n_i + k)

        # Shrinkage: u_shrunk = λ * u_mle + (1-λ) * μ_0
        mu_0 = self.config.prior_mean
        u_shrunk = lambda_i * u_mle + (1 - lambda_i) * mu_0

        return u_shrunk

    def get_slopes(
        self,
        participant_id: str,
        fixed_head: nn.Module,
        create_if_missing: bool = True,
    ) -> nn.Module:
        """Get random slopes (model head) for participant.

        Behavior:
        - Known participant: Return learned head
        - Unknown participant:
          - If create_if_missing=True: Clone fixed_head and add noise
          - If create_if_missing=False: Return clone of fixed_head

        Parameters
        ----------
        participant_id : str
            Participant identifier.
        fixed_head : nn.Module
            Fixed effects head to clone for new participants.
        create_if_missing : bool, default=True
            Whether to create new slopes for unknown participants.

        Returns
        -------
        nn.Module
            Model head for this participant.

        Raises
        ------
        ValueError
            If mode is not 'random_slopes'.

        Examples
        --------
        >>> fixed_head = nn.Linear(768, 3)
        >>> # Training: Create participant-specific head
        >>> head_alice = manager.get_slopes("alice", fixed_head, create_if_missing=True)

        >>> # Prediction: Use fixed head for unknown
        >>> head_unknown = manager.get_slopes(
        ...     "unknown", fixed_head, create_if_missing=False
        ... )
        """
        if self.config.mode != "random_slopes":
            raise ValueError(
                f"get_slopes() called but mode is '{self.config.mode}', "
                f"expected 'random_slopes'"
            )

        # Known participant: return learned slopes
        if participant_id in self.slopes:
            return self.slopes[participant_id]

        # Unknown participant: return clone of fixed head
        if not create_if_missing:
            return copy.deepcopy(fixed_head)

        # Create new slopes: φ_i = θ + noise
        # Clone fixed head and add Gaussian noise to parameters
        participant_head = copy.deepcopy(fixed_head)

        with torch.no_grad():
            for param in participant_head.parameters():
                noise = torch.randn_like(param) * np.sqrt(self.config.prior_variance)
                param.add_(noise)

        self.slopes[participant_id] = participant_head
        return participant_head

    def estimate_variance_components(self) -> VarianceComponents | None:
        """Estimate variance components (G matrix) from random effects.

        Implements Maximum Likelihood Estimation (MLE):

            σ²_u = Var(u_i) across all participants

        For random intercepts:
            σ²_u = (1/K) Σ_i ||u_i - mean(u)||²

        For random slopes:
            σ²_u = (1/K) Σ_i ||φ_i - mean(φ)||²

        Returns variance component estimates for:
        - Model diagnostics (how much participant variation?)
        - Shrinkage estimation (λ_i depends on σ²_u)
        - Model comparison (compare σ²_u across models)

        Returns
        -------
        VarianceComponents | None
            Variance component estimates, or None if mode='fixed'.

        Examples
        --------
        >>> # After training with random intercepts
        >>> var_comp = manager.estimate_variance_components()
        >>> print(f"Participant variance: {var_comp.variance:.3f}")
        >>> print(f"Number of participants: {var_comp.n_groups}")

        >>> # Interpret variance:
        >>> # σ²_u = 0.01 → participants very similar
        >>> # (little benefit from mixed effects)
        >>> # σ²_u = 1.00 → substantial participant variation (mixed effects helpful)
        >>> # σ²_u = 10.0 → large participant differences (mixed effects essential)
        """
        if self.config.mode == "fixed":
            return None

        if self.config.mode == "random_intercepts":
            if not self.intercepts:
                return None

            # Stack all intercepts: shape (n_participants, n_classes)
            all_intercepts = torch.stack(list(self.intercepts.values()))

            # Compute variance across participants
            # MLE estimate: σ²_u = Var(u_i)
            variance = torch.var(all_intercepts, unbiased=True).item()

            var_comp = VarianceComponents(
                grouping_factor="participant",
                effect_type="intercept",
                variance=variance,
                n_groups=len(self.intercepts),
                n_observations_per_group=self.participant_sample_counts.copy(),
            )

            # Update tracking
            self.variance_components = var_comp
            self.variance_history.append(var_comp)

            return var_comp

        elif self.config.mode == "random_slopes":
            if not self.slopes:
                return None

            # Compute variance of parameters across participants
            # Flatten all parameters from all participant heads
            all_params: list[torch.Tensor] = []
            for head in self.slopes.values():
                params_flat = torch.cat([p.flatten() for p in head.parameters()])
                all_params.append(params_flat)

            all_params_tensor = torch.stack(all_params)
            variance = torch.var(all_params_tensor, unbiased=True).item()

            var_comp = VarianceComponents(
                grouping_factor="participant",
                effect_type="slope",
                variance=variance,
                n_groups=len(self.slopes),
                n_observations_per_group=self.participant_sample_counts.copy(),
            )

            self.variance_components = var_comp
            self.variance_history.append(var_comp)

            return var_comp

        return None

    def compute_prior_loss(self) -> torch.Tensor:
        """Compute regularization loss toward prior.

        Implements adaptive regularization:

            L_prior = λ * Σ_i w_i * ||u_i - μ_0||²

        where:
            w_i = 1 / max(n_i, min_samples)  (adaptive weighting)
            λ = regularization_strength

        Participants with fewer samples get stronger regularization.
        This prevents overfitting when participant has little data.

        Returns
        -------
        torch.Tensor
            Scalar regularization loss to add to training loss.

        Examples
        --------
        >>> # During training:
        >>> loss_data = cross_entropy(logits, labels)
        >>> loss_prior = manager.compute_prior_loss()
        >>> loss_total = loss_data + loss_prior
        >>> loss_total.backward()
        """
        if self.config.mode == "fixed":
            return torch.tensor(0.0)

        loss = torch.tensor(0.0)

        if self.config.mode == "random_intercepts":
            for participant_id, bias in self.intercepts.items():
                # Deviation from prior mean
                deviation = bias - self.config.prior_mean
                squared_dev = torch.sum(deviation**2)

                # Adaptive weight: stronger for participants with fewer samples
                if self.config.adaptive_regularization:
                    n_samples = self.participant_sample_counts.get(participant_id, 1)
                    weight = 1.0 / max(
                        n_samples, self.config.min_samples_for_random_effects
                    )
                else:
                    weight = 1.0

                loss += weight * squared_dev

        elif self.config.mode == "random_slopes":
            for participant_id, head in self.slopes.items():
                # Sum squared parameters (deviation from 0)
                squared_dev = sum(torch.sum(param**2) for param in head.parameters())

                # Adaptive weight
                if self.config.adaptive_regularization:
                    n_samples = self.participant_sample_counts.get(participant_id, 1)
                    weight = 1.0 / max(
                        n_samples, self.config.min_samples_for_random_effects
                    )
                else:
                    weight = 1.0

                loss += weight * squared_dev

        return self.config.regularization_strength * loss

    def save(self, path: Path) -> None:
        """Save random effects to disk.

        Saves:
        - intercepts.pt (if mode='random_intercepts')
        - slopes.pt (if mode='random_slopes')
        - sample_counts.json
        - variance_history.json (NEW: for diagnostics)

        Parameters
        ----------
        path : Path
            Directory to save random effects.

        Examples
        --------
        >>> manager.save(Path("model_checkpoint/random_effects"))
        """
        path.mkdir(parents=True, exist_ok=True)

        # Save intercepts
        if self.config.mode == "random_intercepts" and self.intercepts:
            torch.save(self.intercepts, path / "intercepts.pt")

        # Save slopes
        if self.config.mode == "random_slopes" and self.slopes:
            slopes_state = {pid: head.state_dict() for pid, head in self.slopes.items()}
            torch.save(slopes_state, path / "slopes.pt")

        # Save sample counts
        with open(path / "sample_counts.json", "w") as f:
            json.dump(self.participant_sample_counts, f)

        # NEW: Save variance component history
        if self.variance_history:
            variance_data = [vc.model_dump() for vc in self.variance_history]
            with open(path / "variance_history.json", "w") as f:
                json.dump(variance_data, f, indent=2)

    def load(self, path: Path, fixed_head: nn.Module | None = None) -> None:
        """Load random effects from disk.

        Parameters
        ----------
        path : Path
            Directory to load from.
        fixed_head : nn.Module | None
            Fixed head (required if mode='random_slopes').

        Raises
        ------
        FileNotFoundError
            If path doesn't exist.
        ValueError
            If mode='random_slopes' but fixed_head is None.

        Examples
        --------
        >>> manager.load(Path("model_checkpoint/random_effects"))
        """
        if not path.exists():
            raise FileNotFoundError(f"Random effects directory not found: {path}")

        # Load intercepts
        if self.config.mode == "random_intercepts":
            intercepts_path = path / "intercepts.pt"
            if intercepts_path.exists():
                self.intercepts = torch.load(intercepts_path)

        # Load slopes
        if self.config.mode == "random_slopes":
            if fixed_head is None:
                raise ValueError(
                    "fixed_head is required when loading random slopes. "
                    "Pass the fixed effects head to load()."
                )

            slopes_path = path / "slopes.pt"
            if slopes_path.exists():
                slopes_state = torch.load(slopes_path)
                self.slopes = {}
                for pid, state_dict in slopes_state.items():
                    head = copy.deepcopy(fixed_head)
                    head.load_state_dict(state_dict)
                    self.slopes[pid] = head

        # Load sample counts
        sample_counts_path = path / "sample_counts.json"
        if sample_counts_path.exists():
            with open(sample_counts_path) as f:
                self.participant_sample_counts = json.load(f)

        # NEW: Load variance history
        variance_history_path = path / "variance_history.json"
        if variance_history_path.exists():
            with open(variance_history_path) as f:
                variance_data = json.load(f)
                self.variance_history = [
                    VarianceComponents(**vc) for vc in variance_data
                ]
                if self.variance_history:
                    self.variance_components = self.variance_history[-1]
