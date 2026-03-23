"""Temperature-based noise model."""

from __future__ import annotations

import numpy as np

from bead.simulation.noise_models.base import NoiseModel


class TemperatureNoiseModel(NoiseModel):
    """Temperature scaling for probability distributions.

    Scales logits or probabilities by temperature before sampling:
        - temperature < 1.0: More deterministic (sharper distribution)
        - temperature = 1.0: No change
        - temperature > 1.0: More random (flatter distribution)

    For forced choice, modifies the softmax:
        P_i = exp(score_i / T) / sum(exp(score_j / T))

    Parameters
    ----------
    temperature
        Temperature scaling factor (> 0). Default: 1.0.

    Raises
    ------
    ValueError
        If temperature <= 0.

    Examples
    --------
    >>> noise_model = TemperatureNoiseModel(temperature=2.0)
    >>> # More random decisions
    """

    def __init__(self, temperature: float = 1.0) -> None:
        if temperature <= 0:
            msg = "Temperature must be positive"
            raise ValueError(msg)
        self.temperature = temperature

    def apply(
        self,
        value: str | int | float | list[str],
        context: dict[str, str | int | float | bool | list[str]],
        rng: np.random.RandomState,
    ) -> str | int | float | list[str]:
        """Apply temperature scaling.

        For forced_choice, re-samples with scaled probabilities.
        For ordinal_scale, adds scaled noise to value.

        Parameters
        ----------
        value : str | int | float | list[str]
            Original value (choice, rating, etc.).
        context : dict[str, str | int | float | bool | list[str]]
            Context with item, template, strategy.
        rng : np.random.RandomState
            Random number generator.

        Returns
        -------
        str | int | float | list[str]
            Value with temperature applied.
        """
        strategy = context.get("strategy")

        if strategy and hasattr(strategy, "supported_task_type"):
            task_type = strategy.supported_task_type

            if task_type == "forced_choice":
                # for forced choice, temperature is already handled in strategy
                # by applying it to the softmax computation; return value as-is
                return value

            elif task_type == "ordinal_scale":
                # for ordinal, add temperature-scaled gaussian noise
                if isinstance(value, int | float):
                    noise = rng.normal(0, self.temperature * 0.5)
                    return value + noise
                return value

        # default: no modification
        return value
