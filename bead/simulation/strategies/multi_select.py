"""Multi-select simulation strategy."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from bead.simulation.strategies.base import SimulationStrategy

if TYPE_CHECKING:
    from bead.items.item import Item
    from bead.items.item_template import ItemTemplate


class MultiSelectStrategy(SimulationStrategy):
    """Strategy for multi_select tasks.

    Handles tasks where multiple options can be selected independently.
    Uses model outputs to compute independent selection probabilities
    for each option via sigmoid.

    For each option i:
        P(select option i) = sigmoid(score_i / temperature)

    Parameters
    ----------
    threshold
        Probability threshold for selection. Default: 0.5.
    temperature
        Temperature for scaling decisions. Default: 1.0.

    Examples
    --------
    >>> strategy = MultiSelectStrategy()
    >>> strategy.supported_task_type
    'multi_select'
    """

    def __init__(self, threshold: float = 0.5, temperature: float = 1.0) -> None:
        self.threshold = threshold
        self.temperature = temperature

    @property
    def supported_task_type(self) -> str:
        """Return 'multi_select'."""
        return "multi_select"

    def validate_item(self, item: Item, item_template: ItemTemplate) -> None:
        """Validate item for multi-select.

        Checks:
        - task_type is 'multi_select'
        - task_spec.options is defined
        - At least 2 options

        Parameters
        ----------
        item : Item
            Item to validate.
        item_template : ItemTemplate
            Template defining task.

        Raises
        ------
        ValueError
            If validation fails.
        """
        if item_template.task_type != "multi_select":
            msg = f"Expected task_type 'multi_select', got '{item_template.task_type}'"
            raise ValueError(msg)

        if not item_template.task_spec.options:
            raise ValueError("task_spec.options must be defined for multi_select")

        if len(item_template.task_spec.options) < 2:
            raise ValueError("multi_select requires at least 2 options")

    def simulate_response(
        self,
        item: Item,
        item_template: ItemTemplate,
        model_output_key: str,
        rng: np.random.RandomState,
    ) -> list[str]:
        """Generate multi-select response.

        Parameters
        ----------
        item : Item
            Item to respond to.
        item_template : ItemTemplate
            Template defining task.
        model_output_key : str
            Key for model outputs (e.g., "lm_score").
        rng : np.random.RandomState
            Random number generator.

        Returns
        -------
        list[str]
            List of selected option names.
        """
        options = item_template.task_spec.options
        assert options is not None, "options validated in validate()"
        n_options = len(options)

        # extract model outputs for each option
        scores = self.extract_model_outputs(item, model_output_key, n_options)

        if scores is None:
            # fallback to random selection (each option has threshold probability)
            selected = []
            for option in options:
                if rng.random() < self.threshold:
                    selected.append(option)
            return selected

        # compute selection probability for each option using sigmoid
        selected = []
        for option, score in zip(options, scores, strict=True):
            # sigmoid(score / temperature)
            prob = 1.0 / (1.0 + np.exp(-score / self.temperature))

            # sample selection
            if rng.random() < prob:
                selected.append(option)

        return selected
