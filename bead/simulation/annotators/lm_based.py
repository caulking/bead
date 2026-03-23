"""LM score-based annotator."""

from __future__ import annotations

from typing import TYPE_CHECKING

from bead.simulation.annotators.base import SimulatedAnnotator
from bead.simulation.noise_models.temperature import TemperatureNoiseModel
from bead.simulation.strategies.binary import BinaryStrategy
from bead.simulation.strategies.categorical import CategoricalStrategy
from bead.simulation.strategies.cloze import ClozeStrategy
from bead.simulation.strategies.forced_choice import ForcedChoiceStrategy
from bead.simulation.strategies.free_text import FreeTextStrategy
from bead.simulation.strategies.magnitude import MagnitudeStrategy
from bead.simulation.strategies.multi_select import MultiSelectStrategy
from bead.simulation.strategies.ordinal_scale import OrdinalScaleStrategy

if TYPE_CHECKING:
    from bead.config.simulation import SimulatedAnnotatorConfig
    from bead.items.item import Item
    from bead.items.item_template import ItemTemplate


class LMBasedAnnotator(SimulatedAnnotator):
    """Annotator using language model scores for decisions.

    Uses LM log probabilities or scores from Item.model_outputs
    to make informed decisions. Applies noise model for variability.

    Supports all task types via pluggable strategies.

    Parameters
    ----------
    config
        Configuration for annotator.

    Examples
    --------
    >>> from bead.config.simulation import SimulatedAnnotatorConfig, NoiseModelConfig
    >>> config = SimulatedAnnotatorConfig(
    ...     strategy="lm_score",
    ...     model_output_key="lm_score",
    ...     noise_model=NoiseModelConfig(noise_type="temperature", temperature=1.5)
    ... )
    >>> annotator = LMBasedAnnotator(config)
    >>> # judgment = annotator.annotate(item, template)
    """

    def __init__(self, config: SimulatedAnnotatorConfig) -> None:
        super().__init__(config)

        # initialize strategies for different task types
        self.strategies = {
            "forced_choice": ForcedChoiceStrategy(),
            "binary": BinaryStrategy(),
            "ordinal_scale": OrdinalScaleStrategy(),
            "categorical": CategoricalStrategy(),
            "magnitude": MagnitudeStrategy(),
            "multi_select": MultiSelectStrategy(),
            "free_text": FreeTextStrategy(),
            "cloze": ClozeStrategy(),
        }

        # initialize noise model
        if config.noise_model.noise_type == "temperature":
            self.noise_model = TemperatureNoiseModel(
                temperature=config.noise_model.temperature
            )
        elif config.noise_model.noise_type == "none":
            self.noise_model = None
        else:
            # default: no noise
            self.noise_model = None

    def annotate(
        self, item: Item, item_template: ItemTemplate
    ) -> str | int | float | list[str]:
        """Generate annotation using LM scores.

        Parameters
        ----------
        item : Item
            Item to annotate.
        item_template : ItemTemplate
            Template defining task.

        Returns
        -------
        str | int | float | list[str]
            Annotation (format depends on task type).
        """
        # get strategy for task type
        strategy = self.get_strategy(item_template.task_type)

        # validate item
        strategy.validate_item(item, item_template)

        # generate base response
        response = strategy.simulate_response(
            item=item,
            item_template=item_template,
            model_output_key=self.config.model_output_key,
            rng=self.rng,
        )

        # apply noise model if configured
        if self.noise_model is not None:
            response = self.noise_model.apply(
                value=response,
                context={"item": item, "template": item_template, "strategy": strategy},
                rng=self.rng,
            )

        return response
