"""Simulation framework for generating synthetic human judgments.

This module provides comprehensive simulation infrastructure for testing
active learning pipelines without requiring real human data. The framework:

- Supports all task types (forced_choice, ordinal_scale, magnitude, etc.)
- Respects task specifications from ItemTemplate
- Uses model outputs (LM scores, embeddings) for informed decisions
- Provides configurable noise models
- Extends the existing DSL for custom simulation logic

Examples
--------
>>> from sash.simulation import SimulatedAnnotator
>>> from sash.config.models import SimulatedAnnotatorConfig, NoiseModelConfig
>>>
>>> # Create annotator with configuration
>>> config = SimulatedAnnotatorConfig(
...     strategy="lm_score",
...     noise_model=NoiseModelConfig(noise_type="temperature", temperature=1.0)
... )
>>> annotator = SimulatedAnnotator.from_config(config)
>>>
>>> # Annotate items (automatically handles any task type)
>>> judgments = annotator.annotate_batch(items)
"""

from sash.simulation.annotators.base import SimulatedAnnotator
from sash.simulation.annotators.lm_based import LMBasedAnnotator
from sash.simulation.noise_models.base import NoiseModel
from sash.simulation.noise_models.temperature import TemperatureNoiseModel
from sash.simulation.runner import SimulationRunner
from sash.simulation.strategies.base import SimulationStrategy
from sash.simulation.strategies.binary import BinaryStrategy
from sash.simulation.strategies.categorical import CategoricalStrategy
from sash.simulation.strategies.forced_choice import ForcedChoiceStrategy
from sash.simulation.strategies.ordinal_scale import OrdinalScaleStrategy

__all__ = [
    "SimulatedAnnotator",
    "LMBasedAnnotator",
    "NoiseModel",
    "TemperatureNoiseModel",
    "SimulationRunner",
    "SimulationStrategy",
    "BinaryStrategy",
    "CategoricalStrategy",
    "ForcedChoiceStrategy",
    "OrdinalScaleStrategy",
]
