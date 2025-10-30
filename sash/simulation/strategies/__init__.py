"""Task-specific simulation strategies."""

from sash.simulation.strategies.base import SimulationStrategy
from sash.simulation.strategies.binary import BinaryStrategy
from sash.simulation.strategies.categorical import CategoricalStrategy
from sash.simulation.strategies.cloze import ClozeStrategy
from sash.simulation.strategies.forced_choice import ForcedChoiceStrategy
from sash.simulation.strategies.free_text import FreeTextStrategy
from sash.simulation.strategies.magnitude import MagnitudeStrategy
from sash.simulation.strategies.multi_select import MultiSelectStrategy
from sash.simulation.strategies.ordinal_scale import OrdinalScaleStrategy

__all__ = [
    "SimulationStrategy",
    "BinaryStrategy",
    "CategoricalStrategy",
    "ClozeStrategy",
    "ForcedChoiceStrategy",
    "FreeTextStrategy",
    "MagnitudeStrategy",
    "MultiSelectStrategy",
    "OrdinalScaleStrategy",
]
