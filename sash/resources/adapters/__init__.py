"""External resource adapters for linguistic databases.

This module provides adapters for fetching lexical items from external
linguistic databases including VerbNet, PropBank, FrameNet (via glazing),
and UniMorph morphological paradigms.
"""

from sash.resources.adapters.base import ResourceAdapter
from sash.resources.adapters.cache import AdapterCache
from sash.resources.adapters.glazing import GlazingAdapter
from sash.resources.adapters.registry import AdapterRegistry
from sash.resources.adapters.unimorph import UniMorphAdapter

__all__ = [
    "ResourceAdapter",
    "AdapterCache",
    "GlazingAdapter",
    "UniMorphAdapter",
    "AdapterRegistry",
]
