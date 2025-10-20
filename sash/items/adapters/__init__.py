"""Model adapters for judgment prediction during item construction.

This module provides adapters for various model types used to compute
constraints during Stage 3 (Item Construction). Each adapter integrates
with the ModelOutputCache (Phase 13) for efficient caching.

These are SEPARATE from template filling adapters (sash.templates.models),
which are used in Stage 2.
"""

from sash.items.adapters.base import ModelAdapter
from sash.items.adapters.huggingface import (
    HuggingFaceLanguageModel,
    HuggingFaceMaskedLanguageModel,
    HuggingFaceNLI,
)
from sash.items.adapters.sentence_transformers import HuggingFaceSentenceTransformer

__all__ = [
    "ModelAdapter",
    "HuggingFaceLanguageModel",
    "HuggingFaceMaskedLanguageModel",
    "HuggingFaceNLI",
    "HuggingFaceSentenceTransformer",
]
