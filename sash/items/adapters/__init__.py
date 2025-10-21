"""Model adapters for judgment prediction during item construction.

This module provides adapters for various model types used to compute
constraints during Stage 3 (Item Construction). Each adapter integrates
with the ModelOutputCache (Phase 13) for efficient caching.

These are SEPARATE from template filling adapters (sash.templates.models),
which are used in Stage 2.

Available Adapters
------------------
Local Models:
- HuggingFaceLanguageModel: Causal language models (GPT-2, LLaMA, etc.)
- HuggingFaceMaskedLanguageModel: Masked language models (BERT, RoBERTa, etc.)
- HuggingFaceNLI: Natural language inference models (MNLI, etc.)
- HuggingFaceSentenceTransformer: Sentence embedding models

API Models:
- OpenAIAdapter: OpenAI GPT models via API
- AnthropicAdapter: Anthropic Claude models via API
- GoogleAdapter: Google Gemini models via API
- TogetherAIAdapter: Together AI models via API

Registry:
- ModelAdapterRegistry: Centralized registry for all adapters
- default_registry: Pre-configured registry with all built-in adapters
"""

# API utilities
from sash.items.adapters.api_utils import (  # noqa: F401
    RateLimiter,
    rate_limit,
    retry_with_backoff,
)
from sash.items.adapters.base import ModelAdapter  # noqa: F401
from sash.items.adapters.huggingface import (  # noqa: F401
    HuggingFaceLanguageModel,
    HuggingFaceMaskedLanguageModel,
    HuggingFaceNLI,
)

# Registry
from sash.items.adapters.registry import (  # noqa: F401
    ModelAdapterRegistry,
    default_registry,
)
from sash.items.adapters.sentence_transformers import (  # noqa: F401
    HuggingFaceSentenceTransformer,
)

# API adapters (optional, may not be available if dependencies not installed)
_api_adapters: list[str] = []

try:
    from sash.items.adapters.openai import OpenAIAdapter  # noqa: F401

    _api_adapters.append("OpenAIAdapter")
except ImportError:
    pass

try:
    from sash.items.adapters.anthropic import AnthropicAdapter  # noqa: F401

    _api_adapters.append("AnthropicAdapter")
except ImportError:
    pass

try:
    from sash.items.adapters.google import GoogleAdapter  # noqa: F401

    _api_adapters.append("GoogleAdapter")
except ImportError:
    pass

try:
    from sash.items.adapters.togetherai import TogetherAIAdapter  # noqa: F401

    _api_adapters.append("TogetherAIAdapter")
except ImportError:
    pass

__all__ = [
    # Base
    "ModelAdapter",
    # HuggingFace adapters
    "HuggingFaceLanguageModel",
    "HuggingFaceMaskedLanguageModel",
    "HuggingFaceNLI",
    "HuggingFaceSentenceTransformer",
    # API utilities
    "RateLimiter",
    "rate_limit",
    "retry_with_backoff",
    # Registry
    "ModelAdapterRegistry",
    "default_registry",
] + _api_adapters
