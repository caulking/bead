"""Template filling model adapters.

This module provides adapters for masked language models used in template filling.
These models are SEPARATE from judgment prediction models used later in the pipeline.
"""

from __future__ import annotations

from sash.templates.models.adapter import TemplateFillingModelAdapter
from sash.templates.models.cache import ModelOutputCache
from sash.templates.models.huggingface_mlm import HuggingFaceMLMAdapter

__all__ = [
    "TemplateFillingModelAdapter",
    "ModelOutputCache",
    "HuggingFaceMLMAdapter",
]
