"""Tokenizer configuration model.

Aligned with the existing ChunkingSpec pattern in bead.items.item_template,
which already supports ``parser: Literal["stanza", "spacy"]``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

TokenizerBackend = Literal["spacy", "stanza", "whitespace"]


class TokenizerConfig(BaseModel):
    """Configuration for display-level tokenization.

    Controls how text is split into word-level tokens for span annotation
    and UI display. Supports multiple NLP backends for multilingual coverage.

    Attributes
    ----------
    backend : TokenizerBackend
        Tokenization backend to use. "spacy" (default) supports 49+ languages
        and is fast and production-grade. "stanza" supports 80+ languages
        with better coverage for low-resource and morphologically rich
        languages. "whitespace" is a simple fallback for pre-tokenized text.
    language : str
        ISO 639 language code (e.g. "en", "zh", "de", "ar").
    model_name : str | None
        Explicit model name (e.g. "en_core_web_sm", "zh_core_web_sm").
        When None, auto-resolved from language and backend.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    backend: TokenizerBackend = Field(
        default="spacy", description="Tokenization backend"
    )
    language: str = Field(default="en", description="ISO 639 language code")
    model_name: str | None = Field(
        default=None,
        description="Explicit model name; auto-resolved when None",
    )
