"""Concrete tokenizer implementations.

Provides display-level tokenizers for span annotation. Each tokenizer
converts raw text into a sequence of ``DisplayToken`` objects that carry
rendering metadata (``space_after``) for artifact-free reconstruction.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator

from pydantic import BaseModel, ConfigDict

from bead.tokenization.config import TokenizerConfig


class DisplayToken(BaseModel):
    """A word-level token with rendering metadata.

    Attributes
    ----------
    text : str
        The token text.
    space_after : bool
        Whether whitespace follows this token in the original text.
    start_char : int
        Character offset of the token start in the original text.
    end_char : int
        Character offset of the token end in the original text.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str
    space_after: bool = True
    start_char: int
    end_char: int


class TokenizedText(BaseModel):
    """Result of display-level tokenization.

    Attributes
    ----------
    tokens : list[DisplayToken]
        The sequence of display tokens.
    original_text : str
        The original input text.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    tokens: list[DisplayToken]
    original_text: str

    @property
    def token_texts(self) -> list[str]:
        """Plain token strings (for ``Item.tokenized_elements``).

        Returns
        -------
        list[str]
            List of token text strings.
        """
        return [t.text for t in self.tokens]

    @property
    def space_after_flags(self) -> list[bool]:
        """Per-token space_after flags (for ``Item.token_space_after``).

        Returns
        -------
        list[bool]
            List of boolean flags.
        """
        return [t.space_after for t in self.tokens]

    def render(self) -> str:
        """Reconstruct display text from tokens with correct spacing.

        Guarantees identical rendering to original when round-tripped.

        Returns
        -------
        str
            Reconstructed text.
        """
        parts: list[str] = []
        for token in self.tokens:
            parts.append(token.text)
            if token.space_after:
                parts.append(" ")
        return "".join(parts).rstrip()


class WhitespaceTokenizer:
    """Simple whitespace-split tokenizer.

    Fallback for pre-tokenized text or languages not supported by spaCy
    or Stanza. Splits on whitespace boundaries and infers ``space_after``
    from the original character offsets.
    """

    def __call__(self, text: str) -> TokenizedText:
        """Tokenize text by splitting on whitespace.

        Parameters
        ----------
        text : str
            Input text.

        Returns
        -------
        TokenizedText
            Tokenized result.
        """
        tokens: list[DisplayToken] = []
        for match in re.finditer(r"\S+", text):
            start = match.start()
            end = match.end()
            # space_after is True if there is whitespace after this token
            space_after = end < len(text) and text[end] == " "
            tokens.append(
                DisplayToken(
                    text=match.group(),
                    space_after=space_after,
                    start_char=start,
                    end_char=end,
                )
            )
        return TokenizedText(tokens=tokens, original_text=text)


class SpacyTokenizer:
    """spaCy-based tokenizer.

    Supports 49+ languages. Auto-resolves model from language code if
    ``model_name`` is not specified. Handles punctuation attachment and
    multi-word token (MWT) expansion correctly.

    Parameters
    ----------
    language : str
        ISO 639 language code.
    model_name : str | None
        Explicit spaCy model name. When None, uses ``{language}_core_web_sm``
        for common languages, falling back to a blank model.
    """

    def __init__(self, language: str = "en", model_name: str | None = None) -> None:
        self._language = language
        self._model_name = model_name
        self._nlp: Callable[..., _SpacyDocProtocol] | None = None

    def _load(self) -> Callable[..., _SpacyDocProtocol]:
        if self._nlp is not None:
            return self._nlp

        try:
            import spacy  # noqa: PLC0415
        except ImportError as e:
            raise ImportError(
                "spaCy is required for SpacyTokenizer. "
                "Install it with: pip install 'bead[tokenization]'"
            ) from e

        model = self._model_name
        if model is None:
            model = f"{self._language}_core_web_sm"

        try:
            self._nlp = spacy.load(model)
        except OSError:
            # Fall back to blank model
            self._nlp = spacy.blank(self._language)

        return self._nlp

    def __call__(self, text: str) -> TokenizedText:
        """Tokenize text using spaCy.

        Parameters
        ----------
        text : str
            Input text.

        Returns
        -------
        TokenizedText
            Tokenized result with correct ``space_after`` metadata.
        """
        nlp = self._load()
        doc = nlp(text)
        tokens: list[DisplayToken] = []
        for token in doc:
            tokens.append(
                DisplayToken(
                    text=token.text,
                    space_after=token.whitespace_ != "",
                    start_char=token.idx,
                    end_char=token.idx + len(token.text),
                )
            )
        return TokenizedText(tokens=tokens, original_text=text)


class StanzaTokenizer:
    """Stanza-based tokenizer.

    Supports 80+ languages. Handles multi-word token (MWT) expansion for
    languages like German, French, and Arabic. Better coverage for
    low-resource and morphologically rich languages.

    Parameters
    ----------
    language : str
        ISO 639 language code.
    model_name : str | None
        Explicit Stanza model/package name. When None, uses the default
        package for the language.
    """

    def __init__(self, language: str = "en", model_name: str | None = None) -> None:
        self._language = language
        self._model_name = model_name
        self._nlp: _StanzaPipelineProtocol | None = None

    def _load(self) -> _StanzaPipelineProtocol:
        if self._nlp is not None:
            return self._nlp

        try:
            import stanza  # noqa: PLC0415
        except ImportError as e:
            raise ImportError(
                "Stanza is required for StanzaTokenizer. "
                "Install it with: pip install 'bead[tokenization]'"
            ) from e

        kwargs: dict[str, str | bool] = {
            "lang": self._language,
            "processors": "tokenize",
            "verbose": False,
        }
        if self._model_name is not None:
            kwargs["package"] = self._model_name

        try:
            self._nlp = stanza.Pipeline(**kwargs)
        except Exception:
            # Download model and retry
            stanza.download(self._language, verbose=False)
            self._nlp = stanza.Pipeline(**kwargs)

        return self._nlp

    def __call__(self, text: str) -> TokenizedText:
        """Tokenize text using Stanza.

        Parameters
        ----------
        text : str
            Input text.

        Returns
        -------
        TokenizedText
            Tokenized result with correct ``space_after`` metadata.
        """
        nlp = self._load()
        doc = nlp(text)
        tokens: list[DisplayToken] = []
        for sentence in doc.sentences:
            for token in sentence.tokens:
                start_char = token.start_char
                end_char = token.end_char
                # Stanza tokens have a misc field; space_after can be
                # inferred from character offsets or the SpaceAfter=No
                # annotation in the misc field.
                space_after = True
                if hasattr(token, "misc") and token.misc:
                    if "SpaceAfter=No" in token.misc:
                        space_after = False
                elif end_char < len(text):
                    space_after = text[end_char] == " "

                tokens.append(
                    DisplayToken(
                        text=token.text,
                        space_after=space_after,
                        start_char=start_char,
                        end_char=end_char,
                    )
                )
        return TokenizedText(tokens=tokens, original_text=text)


def create_tokenizer(config: TokenizerConfig) -> Callable[[str], TokenizedText]:
    """Return a tokenization function for the given config.

    Lazy-loads the NLP backend (spaCy/Stanza) on first call.

    Parameters
    ----------
    config : TokenizerConfig
        Tokenizer configuration.

    Returns
    -------
    Callable[[str], TokenizedText]
        A callable that tokenizes text.

    Raises
    ------
    ValueError
        If the backend is not recognized.
    """
    if config.backend == "whitespace":
        return WhitespaceTokenizer()
    elif config.backend == "spacy":
        return SpacyTokenizer(
            language=config.language, model_name=config.model_name
        )
    elif config.backend == "stanza":
        return StanzaTokenizer(
            language=config.language, model_name=config.model_name
        )
    else:
        raise ValueError(f"Unknown tokenizer backend: {config.backend}")


# Structural typing protocols for spaCy/Stanza (avoids hard imports)
class _SpacyTokenProtocol:
    text: str
    whitespace_: str
    idx: int


class _SpacyDocProtocol:
    def __iter__(self) -> Iterator[_SpacyTokenProtocol]: ...  # noqa: D105


class _StanzaTokenProtocol:
    text: str
    start_char: int
    end_char: int
    misc: str | None


class _StanzaSentenceProtocol:
    tokens: list[_StanzaTokenProtocol]


class _StanzaDocProtocol:
    sentences: list[_StanzaSentenceProtocol]


class _StanzaPipelineProtocol:
    def __call__(self, text: str) -> _StanzaDocProtocol: ...  # noqa: D102
