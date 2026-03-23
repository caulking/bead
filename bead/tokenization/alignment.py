"""Alignment between display tokens and subword model tokens.

Maps display-token-level span indices to subword-token indices so that
active learning models can consume span annotations created in
display-token space.
"""

from __future__ import annotations

from typing import Protocol


def align_display_to_subword(
    display_tokens: list[str],
    subword_tokenizer: _PreTrainedTokenizerProtocol,
) -> list[list[int]]:
    """Map each display token index to its corresponding subword token indices.

    Parameters
    ----------
    display_tokens : list[str]
        Display-level token strings (word-level).
    subword_tokenizer : _PreTrainedTokenizerProtocol
        A HuggingFace-compatible tokenizer with ``__call__`` and
        ``convert_ids_to_tokens`` methods.

    Returns
    -------
    list[list[int]]
        A list where ``entry[i]`` is the list of subword token indices
        for display token ``i``. Special tokens (CLS, SEP, etc.) are
        excluded.
    """
    alignment: list[list[int]] = []
    # tokenize each display token individually to get the mapping
    subword_offset = 0

    # first, tokenize the full text to get the complete subword sequence
    full_text = " ".join(display_tokens)
    full_encoding = subword_tokenizer(full_text, add_special_tokens=False)
    full_ids: list[int] = full_encoding["input_ids"]
    full_subword_tokens = subword_tokenizer.convert_ids_to_tokens(full_ids)

    # now align by tokenizing each display token
    for display_token in display_tokens:
        token_encoding = subword_tokenizer(display_token, add_special_tokens=False)
        token_ids: list[int] = token_encoding["input_ids"]
        n_subwords = len(token_ids)

        # map to indices in the full subword sequence
        indices = list(range(subword_offset, subword_offset + n_subwords))
        # clamp to valid range
        indices = [i for i in indices if i < len(full_subword_tokens)]
        alignment.append(indices)
        subword_offset += n_subwords

    return alignment


def convert_span_indices(
    span_indices: list[int],
    alignment: list[list[int]],
) -> list[int]:
    """Convert display-token span indices to subword-token indices.

    Parameters
    ----------
    span_indices : list[int]
        Display-token indices forming the span.
    alignment : list[list[int]]
        Alignment from ``align_display_to_subword``.

    Returns
    -------
    list[int]
        Corresponding subword-token indices.

    Raises
    ------
    IndexError
        If any span index is out of range of the alignment.
    """
    subword_indices: list[int] = []
    for idx in span_indices:
        if idx < 0 or idx >= len(alignment):
            raise IndexError(
                f"Span index {idx} is out of range. "
                f"Alignment covers {len(alignment)} display tokens."
            )
        subword_indices.extend(alignment[idx])
    return sorted(set(subword_indices))


class _PreTrainedTokenizerProtocol(Protocol):
    """Structural typing protocol for HuggingFace tokenizers.

    Defines the minimal interface expected from a HuggingFace
    ``PreTrainedTokenizerBase`` instance: callable tokenization
    and ID-to-token conversion.
    """

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = True,
    ) -> dict[str, list[int]]: ...

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]: ...
