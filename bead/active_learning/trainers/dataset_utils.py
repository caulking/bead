"""Utilities for converting items to HuggingFace datasets.

This module provides functions to convert bead Items to HuggingFace Dataset
format for use with HuggingFace Trainer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from datasets import Dataset

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from bead.items.item import Item


def items_to_dataset(
    items: list[Item],
    labels: list[str | int | float],
    participant_ids: list[str] | None,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 128,
    text_key: str = "text",
) -> Dataset:
    """Convert items and labels to HuggingFace Dataset.

    Parameters
    ----------
    items : list[Item]
        Items to convert.
    labels : list[str | int | float]
        Labels for items.
    participant_ids : list[str] | None
        Participant IDs for each item (required for mixed effects).
    tokenizer : PreTrainedTokenizerBase
        HuggingFace tokenizer.
    max_length : int
        Maximum sequence length for tokenization.
    text_key : str
        Key in rendered_elements to use as text (default: "text").

    Returns
    -------
    Dataset
        HuggingFace Dataset with tokenized inputs, labels, and participant_ids.

    Examples
    --------
    >>> from transformers import AutoTokenizer
    >>> tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    >>> dataset = items_to_dataset(
    ...     items=items,
    ...     labels=['yes', 'no', 'yes'],
    ...     participant_ids=['p1', 'p1', 'p2'],
    ...     tokenizer=tokenizer
    ... )
    >>> len(dataset)
    3
    """
    # Extract texts from items
    texts: list[str] = []
    for item in items:
        # Try to get text from rendered_elements
        if text_key in item.rendered_elements:
            text = item.rendered_elements[text_key]
        else:
            # Fallback: concatenate all rendered elements
            text = " ".join(str(v) for v in item.rendered_elements.values())
        texts.append(text)

    # Tokenize texts
    tokenized = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors=None,  # Return lists, not tensors
    )

    # Build dataset dict
    dataset_dict: dict[str, list[str | int | float]] = {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
    }

    # Add token_type_ids if present
    if "token_type_ids" in tokenized:
        dataset_dict["token_type_ids"] = tokenized["token_type_ids"]

    # Add labels
    dataset_dict["labels"] = labels

    # Add participant IDs if provided
    if participant_ids is not None:
        dataset_dict["participant_id"] = participant_ids

    return Dataset.from_dict(dataset_dict)


def cloze_items_to_dataset(
    items: list[Item],
    labels: list[list[str]],
    participant_ids: list[str] | None,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 128,
    text_key: str = "text",
) -> Dataset:
    """Convert cloze items and labels to HuggingFace Dataset with masking.

    For cloze tasks, this function:
    1. Extracts text from items
    2. Tokenizes and identifies masked positions (from "___" placeholders)
    3. Replaces "___" with [MASK] tokens
    4. Stores masked positions and target token IDs for loss computation

    Parameters
    ----------
    items : list[Item]
        Items with unfilled_slots (cloze items).
    labels : list[list[str]]
        Labels as list of lists. Each inner list contains one token per unfilled slot.
    participant_ids : list[str] | None
        Participant IDs for each item.
    tokenizer : PreTrainedTokenizerBase
        HuggingFace tokenizer.
    max_length : int
        Maximum sequence length for tokenization.
    text_key : str
        Key in rendered_elements to use as text (default: "text").

    Returns
    -------
    Dataset
        HuggingFace Dataset with:
        - input_ids: Tokenized text with [MASK] tokens
        - attention_mask: Attention mask
        - masked_positions: List of masked token positions per item
        - target_token_ids: List of target token IDs per masked position
        - participant_id: Participant IDs

    Examples
    --------
    >>> from transformers import AutoTokenizer
    >>> tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    >>> items = [Item(..., rendered_elements={"text": "The cat ___."}, ...)]
    >>> labels = [["ran"]]
    >>> dataset = cloze_items_to_dataset(
    ...     items=items,
    ...     labels=[["ran"]],
    ...     participant_ids=['p1'],
    ...     tokenizer=tokenizer
    ... )
    >>> len(dataset)
    1
    """
    mask_token_id = tokenizer.mask_token_id
    texts: list[str] = []
    all_masked_positions: list[list[int]] = []
    all_target_token_ids: list[list[int]] = []

    for item, label_list in zip(items, labels, strict=True):
        # Get text
        if text_key in item.rendered_elements:
            text = item.rendered_elements[text_key]
        else:
            text = " ".join(str(v) for v in item.rendered_elements.values())
        texts.append(text)

        # Tokenize to find "___" positions
        # First tokenize the full text to get the actual token IDs
        full_tokenized = tokenizer(
            text, add_special_tokens=True, return_offsets_mapping=False
        )
        full_tokens = tokenizer.convert_ids_to_tokens(full_tokenized["input_ids"])

        # Now find "___" positions in the tokenized sequence
        masked_indices: list[int] = []
        target_ids: list[int] = []

        # Track which tokens are part of "___" to avoid duplicates
        in_blank = False
        label_idx = 0

        # Skip [CLS] token (index 0)
        for j in range(1, len(full_tokens)):
            token = full_tokens[j]
            # Check if this token is part of a "___" placeholder
            if "_" in token and not in_blank:
                # Start of a new blank - record this position
                masked_indices.append(j)
                in_blank = True

                # Get target token ID for this label
                if label_idx < len(label_list):
                    target_token = label_list[label_idx]
                    # Tokenize the target token
                    target_tokenized = tokenizer.encode(
                        target_token, add_special_tokens=False
                    )
                    if target_tokenized:
                        target_ids.append(target_tokenized[0])
                    else:
                        # Fallback: use UNK token
                        target_ids.append(tokenizer.unk_token_id)
                    label_idx += 1
            elif "_" in token and in_blank:
                # Continuation of current blank - also mask but don't record again
                pass
            else:
                # Not a blank token - reset in_blank
                in_blank = False

        # Verify we found the expected number of masked positions
        expected_slots = len(item.unfilled_slots)
        if len(masked_indices) != expected_slots:
            raise ValueError(
                f"Mismatch between masked positions and unfilled_slots "
                f"for item: found {len(masked_indices)} '___' "
                f"placeholders in text but item has {expected_slots} "
                f"unfilled_slots. Ensure rendered text uses exactly one "
                f"'___' per unfilled_slot. Text: '{text}'"
            )

        all_masked_positions.append(masked_indices)
        all_target_token_ids.append(target_ids)

    # Tokenize all texts (this will include "___" which we'll replace)
    tokenized = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors=None,  # Return lists, not tensors
    )

    # Replace "___" tokens with [MASK] in input_ids
    input_ids = tokenized["input_ids"]
    for i, masked_pos in enumerate(all_masked_positions):
        for pos in masked_pos:
            if pos < len(input_ids[i]):
                input_ids[i][pos] = mask_token_id

    # Build dataset dict
    dataset_dict: dict[str, list[str | int | float | list[int]]] = {
        "input_ids": input_ids,
        "attention_mask": tokenized["attention_mask"],
        "masked_positions": all_masked_positions,
        "target_token_ids": all_target_token_ids,
    }

    # Add token_type_ids if present
    if "token_type_ids" in tokenized:
        dataset_dict["token_type_ids"] = tokenized["token_type_ids"]

    # Add participant IDs if provided
    if participant_ids is not None:
        dataset_dict["participant_id"] = participant_ids

    return Dataset.from_dict(dataset_dict)
