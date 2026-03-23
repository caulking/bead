"""Data collator for mixed effects training.

This module provides a custom data collator that handles participant_ids
along with standard tokenization and padding.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from transformers import DataCollatorWithPadding

if TYPE_CHECKING:
    import torch


class MixedEffectsDataCollator(DataCollatorWithPadding):
    """Data collator that preserves participant_ids for mixed effects.

    Extends DataCollatorWithPadding to handle participant_ids as strings
    (not tensors) and pass them through to the training batch.

    Parameters
    ----------
    tokenizer : PreTrainedTokenizerBase
        HuggingFace tokenizer.
    padding : bool | str
        Padding strategy (default: True).
    max_length : int | None
        Maximum sequence length (optional).
    pad_to_multiple_of : int | None
        Pad to multiple of this value (optional).

    Examples
    --------
    >>> from transformers import AutoTokenizer
    >>> tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    >>> collator = MixedEffectsDataCollator(tokenizer)
    >>> batch = collator([{'input_ids': [1, 2, 3], 'participant_id': 'alice'}])
    >>> 'participant_id' in batch
    True
    """

    def __call__(
        self, features: list[dict[str, torch.Tensor | str | int | float]]
    ) -> dict[str, torch.Tensor | list[str]]:
        """Collate batch with participant_ids preserved.

        Parameters
        ----------
        features : list[dict[str, torch.Tensor | str | int | float]]
            List of feature dictionaries from dataset.

        Returns
        -------
        dict[str, torch.Tensor | list[str]]
            Collated batch with participant_ids as list[str].
        """
        # Extract participant_ids before padding
        participant_ids: list[str] = []
        for feat in features:
            pid = feat.get("participant_id", "_fixed_")
            participant_ids.append(str(pid))

        # Remove participant_id from features for standard collation
        features_for_collation = [
            {k: v for k, v in feat.items() if k != "participant_id"}
            for feat in features
        ]

        # Use parent collator for tokenization/padding
        batch = super().__call__(features_for_collation)

        # Add participant_ids back as list (not tensor)
        batch["participant_id"] = participant_ids

        return batch


class ClozeDataCollator(MixedEffectsDataCollator):
    """Data collator for cloze (MLM) tasks with custom masking.

    Extends MixedEffectsDataCollator to handle:
    - masked_positions: List of masked token positions per item
    - target_token_ids: List of target token IDs per masked position
    - Preserves these for loss computation in the trainer

    Parameters
    ----------
    tokenizer : PreTrainedTokenizerBase
        HuggingFace tokenizer.
    padding : bool | str
        Padding strategy (default: True).
    max_length : int | None
        Maximum sequence length (optional).
    pad_to_multiple_of : int | None
        Pad to multiple of this value (optional).

    Examples
    --------
    >>> from transformers import AutoTokenizer
    >>> tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    >>> collator = ClozeDataCollator(tokenizer)
    >>> batch = collator([{
    ...     'input_ids': [1, 2, 103, 4],  # 103 is [MASK]
    ...     'masked_positions': [2],
    ...     'target_token_ids': [1234],
    ...     'participant_id': 'alice'
    ... }])
    >>> 'masked_positions' in batch
    True
    """

    def __call__(
        self, features: list[dict[str, torch.Tensor | str | int | float | list[int]]]
    ) -> dict[str, torch.Tensor | list[str] | list[list[int]]]:
        """Collate batch with masked positions and target token IDs preserved.

        Parameters
        ----------
        features : list[dict[str, torch.Tensor | str | int | float | list[int]]]
            List of feature dictionaries from dataset.

        Returns
        -------
        dict[str, torch.Tensor | list[str] | list[list[int]]]
            Collated batch with:
            - Standard tokenized inputs (input_ids, attention_mask, etc.)
            - participant_ids as list[str]
            - masked_positions as list[list[int]]
            - target_token_ids as list[list[int]]
        """
        # Extract cloze-specific fields before padding
        participant_ids: list[str] = []
        masked_positions: list[list[int]] = []
        target_token_ids: list[list[int]] = []

        for feat in features:
            pid = feat.get("participant_id", "_fixed_")
            participant_ids.append(str(pid))

            masked_pos = feat.get("masked_positions", [])
            if isinstance(masked_pos, list):
                masked_positions.append(masked_pos)
            else:
                masked_positions.append([])

            target_ids = feat.get("target_token_ids", [])
            if isinstance(target_ids, list):
                target_token_ids.append(target_ids)
            else:
                target_token_ids.append([])

        # Remove cloze-specific fields for standard collation
        features_for_collation = [
            {
                k: v
                for k, v in feat.items()
                if k not in ("participant_id", "masked_positions", "target_token_ids")
            }
            for feat in features
        ]

        # Use parent collator for tokenization/padding
        batch = super().__call__(features_for_collation)

        # Add cloze-specific fields back
        batch["participant_id"] = participant_ids
        batch["masked_positions"] = masked_positions
        batch["target_token_ids"] = target_token_ids

        return batch
