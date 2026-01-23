"""Mixed effects trainer for HuggingFace models.

This module provides a custom Trainer that handles participant-level
random effects (intercepts and slopes) while using HuggingFace's Trainer
infrastructure for optimization, checkpointing, and device management.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
import torch.nn.functional
from transformers import Trainer, TrainingArguments
from transformers.modeling_outputs import MaskedLMOutput, SequenceClassifierOutput

from bead.active_learning.models.random_effects import RandomEffectsManager

if TYPE_CHECKING:
    from collections.abc import Mapping

    import torch
    from torch.utils.data import Dataset
    from transformers import PreTrainedTokenizerBase


class MixedEffectsTrainer(Trainer):
    """HuggingFace Trainer with mixed effects support.

    Extends HuggingFace Trainer to handle participant-level random effects
    (random intercepts and random slopes) while leveraging Trainer's
    optimization, checkpointing, and device management.

    The key innovation is overriding compute_loss to apply participant-specific
    adjustments to model outputs before computing the loss. This preserves
    the mixed effects functionality while using HuggingFace infrastructure.

    Parameters
    ----------
    model : torch.nn.Module
        The model to train (must support mixed effects).
    args : TrainingArguments
        HuggingFace training arguments.
    train_dataset : Dataset
        Training dataset (must include 'participant_id' field).
    eval_dataset : Dataset | None
        Evaluation dataset (optional).
    random_effects_manager : RandomEffectsManager
        Manager for participant-level random effects.
    data_collator : Callable | None
        Data collator (optional, uses default if None).
    tokenizer : PreTrainedTokenizerBase | None
        Tokenizer (optional, for data collation).
    compute_metrics : Callable[[object], dict[str, float]] | None
        Metrics computation function (optional).

    Attributes
    ----------
    random_effects_manager : RandomEffectsManager
        Manager for random effects.
    mixed_effects_config : MixedEffectsConfig
        Mixed effects configuration.

    Examples
    --------
    >>> from transformers import AutoModelForSequenceClassification, TrainingArguments
    >>> from datasets import Dataset
    >>> config = MixedEffectsConfig(mode='random_intercepts')
    >>> manager = RandomEffectsManager(config, n_classes=2)
    >>> model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')
    >>> trainer = MixedEffectsTrainer(
    ...     model=model,
    ...     args=TrainingArguments(output_dir='./output'),
    ...     train_dataset=dataset,
    ...     random_effects_manager=manager
    ... )
    >>> trainer.train()
    """

    def __init__(
        self,
        model: torch.nn.Module,
        args: TrainingArguments,
        train_dataset: Dataset,
        random_effects_manager: RandomEffectsManager,
        eval_dataset: Dataset | None = None,
        data_collator: (
            Callable[
                [list[dict[str, torch.Tensor | str | int | float]]],
                dict[str, torch.Tensor | list[str]],
            ]
            | None
        ) = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        compute_metrics: Callable[[object], dict[str, float]] | None = None,
    ) -> None:
        """Initialize mixed effects trainer.

        Parameters
        ----------
        model : torch.nn.Module
            Model to train.
        args : TrainingArguments
            Training arguments.
        train_dataset : Dataset
            Training dataset.
        random_effects_manager : RandomEffectsManager
            Random effects manager.
        eval_dataset : Dataset | None
            Evaluation dataset.
        data_collator : Callable | None
            Data collator.
        tokenizer : PreTrainedTokenizerBase | None
            Tokenizer.
        compute_metrics : Callable[[object], dict[str, float]] | None
            Metrics computation function.
        """
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
        self.random_effects_manager = random_effects_manager
        self.mixed_effects_config = random_effects_manager.config

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: Mapping[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Compute loss with mixed effects adjustments.

        Overrides HuggingFace Trainer's compute_loss to:
        1. Get standard model outputs
        2. Apply participant-specific adjustments (intercepts/slopes)
        3. Compute loss with prior regularization
        4. Return loss (and optionally outputs)

        Parameters
        ----------
        model : torch.nn.Module
            Model to compute loss for.
        inputs : Mapping[str, torch.Tensor]
            Input batch (must include 'participant_id' if mixed effects).
            participant_id should be a list[str] in the dataset, but will be
            converted to tensor by data collator.
        return_outputs : bool
            Whether to return model outputs.

        Returns
        -------
        torch.Tensor | tuple[torch.Tensor, torch.Tensor]
            Loss tensor, or (loss, outputs) if return_outputs=True.
        """
        # Get labels and participant IDs
        labels = inputs.get("labels")
        participant_ids = inputs.get("participant_id")

        # For random_slopes mode, pass participant_id to model (wrapper handles routing)
        # For other modes, remove participant_id from inputs
        if self.mixed_effects_config.mode == "random_slopes":
            # RandomSlopesModelWrapper expects participant_id in forward()
            model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        else:
            excluded = ("labels", "participant_id")
            model_inputs = {k: v for k, v in inputs.items() if k not in excluded}

        # Standard forward pass
        outputs = model(**model_inputs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs

        # Apply mixed effects adjustments
        if self.mixed_effects_config.mode == "random_intercepts":
            # Apply participant-specific biases to logits
            if participant_ids is not None:
                batch_size = logits.shape[0]
                # Handle participant_ids: could be tensor of indices or list of strings
                # In our case, we store participant_ids as strings in dataset
                # The data collator will need to handle this specially
                for i in range(batch_size):
                    # Extract participant ID - data collator provides as list[str]
                    if isinstance(participant_ids, list):
                        pid_str = str(participant_ids[i])
                    elif isinstance(participant_ids, torch.Tensor):
                        # Fallback: if somehow tensor, convert
                        pid_elem = participant_ids[i]
                        pid_raw = pid_elem.item() if pid_elem.numel() == 1 else pid_elem
                        pid_str = str(pid_raw)
                    else:
                        pid_str = str(participant_ids[i])

                    # Get bias for this participant
                    # For binary: n_classes=1 (scalar bias)
                    n_classes = logits.shape[1] if logits.dim() > 1 else 1
                    bias = self.random_effects_manager.get_intercepts(
                        pid_str,
                        n_classes=n_classes,
                        param_name="mu",
                        create_if_missing=True,
                    )
                    # Ensure bias is on same device as logits
                    bias = bias.to(logits.device)
                    # For binary, bias is scalar, add to logits
                    if logits.dim() == 1:
                        bias_val = bias[0] if bias.numel() > 0 else 0
                        logits[i] = logits[i] + bias_val
                    else:
                        logits[i] = logits[i] + bias

        elif self.mixed_effects_config.mode == "random_slopes":
            # Random slopes are handled by RandomSlopesModelWrapper in forward()
            # The model routes each sample through participant-specific heads
            # Logits already incorporate random slopes - nothing to do here
            pass

        # Compute data loss
        if labels is not None:
            # Check if this is regression (continuous labels) or classification
            # Regression: labels are float, logits shape is (batch, 1)
            # Classification: labels are int/long, logits shape varies by task
            if labels.dtype.is_floating_point:
                # Regression task: use MSE loss
                if logits.dim() == 2 and logits.shape[1] == 1:
                    # Squeeze to (batch,)
                    preds = logits.squeeze(1)
                elif logits.dim() == 1:
                    preds = logits
                else:
                    # Unexpected shape, use first column
                    preds = logits[:, 0]
                loss = torch.nn.functional.mse_loss(preds, labels.float())
            elif logits.dim() == 1 or (logits.dim() == 2 and logits.shape[1] == 1):
                # Binary classification
                loss_fct = torch.nn.functional.binary_cross_entropy_with_logits
                if labels.dim() == 0:
                    labels = labels.unsqueeze(0)
                if logits.dim() == 1:
                    logits = logits.unsqueeze(1)
                loss = loss_fct(logits.squeeze(1), labels.float())
            else:
                # Multi-class classification
                loss_fct = torch.nn.functional.cross_entropy
                loss = loss_fct(logits, labels.long())
        else:
            # No labels provided (unsupervised)
            loss = torch.tensor(0.0, device=logits.device)

        # Add prior regularization loss
        loss_prior = self.random_effects_manager.compute_prior_loss()
        if loss_prior.device != loss.device:
            loss_prior = loss_prior.to(loss.device)
        loss = loss + loss_prior

        if return_outputs:
            # Create output object with adjusted logits
            adjusted_outputs = SequenceClassifierOutput(logits=logits)
            return (loss, adjusted_outputs)
        return loss

    def create_optimizer(self) -> None:
        """Create optimizer with all parameters including participant heads.

        For random_slopes mode, this method collects parameters from:
        1. The encoder (via model.encoder or model.model.encoder)
        2. The fixed classifier head
        3. All participant-specific heads (slopes)

        For other modes, delegates to parent implementation.
        """
        if self.optimizer is not None:
            # Optimizer already exists
            return

        if self.mixed_effects_config.mode == "random_slopes":
            # Collect parameters for random_slopes mode
            optimizer_grouped_parameters: list[dict[str, object]] = []

            # Check if model has get_all_parameters method (RandomSlopesModelWrapper)
            if hasattr(self.model, "get_all_parameters"):
                all_params = self.model.get_all_parameters()
                optimizer_grouped_parameters.append(
                    {
                        "params": all_params,
                        "lr": self.args.learning_rate,
                    }
                )
            else:
                # Fallback: collect standard model parameters plus slope parameters
                optimizer_grouped_parameters.append(
                    {
                        "params": list(self.model.parameters()),
                        "lr": self.args.learning_rate,
                    }
                )

                # Add participant head parameters from random_effects_manager
                if hasattr(self.random_effects_manager, "slopes"):
                    for head in self.random_effects_manager.slopes.values():
                        if hasattr(head, "parameters"):
                            optimizer_grouped_parameters.append(
                                {
                                    "params": list(head.parameters()),
                                    "lr": self.args.learning_rate,
                                }
                            )

            # Create AdamW optimizer
            self.optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
            )
        else:
            # Use parent implementation for other modes
            super().create_optimizer()


class ClozeMLMTrainer(MixedEffectsTrainer):
    """Custom trainer for cloze (MLM) tasks with custom masking positions.

    Extends MixedEffectsTrainer to handle MLM loss computation only on
    specific masked positions (from unfilled_slots) rather than all positions.

    Parameters
    ----------
    model : torch.nn.Module
        MLM model (AutoModelForMaskedLM or wrapper).
    args : TrainingArguments
        Training arguments.
    train_dataset : Dataset
        Training dataset (must include 'masked_positions' and 'target_token_ids').
    random_effects_manager : RandomEffectsManager
        Random effects manager.
        eval_dataset : Dataset | None
            Evaluation dataset.
        data_collator : Callable | None
            Data collator (should be ClozeDataCollator).
        tokenizer : PreTrainedTokenizerBase | None
            Tokenizer.
        compute_metrics : Callable[[object], dict[str, float]] | None
            Metrics computation function.

    Examples
    --------
    >>> from transformers import AutoModelForMaskedLM, TrainingArguments
    >>> model = AutoModelForMaskedLM.from_pretrained('bert-base-uncased')
    >>> trainer = ClozeMLMTrainer(
    ...     model=model,
    ...     args=TrainingArguments(output_dir='./output'),
    ...     train_dataset=dataset,
    ...     random_effects_manager=manager
    ... )
    >>> trainer.train()
    """

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: Mapping[str, torch.Tensor | list[str] | list[list[int]]],
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Compute MLM loss only on masked positions.

        Overrides MixedEffectsTrainer's compute_loss to:
        1. Get model outputs (logits for all positions)
        2. Apply participant-specific adjustments (intercepts) if needed
        3. Compute cross-entropy loss only on masked positions
        4. Add prior regularization
        5. Return loss (and optionally outputs)

        Parameters
        ----------
        model : torch.nn.Module
            Model to compute loss for.
        inputs : Mapping[str, torch.Tensor | list[str] | list[list[int]]]
            Input batch with:
            - Standard tokenized inputs (input_ids, attention_mask, etc.)
            - participant_id: list[str]
            - masked_positions: list[list[int]] - masked token positions per item
            - target_token_ids: list[list[int]] - target token IDs per masked position
        return_outputs : bool
            Whether to return model outputs.
        num_items_in_batch : int | None
            Unused, kept for compatibility.

        Returns
        -------
        torch.Tensor | tuple[torch.Tensor, torch.Tensor]
            Loss tensor, or (loss, outputs) if return_outputs=True.
        """
        # Extract cloze-specific fields
        participant_ids = inputs.get("participant_id")
        masked_positions = inputs.get("masked_positions", [])
        target_token_ids = inputs.get("target_token_ids", [])

        # Remove these from inputs for model forward pass
        excluded = ("labels", "participant_id", "masked_positions", "target_token_ids")
        model_inputs = {k: v for k, v in inputs.items() if k not in excluded}

        # Standard forward pass
        outputs = model(**model_inputs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        # logits shape: (batch, seq_len, vocab_size)

        # Apply mixed effects adjustments for random_intercepts
        if self.mixed_effects_config.mode == "random_intercepts":
            if participant_ids is not None and isinstance(participant_ids, list):
                vocab_size = logits.shape[2]
                batch_size = logits.shape[0]
                for i in range(batch_size):
                    pid_str = str(participant_ids[i])
                    # Get bias for this participant (vocab_size,)
                    bias = self.random_effects_manager.get_intercepts(
                        pid_str,
                        n_classes=vocab_size,
                        param_name="mu",
                        create_if_missing=True,
                    )
                    bias = bias.to(logits.device)
                    # Add bias to all masked positions for this item
                    in_range = i < len(masked_positions)
                    if in_range and isinstance(masked_positions[i], list):
                        for pos in masked_positions[i]:
                            if pos < logits.shape[1]:
                                logits[i, pos] = logits[i, pos] + bias

        # Compute loss only on masked positions
        losses: list[torch.Tensor] = []
        if isinstance(masked_positions, list) and isinstance(target_token_ids, list):
            for j, (masked_pos, target_ids) in enumerate(
                zip(masked_positions, target_token_ids, strict=True)
            ):
                if j >= logits.shape[0]:
                    continue
                if isinstance(masked_pos, list) and isinstance(target_ids, list):
                    for pos, target_id in zip(masked_pos, target_ids, strict=True):
                        if pos < logits.shape[1]:
                            # Cross-entropy loss for this position
                            # logits[j, pos] shape: (vocab_size,)
                            # target_id: int
                            # Need shape (1, vocab_size) for logits and (1,) for target
                            pos_logits = logits[j, pos].unsqueeze(0)  # (1, vocab_size)
                            pos_target = torch.tensor(
                                [target_id], device=logits.device, dtype=torch.long
                            )  # (1,)
                            loss_j = torch.nn.functional.cross_entropy(
                                pos_logits, pos_target
                            )
                            losses.append(loss_j)

        if losses:
            loss_nll = torch.stack(losses).mean()
        else:
            loss_nll = torch.tensor(0.0, device=logits.device)

        # Add prior regularization loss
        loss_prior = self.random_effects_manager.compute_prior_loss()
        if loss_prior.device != loss_nll.device:
            loss_prior = loss_prior.to(loss_nll.device)
        loss = loss_nll + loss_prior

        if return_outputs:
            # Return outputs with logits
            adjusted_outputs = MaskedLMOutput(logits=logits)
            return (loss, adjusted_outputs)
        return loss

    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor | list[str] | list[list[int]]],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Perform a prediction step with cloze-specific label encoding.

        Creates labels tensor encoding target_token_ids at masked_positions
        with -100 elsewhere (HuggingFace ignore index convention). This enables
        compute_cloze_metrics() to evaluate predictions at the correct positions.

        Parameters
        ----------
        model : torch.nn.Module
            Model to use for prediction.
        inputs : dict[str, torch.Tensor | list[str] | list[list[int]]]
            Input batch with:
            - Standard tokenized inputs (input_ids, attention_mask, etc.)
            - participant_id: list[str]
            - masked_positions: list[list[int]] - masked token positions per item
            - target_token_ids: list[list[int]] - target token IDs per position
        prediction_loss_only : bool
            Whether to only return loss.
        ignore_keys : list[str] | None
            Keys to ignore (unused).

        Returns
        -------
        tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]
            (loss, logits, labels) tuple where labels encodes target tokens
            at masked positions with -100 elsewhere.
        """
        # Extract cloze-specific fields
        masked_positions = inputs.get("masked_positions", [])
        target_token_ids = inputs.get("target_token_ids", [])

        # Filter inputs for model forward pass
        model_inputs = {
            k: v
            for k, v in inputs.items()
            if k not in ("participant_id", "masked_positions", "target_token_ids")
        }

        # Get predictions from parent (which handles compute_loss internally)
        loss, logits, _ = super().prediction_step(
            model, model_inputs, prediction_loss_only, ignore_keys
        )

        if prediction_loss_only:
            return (loss, None, None)

        # Build labels tensor: (batch_size, seq_len) with -100 default
        labels = None
        has_masks = isinstance(masked_positions, list)
        has_targets = isinstance(target_token_ids, list)
        if logits is not None and has_masks and has_targets:
            batch_size, seq_len = logits.shape[:2]
            labels = torch.full(
                (batch_size, seq_len), -100, dtype=torch.long, device=logits.device
            )

            # Fill in target token IDs at masked positions
            for i, (positions, targets) in enumerate(
                zip(masked_positions, target_token_ids, strict=False)
            ):
                if i >= batch_size:
                    break
                if isinstance(positions, list) and isinstance(targets, list):
                    for pos, target_id in zip(positions, targets, strict=False):
                        if isinstance(pos, int) and isinstance(target_id, int):
                            if 0 <= pos < seq_len:
                                labels[i, pos] = target_id

        return (loss, logits, labels)
