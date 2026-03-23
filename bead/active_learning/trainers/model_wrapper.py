"""Model wrapper for HuggingFace Trainer integration.

This module provides wrapper models that combine encoder and classifier
head into a single model compatible with HuggingFace Trainer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from transformers.modeling_outputs import MaskedLMOutput, SequenceClassifierOutput

if TYPE_CHECKING:
    from transformers import PreTrainedModel


class EncoderClassifierWrapper(nn.Module):
    """Wrapper that combines encoder and classifier for HuggingFace Trainer.

    This wrapper takes a transformer encoder and a classifier head and
    combines them into a single model that HuggingFace Trainer can use.
    The forward method takes standard HuggingFace inputs (input_ids, etc.)
    and returns outputs with .logits attribute.

    Parameters
    ----------
    encoder : PreTrainedModel
        Transformer encoder (e.g., BERT, RoBERTa).
    classifier_head : nn.Module
        Classification head that takes encoder outputs.

    Attributes
    ----------
    encoder : PreTrainedModel
        Transformer encoder.
    classifier_head : nn.Module
        Classification head.

    Examples
    --------
    >>> from transformers import AutoModel, AutoModelForSequenceClassification
    >>> encoder = AutoModel.from_pretrained('bert-base-uncased')
    >>> classifier = nn.Linear(768, 1)  # Binary classification
    >>> model = EncoderClassifierWrapper(encoder, classifier)
    >>> outputs = model(input_ids=..., attention_mask=...)
    >>> logits = outputs.logits
    """

    def __init__(
        self,
        encoder: PreTrainedModel,
        classifier_head: nn.Module,
    ) -> None:
        """Initialize wrapper.

        Parameters
        ----------
        encoder : PreTrainedModel
            Transformer encoder.
        classifier_head : nn.Module
            Classification head.
        """
        super().__init__()
        self.encoder = encoder
        self.classifier_head = classifier_head

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        **kwargs: torch.Tensor,
    ) -> SequenceClassifierOutput:
        """Forward pass through encoder and classifier.

        Parameters
        ----------
        input_ids : torch.Tensor | None
            Token IDs.
        attention_mask : torch.Tensor | None
            Attention mask.
        token_type_ids : torch.Tensor | None
            Token type IDs (for BERT-style models).
        **kwargs : torch.Tensor
            Additional model inputs.

        Returns
        -------
        SequenceClassifierOutput
            Outputs with .logits attribute (for HuggingFace compatibility).
        """
        # Encoder forward pass
        encoder_inputs: dict[str, torch.Tensor] = {}
        if input_ids is not None:
            encoder_inputs["input_ids"] = input_ids
        if attention_mask is not None:
            encoder_inputs["attention_mask"] = attention_mask
        if token_type_ids is not None:
            encoder_inputs["token_type_ids"] = token_type_ids

        # Add any other kwargs that encoder might accept
        for key, value in kwargs.items():
            if key not in ("labels", "participant_id"):
                encoder_inputs[key] = value

        encoder_outputs = self.encoder(**encoder_inputs)

        # Extract [CLS] token representation (first token)
        # Shape: (batch_size, hidden_size)
        if hasattr(encoder_outputs, "last_hidden_state"):
            cls_embedding = encoder_outputs.last_hidden_state[:, 0, :]
        elif hasattr(encoder_outputs, "pooler_output"):
            cls_embedding = encoder_outputs.pooler_output
        else:
            # Fallback: use first token from sequence
            cls_embedding = encoder_outputs[0][:, 0, :]

        # Classifier forward pass
        logits = self.classifier_head(cls_embedding)

        # Return SequenceClassifierOutput for HuggingFace compatibility
        # This is the standard output format that Trainer expects
        return SequenceClassifierOutput(logits=logits)


class EncoderRegressionWrapper(nn.Module):
    """Wrapper that combines encoder and regression head for HuggingFace Trainer.

    This wrapper takes a transformer encoder and a regression head and
    combines them into a single model that HuggingFace Trainer can use.
    The forward method takes standard HuggingFace inputs (input_ids, etc.)
    and returns outputs with .logits attribute (for regression, logits
    represents continuous values).

    Parameters
    ----------
    encoder : PreTrainedModel
        Transformer encoder (e.g., BERT, RoBERTa).
    regression_head : nn.Module
        Regression head that takes encoder outputs and outputs continuous values.

    Attributes
    ----------
    encoder : PreTrainedModel
        Transformer encoder.
    regression_head : nn.Module
        Regression head.

    Examples
    --------
    >>> from transformers import AutoModel
    >>> encoder = AutoModel.from_pretrained('bert-base-uncased')
    >>> regressor = nn.Linear(768, 1)  # Single continuous output
    >>> model = EncoderRegressionWrapper(encoder, regressor)
    >>> outputs = model(input_ids=..., attention_mask=...)
    >>> predictions = outputs.logits.squeeze()  # Continuous values
    """

    def __init__(
        self,
        encoder: PreTrainedModel,
        regression_head: nn.Module,
    ) -> None:
        """Initialize wrapper.

        Parameters
        ----------
        encoder : PreTrainedModel
            Transformer encoder.
        regression_head : nn.Module
            Regression head.
        """
        super().__init__()
        self.encoder = encoder
        self.regression_head = regression_head

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        **kwargs: torch.Tensor,
    ) -> SequenceClassifierOutput:
        """Forward pass through encoder and regression head.

        Parameters
        ----------
        input_ids : torch.Tensor | None
            Token IDs.
        attention_mask : torch.Tensor | None
            Attention mask.
        token_type_ids : torch.Tensor | None
            Token type IDs (for BERT-style models).
        **kwargs : torch.Tensor
            Additional model inputs.

        Returns
        -------
        SequenceClassifierOutput
            Outputs with .logits attribute containing continuous values.
        """
        # Encoder forward pass
        encoder_inputs: dict[str, torch.Tensor] = {}
        if input_ids is not None:
            encoder_inputs["input_ids"] = input_ids
        if attention_mask is not None:
            encoder_inputs["attention_mask"] = attention_mask
        if token_type_ids is not None:
            encoder_inputs["token_type_ids"] = token_type_ids

        # Add any other kwargs that encoder might accept
        for key, value in kwargs.items():
            if key not in ("labels", "participant_id"):
                encoder_inputs[key] = value

        encoder_outputs = self.encoder(**encoder_inputs)

        # Extract [CLS] token representation (first token)
        if hasattr(encoder_outputs, "last_hidden_state"):
            cls_embedding = encoder_outputs.last_hidden_state[:, 0, :]
        elif hasattr(encoder_outputs, "pooler_output"):
            cls_embedding = encoder_outputs.pooler_output
        else:
            # Fallback: use first token from sequence
            cls_embedding = encoder_outputs[0][:, 0, :]

        # Regression head forward pass
        # Output shape: (batch_size, 1) for single continuous value
        logits = self.regression_head(cls_embedding)

        # Return SequenceClassifierOutput for HuggingFace compatibility
        # For regression, logits represents continuous values
        return SequenceClassifierOutput(logits=logits)


class MLMModelWrapper(nn.Module):
    """Wrapper for MLM models to work with HuggingFace Trainer.

    This wrapper takes an AutoModelForMaskedLM and makes it compatible
    with the Trainer while allowing access to encoder and mlm_head separately
    for mixed effects adjustments.

    Parameters
    ----------
    model : PreTrainedModel
        AutoModelForMaskedLM model.

    Attributes
    ----------
    model : PreTrainedModel
        The MLM model.
    encoder : nn.Module
        Encoder module (extracted from model).
    mlm_head : nn.Module
        MLM head (extracted from model).

    Examples
    --------
    >>> from transformers import AutoModelForMaskedLM
    >>> model = AutoModelForMaskedLM.from_pretrained('bert-base-uncased')
    >>> wrapped = MLMModelWrapper(model)
    >>> outputs = wrapped(input_ids=..., attention_mask=...)
    >>> logits = outputs.logits  # (batch, seq_len, vocab_size)
    """

    def __init__(self, model: PreTrainedModel) -> None:
        """Initialize wrapper.

        Parameters
        ----------
        model : PreTrainedModel
            AutoModelForMaskedLM model.
        """
        super().__init__()
        self.model = model

        # Extract encoder and MLM head
        if hasattr(model, "bert"):
            self.encoder = model.bert
            self.mlm_head = model.cls
        elif hasattr(model, "roberta"):
            self.encoder = model.roberta
            self.mlm_head = model.lm_head
        else:
            # Fallback: try base_model and lm_head
            self.encoder = model.base_model
            self.mlm_head = model.lm_head

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        **kwargs: torch.Tensor,
    ) -> MaskedLMOutput:
        """Forward pass through MLM model.

        Parameters
        ----------
        input_ids : torch.Tensor | None
            Token IDs.
        attention_mask : torch.Tensor | None
            Attention mask.
        token_type_ids : torch.Tensor | None
            Token type IDs (for BERT-style models).
        **kwargs : torch.Tensor
            Additional model inputs.

        Returns
        -------
        MaskedLMOutput
            Model outputs with .logits attribute (shape: batch, seq_len, vocab_size).
        """
        # Forward through full model
        encoder_inputs: dict[str, torch.Tensor] = {}
        if input_ids is not None:
            encoder_inputs["input_ids"] = input_ids
        if attention_mask is not None:
            encoder_inputs["attention_mask"] = attention_mask
        if token_type_ids is not None:
            encoder_inputs["token_type_ids"] = token_type_ids

        # Add any other kwargs that model might accept
        for key, value in kwargs.items():
            if key not in (
                "labels",
                "participant_id",
                "masked_positions",
                "target_token_ids",
            ):
                encoder_inputs[key] = value

        # Use the full model's forward pass
        outputs = self.model(**encoder_inputs)
        return outputs


class RandomSlopesModelWrapper(nn.Module):
    """Wrapper for random slopes with per-participant classifier heads.

    This wrapper combines:
    - A shared encoder (transformer backbone)
    - A fixed classifier head (population-level)
    - Per-participant heads via RandomEffectsManager

    During forward pass, each sample is routed through its participant's
    specific classifier head. New participant heads are created on-demand
    by cloning the fixed head.

    Parameters
    ----------
    encoder : PreTrainedModel
        Transformer encoder (e.g., BERT, RoBERTa).
    classifier_head : nn.Module
        Fixed/population-level classification head.
    random_effects_manager : object
        RandomEffectsManager instance that stores participant slopes.

    Attributes
    ----------
    encoder : PreTrainedModel
        Transformer encoder.
    classifier_head : nn.Module
        Fixed classification head (used as template for new participants).
    random_effects_manager : object
        Manager for participant-specific heads.

    Examples
    --------
    >>> from transformers import AutoModel
    >>> from bead.active_learning.models.random_effects import RandomEffectsManager
    >>> encoder = AutoModel.from_pretrained('bert-base-uncased')
    >>> classifier = nn.Linear(768, 2)  # Binary classification
    >>> manager = RandomEffectsManager(config, n_classes=2)
    >>> model = RandomSlopesModelWrapper(encoder, classifier, manager)
    >>> outputs = model(input_ids=..., attention_mask=..., participant_id=['p1', 'p2'])
    >>> logits = outputs.logits
    """

    def __init__(
        self,
        encoder: PreTrainedModel,
        classifier_head: nn.Module,
        random_effects_manager: object,
    ) -> None:
        """Initialize wrapper.

        Parameters
        ----------
        encoder : PreTrainedModel
            Transformer encoder.
        classifier_head : nn.Module
            Fixed classification head.
        random_effects_manager : object
            RandomEffectsManager for participant heads.
        """
        super().__init__()
        self.encoder = encoder
        self.classifier_head = classifier_head
        self.random_effects_manager = random_effects_manager

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        participant_id: list[str] | None = None,
        **kwargs: torch.Tensor,
    ) -> SequenceClassifierOutput:
        """Forward pass through encoder and participant-specific heads.

        Each sample is routed through its participant's classifier head.
        If participant_id is None, uses the fixed (population) head.

        Parameters
        ----------
        input_ids : torch.Tensor | None
            Token IDs.
        attention_mask : torch.Tensor | None
            Attention mask.
        token_type_ids : torch.Tensor | None
            Token type IDs (for BERT-style models).
        participant_id : list[str] | None
            List of participant IDs for each sample in the batch.
            If None, uses fixed head for all samples.
        **kwargs : torch.Tensor
            Additional model inputs.

        Returns
        -------
        SequenceClassifierOutput
            Outputs with .logits attribute (for HuggingFace compatibility).
        """
        # Encoder forward pass
        encoder_inputs: dict[str, torch.Tensor] = {}
        if input_ids is not None:
            encoder_inputs["input_ids"] = input_ids
        if attention_mask is not None:
            encoder_inputs["attention_mask"] = attention_mask
        if token_type_ids is not None:
            encoder_inputs["token_type_ids"] = token_type_ids

        # Add any other kwargs that encoder might accept
        for key, value in kwargs.items():
            if key not in ("labels", "participant_id"):
                encoder_inputs[key] = value

        encoder_outputs = self.encoder(**encoder_inputs)

        # Extract [CLS] token representation (first token)
        if hasattr(encoder_outputs, "last_hidden_state"):
            cls_embedding = encoder_outputs.last_hidden_state[:, 0, :]
        elif hasattr(encoder_outputs, "pooler_output"):
            cls_embedding = encoder_outputs.pooler_output
        else:
            # Fallback: use first token from sequence
            cls_embedding = encoder_outputs[0][:, 0, :]

        # Route through participant-specific heads
        if participant_id is None:
            # No participant IDs - use fixed head for all
            logits = self.classifier_head(cls_embedding)
        else:
            # Per-participant routing
            logits_list: list[torch.Tensor] = []
            for i, pid in enumerate(participant_id):
                # Get or create participant-specific head
                participant_head = self.random_effects_manager.get_slopes(
                    pid,
                    fixed_head=self.classifier_head,
                    create_if_missing=True,
                )
                # Forward single sample through participant's head
                sample_embedding = cls_embedding[i : i + 1]  # Keep batch dimension
                sample_logits = participant_head(sample_embedding)
                logits_list.append(sample_logits)

            # Concatenate all logits
            logits = torch.cat(logits_list, dim=0)

        # Return SequenceClassifierOutput for HuggingFace compatibility
        return SequenceClassifierOutput(logits=logits)

    def get_all_parameters(self) -> list[nn.Parameter]:
        """Get all parameters including dynamically created participant heads.

        This method collects parameters from:
        1. The encoder
        2. The fixed classifier head
        3. All participant-specific heads (slopes)

        Returns
        -------
        list[nn.Parameter]
            List of all model parameters.
        """
        params: list[nn.Parameter] = []
        params.extend(self.encoder.parameters())
        params.extend(self.classifier_head.parameters())

        # Add participant head parameters if available
        if hasattr(self.random_effects_manager, "slopes"):
            for head in self.random_effects_manager.slopes.values():
                if hasattr(head, "parameters"):
                    params.extend(head.parameters())

        return params
