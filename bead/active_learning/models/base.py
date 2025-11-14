"""Base interfaces for active learning models with mixed effects support.

This module implements Generalized Linear Mixed Effects Models (GLMMs) following
the standard formulation:

    y = Xβ + Zu + ε

Where:
- Xβ: Fixed effects (population-level parameters, shared across all groups)
- Zu: Random effects (group-specific parameters, e.g., per-participant)
- u ~ N(0, G): Random effects with variance-covariance matrix G
- ε: Residuals

The implementation supports three modeling modes:
1. Fixed effects: Standard model, ignores grouping structure
2. Random intercepts: Per-group biases (Zu = bias vector per group)
3. Random slopes: Per-group model parameters (Zu = separate model head per group)

References
----------
- Bates et al. (2015). "Fitting Linear Mixed-Effects Models using lme4"
- Simchoni & Rosset (2022). "Integrating Random Effects in Deep Neural Networks"
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from bead.active_learning.config import (
    MixedEffectsConfig,
    RandomEffectsSpec,
    VarianceComponents,
)
from bead.data.base import BeadBaseModel
from bead.items.item import Item

if TYPE_CHECKING:
    from bead.items.item_template import ItemTemplate, TaskType

__all__ = [
    "ActiveLearningModel",
    "ModelPrediction",
    "MixedEffectsConfig",
    "VarianceComponents",
    "RandomEffectsSpec",
]


class ModelPrediction(BeadBaseModel):
    """Prediction output for a single item.

    Attributes
    ----------
    item_id : str
        Unique identifier for the item.
    probabilities : dict[str, float]
        Predicted probabilities for each class/option.
        Keys are option names (e.g., "option_a", "option_b") or class labels.
    predicted_class : str
        The predicted class/option with highest probability.
    confidence : float
        Confidence score (max probability).

    Examples
    --------
    >>> prediction = ModelPrediction(
    ...     item_id="abc123",
    ...     probabilities={"option_a": 0.7, "option_b": 0.3},
    ...     predicted_class="option_a",
    ...     confidence=0.7
    ... )
    >>> prediction.predicted_class
    'option_a'
    """

    item_id: str
    probabilities: dict[str, float]
    predicted_class: str
    confidence: float


class ActiveLearningModel(ABC):
    """Base class for all active learning models with mixed effects support.

    Implements GLMM-based active learning: y = Xβ + Zu + ε

    All models must:
    1. Support mixed effects (fixed, random_intercepts, random_slopes modes)
    2. Accept participant_ids in train/predict/predict_proba (None for fixed effects)
    3. Validate items match supported task types
    4. Track variance components (if estimate_variance_components=True)

    Attributes
    ----------
    config : Any
        Model configuration (task-type-specific).
        Must include a `mixed_effects: MixedEffectsConfig` field.
    supported_task_types : list[TaskType]
        List of task types this model can handle.

    Examples
    --------
    >>> class MyModel(ActiveLearningModel):
    ...     def __init__(self, config):
    ...         super().__init__(config)  # Validates mixed_effects field
    ...     @property
    ...     def supported_task_types(self):
    ...         return ["forced_choice"]
    ...     def validate_item_compatibility(self, item, item_template):
    ...         pass
    ...     def train(self, items, labels, participant_ids):
    ...         return {}
    ...     def predict(self, items, participant_ids):
    ...         return []
    ...     def predict_proba(self, items, participant_ids):
    ...         return np.array([])
    ...     def save(self, path):
    ...         pass
    ...     def load(self, path):
    ...         pass
    """

    def __init__(self, config: any) -> None:
        """Initialize model with configuration.

        Parameters
        ----------
        config : Any
            Model configuration. Must have a `mixed_effects` field of type
            MixedEffectsConfig.

        Raises
        ------
        ValueError
            If config is invalid or missing required fields.

        Examples
        --------
        >>> from bead.config.active_learning import ForcedChoiceModelConfig
        >>> config = ForcedChoiceModelConfig(
        ...     n_classes=2,
        ...     mixed_effects=MixedEffectsConfig(mode='fixed')
        ... )
        >>> model = ForcedChoiceModel(config)  # doctest: +SKIP
        """
        self.config = config

        # Validate mixed_effects field exists
        if not hasattr(config, "mixed_effects"):
            raise ValueError(
                f"Model config must have a 'mixed_effects' field of type "
                f"MixedEffectsConfig, but {type(config).__name__} has no such field. "
                f"Add: mixed_effects: MixedEffectsConfig = "
                f"Field(default_factory=MixedEffectsConfig)"
            )

        # Validate mixed_effects is correct type
        if not isinstance(config.mixed_effects, MixedEffectsConfig):
            raise ValueError(
                f"config.mixed_effects must be MixedEffectsConfig, but got "
                f"{type(config.mixed_effects).__name__}. "
                f"Ensure the field is properly typed: mixed_effects: MixedEffectsConfig"
            )

    @property
    @abstractmethod
    def supported_task_types(self) -> list[TaskType]:
        """Get list of task types this model supports.

        Returns
        -------
        list[TaskType]
            List of supported TaskType literals from items.models.

        Examples
        --------
        >>> model.supported_task_types
        ['forced_choice']
        """
        pass

    @abstractmethod
    def validate_item_compatibility(
        self, item: Item, item_template: ItemTemplate
    ) -> None:
        """Validate that an item is compatible with this model.

        Parameters
        ----------
        item : Item
            Item to validate.
        item_template : ItemTemplate
            Template the item was constructed from.

        Raises
        ------
        ValueError
            If item's task_type is not in supported_task_types.
        ValueError
            If item is missing required elements.
        ValueError
            If item structure is incompatible with model.

        Examples
        --------
        >>> model.validate_item_compatibility(item, template)  # doctest: +SKIP
        """
        pass

    @abstractmethod
    def train(
        self,
        items: list[Item],
        labels: list[str],
        participant_ids: list[str] | None = None,
        validation_items: list[Item] | None = None,
        validation_labels: list[str] | None = None,
    ) -> dict[str, float]:
        """Train model on labeled items with participant identifiers.

        Parameters
        ----------
        items : list[Item]
            Training items.
        labels : list[str]
            Training labels (format depends on task type).
        participant_ids : list[str] | None
            Participant identifier for each item.
            - For fixed effects (mode='fixed'): Pass None (automatically handled).
            - For mixed effects (mode='random_intercepts' or 'random_slopes'):
              Must provide list[str] with same length as items.
            Must not contain empty strings.
        validation_items : list[Item] | None
            Optional validation items.
        validation_labels : list[str] | None
            Optional validation labels.

        Returns
        -------
        dict[str, float]
            Training metrics including:
            - "train_accuracy", "train_loss": Standard metrics
            - "participant_variance": σ²_u (if estimate_variance_components=True)
            - "n_participants": Number of unique participants
            - "residual_variance": σ²_ε (if estimated)

        Raises
        ------
        ValueError
            If participant_ids is None when mode is 'random_intercepts' or 'random_slopes'.
        ValueError
            If items, labels, and participant_ids have different lengths.
        ValueError
            If participant_ids contains empty strings.
        ValueError
            If validation data is incomplete.
        ValueError
            If labels are invalid for this task type.

        Examples
        --------
        >>> # Fixed effects
        >>> metrics = model.train(items, labels, participant_ids=None)  # doctest: +SKIP

        >>> # Mixed effects
        >>> metrics = model.train(items, labels, participant_ids=["alice", "bob", "alice"])  # doctest: +SKIP
        >>> print(f"σ²_u: {metrics['participant_variance']:.3f}")  # doctest: +SKIP
        """
        pass

    @abstractmethod
    def predict(
        self, items: list[Item], participant_ids: list[str] | None = None
    ) -> list[ModelPrediction]:
        """Predict class labels for items with participant identifiers.

        Parameters
        ----------
        items : list[Item]
            Items to predict.
        participant_ids : list[str] | None
            Participant identifier for each item.
            - For fixed effects (mode='fixed'): Pass None.
            - For mixed effects: Must provide list[str] with same length as items.
            - For unknown participants: Use population mean (prior) for random effects.

        Returns
        -------
        list[ModelPrediction]
            Predictions with probabilities and predicted class for each item.

        Raises
        ------
        ValueError
            If model has not been trained.
        ValueError
            If participant_ids is None when mode requires mixed effects.
        ValueError
            If items and participant_ids have different lengths.
        ValueError
            If participant_ids contains empty strings.
        ValueError
            If items are incompatible with model.

        Examples
        --------
        >>> predictions = model.predict(test_items, participant_ids=None)  # doctest: +SKIP
        >>> predictions = model.predict(test_items, participant_ids=["alice"] * len(test_items))  # doctest: +SKIP
        >>> for pred in predictions:  # doctest: +SKIP
        ...     print(f"{pred.predicted_class}: {pred.confidence:.2f}")
        """
        pass

    @abstractmethod
    def predict_proba(
        self, items: list[Item], participant_ids: list[str] | None = None
    ) -> np.ndarray:
        """Predict class probabilities for items with participant identifiers.

        Parameters
        ----------
        items : list[Item]
            Items to predict.
        participant_ids : list[str] | None
            Participant identifier for each item.
            - For fixed effects (mode='fixed'): Pass None.
            - For mixed effects: Must provide list[str] with same length as items.

        Returns
        -------
        np.ndarray
            Array of shape (n_items, n_classes) with probabilities.
            Each row sums to 1.0 for classification tasks.

        Raises
        ------
        ValueError
            If model has not been trained.
        ValueError
            If participant_ids is None when mode requires mixed effects.
        ValueError
            If items and participant_ids have different lengths.
        ValueError
            If participant_ids contains empty strings.
        ValueError
            If items are incompatible with model.

        Examples
        --------
        >>> proba = model.predict_proba(test_items, participant_ids=None)  # doctest: +SKIP
        >>> proba.shape  # doctest: +SKIP
        (10, 2)
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save model to disk.

        Parameters
        ----------
        path : str
            File or directory path to save the model.

        Raises
        ------
        ValueError
            If model has not been trained.

        Examples
        --------
        >>> model.save("/path/to/model")  # doctest: +SKIP
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model from disk.

        Parameters
        ----------
        path : str
            File or directory path to load the model from.

        Raises
        ------
        FileNotFoundError
            If model file/directory does not exist.

        Examples
        --------
        >>> model.load("/path/to/model")  # doctest: +SKIP
        """
        pass
