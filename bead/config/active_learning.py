"""Active learning configuration models for the bead package."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from bead.active_learning.config import MixedEffectsConfig


class ForcedChoiceModelConfig(BaseModel):
    """Configuration for forced choice active learning models.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.
    max_length : int
        Maximum sequence length for tokenization.
    encoder_mode : Literal["single_encoder", "dual_encoder"]
        Encoding strategy for options.
    include_instructions : bool
        Whether to include task instructions.
    learning_rate : float
        Learning rate for AdamW optimizer.
    batch_size : int
        Batch size for training.
    num_epochs : int
        Number of training epochs.
    device : Literal["cpu", "cuda", "mps"]
        Device to train on.
    mixed_effects : MixedEffectsConfig
        Mixed effects configuration for participant-level modeling.

    Examples
    --------
    >>> config = ForcedChoiceModelConfig()
    >>> config.model_name
    'bert-base-uncased'
    >>> config.batch_size
    16
    >>> config.mixed_effects.mode
    'fixed'
    """

    model_name: str = Field(
        default="bert-base-uncased",
        description="HuggingFace model identifier",
    )
    max_length: int = Field(
        default=128,
        description="Maximum sequence length for tokenization",
        gt=0,
    )
    encoder_mode: Literal["single_encoder", "dual_encoder"] = Field(
        default="single_encoder",
        description="Encoding strategy for options",
    )
    include_instructions: bool = Field(
        default=False,
        description="Whether to include task instructions",
    )
    learning_rate: float = Field(
        default=2e-5,
        description="Learning rate for AdamW optimizer",
        gt=0,
    )
    batch_size: int = Field(
        default=16,
        description="Batch size for training",
        gt=0,
    )
    num_epochs: int = Field(
        default=3,
        description="Number of training epochs",
        gt=0,
    )
    device: Literal["cpu", "cuda", "mps"] = Field(
        default="cpu",
        description="Device to train on",
    )
    mixed_effects: MixedEffectsConfig = Field(
        default_factory=MixedEffectsConfig,
        description="Mixed effects configuration for participant-level modeling",
    )


class UncertaintySamplerConfig(BaseModel):
    """Configuration for uncertainty sampling strategies.

    Parameters
    ----------
    method : str
        Uncertainty method to use ("entropy", "margin", "least_confidence").
    batch_size : int | None
        Number of items to select per iteration. If None, uses the
        budget_per_iteration from ActiveLearningLoopConfig.

    Examples
    --------
    >>> config = UncertaintySamplerConfig()
    >>> config.method
    'entropy'
    >>> config = UncertaintySamplerConfig(method="margin", batch_size=50)
    >>> config.method
    'margin'
    """

    method: Literal["entropy", "margin", "least_confidence"] = Field(
        default="entropy",
        description="Uncertainty sampling method",
    )
    batch_size: int | None = Field(
        default=None,
        description="Number of items to select per iteration",
        gt=0,
    )


class ActiveLearningLoopConfig(BaseModel):
    """Configuration for active learning loop orchestration.

    Parameters
    ----------
    max_iterations : int
        Maximum number of AL iterations to run.
    budget_per_iteration : int
        Number of items to select per iteration.
    stopping_criterion : str
        Stopping criterion.
    performance_threshold : float | None
        Performance threshold for stopping.
    metric_name : str
        Metric name for convergence/threshold checks.
    convergence_patience : int
        Iterations to wait before declaring convergence.
    convergence_threshold : float
        Minimum improvement to avoid convergence.

    Examples
    --------
    >>> config = ActiveLearningLoopConfig()
    >>> config.max_iterations
    10
    >>> config.budget_per_iteration
    100
    """

    max_iterations: int = Field(
        default=10,
        description="Maximum number of iterations",
        gt=0,
    )
    budget_per_iteration: int = Field(
        default=100,
        description="Number of items to select per iteration",
        gt=0,
    )
    stopping_criterion: Literal[
        "max_iterations", "convergence", "performance_threshold"
    ] = Field(
        default="max_iterations",
        description="Stopping criterion for the loop",
    )
    performance_threshold: float | None = Field(
        default=None,
        description="Performance threshold for stopping",
        ge=0,
        le=1,
    )
    metric_name: str = Field(
        default="accuracy",
        description="Metric name for convergence/threshold checks",
    )
    convergence_patience: int = Field(
        default=3,
        description="Iterations to wait before declaring convergence",
        gt=0,
    )
    convergence_threshold: float = Field(
        default=0.01,
        description="Minimum improvement to avoid convergence",
        gt=0,
    )


class TrainerConfig(BaseModel):
    """Configuration for active learning trainers (HuggingFace, Lightning, etc.).

    Parameters
    ----------
    trainer_type : str
        Trainer type ("huggingface", "lightning").
    epochs : int
        Number of training epochs.
    eval_strategy : str
        Evaluation strategy.
    save_strategy : str
        Save strategy.
    logging_dir : Path
        Logging directory.
    use_wandb : bool
        Whether to use Weights & Biases.
    wandb_project : str | None
        W&B project name.

    Examples
    --------
    >>> config = TrainerConfig()
    >>> config.trainer_type
    'huggingface'
    >>> config.epochs
    3
    """

    trainer_type: Literal["huggingface", "lightning"] = Field(
        default="huggingface",
        description="Trainer type",
    )
    epochs: int = Field(default=3, description="Training epochs", gt=0)
    eval_strategy: str = Field(default="epoch", description="Evaluation strategy")
    save_strategy: str = Field(default="epoch", description="Save strategy")
    logging_dir: Path = Field(default=Path("logs"), description="Logging directory")
    use_wandb: bool = Field(default=False, description="Use Weights & Biases")
    wandb_project: str | None = Field(default=None, description="W&B project name")


class CategoricalModelConfig(BaseModel):
    """Configuration for categorical active learning models.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.
    max_length : int
        Maximum sequence length for tokenization.
    encoder_mode : Literal["single_encoder", "dual_encoder"]
        Encoding strategy for categories.
    include_instructions : bool
        Whether to include task instructions.
    learning_rate : float
        Learning rate for AdamW optimizer.
    batch_size : int
        Batch size for training.
    num_epochs : int
        Number of training epochs.
    device : Literal["cpu", "cuda", "mps"]
        Device to train on.
    mixed_effects : MixedEffectsConfig
        Mixed effects configuration for participant-level modeling.

    Examples
    --------
    >>> config = CategoricalModelConfig()
    >>> config.model_name
    'bert-base-uncased'
    >>> config.mixed_effects.mode
    'fixed'
    """

    model_name: str = Field(
        default="bert-base-uncased",
        description="HuggingFace model identifier",
    )
    max_length: int = Field(
        default=128,
        description="Maximum sequence length for tokenization",
        gt=0,
    )
    encoder_mode: Literal["single_encoder", "dual_encoder"] = Field(
        default="single_encoder",
        description="Encoding strategy for categories",
    )
    include_instructions: bool = Field(
        default=False,
        description="Whether to include task instructions",
    )
    learning_rate: float = Field(
        default=2e-5,
        description="Learning rate for AdamW optimizer",
        gt=0,
    )
    batch_size: int = Field(
        default=16,
        description="Batch size for training",
        gt=0,
    )
    num_epochs: int = Field(
        default=3,
        description="Number of training epochs",
        gt=0,
    )
    device: Literal["cpu", "cuda", "mps"] = Field(
        default="cpu",
        description="Device to train on",
    )
    mixed_effects: MixedEffectsConfig = Field(
        default_factory=MixedEffectsConfig,
        description="Mixed effects configuration for participant-level modeling",
    )


class BinaryModelConfig(BaseModel):
    """Configuration for binary active learning models.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.
    max_length : int
        Maximum sequence length for tokenization.
    encoder_mode : Literal["single_encoder", "dual_encoder"]
        Encoding strategy for binary classification.
    include_instructions : bool
        Whether to include task instructions.
    learning_rate : float
        Learning rate for AdamW optimizer.
    batch_size : int
        Batch size for training.
    num_epochs : int
        Number of training epochs.
    device : Literal["cpu", "cuda", "mps"]
        Device to train on.
    mixed_effects : MixedEffectsConfig
        Mixed effects configuration for participant-level modeling.

    Examples
    --------
    >>> config = BinaryModelConfig()
    >>> config.model_name
    'bert-base-uncased'
    >>> config.mixed_effects.mode
    'fixed'
    """

    model_name: str = Field(
        default="bert-base-uncased",
        description="HuggingFace model identifier",
    )
    max_length: int = Field(
        default=128,
        description="Maximum sequence length for tokenization",
        gt=0,
    )
    encoder_mode: Literal["single_encoder", "dual_encoder"] = Field(
        default="single_encoder",
        description="Encoding strategy for binary classification",
    )
    include_instructions: bool = Field(
        default=False,
        description="Whether to include task instructions",
    )
    learning_rate: float = Field(
        default=2e-5,
        description="Learning rate for AdamW optimizer",
        gt=0,
    )
    batch_size: int = Field(
        default=16,
        description="Batch size for training",
        gt=0,
    )
    num_epochs: int = Field(
        default=3,
        description="Number of training epochs",
        gt=0,
    )
    device: Literal["cpu", "cuda", "mps"] = Field(
        default="cpu",
        description="Device to train on",
    )
    mixed_effects: MixedEffectsConfig = Field(
        default_factory=MixedEffectsConfig,
        description="Mixed effects configuration for participant-level modeling",
    )


class MultiSelectModelConfig(BaseModel):
    """Configuration for multi-select active learning models.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.
    max_length : int
        Maximum sequence length for tokenization.
    encoder_mode : Literal["single_encoder", "dual_encoder"]
        Encoding strategy for multi-select options.
    include_instructions : bool
        Whether to include task instructions.
    learning_rate : float
        Learning rate for AdamW optimizer.
    batch_size : int
        Batch size for training.
    num_epochs : int
        Number of training epochs.
    device : Literal["cpu", "cuda", "mps"]
        Device to train on.
    mixed_effects : MixedEffectsConfig
        Mixed effects configuration for participant-level modeling.

    Examples
    --------
    >>> config = MultiSelectModelConfig()
    >>> config.model_name
    'bert-base-uncased'
    >>> config.mixed_effects.mode
    'fixed'
    """

    model_name: str = Field(
        default="bert-base-uncased",
        description="HuggingFace model identifier",
    )
    max_length: int = Field(
        default=128,
        description="Maximum sequence length for tokenization",
        gt=0,
    )
    encoder_mode: Literal["single_encoder", "dual_encoder"] = Field(
        default="single_encoder",
        description="Encoding strategy for multi-select options",
    )
    include_instructions: bool = Field(
        default=False,
        description="Whether to include task instructions",
    )
    learning_rate: float = Field(
        default=2e-5,
        description="Learning rate for AdamW optimizer",
        gt=0,
    )
    batch_size: int = Field(
        default=16,
        description="Batch size for training",
        gt=0,
    )
    num_epochs: int = Field(
        default=3,
        description="Number of training epochs",
        gt=0,
    )
    device: Literal["cpu", "cuda", "mps"] = Field(
        default="cpu",
        description="Device to train on",
    )
    mixed_effects: MixedEffectsConfig = Field(
        default_factory=MixedEffectsConfig,
        description="Mixed effects configuration for participant-level modeling",
    )


class ActiveLearningConfig(BaseModel):
    """Configuration for active learning infrastructure.

    Reflects the bead/active_learning/ module structure:
    - models: Active learning models (ForcedChoiceModel, etc.)
    - trainers: Training infrastructure (HuggingFace, Lightning)
    - loop: Active learning loop orchestration
    - selection: Item selection strategies (uncertainty sampling, etc.)

    Parameters
    ----------
    forced_choice_model : ForcedChoiceModelConfig
        Configuration for forced choice models.
    trainer : TrainerConfig
        Configuration for trainers (HuggingFace, Lightning).
    loop : ActiveLearningLoopConfig
        Configuration for active learning loop.
    uncertainty_sampler : UncertaintySamplerConfig
        Configuration for uncertainty sampling strategies.

    Examples
    --------
    >>> config = ActiveLearningConfig()
    >>> config.forced_choice_model.model_name
    'bert-base-uncased'
    >>> config.trainer.trainer_type
    'huggingface'
    >>> config.loop.max_iterations
    10
    >>> config.uncertainty_sampler.method
    'entropy'
    """

    forced_choice_model: ForcedChoiceModelConfig = Field(
        default_factory=ForcedChoiceModelConfig,
        description="Forced choice model configuration",
    )
    trainer: TrainerConfig = Field(
        default_factory=TrainerConfig,
        description="Trainer configuration",
    )
    loop: ActiveLearningLoopConfig = Field(
        default_factory=ActiveLearningLoopConfig,
        description="Active learning loop configuration",
    )
    uncertainty_sampler: UncertaintySamplerConfig = Field(
        default_factory=UncertaintySamplerConfig,
        description="Uncertainty sampler configuration",
    )
