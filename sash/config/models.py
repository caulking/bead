"""Configuration models for the sash package.

This module provides Pydantic models for all configuration sections
with comprehensive validation rules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class PathsConfig(BaseModel):
    """Configuration for file system paths.

    Parameters
    ----------
    data_dir : Path
        Base directory for data files.
    output_dir : Path
        Base directory for outputs.
    cache_dir : Path
        Cache directory.
    temp_dir : Path | None
        Temporary directory. If None, uses system temp.
    create_dirs : bool
        Whether to create directories if they don't exist.

    Examples
    --------
    >>> config = PathsConfig()
    >>> config.data_dir
    PosixPath('data')
    >>> config = PathsConfig(data_dir=Path("/absolute/path"))
    >>> config.data_dir
    PosixPath('/absolute/path')
    """

    data_dir: Path = Field(
        default=Path("data"), description="Base directory for data files"
    )
    output_dir: Path = Field(
        default=Path("output"), description="Base directory for outputs"
    )
    cache_dir: Path = Field(default=Path(".cache"), description="Cache directory")
    temp_dir: Path | None = Field(default=None, description="Temporary directory")
    create_dirs: bool = Field(
        default=True, description="Create directories if they don't exist"
    )


class ResourceConfig(BaseModel):
    """Configuration for external resources.

    Parameters
    ----------
    lexicon_path : Path | None
        Path to lexicon file.
    templates_path : Path | None
        Path to templates file.
    constraints_path : Path | None
        Path to constraints file.
    external_adapters : list[str]
        List of external adapters to enable.
    cache_external : bool
        Whether to cache external resource lookups.

    Examples
    --------
    >>> config = ResourceConfig()
    >>> config.cache_external
    True
    >>> config.external_adapters
    []
    """

    lexicon_path: Path | None = Field(default=None, description="Path to lexicon file")
    templates_path: Path | None = Field(
        default=None, description="Path to templates file"
    )
    constraints_path: Path | None = Field(
        default=None, description="Path to constraints file"
    )
    external_adapters: list[str] = Field(
        default_factory=list, description="External adapters to enable"
    )
    cache_external: bool = Field(
        default=True, description="Cache external resource lookups"
    )


class TemplateConfig(BaseModel):
    """Configuration for template filling.

    Parameters
    ----------
    filling_strategy : str
        Strategy name for filling templates.
    batch_size : int
        Batch size for filling operations.
    max_combinations : int | None
        Maximum combinations to generate.
    random_seed : int | None
        Random seed for reproducibility.
    stream_mode : bool
        Use streaming for large templates.

    Examples
    --------
    >>> config = TemplateConfig()
    >>> config.filling_strategy
    'exhaustive'
    >>> config.batch_size
    1000
    """

    filling_strategy: Literal["exhaustive", "random", "stratified"] = Field(
        default="exhaustive", description="Strategy for filling templates"
    )
    batch_size: int = Field(default=1000, description="Batch size for filling", gt=0)
    max_combinations: int | None = Field(
        default=None, description="Max combinations to generate"
    )
    random_seed: int | None = Field(
        default=None, description="Random seed for reproducibility"
    )
    stream_mode: bool = Field(
        default=False, description="Use streaming for large templates"
    )

    @field_validator("max_combinations")
    @classmethod
    def validate_max_combinations(cls, v: int | None) -> int | None:
        """Validate max_combinations is positive.

        Parameters
        ----------
        v : int | None
            Max combinations value.

        Returns
        -------
        int | None
            Validated value.

        Raises
        ------
        ValueError
            If value is not positive.
        """
        if v is not None and v <= 0:
            msg = f"max_combinations must be positive, got {v}"
            raise ValueError(msg)
        return v


class ModelConfig(BaseModel):
    """Configuration for language models.

    Parameters
    ----------
    provider : str
        Model provider name.
    model_name : str
        Model identifier.
    batch_size : int
        Inference batch size.
    device : str
        Device to use for computation.
    max_length : int
        Maximum sequence length.
    temperature : float
        Sampling temperature.
    cache_outputs : bool
        Whether to cache model outputs.

    Examples
    --------
    >>> config = ModelConfig()
    >>> config.provider
    'huggingface'
    >>> config.device
    'cpu'
    """

    provider: Literal["huggingface", "openai", "anthropic"] = Field(
        default="huggingface", description="Model provider"
    )
    model_name: str = Field(default="gpt2", description="Model identifier")
    batch_size: int = Field(default=8, description="Inference batch size", gt=0)
    device: Literal["cpu", "cuda", "mps"] = Field(
        default="cpu", description="Device to use"
    )
    max_length: int = Field(default=512, description="Max sequence length", gt=0)
    temperature: float = Field(default=1.0, description="Sampling temperature", ge=0)
    cache_outputs: bool = Field(default=True, description="Cache model outputs")


class ItemConfig(BaseModel):
    """Configuration for item generation.

    Parameters
    ----------
    model : ModelConfig
        Model configuration.
    apply_constraints : bool
        Whether to apply model-based constraints.
    track_metadata : bool
        Whether to track item metadata.
    parallel_processing : bool
        Whether to use parallel processing.
    num_workers : int
        Number of workers for parallel processing.

    Examples
    --------
    >>> config = ItemConfig()
    >>> config.apply_constraints
    True
    >>> config.num_workers
    4
    """

    model: ModelConfig = Field(
        default_factory=ModelConfig, description="Model configuration"
    )
    apply_constraints: bool = Field(
        default=True, description="Apply model-based constraints"
    )
    track_metadata: bool = Field(default=True, description="Track item metadata")
    parallel_processing: bool = Field(
        default=False, description="Use parallel processing"
    )
    num_workers: int = Field(default=4, description="Number of workers", gt=0)


class ListConfig(BaseModel):
    """Configuration for list partitioning.

    Parameters
    ----------
    partitioning_strategy : str
        Strategy name for partitioning.
    num_lists : int
        Number of lists to create.
    items_per_list : int | None
        Items per list.
    balance_by : list[str]
        Fields to balance on.
    ensure_uniqueness : bool
        Whether to ensure items are unique across lists.
    random_seed : int | None
        Random seed for reproducibility.

    Examples
    --------
    >>> config = ListConfig()
    >>> config.partitioning_strategy
    'balanced'
    >>> config.num_lists
    1
    """

    partitioning_strategy: str = Field(
        default="balanced", description="Partitioning strategy"
    )
    num_lists: int = Field(default=1, description="Number of lists to create", gt=0)
    items_per_list: int | None = Field(default=None, description="Items per list")
    balance_by: list[str] = Field(
        default_factory=list, description="Fields to balance on"
    )
    ensure_uniqueness: bool = Field(
        default=True, description="Ensure items unique across lists"
    )
    random_seed: int | None = Field(default=None, description="Random seed")

    @field_validator("items_per_list")
    @classmethod
    def validate_items_per_list(cls, v: int | None) -> int | None:
        """Validate items_per_list is positive.

        Parameters
        ----------
        v : int | None
            Items per list value.

        Returns
        -------
        int | None
            Validated value.

        Raises
        ------
        ValueError
            If value is not positive.
        """
        if v is not None and v <= 0:
            msg = f"items_per_list must be positive, got {v}"
            raise ValueError(msg)
        return v


class DeploymentConfig(BaseModel):
    """Configuration for experiment deployment.

    Parameters
    ----------
    platform : str
        Deployment platform.
    jspsych_version : str
        jsPsych version to use.
    apply_material_design : bool
        Whether to use Material Design.
    include_demographics : bool
        Whether to include demographics survey.
    include_attention_checks : bool
        Whether to include attention checks.
    jatos_export : bool
        Whether to export to JATOS.

    Examples
    --------
    >>> config = DeploymentConfig()
    >>> config.platform
    'jspsych'
    >>> config.jspsych_version
    '7.3.0'
    """

    platform: str = Field(default="jspsych", description="Deployment platform")
    jspsych_version: str = Field(default="7.3.0", description="jsPsych version")
    apply_material_design: bool = Field(default=True, description="Use Material Design")
    include_demographics: bool = Field(
        default=True, description="Include demographics survey"
    )
    include_attention_checks: bool = Field(
        default=True, description="Include attention checks"
    )
    jatos_export: bool = Field(default=False, description="Export to JATOS")


class TrainingConfig(BaseModel):
    """Configuration for model training.

    Parameters
    ----------
    trainer : str
        Trainer type.
    epochs : int
        Number of training epochs.
    learning_rate : float
        Learning rate.
    batch_size : int
        Training batch size.
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
    >>> config = TrainingConfig()
    >>> config.trainer
    'huggingface'
    >>> config.epochs
    3
    """

    trainer: str = Field(default="huggingface", description="Trainer type")
    epochs: int = Field(default=3, description="Training epochs", gt=0)
    learning_rate: float = Field(default=2e-5, description="Learning rate", gt=0)
    batch_size: int = Field(default=16, description="Training batch size", gt=0)
    eval_strategy: str = Field(default="epoch", description="Evaluation strategy")
    save_strategy: str = Field(default="epoch", description="Save strategy")
    logging_dir: Path = Field(default=Path("logs"), description="Logging directory")
    use_wandb: bool = Field(default=False, description="Use Weights & Biases")
    wandb_project: str | None = Field(default=None, description="W&B project name")


class LoggingConfig(BaseModel):
    """Configuration for logging.

    Parameters
    ----------
    level : str
        Log level.
    format : str
        Log format string.
    file : Path | None
        Log file path.
    console : bool
        Whether to log to console.

    Examples
    --------
    >>> config = LoggingConfig()
    >>> config.level
    'INFO'
    >>> config.console
    True
    """

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Log level"
    )
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format",
    )
    file: Path | None = Field(default=None, description="Log file path")
    console: bool = Field(default=True, description="Log to console")


class SashConfig(BaseModel):
    """Main configuration for the sash package.

    Parameters
    ----------
    profile : str
        Configuration profile name.
    paths : PathsConfig
        Paths configuration.
    resources : ResourceConfig
        Resources configuration.
    templates : TemplateConfig
        Templates configuration.
    items : ItemConfig
        Items configuration.
    lists : ListConfig
        Lists configuration.
    deployment : DeploymentConfig
        Deployment configuration.
    training : TrainingConfig
        Training configuration.
    logging : LoggingConfig
        Logging configuration.

    Examples
    --------
    >>> config = SashConfig()
    >>> config.profile
    'default'
    >>> config.paths.data_dir
    PosixPath('data')
    """

    profile: str = Field(default="default", description="Configuration profile name")
    paths: PathsConfig = Field(
        default_factory=PathsConfig, description="Paths configuration"
    )
    resources: ResourceConfig = Field(
        default_factory=ResourceConfig, description="Resources configuration"
    )
    templates: TemplateConfig = Field(
        default_factory=TemplateConfig, description="Templates configuration"
    )
    items: ItemConfig = Field(
        default_factory=ItemConfig, description="Items configuration"
    )
    lists: ListConfig = Field(
        default_factory=ListConfig, description="Lists configuration"
    )
    deployment: DeploymentConfig = Field(
        default_factory=DeploymentConfig, description="Deployment configuration"
    )
    training: TrainingConfig = Field(
        default_factory=TrainingConfig, description="Training configuration"
    )
    logging: LoggingConfig = Field(
        default_factory=LoggingConfig, description="Logging configuration"
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Returns
        -------
        dict[str, Any]
            Configuration as a dictionary.

        Examples
        --------
        >>> config = SashConfig()
        >>> d = config.to_dict()
        >>> d["profile"]
        'default'
        """
        return self.model_dump()

    def to_yaml(self) -> str:
        """Convert configuration to YAML string.

        Returns
        -------
        str
            Configuration as YAML string.

        Raises
        ------
        NotImplementedError
            This feature will be implemented in Phase 4.

        Notes
        -----
        This method is a placeholder for Phase 4 implementation.
        """
        msg = "to_yaml() will be implemented in Phase 4"
        raise NotImplementedError(msg)

    def validate_paths(self) -> list[str]:
        """Validate all path fields exist.

        Returns
        -------
        list[str]
            List of validation errors. Empty if all paths are valid.

        Examples
        --------
        >>> config = SashConfig()
        >>> errors = config.validate_paths()
        >>> len(errors)
        0
        """
        errors: list[str] = []

        # Check paths config
        if not self.paths.data_dir.exists() and self.paths.data_dir.is_absolute():
            errors.append(f"data_dir does not exist: {self.paths.data_dir}")
        if not self.paths.output_dir.exists() and self.paths.output_dir.is_absolute():
            errors.append(f"output_dir does not exist: {self.paths.output_dir}")
        if not self.paths.cache_dir.exists() and self.paths.cache_dir.is_absolute():
            errors.append(f"cache_dir does not exist: {self.paths.cache_dir}")
        if self.paths.temp_dir is not None and not self.paths.temp_dir.exists():
            errors.append(f"temp_dir does not exist: {self.paths.temp_dir}")

        # Check resource paths
        if (
            self.resources.lexicon_path is not None
            and not self.resources.lexicon_path.exists()
        ):
            errors.append(f"lexicon_path does not exist: {self.resources.lexicon_path}")
        if (
            self.resources.templates_path is not None
            and not self.resources.templates_path.exists()
        ):
            errors.append(
                f"templates_path does not exist: {self.resources.templates_path}"
            )
        if (
            self.resources.constraints_path is not None
            and not self.resources.constraints_path.exists()
        ):
            errors.append(
                f"constraints_path does not exist: {self.resources.constraints_path}"
            )

        # Check training logging dir
        if (
            not self.training.logging_dir.exists()
            and self.training.logging_dir.is_absolute()
        ):
            errors.append(f"logging_dir does not exist: {self.training.logging_dir}")

        # Check logging file
        if self.logging.file is not None and not self.logging.file.parent.exists():
            parent_dir = self.logging.file.parent
            errors.append(f"logging file parent directory does not exist: {parent_dir}")

        return errors
