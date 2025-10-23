"""Main CLI entry point for sash package.

This module provides the main CLI command group and the init command for
project scaffolding.
"""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console

from sash import __version__
from sash.cli.utils import confirm, print_error, print_info, print_success

console = Console()


@click.group()
@click.version_option(version=__version__, prog_name="sash")
@click.option(
    "--config-file",
    "-c",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to configuration file (default: use profile defaults)",
)
@click.option(
    "--profile",
    "-p",
    type=click.Choice(["default", "dev", "prod", "test"], case_sensitive=False),
    default="default",
    help="Configuration profile to use",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Enable verbose output",
)
@click.option(
    "--quiet",
    "-q",
    is_flag=True,
    default=False,
    help="Suppress all output except errors",
)
@click.pass_context
def cli(
    ctx: click.Context,
    config_file: Path | None,
    profile: str,
    verbose: bool,
    quiet: bool,
) -> None:
    r"""Sash - Semantic Acceptability and Similarity Heuristics.

    A comprehensive CLI for constructing, deploying, and analyzing large-scale
    linguistic judgment experiments.

    \b
    Examples:
        # Show version
        $ sash --version

        # Use custom config file
        $ sash --config-file my-config.yaml config show

        # Use development profile
        $ sash --profile dev config show

        # Initialize new project
        $ sash init my-experiment

        # Show current configuration
        $ sash config show

        # Validate configuration
        $ sash config validate

    For more information, visit: https://github.com/aaronstevenwhite/sash
    """
    # Store options in context for subcommands
    ctx.ensure_object(dict)
    ctx.obj["config_file"] = config_file
    ctx.obj["profile"] = profile
    ctx.obj["verbose"] = verbose
    ctx.obj["quiet"] = quiet


@cli.command()
@click.argument("project_name", required=False, default=None)
@click.option(
    "--profile",
    type=click.Choice(["default", "dev", "prod", "test"], case_sensitive=False),
    default="default",
    help="Initialize with specific profile",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Overwrite existing directory",
)
def init(project_name: str | None, profile: str, force: bool) -> None:
    r"""Initialize a new sash project with scaffolding.

    Creates a new project directory with the following structure:

    \b
    project-name/
    ├── sash.yaml          # Default configuration
    ├── .gitignore         # Sash-specific ignores
    ├── lexicons/          # Lexical resources
    ├── templates/         # Template definitions
    ├── filled_templates/  # Generated filled templates
    ├── items/             # Generated items
    ├── lists/             # Generated experiment lists
    ├── experiments/       # Generated experiments
    ├── data/              # Collected data
    └── models/            # Trained models

    \b
    Examples:
        $ sash init my-experiment
        $ sash init my-experiment --profile dev
        $ sash init --force  # Overwrite current directory
    """
    # Use current directory if no project name provided
    if project_name is None:
        project_dir = Path.cwd()
        project_name = project_dir.name
        print_info(f"Using current directory: {project_dir}")
    else:
        # Validate project name
        if not _is_valid_project_name(project_name):
            print_error(
                f"Invalid project name: '{project_name}'. "
                "Use only letters, numbers, hyphens, and underscores."
            )
            return

        project_dir = Path.cwd() / project_name

    # Check if directory exists
    if project_dir.exists() and not force:
        if list(project_dir.iterdir()):
            if not confirm(
                f"Directory '{project_dir}' already exists and is not empty. "
                "Continue anyway?",
                default=False,
            ):
                print_info("Initialization cancelled.")
                return
    elif not project_dir.exists():
        project_dir.mkdir(parents=True)

    try:
        _create_project_structure(project_dir, project_name, profile)
        print_success(f"Project initialized: {project_dir}")
        print_info("\nNext steps:")
        print_info(
            f"  1. cd {project_name if project_name != project_dir.name else '.'}"
        )
        print_info("  2. Edit sash.yaml to configure your experiment")
        print_info("  3. Create lexicons in lexicons/")
        print_info("  4. Create templates in templates/")
        print_info("  5. Run: sash config validate")

    except Exception as e:
        print_error(f"Failed to initialize project: {e}")


def _is_valid_project_name(name: str) -> bool:
    """Validate project name.

    Parameters
    ----------
    name : str
        Project name to validate.

    Returns
    -------
    bool
        True if valid, False otherwise.
    """
    import re

    # Allow letters, numbers, hyphens, underscores
    return bool(re.match(r"^[a-zA-Z0-9_-]+$", name))


def _create_project_structure(
    project_dir: Path,
    project_name: str,
    profile: str,
) -> None:
    """Create project directory structure.

    Parameters
    ----------
    project_dir : Path
        Project directory path.
    project_name : str
        Project name.
    profile : str
        Configuration profile.
    """
    # Create subdirectories
    subdirs = [
        "lexicons",
        "templates",
        "filled_templates",
        "items",
        "lists",
        "experiments",
        "data",
        "models",
    ]

    for subdir in subdirs:
        (project_dir / subdir).mkdir(parents=True, exist_ok=True)

    # Create default sash.yaml
    config_content = _generate_default_config(project_name, profile)
    config_file = project_dir / "sash.yaml"
    config_file.write_text(config_content)

    # Create .gitignore
    gitignore_content = _generate_gitignore()
    gitignore_file = project_dir / ".gitignore"
    gitignore_file.write_text(gitignore_content)


def _generate_default_config(project_name: str, profile: str) -> str:
    """Generate default configuration file content.

    Parameters
    ----------
    project_name : str
        Project name.
    profile : str
        Configuration profile.

    Returns
    -------
    str
        YAML configuration content.
    """
    return f"""# sash Configuration
# Generated by: sash init

# Project metadata
profile: {profile}

# Logging configuration
logging:
  level: INFO
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: null  # Set to path for file logging

# Path configuration
paths:
  data_dir: .
  lexicons_dir: lexicons
  templates_dir: templates
  filled_templates_dir: filled_templates
  items_dir: items
  lists_dir: lists
  experiments_dir: experiments
  results_dir: data
  models_dir: models
  cache_dir: .cache

# Resource configuration
resources:
  auto_download: true
  cache_resources: true
  default_language: eng

# Template filling configuration
templates:
  filling_strategy: exhaustive
  max_combinations: null
  random_seed: null
  use_streaming: true
  constraint_cache_size: 10000

# Model configuration
models:
  default_language_model: gpt2
  default_nli_model: facebook/bart-large-mnli
  default_sentence_transformer: sentence-transformers/all-mpnet-base-v2
  use_gpu: true
  cache_model_outputs: true

# Item configuration
items:
  validation_enabled: true
  auto_save: true

# List configuration
lists:
  partitioning_strategy: balanced
  n_lists: 1
  quantile_balancing: false

# Deployment configuration
deployment:
  platform: jatos
  jspsych_version: "8.0.0"
  plugins: []
  include_attention_checks: true

# Training configuration (Stage 6)
training:
  framework: huggingface
  epochs: 10
  batch_size: 32
  learning_rate: 2.0e-5
  early_stopping: true
  early_stopping_patience: 3
"""


def _generate_gitignore() -> str:
    """Generate .gitignore file content.

    Returns
    -------
    str
        .gitignore content.
    """
    return """# sash-specific ignores

# Cache
.cache/
__pycache__/
*.pyc

# Model outputs
models/*.pt
models/*.pth
models/*.bin

# Data (comment out if you want to track data)
data/
*.jsonl

# Experiments
experiments/

# Python
*.egg-info/
dist/
build/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
"""


# Import command groups
from sash.cli.config import config  # noqa: E402
from sash.cli.deployment import deployment  # noqa: E402
from sash.cli.items import items  # noqa: E402
from sash.cli.lists import lists  # noqa: E402
from sash.cli.resources import resources  # noqa: E402
from sash.cli.templates import templates  # noqa: E402
from sash.cli.training import training  # noqa: E402

cli.add_command(config)
cli.add_command(resources)
cli.add_command(templates)
cli.add_command(items)
cli.add_command(lists)
cli.add_command(deployment)
cli.add_command(training)
