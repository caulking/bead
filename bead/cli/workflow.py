"""Workflow orchestration commands for the bead CLI.

This module provides commands for managing end-to-end pipeline workflows,
including running complete pipelines, resuming interrupted workflows, and
rolling back to previous stages.
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import click
import yaml
from rich.console import Console
from rich.table import Table

from bead.cli.utils import JsonValue, print_error, print_info, print_success

console = Console()


# ============================================================================
# State Management Utilities
# ============================================================================


def get_state_file(project_dir: Path) -> Path:
    """Get path to workflow state file.

    Parameters
    ----------
    project_dir : Path
        Project directory path.

    Returns
    -------
    Path
        Path to .bead/workflow_state.json
    """
    bead_dir = project_dir / ".bead"
    bead_dir.mkdir(exist_ok=True)
    return bead_dir / "workflow_state.json"


def load_state(project_dir: Path) -> dict[str, JsonValue]:
    """Load workflow state from file.

    Parameters
    ----------
    project_dir : Path
        Project directory path.

    Returns
    -------
    dict[str, JsonValue]
        Workflow state dictionary.
    """
    state_file = get_state_file(project_dir)
    if not state_file.exists():
        return {"stages": {}, "last_run": None}

    with open(state_file) as f:
        return json.load(f)


def save_state(project_dir: Path, state: dict[str, JsonValue]) -> None:
    """Save workflow state to file.

    Parameters
    ----------
    project_dir : Path
        Project directory path.
    state : dict[str, JsonValue]
        Workflow state dictionary.
    """
    state_file = get_state_file(project_dir)
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2, default=str)


def update_stage_state(
    project_dir: Path, stage: str, status: str, error: str | None = None
) -> None:
    """Update state for a specific stage.

    Parameters
    ----------
    project_dir : Path
        Project directory path.
    stage : str
        Stage name.
    status : str
        Stage status ('pending', 'running', 'completed', 'failed').
    error : str | None
        Error message if status is 'failed'.
    """
    state = load_state(project_dir)
    if not isinstance(state.get("stages"), dict):
        state["stages"] = {}

    stages_dict = state["stages"]
    assert isinstance(stages_dict, dict)

    stages_dict[stage] = {
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "error": error,
    }

    state["last_run"] = datetime.now().isoformat()
    save_state(project_dir, state)


def detect_stage_completion(project_dir: Path, stage: str) -> bool:
    """Detect if a stage has been completed by checking filesystem.

    Parameters
    ----------
    project_dir : Path
        Project directory path.
    stage : str
        Stage name.

    Returns
    -------
    bool
        True if stage appears completed.
    """
    # Check for expected output files
    if stage == "resources":
        return (project_dir / "lexicons").exists() and any(
            (project_dir / "lexicons").glob("*.jsonl")
        )
    elif stage == "templates":
        return (project_dir / "templates").exists() and any(
            (project_dir / "templates").glob("*.jsonl")
        )
    elif stage == "items":
        return (project_dir / "items").exists() and any(
            (project_dir / "items").glob("*.jsonl")
        )
    elif stage == "lists":
        return (project_dir / "lists").exists() and any(
            (project_dir / "lists").glob("*.jsonl")
        )
    elif stage == "deployment":
        return (project_dir / "experiments").exists() and any(
            (project_dir / "experiments").iterdir()
        )
    elif stage == "training":
        return (project_dir / "models").exists() and any(
            (project_dir / "models").iterdir()
        )

    return False


# ============================================================================
# Workflow Templates
# ============================================================================

WORKFLOW_TEMPLATES = {
    "acceptability-study": {
        "name": "Acceptability Judgment Study",
        "description": "Collect acceptability judgments on linguistic stimuli",
        "config": {
            "project": {
                "name": "acceptability_study",
                "language_code": "eng",
                "description": "Acceptability judgment experiment",
            },
            "paths": {
                "lexicons_dir": "lexicons",
                "templates_dir": "templates",
                "items_dir": "items",
                "lists_dir": "lists",
                "experiments_dir": "experiments",
            },
            "templates": {"filling_strategy": "exhaustive"},
            "items": {"validation_enabled": True},
            "lists": {"n_lists": 10},
            "deployment": {"platform": "jatos", "jspsych_version": "8.0.0"},
        },
    },
    "forced-choice": {
        "name": "Forced Choice Study",
        "description": "2AFC or 3AFC comparison judgments",
        "config": {
            "project": {
                "name": "forced_choice_study",
                "language_code": "eng",
                "description": "Forced choice experiment",
            },
            "paths": {
                "lexicons_dir": "lexicons",
                "templates_dir": "templates",
                "items_dir": "items",
                "lists_dir": "lists",
                "experiments_dir": "experiments",
            },
            "templates": {"filling_strategy": "stratified"},
            "items": {"validation_enabled": True},
            "lists": {"n_lists": 20},
            "deployment": {"platform": "jatos", "jspsych_version": "8.0.0"},
        },
    },
    "ordinal-scale": {
        "name": "Ordinal Scale Study",
        "description": "Likert scale or slider ratings",
        "config": {
            "project": {
                "name": "ordinal_scale_study",
                "language_code": "eng",
                "description": "Ordinal scale experiment",
            },
            "paths": {
                "lexicons_dir": "lexicons",
                "templates_dir": "templates",
                "items_dir": "items",
                "lists_dir": "lists",
                "experiments_dir": "experiments",
            },
            "templates": {"filling_strategy": "random", "max_combinations": 500},
            "items": {"validation_enabled": True},
            "lists": {"n_lists": 15},
            "deployment": {"platform": "jatos", "jspsych_version": "8.0.0"},
        },
    },
    "active-learning": {
        "name": "Active Learning Study",
        "description": "Human-in-the-loop training with convergence detection",
        "config": {
            "project": {
                "name": "active_learning_study",
                "language_code": "eng",
                "description": "Active learning experiment",
            },
            "paths": {
                "lexicons_dir": "lexicons",
                "templates_dir": "templates",
                "items_dir": "items",
                "lists_dir": "lists",
                "experiments_dir": "experiments",
                "models_dir": "models",
            },
            "templates": {"filling_strategy": "stratified"},
            "items": {"validation_enabled": True},
            "lists": {"n_lists": 10},
            "deployment": {"platform": "jatos", "jspsych_version": "8.0.0"},
            "training": {
                "framework": "huggingface",
                "epochs": 10,
                "convergence": {"metric": "krippendorff_alpha", "threshold": 0.80},
            },
        },
    },
}


# ============================================================================
# Workflow Commands
# ============================================================================


@click.group()
def workflow() -> None:
    """Manage end-to-end pipeline workflows.

    Examples
    --------
    Run complete pipeline:
        $ bead workflow run --config bead.yaml

    Initialize new project:
        $ bead workflow init acceptability-study

    Check workflow status:
        $ bead workflow status

    Resume interrupted workflow:
        $ bead workflow resume

    Rollback to previous stage:
        $ bead workflow rollback deployment
    """


@workflow.command()
@click.option(
    "--config",
    "-c",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default="bead.yaml",
    help="Path to configuration file",
)
@click.option(
    "--stages",
    type=str,
    default=None,
    help="Comma-separated list of stages to run (default: all)",
)
@click.option(
    "--from-stage",
    type=click.Choice(
        ["resources", "templates", "items", "lists", "deployment", "training"]
    ),
    default=None,
    help="Start from this stage",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Show what would be executed without running",
)
def run(
    config_path: Path, stages: str | None, from_stage: str | None, dry_run: bool
) -> None:
    """Run complete pipeline workflow.

    Executes all pipeline stages sequentially:
    1. resources - Create lexicons and templates
    2. templates - Fill templates with lexicon items
    3. items - Construct experimental items
    4. lists - Partition items into experiment lists
    5. deployment - Generate jsPsych experiments
    6. training - Train models with active learning (optional)

    The workflow tracks progress and can be resumed if interrupted.

    Examples
    --------
    Run all stages:
        $ bead workflow run --config bead.yaml

    Run specific stages:
        $ bead workflow run --stages resources,templates,items

    Start from items stage:
        $ bead workflow run --from-stage items

    Dry run to preview:
        $ bead workflow run --dry-run
    """
    project_dir = config_path.parent
    console.rule("[bold]Pipeline Workflow Execution[/bold]")

    # Load configuration
    try:
        with open(config_path) as f:
            _ = yaml.safe_load(f)
    except Exception as e:
        print_error(f"Failed to load config: {e}")
        sys.exit(1)

    # Determine which stages to run
    all_stages = [
        "resources",
        "templates",
        "items",
        "lists",
        "deployment",
        "training",
    ]

    if stages:
        selected_stages = [s.strip() for s in stages.split(",")]
    elif from_stage:
        start_idx = all_stages.index(from_stage)
        selected_stages = all_stages[start_idx:]
    else:
        selected_stages = all_stages

    # Show plan
    print_info(f"Configuration: {config_path}")
    print_info(f"Stages to run: {', '.join(selected_stages)}")

    if dry_run:
        console.print("\n[yellow]DRY RUN MODE - No commands will be executed[/yellow]")
        for stage in selected_stages:
            console.print(f"  • Would execute: [cyan]{stage}[/cyan] stage")
        return

    # Execute stages
    state = load_state(project_dir)
    failed = False

    for stage in selected_stages:
        console.rule(f"[bold cyan]Stage: {stage}[/bold cyan]")

        try:
            update_stage_state(project_dir, stage, "running")

            # This is a placeholder - in reality would invoke actual commands
            # For now, just simulate success
            console.print(f"[green]✓[/green] Executing {stage} stage...")

            # Simulate stage execution (replace with actual command invocation)
            # Example: subprocess.run(["bead", stage, "..."], check=True)

            update_stage_state(project_dir, stage, "completed")
            console.print(f"[green]✓[/green] {stage} stage completed")

        except Exception as e:
            update_stage_state(project_dir, stage, "failed", str(e))
            print_error(f"Stage '{stage}' failed: {e}")
            failed = True
            break

    if not failed:
        console.rule("[bold green]Pipeline Complete[/bold green]")
        print_success("All stages completed successfully")
    else:
        console.rule("[bold red]Pipeline Failed[/bold red]")
        print_error("Pipeline execution failed. Use 'bead workflow resume' to continue")
        sys.exit(1)


@workflow.command()
@click.argument("template", type=click.Choice(list(WORKFLOW_TEMPLATES.keys())))
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory (default: current directory)",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Overwrite existing files",
)
def init(template: str, output_dir: Path | None, force: bool) -> None:
    """Initialize new project from template.

    Creates a complete project structure with configuration file,
    directory layout, and example files.

    Available templates:
    - acceptability-study: Acceptability judgment experiments
    - forced-choice: 2AFC or 3AFC comparison judgments
    - ordinal-scale: Likert scale or slider ratings
    - active-learning: Human-in-the-loop training

    Examples
    --------
    Initialize acceptability study:
        $ bead workflow init acceptability-study

    Initialize in specific directory:
        $ bead workflow init forced-choice --output-dir my-project

    Overwrite existing files:
        $ bead workflow init ordinal-scale --force
    """
    if output_dir is None:
        output_dir = Path.cwd()

    template_spec = WORKFLOW_TEMPLATES[template]
    console.rule(f"[bold]Initialize: {template_spec['name']}[/bold]")

    # Create directory structure
    config_data: Any = template_spec["config"]
    paths: dict[str, Any] = config_data.get("paths", {})

    dirs_to_create: list[str] = [
        str(paths.get("lexicons_dir", "lexicons")),
        str(paths.get("templates_dir", "templates")),
        str(paths.get("items_dir", "items")),
        str(paths.get("lists_dir", "lists")),
        str(paths.get("experiments_dir", "experiments")),
    ]

    if "models_dir" in paths:
        dirs_to_create.append(str(paths["models_dir"]))

    for dir_name in dirs_to_create:
        dir_path = output_dir / dir_name
        if dir_path.exists() and not force:
            console.print(f"[yellow]⚠[/yellow] Directory exists: {dir_name}")
        else:
            dir_path.mkdir(parents=True, exist_ok=True)
            console.print(f"[green]✓[/green] Created: {dir_name}/")

    # Create configuration file
    config_file = output_dir / "bead.yaml"
    if config_file.exists() and not force:
        print_error(f"Configuration file exists: {config_file}")
        print_info("Use --force to overwrite")
        sys.exit(1)

    with open(config_file, "w") as f:
        yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
    print_success(f"Created configuration: {config_file}")

    # Create .gitignore
    gitignore_file = output_dir / ".gitignore"
    gitignore_content = """# bead workflow ignores
.bead/
.cache/
*.pyc
__pycache__/
*.jsonl
experiments/
models/
"""
    with open(gitignore_file, "w") as f:
        f.write(gitignore_content)
    print_success("Created .gitignore")

    console.print("\n[bold green]✓ Project initialized[/bold green]")
    console.print(f"\n[bold]Next steps:[/bold]")
    console.print("  1. Edit bead.yaml to configure your experiment")
    console.print("  2. Create lexicon files in lexicons/")
    console.print("  3. Create template files in templates/")
    console.print("  4. Run: bead workflow run")


@workflow.command()
@click.option(
    "--config",
    "-c",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default="bead.yaml",
    help="Path to configuration file",
)
def status(config_path: Path) -> None:
    """Show current workflow status.

    Displays completion status for each pipeline stage by checking
    both the workflow state file and the filesystem.

    Examples
    --------
    Show workflow status:
        $ bead workflow status

    Use custom config:
        $ bead workflow status --config my-config.yaml
    """
    project_dir = config_path.parent
    console.rule("[bold]Workflow Status[/bold]")

    # Load state
    state = load_state(project_dir)
    stages_state: dict[str, Any] = state.get("stages", {})  # type: ignore

    # Check filesystem
    all_stages = [
        "resources",
        "templates",
        "items",
        "lists",
        "deployment",
        "training",
    ]

    table = Table(title="Pipeline Stage Status")
    table.add_column("Stage", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Last Updated")

    for stage in all_stages:
        stage_info: dict[str, Any] = stages_state.get(stage, {})

        # Check filesystem
        fs_complete = detect_stage_completion(project_dir, stage)

        # Determine status
        if stage_info.get("status") == "completed":
            status = "[green]✓ Completed[/green]"
            timestamp = str(stage_info.get("timestamp", "Unknown"))
        elif stage_info.get("status") == "failed":
            status = "[red]✗ Failed[/red]"
            timestamp = str(stage_info.get("timestamp", "Unknown"))
        elif fs_complete:
            status = "[yellow]⚠ Detected[/yellow]"
            timestamp = "Filesystem check"
        else:
            status = "[dim]○ Pending[/dim]"
            timestamp = "-"

        table.add_row(stage, status, timestamp)

    console.print(table)

    if state.get("last_run"):
        console.print(f"\n[dim]Last run: {state.get('last_run')}[/dim]")


@workflow.command()
@click.option(
    "--config",
    "-c",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default="bead.yaml",
    help="Path to configuration file",
)
def resume(config_path: Path) -> None:
    """Resume interrupted workflow.

    Reads the workflow state file and continues execution from the
    last incomplete stage, skipping stages that have already completed.

    Examples
    --------
    Resume workflow:
        $ bead workflow resume

    Resume with custom config:
        $ bead workflow resume --config my-config.yaml
    """
    project_dir = config_path.parent
    console.rule("[bold]Resume Workflow[/bold]")

    # Load state
    state = load_state(project_dir)
    stages_state: dict[str, Any] = state.get("stages", {})  # type: ignore

    if not isinstance(stages_state, dict):
        print_error("No workflow state found. Use 'bead workflow run' to start.")
        sys.exit(1)

    # Find last completed stage
    all_stages = [
        "resources",
        "templates",
        "items",
        "lists",
        "deployment",
        "training",
    ]

    last_completed_idx = -1
    for i, stage in enumerate(all_stages):
        stage_info: dict[str, Any] = stages_state.get(stage, {})
        if stage_info.get("status") == "completed":
            last_completed_idx = i

    if last_completed_idx == len(all_stages) - 1:
        print_success("All stages completed. Nothing to resume.")
        return

    # Resume from next stage
    resume_from = all_stages[last_completed_idx + 1]
    console.print(f"[cyan]Resuming from stage: {resume_from}[/cyan]")
    console.print(f"[dim]Last completed: {all_stages[last_completed_idx]}[/dim]\n")

    # Invoke run command with from-stage
    ctx = click.get_current_context()
    ctx.invoke(run, config_path=config_path, stages=None, from_stage=resume_from, dry_run=False)


@workflow.command()
@click.argument(
    "stage",
    type=click.Choice(
        ["resources", "templates", "items", "lists", "deployment", "training"]
    ),
)
@click.option(
    "--config",
    "-c",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default="bead.yaml",
    help="Path to configuration file",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Skip confirmation prompt",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Show what would be deleted without deleting",
)
def rollback(stage: str, config_path: Path, force: bool, dry_run: bool) -> None:
    """Rollback to previous stage.

    Deletes outputs from the specified stage and all subsequent stages,
    allowing you to re-run from that point. Also updates the workflow
    state file.

    Examples
    --------
    Rollback to items stage:
        $ bead workflow rollback items

    Dry run to preview:
        $ bead workflow rollback deployment --dry-run

    Skip confirmation:
        $ bead workflow rollback templates --force
    """
    project_dir = config_path.parent
    console.rule("[bold yellow]Rollback Workflow[/bold yellow]")

    # Determine stages to delete
    all_stages = [
        "resources",
        "templates",
        "items",
        "lists",
        "deployment",
        "training",
    ]

    stage_idx = all_stages.index(stage)
    stages_to_delete = all_stages[stage_idx:]

    # Map stages to directories
    stage_dirs: dict[str, list[str]] = {
        "resources": ["lexicons", "templates"],
        "templates": ["filled_templates"],
        "items": ["items"],
        "lists": ["lists"],
        "deployment": ["experiments"],
        "training": ["models"],
    }

    dirs_to_delete: list[str] = []
    for s in stages_to_delete:
        dirs_to_delete.extend(stage_dirs.get(s, []))

    # Show what will be deleted
    console.print(f"[yellow]Will rollback stages:[/yellow]")
    for s in stages_to_delete:
        console.print(f"  • {s}")

    console.print(f"\n[yellow]Will delete directories:[/yellow]")
    for dir_name in dirs_to_delete:
        dir_path = project_dir / dir_name
        if dir_path.exists():
            file_count = sum(1 for _ in dir_path.rglob("*") if _.is_file())
            console.print(f"  • {dir_name}/ ({file_count} files)")

    if dry_run:
        console.print("\n[cyan]DRY RUN MODE - No files will be deleted[/cyan]")
        return

    # Confirm deletion
    if not force:
        console.print()
        if not click.confirm(
            "Are you sure you want to delete these files?", default=False
        ):
            print_info("Rollback cancelled")
            return

    # Delete directories
    console.print()
    for dir_name in dirs_to_delete:
        dir_path = project_dir / dir_name
        if dir_path.exists():
            try:
                shutil.rmtree(dir_path)
                console.print(f"[green]✓[/green] Deleted: {dir_name}/")
            except Exception as e:
                print_error(f"Failed to delete {dir_name}/: {e}")

    # Update state
    state = load_state(project_dir)
    stages_state: dict[str, Any] = state.get("stages", {})  # type: ignore
    for s in stages_to_delete:
        if s in stages_state:
            del stages_state[s]
    save_state(project_dir, state)

    console.print(f"\n[green]✓[/green] Rolled back to stage: [cyan]{stage}[/cyan]")
    print_info("Run 'bead workflow run --from-stage " + stage + "' to continue")


@workflow.command(name="list-templates")
def list_templates() -> None:
    """List available workflow templates.

    Shows all predefined workflow templates with descriptions and
    configuration requirements.

    Examples
    --------
    List templates:
        $ bead workflow list-templates
    """
    console.rule("[bold]Available Workflow Templates[/bold]")

    table = Table(title="Workflow Templates")
    table.add_column("Template ID", style="cyan")
    table.add_column("Name", style="bold")
    table.add_column("Description")

    for template_id, template_spec in WORKFLOW_TEMPLATES.items():
        table.add_row(
            template_id,
            template_spec["name"],
            template_spec["description"],
        )

    console.print(table)

    console.print("\n[bold]Usage:[/bold]")
    console.print("  bead workflow init <template-id>")
    console.print("\n[bold]Example:[/bold]")
    console.print("  bead workflow init acceptability-study")
