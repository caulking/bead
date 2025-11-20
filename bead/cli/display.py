"""Rich display utilities for CLI commands.

This module provides centralized Rich display utilities for beautiful terminal output
across all bead CLI commands. All CLI modules should import from this module for
consistent formatting.

Examples
--------
>>> from bead.cli.display import print_header, print_success, create_summary_table
>>> print_header("Stage 1: Resources")
>>> print_success("Loaded 1,234 items")
>>> table = create_summary_table({"Items": "1,234", "Time": "2.3s"})
>>> console.print(table)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.spinner import Spinner
from rich.table import Table
from rich.traceback import install

# Install rich traceback globally for CLI
install(show_locals=True, width=100, word_wrap=True)

# Shared console instance
console = Console()


def print_header(text: str) -> None:
    """Print command header with horizontal rule.

    Parameters
    ----------
    text : str
        Header text to display.

    Examples
    --------
    >>> print_header("Stage 1: Resources")
    ═══════════════════════════════════════
                 Stage 1: Resources
    ═══════════════════════════════════════
    """
    console.rule(f"[bold]{text}[/bold]")


def print_success(text: str) -> None:
    """Print success message with green checkmark.

    Parameters
    ----------
    text : str
        Success message to display.

    Examples
    --------
    >>> print_success("Loaded 1,234 items")
    ✓ Loaded 1,234 items
    """
    console.print(f"[green]✓[/green] {text}")


def print_error(text: str) -> None:
    """Print error message with red X.

    Parameters
    ----------
    text : str
        Error message to display.

    Examples
    --------
    >>> print_error("Failed to load config")
    ✗ Failed to load config
    """
    console.print(f"[red]✗[/red] {text}")


def print_warning(text: str) -> None:
    """Print warning message with yellow warning sign.

    Parameters
    ----------
    text : str
        Warning message to display.

    Examples
    --------
    >>> print_warning("Using default strategy")
    ⚠ Warning: Using default strategy
    """
    console.print(f"[yellow]⚠[/yellow] {text}")


def print_info(text: str) -> None:
    """Print info message with blue info icon.

    Parameters
    ----------
    text : str
        Info message to display.

    Examples
    --------
    >>> print_info("Next step: Run partition command")
    ℹ Next step: Run partition command
    """
    console.print(f"[blue]ℹ[/blue] {text}")


def create_summary_table(
    data: dict[str, str],
    title: str | None = None,
    show_header: bool = False,
) -> Table:
    """Create formatted summary table.

    Parameters
    ----------
    data : dict[str, str]
        Dictionary mapping metric names to values.
    title : str | None, optional
        Table title. Default is None.
    show_header : bool, optional
        Whether to show table header. Default is False.

    Returns
    -------
    Table
        Formatted Rich Table object.

    Examples
    --------
    >>> table = create_summary_table({
    ...     "Items processed": "1,234",
    ...     "Success rate": "98.5%",
    ...     "Time": "45.2s"
    ... }, title="Summary")
    >>> console.print(table)
    """
    table = Table(show_header=show_header, title=title)
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", justify="right", style="green")

    for key, value in data.items():
        table.add_row(key, value)

    return table


def create_progress() -> Progress:
    """Create standard progress bar for CLI operations.

    Returns
    -------
    Progress
        Configured Rich Progress instance.

    Examples
    --------
    >>> with create_progress() as progress:
    ...     task = progress.add_task("Loading items...", total=1000)
    ...     for i in range(1000):
    ...         progress.advance(task)
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    )


def create_spinner_progress() -> Progress:
    """Create spinner-only progress (for indeterminate operations).

    Returns
    -------
    Progress
        Configured Rich Progress instance with spinner only.

    Examples
    --------
    >>> with create_spinner_progress() as progress:
    ...     progress.add_task("Loading model weights...", total=None)
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    )


def create_live_status(message: str) -> Live:
    """Create live status display with spinner.

    Parameters
    ----------
    message : str
        Status message to display.

    Returns
    -------
    Live
        Rich Live instance for status updates.

    Examples
    --------
    >>> with create_live_status("Training model..."):
    ...     model.train()
    """
    return Live(Spinner("dots", text=message), console=console)


def create_panel(
    content: str,
    title: str | None = None,
    style: str = "cyan",
) -> Panel:
    """Create formatted panel for important messages.

    Parameters
    ----------
    content : str
        Panel content.
    title : str | None, optional
        Panel title. Default is None.
    style : str, optional
        Panel border style color. Default is "cyan".

    Returns
    -------
    Panel
        Rich Panel object.

    Examples
    --------
    >>> panel = create_panel(
    ...     "Generating lexicons from VerbNet...",
    ...     title="In Progress"
    ... )
    >>> console.print(panel)
    """
    return Panel(content, title=f"[{style}]{title}[/{style}]" if title else None)


def display_file_stats(file_path: Path, count: int, item_type: str = "items") -> None:
    """Display statistics for a saved file.

    Parameters
    ----------
    file_path : Path
        Path to saved file.
    count : int
        Number of items in file.
    item_type : str, optional
        Type of items (for display). Default is "items".

    Examples
    --------
    >>> display_file_stats(Path("items.jsonl"), 1234, "items")
    ✓ Saved 1,234 items to items.jsonl
    """
    print_success(f"Saved {count:,} {item_type} to {file_path}")


def display_validation_errors(
    errors: list[str],
    max_display: int = 10,
) -> None:
    """Display validation errors with truncation.

    Parameters
    ----------
    errors : list[str]
        List of error messages.
    max_display : int, optional
        Maximum number of errors to display. Default is 10.

    Examples
    --------
    >>> errors = ["Line 1: Invalid JSON", "Line 5: Missing field"]
    >>> display_validation_errors(errors)
    """
    print_error(f"Validation failed with {len(errors)} error(s):")
    for error in errors[:max_display]:
        console.print(f"  [red]✗[/red] {error}")
    if len(errors) > max_display:
        console.print(f"  ... and {len(errors) - max_display} more error(s)")


def confirm(
    message: str,
    default: bool = False,
) -> bool:
    """Prompt user for yes/no confirmation.

    Parameters
    ----------
    message : str
        Confirmation message.
    default : bool, optional
        Default value if user presses Enter. Default is False.

    Returns
    -------
    bool
        True if user confirmed, False otherwise.

    Examples
    --------
    >>> if confirm("Delete all files?", default=False):
    ...     delete_files()
    """
    suffix = " [Y/n]: " if default else " [y/N]: "
    response = console.input(f"[yellow]?[/yellow] {message}{suffix}")

    if not response:
        return default

    return response.lower() in ("y", "yes")


def display_dry_run_summary(data: dict[str, Any]) -> None:
    """Display dry run summary.

    Parameters
    ----------
    data : dict[str, Any]
        Dictionary of dry run information.

    Examples
    --------
    >>> display_dry_run_summary({
    ...     "Templates": 26,
    ...     "Filled Templates": 1234,
    ...     "Output": "items.jsonl"
    ... })
    """
    print_info("[DRY RUN] Preview of operation:")
    for key, value in data.items():
        console.print(f"  {key}: {value}")
    print_warning("[DRY RUN] No changes will be made")


# Export commonly used items
__all__ = [
    "console",
    "print_header",
    "print_success",
    "print_error",
    "print_warning",
    "print_info",
    "create_summary_table",
    "create_progress",
    "create_spinner_progress",
    "create_live_status",
    "create_panel",
    "display_file_stats",
    "display_validation_errors",
    "confirm",
    "display_dry_run_summary",
]
