"""Active learning commands for bead CLI.

This module provides commands for active learning workflows including item
selection and convergence monitoring.
"""

from __future__ import annotations

import json
from pathlib import Path

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from bead.cli.utils import print_error, print_info, print_success
from bead.evaluation.convergence import ConvergenceDetector

console = Console()


@click.group()
def active_learning() -> None:
    r"""Active learning commands (Phase 5.2 - PARTIAL).

    Commands for convergence detection in active learning workflows.

    \b
    AVAILABLE COMMANDS:
        check-convergence    Check if model converged to human agreement

    \b
    DEFERRED COMMANDS (awaiting full implementation):
        select-items         Requires model.predict_proba implementation
        run                  Requires data collection infrastructure
        monitor-convergence  Requires checkpoint loading infrastructure

    \b
    Examples:
        # Check convergence
        $ bead active-learning check-convergence \\
            --predictions predictions.jsonl \\
            --human-labels labels.jsonl \\
            --metric krippendorff_alpha \\
            --threshold 0.85
    """


@click.command()
@click.option(
    "--predictions",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to model predictions file (JSONL with 'prediction' field)",
)
@click.option(
    "--human-labels",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to human labels file (JSONL with 'label' field per rater)",
)
@click.option(
    "--metric",
    type=click.Choice(
        ["krippendorff_alpha", "fleiss_kappa", "cohens_kappa", "percentage_agreement"],
        case_sensitive=False,
    ),
    default="krippendorff_alpha",
    help="Agreement metric to use (default: krippendorff_alpha)",
)
@click.option(
    "--threshold",
    type=float,
    default=0.80,
    help="Convergence threshold (default: 0.80)",
)
@click.option(
    "--min-iterations",
    type=int,
    default=1,
    help="Minimum iterations before checking convergence (default: 1)",
)
@click.pass_context
def check_convergence(
    ctx: click.Context,
    predictions: Path,
    human_labels: Path,
    metric: str,
    threshold: float,
    min_iterations: int,
) -> None:
    r"""Check if model has converged to human agreement level.

    Compares model predictions with human labels using inter-annotator
    agreement metrics to determine convergence. This is a FULLY IMPLEMENTED
    command that uses actual ConvergenceDetector from bead.evaluation.

    Parameters
    ----------
    ctx : click.Context
        Click context object.
    predictions : Path
        Path to model predictions file.
    human_labels : Path
        Path to human labels file.
    metric : str
        Agreement metric name.
    threshold : float
        Convergence threshold.
    min_iterations : int
        Minimum iterations before allowing convergence.

    Examples
    --------
    $ bead active-learning check-convergence \\
        --predictions predictions.jsonl \\
        --human-labels labels.jsonl \\
        --metric krippendorff_alpha \\
        --threshold 0.85

    $ bead active-learning check-convergence \\
        --predictions predictions.jsonl \\
        --human-labels labels.jsonl \\
        --metric fleiss_kappa \\
        --threshold 0.75
    """
    try:
        console.rule("[bold]Convergence Check[/bold]")

        # Load predictions
        print_info(f"Loading predictions from {predictions}")
        with open(predictions, encoding="utf-8") as f:
            pred_records = [json.loads(line) for line in f if line.strip()]

        model_predictions = [r["prediction"] for r in pred_records]
        print_success(f"Loaded {len(model_predictions)} predictions")

        # Load human labels (organized by rater)
        print_info(f"Loading human labels from {human_labels}")
        with open(human_labels, encoding="utf-8") as f:
            label_records = [json.loads(line) for line in f if line.strip()]

        # Organize by rater
        rater_labels: dict[str, list[int | str | float]] = {}
        for record in label_records:
            rater_id = str(record.get("rater_id", "rater_1"))
            label = record["label"]
            if rater_id not in rater_labels:
                rater_labels[rater_id] = []
            rater_labels[rater_id].append(label)

        n_raters = len(rater_labels)
        print_success(f"Loaded labels from {n_raters} raters")

        # Create convergence detector
        print_info(f"Computing {metric}...")
        detector = ConvergenceDetector(
            human_agreement_metric=metric,
            convergence_threshold=threshold,
            min_iterations=min_iterations,
            statistical_test=True,
        )

        # Compute human baseline
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Computing human agreement baseline...", total=None)
            human_baseline = detector.compute_human_baseline(rater_labels)

        print_success(f"Human baseline: {human_baseline:.4f}")

        # Add model as another "rater" for comparison
        all_raters = {**rater_labels, "model": model_predictions}

        # Compute agreement including model
        if metric == "krippendorff_alpha":
            from bead.evaluation.interannotator import InterAnnotatorMetrics
            model_agreement = InterAnnotatorMetrics.krippendorff_alpha(
                all_raters, metric="nominal"
            )
        else:
            # For other metrics, compare model directly to human majority vote
            # Get majority human label for each item
            n_items = len(model_predictions)
            human_votes = []
            for i in range(n_items):
                votes_for_item = [rater_labels[r][i] for r in rater_labels]
                # Simple majority vote
                majority = max(set(votes_for_item), key=votes_for_item.count)
                human_votes.append(majority)

            # Compute agreement between model and human majority
            agreements = sum(p == h for p, h in zip(model_predictions, human_votes))
            model_agreement = agreements / len(model_predictions)

        print_success(f"Model agreement: {model_agreement:.4f}")

        # Check convergence
        converged = detector.check_convergence(
            model_accuracy=model_agreement, iteration=min_iterations
        )

        # Display results
        table = Table(title="Convergence Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green", justify="right")

        table.add_row("Agreement Metric", metric)
        table.add_row("Human Baseline", f"{human_baseline:.4f}")
        table.add_row("Model Agreement", f"{model_agreement:.4f}")
        table.add_row("Threshold", f"{threshold:.4f}")
        table.add_row("Converged", "✓ Yes" if converged else "✗ No")

        if converged:
            table.add_row(
                "Status", "[green]Model has converged to human agreement[/green]"
            )
        else:
            gap = threshold - model_agreement
            table.add_row(
                "Status", f"[yellow]Need {gap:.4f} more to reach threshold[/yellow]"
            )

        console.print(table)

        # Exit with appropriate code
        if converged:
            print_success("Convergence achieved!")
            ctx.exit(0)
        else:
            print_info("Not yet converged. Continue training.")
            ctx.exit(1)

    except FileNotFoundError as e:
        print_error(f"File not found: {e}")
        ctx.exit(1)
    except KeyError as e:
        print_error(f"Missing required field in data: {e}")
        ctx.exit(1)
    except json.JSONDecodeError as e:
        print_error(f"Invalid JSON: {e}")
        ctx.exit(1)
    except Exception as e:
        print_error(f"Convergence check failed: {e}")
        ctx.exit(1)


# Register commands
active_learning.add_command(check_convergence)
