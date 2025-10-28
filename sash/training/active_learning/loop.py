"""Active learning loop orchestration.

This module orchestrates the iterative active learning loop (stages 3-6):
construct items → deploy experiment → collect data → train model → select
next items. It manages convergence detection and coordinates all components.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np

from sash.evaluation.convergence import ConvergenceDetector
from sash.items.models import Item
from sash.training.active_learning.selection import ItemSelector
from sash.training.trainers.base import BaseTrainer, ModelMetadata

if TYPE_CHECKING:
    # Type alias for framework-agnostic ML models
    # Could be transformers.PreTrainedModel, torch.nn.Module,
    # sklearn estimator, etc.
    Model = Any


class IterationResult(TypedDict):
    """Results from a single active learning iteration.

    Attributes
    ----------
    iteration : int
        Iteration number.
    selected_items : list[Item]
        Items selected for annotation in this iteration.
    model : Any
        Updated model after this iteration (framework-agnostic).
    metadata : ModelMetadata | None
        Training metadata if model was retrained, None otherwise.
    """

    iteration: int
    selected_items: list[Item]
    model: Any  # Framework-agnostic model type
    metadata: ModelMetadata | None


class ActiveLearningLoop:
    """Orchestrates the active learning loop (stages 3-6).

    Manages the iterative process of selecting informative items,
    training models on collected data, and determining when to stop.

    Note: Phase 22 (data collection) is not yet implemented, so this
    loop uses placeholder interfaces for deployment and data collection.
    The focus is on the selection logic and loop orchestration.

    Parameters
    ----------
    item_selector : ItemSelector
        Algorithm for selecting informative items.
    trainer : BaseTrainer
        Model trainer for training on collected data.
    predict_fn : Callable[[Any, Item], np.ndarray]
        Function to get prediction probabilities from model.
        Takes (model, item) and returns array of shape (n_classes,).
    max_iterations : int
        Maximum number of AL iterations to run.
    budget_per_iteration : int
        Number of items to select per iteration.

    Attributes
    ----------
    item_selector : ItemSelector
        Item selection algorithm.
    trainer : BaseTrainer
        Model trainer.
    predict_fn : Callable[[Any, Item], np.ndarray]
        Prediction function that takes a model and item, returns class probabilities.
    max_iterations : int
        Maximum iterations.
    budget_per_iteration : int
        Items per iteration.
    iteration_history : list[IterationResult]
        History of all iterations with structured results.

    Examples
    --------
    >>> from sash.training.active_learning.selection import UncertaintySampler
    >>> from sash.training.trainers.base import BaseTrainer
    >>> import numpy as np
    >>> # Create components (mocked for example)
    >>> selector = UncertaintySampler()
    >>> trainer = None  # Mock trainer  # doctest: +SKIP
    >>> def predict_fn(model, item):  # doctest: +SKIP
    ...     return np.array([0.5, 0.5])
    >>> loop = ActiveLearningLoop(  # doctest: +SKIP
    ...     item_selector=selector,
    ...     trainer=trainer,
    ...     predict_fn=predict_fn,
    ...     max_iterations=5,
    ...     budget_per_iteration=100
    ... )
    """

    def __init__(
        self,
        item_selector: ItemSelector,
        trainer: BaseTrainer,
        predict_fn: Callable[[Any, Item], np.ndarray],
        max_iterations: int = 10,
        budget_per_iteration: int = 100,
    ) -> None:
        """Initialize active learning loop.

        Parameters
        ----------
        item_selector : ItemSelector
            Algorithm for selecting items.
        trainer : BaseTrainer
            Model trainer.
        predict_fn : Callable[[Any, Item], np.ndarray]
            Prediction function.
        max_iterations : int
            Maximum number of iterations.
        budget_per_iteration : int
            Items to select per iteration.
        """
        self.item_selector = item_selector
        self.trainer = trainer
        self.predict_fn = predict_fn
        self.max_iterations = max_iterations
        self.budget_per_iteration = budget_per_iteration
        self.iteration_history: list[IterationResult] = []

    def run(
        self,
        initial_items: list[Item],
        initial_model: Any,
        unlabeled_pool: list[Item],
        human_ratings: dict[str, list[Any]] | None = None,
        convergence_detector: ConvergenceDetector | None = None,
        stopping_criterion: str = "max_iterations",
        performance_threshold: float | None = None,
        metric_name: str = "accuracy",
    ) -> list[ModelMetadata]:
        """Run the complete active learning loop.

        Parameters
        ----------
        initial_items : list[Item]
            Initial labeled items for training.
        initial_model : Any
            Initial trained model (or None to train from scratch).
        unlabeled_pool : list[Item]
            Pool of unlabeled items to select from.
        human_ratings : dict[str, list[Any]] | None
            Human ratings for computing inter-rater agreement baseline.
            Dictionary mapping rater IDs to their ratings.
        convergence_detector : ConvergenceDetector | None
            Detector for checking convergence to human-level performance.
            If provided, will check convergence after each iteration.
        stopping_criterion : str
            When to stop ("max_iterations", "performance_threshold", or "convergence").
        performance_threshold : float | None
            Stop when performance exceeds this (if using performance_threshold).
        metric_name : str
            Name of metric to check for performance threshold or convergence.

        Returns
        -------
        list[ModelMetadata]
            Metadata for all trained models across iterations.

        Raises
        ------
        ValueError
            If stopping_criterion is invalid or threshold not provided when needed.

        Examples
        --------
        >>> from uuid import uuid4
        >>> from sash.items.models import Item
        >>> selector = UncertaintySampler()  # doctest: +SKIP
        >>> loop = ActiveLearningLoop(  # doctest: +SKIP
        ...     item_selector=selector,
        ...     trainer=None,
        ...     predict_fn=lambda m, i: np.array([0.5, 0.5]),
        ...     max_iterations=3
        ... )
        >>> # Run would typically be called here with real data
        """
        # Validate inputs
        if (
            stopping_criterion == "performance_threshold"
            and performance_threshold is None
        ):
            raise ValueError(
                "performance_threshold must be provided when using "
                "performance_threshold stopping criterion"
            )

        if stopping_criterion == "convergence" and convergence_detector is None:
            raise ValueError(
                "convergence_detector must be provided when using "
                "convergence stopping criterion"
            )

        valid_criteria = {"max_iterations", "performance_threshold", "convergence"}
        if stopping_criterion not in valid_criteria:
            raise ValueError(
                f"stopping_criterion must be one of {valid_criteria}, "
                f"got '{stopping_criterion}'"
            )

        model_history: list[ModelMetadata] = []
        current_model = initial_model
        current_unlabeled = unlabeled_pool.copy()

        # Check if we have any unlabeled items to start with
        if not current_unlabeled:
            return model_history

        # Run iterations
        for iteration in range(self.max_iterations):
            # Run one iteration
            iteration_result = self.run_iteration(
                iteration=iteration,
                unlabeled_items=current_unlabeled,
                current_model=current_model,
            )

            # Store results
            self.iteration_history.append(iteration_result)

            # Update state
            selected_items = iteration_result["selected_items"]
            current_model = iteration_result["model"]
            metadata = iteration_result["metadata"]

            if metadata is not None:
                model_history.append(metadata)

            # Remove selected items from unlabeled pool
            selected_ids = {item.id for item in selected_items}
            current_unlabeled = [
                item for item in current_unlabeled if item.id not in selected_ids
            ]

            # Check stopping criteria
            if stopping_criterion == "max_iterations":
                # Will stop naturally at max_iterations
                pass
            elif stopping_criterion == "performance_threshold":
                if metadata and metric_name in metadata.metrics:
                    if metadata.metrics[metric_name] >= performance_threshold:  # type: ignore
                        break
            elif stopping_criterion == "convergence":
                if convergence_detector is not None and metadata is not None:
                    # Compute human baseline on first iteration
                    if iteration == 0 and human_ratings is not None:
                        convergence_detector.compute_human_baseline(human_ratings)

                    # Check if converged
                    if metric_name in metadata.metrics:
                        converged = convergence_detector.check_convergence(
                            model_accuracy=metadata.metrics[metric_name],
                            iteration=iteration + 1,
                        )

                        if converged:
                            print(f"✓ Converged at iteration {iteration + 1}")
                            break

            # Check if unlabeled pool is exhausted
            if not current_unlabeled:
                break

        return model_history

    def run_iteration(
        self,
        iteration: int,
        unlabeled_items: list[Item],
        current_model: Any,
    ) -> IterationResult:
        """Run one iteration of the active learning loop.

        Steps:
        1. Select informative items using uncertainty sampling
        2. (Placeholder) Deploy experiment for data collection
        3. (Placeholder) Wait for and collect data
        4. (Placeholder) Train new model on augmented dataset
        5. Return results

        Parameters
        ----------
        iteration : int
            Current iteration number.
        unlabeled_items : list[Item]
            Unlabeled items available for selection.
        current_model : Any
            Current trained model for making predictions.

        Returns
        -------
        IterationResult
            Structured iteration results containing:
            - iteration: Iteration number
            - selected_items: List of selected items
            - model: Updated model (framework-agnostic)
            - metadata: Training metadata if available

        Examples
        --------
        >>> from uuid import uuid4
        >>> from sash.items.models import Item
        >>> import numpy as np
        >>> selector = UncertaintySampler()
        >>> loop = ActiveLearningLoop(
        ...     item_selector=selector,
        ...     trainer=None,
        ...     predict_fn=lambda m, i: np.array([0.5, 0.5]),
        ...     max_iterations=5,
        ...     budget_per_iteration=2
        ... )
        >>> items = [
        ...     Item(item_template_id=uuid4(), rendered_elements={})
        ...     for _ in range(5)
        ... ]
        >>> result = loop.run_iteration(0, items, None)
        >>> len(result["selected_items"])
        2
        >>> result["iteration"]
        0
        """
        # Step 1: Select items using active learning
        budget = min(self.budget_per_iteration, len(unlabeled_items))

        selected_items = self.item_selector.select(
            items=unlabeled_items,
            model=current_model,
            predict_fn=self.predict_fn,
            budget=budget,
        )

        # Step 2: Deploy experiment (PLACEHOLDER - Phase 22 not yet implemented)
        # In the future, this would:
        # - Create experiment lists using ListPartitioner
        # - Generate jsPsych experiment using JsPsychExperimentGenerator
        # - Export to JATOS format
        # - Return deployment info for manual upload

        # Step 3: Collect data (PLACEHOLDER - Phase 22 not yet implemented)
        # In the future, this would:
        # - Wait for participants to complete experiments
        # - Use JATOSDataCollector to download results
        # - Use ProlificDataCollector to get participant metadata
        # - Use DataMerger to merge JATOS and Prolific data

        # Step 4: Train new model (PLACEHOLDER - training data not available)
        # In the future, this would:
        # - Merge old training data with new collected data
        # - Call trainer.train() with augmented dataset
        # - Return updated model and metadata

        # For now, return placeholder results
        return IterationResult(
            iteration=iteration,
            selected_items=selected_items,
            model=current_model,  # Unchanged for now
            metadata=None,  # Would contain training metrics
        )

    def check_convergence(
        self,
        metrics_history: list[dict[str, float]],
        metric_name: str = "accuracy",
        patience: int = 3,
        min_delta: float = 0.01,
    ) -> bool:
        """Check if model performance has converged.

        Uses early stopping logic: if performance hasn't improved by
        at least min_delta for patience iterations, consider converged.

        For metrics where lower is better (like "loss"), the logic checks
        if the best (lowest) value is from more than patience iterations ago.

        Parameters
        ----------
        metrics_history : list[dict[str, float]]
            History of metrics from each iteration.
        metric_name : str
            Name of metric to track.
        patience : int
            Number of iterations without improvement before stopping.
        min_delta : float
            Minimum change to count as improvement.

        Returns
        -------
        bool
            True if converged, False otherwise.

        Examples
        --------
        >>> loop = ActiveLearningLoop(  # doctest: +SKIP
        ...     item_selector=UncertaintySampler(),
        ...     trainer=None,
        ...     predict_fn=lambda m, i: np.array([0.5, 0.5])
        ... )
        >>> # Improving performance - not converged
        >>> history = [
        ...     {"accuracy": 0.7},
        ...     {"accuracy": 0.75},
        ...     {"accuracy": 0.8}
        ... ]
        >>> loop.check_convergence(history, metric_name="accuracy", patience=2)
        False
        >>> # No improvement for 3 iterations - converged
        >>> history = [
        ...     {"accuracy": 0.8},
        ...     {"accuracy": 0.81},
        ...     {"accuracy": 0.805},
        ...     {"accuracy": 0.81}
        ... ]
        >>> loop.check_convergence(
        ...     history, metric_name="accuracy", patience=3, min_delta=0.02
        ... )
        True
        """
        if len(metrics_history) < patience + 1:
            return False

        # Get recent metrics
        recent_metrics = [m[metric_name] for m in metrics_history[-(patience + 1) :]]

        # Determine if lower is better (like loss) or higher is better (like accuracy)
        is_lower_better = metric_name.lower() in ["loss", "error", "mse", "rmse", "mae"]

        if is_lower_better:
            # For loss metrics, best means minimum
            best_metric = min(recent_metrics)
            best_idx = recent_metrics.index(best_metric)

            # If best is from patience or more iterations ago, check convergence
            if best_idx <= len(recent_metrics) - patience - 1:
                # Check that degradation from best to current is < min_delta
                current_metric = recent_metrics[-1]
                degradation = current_metric - best_metric

                if degradation >= min_delta:
                    return True  # Performance degraded, converged

        else:
            # For accuracy metrics, best means maximum
            best_metric = max(recent_metrics)
            best_idx = recent_metrics.index(best_metric)

            # If best is from patience or more iterations ago, converged
            if best_idx <= len(recent_metrics) - patience - 1:
                # Check that improvement from best to current is < min_delta
                current_metric = recent_metrics[-1]
                improvement = best_metric - current_metric

                if improvement >= min_delta:
                    return False  # Still improving

                return True

        return False

    def get_summary(self) -> dict[str, int | dict[str, int]]:
        """Get summary statistics of the active learning loop.

        Returns
        -------
        dict[str, int | dict[str, int]]
            Summary dictionary with the following keys:

            total_iterations : int
                Total number of iterations run.
            total_items_selected : int
                Total items selected across all iterations.
            convergence_info : dict[str, int]
                Configuration parameters (max_iterations, budget_per_iteration).

        Examples
        --------
        >>> selector = UncertaintySampler()
        >>> loop = ActiveLearningLoop(
        ...     item_selector=selector,
        ...     trainer=None,
        ...     predict_fn=lambda m, i: np.array([0.5, 0.5])
        ... )
        >>> summary = loop.get_summary()
        >>> summary["total_iterations"]
        0
        >>> summary["total_items_selected"]
        0
        """
        total_items = sum(
            len(iteration["selected_items"]) for iteration in self.iteration_history
        )

        return {
            "total_iterations": len(self.iteration_history),
            "total_items_selected": total_items,
            "convergence_info": {
                "max_iterations": self.max_iterations,
                "budget_per_iteration": self.budget_per_iteration,
            },
        }
