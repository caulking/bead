"""Tests for active learning loop orchestration.

Tests the ActiveLearningLoop class, verifying iteration management,
convergence detection, and loop orchestration.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from unittest.mock import Mock
from uuid import uuid4

import numpy as np
import pytest

from sash.items.models import Item
from sash.training.active_learning.loop import ActiveLearningLoop
from sash.training.active_learning.selection import RandomSelector, UncertaintySampler


class TestActiveLearningLoop:
    """Test suite for ActiveLearningLoop."""

    def test_initialization(
        self, mock_trainer: Mock, simple_predict_fn: Callable[[Any, Item], np.ndarray]
    ) -> None:
        """Test loop initialization."""
        selector = UncertaintySampler()

        loop = ActiveLearningLoop(
            item_selector=selector,
            trainer=mock_trainer,
            predict_fn=simple_predict_fn,
            max_iterations=5,
            budget_per_iteration=10,
        )

        assert loop.item_selector == selector
        assert loop.trainer == mock_trainer
        assert loop.predict_fn == simple_predict_fn
        assert loop.max_iterations == 5
        assert loop.budget_per_iteration == 10
        assert loop.iteration_history == []

    def test_initialization_defaults(
        self, mock_trainer: Mock, simple_predict_fn: Callable[[Any, Item], np.ndarray]
    ) -> None:
        """Test initialization with default parameters."""
        selector = UncertaintySampler()

        loop = ActiveLearningLoop(
            item_selector=selector,
            trainer=mock_trainer,
            predict_fn=simple_predict_fn,
        )

        assert loop.max_iterations == 10
        assert loop.budget_per_iteration == 100

    def test_run_iteration_basic(
        self,
        mock_items: list[Item],
        mock_trainer: Mock,
        simple_predict_fn: Callable[[Any, Item], np.ndarray],
    ) -> None:
        """Test running single iteration."""
        selector = UncertaintySampler()
        loop = ActiveLearningLoop(
            item_selector=selector,
            trainer=mock_trainer,
            predict_fn=simple_predict_fn,
            budget_per_iteration=3,
        )

        result = loop.run_iteration(
            iteration=0, unlabeled_items=mock_items, current_model=None
        )

        # Check result structure
        assert "iteration" in result
        assert "selected_items" in result
        assert "model" in result
        assert "metadata" in result

        # Check values
        assert result["iteration"] == 0
        assert len(result["selected_items"]) == 3
        assert all(item in mock_items for item in result["selected_items"])

    def test_run_iteration_respects_budget(
        self,
        mock_items: list[Item],
        mock_trainer: Mock,
        simple_predict_fn: Callable[[Any, Item], np.ndarray],
    ) -> None:
        """Test that iteration respects budget."""
        selector = UncertaintySampler()
        loop = ActiveLearningLoop(
            item_selector=selector,
            trainer=mock_trainer,
            predict_fn=simple_predict_fn,
            budget_per_iteration=5,
        )

        result = loop.run_iteration(
            iteration=0, unlabeled_items=mock_items, current_model=None
        )

        # Should select exactly 5 items
        assert len(result["selected_items"]) == 5

    def test_run_iteration_budget_exceeds_pool(
        self,
        mock_items: list[Item],
        mock_trainer: Mock,
        simple_predict_fn: Callable[[Any, Item], np.ndarray],
    ) -> None:
        """Test iteration when budget exceeds available items."""
        selector = UncertaintySampler()
        loop = ActiveLearningLoop(
            item_selector=selector,
            trainer=mock_trainer,
            predict_fn=simple_predict_fn,
            budget_per_iteration=100,
        )

        result = loop.run_iteration(
            iteration=0, unlabeled_items=mock_items, current_model=None
        )

        # Should select all available items
        assert len(result["selected_items"]) == len(mock_items)

    def test_run_iteration_multiple_calls(
        self,
        mock_items: list[Item],
        mock_trainer: Mock,
        simple_predict_fn: Callable[[Any, Item], np.ndarray],
    ) -> None:
        """Test multiple iteration calls."""
        selector = UncertaintySampler()
        loop = ActiveLearningLoop(
            item_selector=selector,
            trainer=mock_trainer,
            predict_fn=simple_predict_fn,
            budget_per_iteration=2,
        )

        # Run first iteration
        result1 = loop.run_iteration(
            iteration=0, unlabeled_items=mock_items, current_model=None
        )

        # Run second iteration
        result2 = loop.run_iteration(
            iteration=1, unlabeled_items=mock_items, current_model=None
        )

        # Should work for both
        assert result1["iteration"] == 0
        assert result2["iteration"] == 1
        assert len(result1["selected_items"]) == 2
        assert len(result2["selected_items"]) == 2

    def test_run_full_loop_max_iterations(
        self, mock_trainer: Mock, simple_predict_fn: Callable[[Any, Item], np.ndarray]
    ) -> None:
        """Test running full loop with max_iterations stopping criterion."""
        # Create items
        items = [
            Item(item_template_id=uuid4(), rendered_elements={}) for _ in range(50)
        ]

        selector = UncertaintySampler()
        loop = ActiveLearningLoop(
            item_selector=selector,
            trainer=mock_trainer,
            predict_fn=simple_predict_fn,
            max_iterations=3,
            budget_per_iteration=10,
        )

        loop.run(
            initial_items=[],
            initial_model=None,
            unlabeled_pool=items,
            stopping_criterion="max_iterations",
        )

        # Should run 3 iterations
        assert len(loop.iteration_history) == 3

        # Should have selected 30 items total (3 iterations * 10 items)
        total_selected = sum(len(it["selected_items"]) for it in loop.iteration_history)
        assert total_selected == 30

    def test_run_full_loop_exhausts_pool(
        self,
        mock_items: list[Item],
        mock_trainer: Mock,
        simple_predict_fn: Callable[[Any, Item], np.ndarray],
    ) -> None:
        """Test loop stops when unlabeled pool is exhausted."""
        selector = UncertaintySampler()
        loop = ActiveLearningLoop(
            item_selector=selector,
            trainer=mock_trainer,
            predict_fn=simple_predict_fn,
            max_iterations=10,
            budget_per_iteration=5,
        )

        loop.run(
            initial_items=[],
            initial_model=None,
            unlabeled_pool=mock_items,
            stopping_criterion="max_iterations",
        )

        # Should stop after 2 iterations (10 items / 5 per iteration)
        assert len(loop.iteration_history) == 2

        # Should have selected all 10 items
        total_selected = sum(len(it["selected_items"]) for it in loop.iteration_history)
        assert total_selected == len(mock_items)

    def test_run_removes_selected_from_pool(
        self,
        mock_items: list[Item],
        mock_trainer: Mock,
        simple_predict_fn: Callable[[Any, Item], np.ndarray],
    ) -> None:
        """Test that selected items are removed from unlabeled pool."""
        selector = UncertaintySampler()
        loop = ActiveLearningLoop(
            item_selector=selector,
            trainer=mock_trainer,
            predict_fn=simple_predict_fn,
            max_iterations=2,
            budget_per_iteration=3,
        )

        len(mock_items)

        loop.run(
            initial_items=[],
            initial_model=None,
            unlabeled_pool=mock_items,
            stopping_criterion="max_iterations",
        )

        # Should have run 2 iterations
        assert len(loop.iteration_history) == 2

        # Should have selected 6 items total
        all_selected_ids = set()
        for iteration in loop.iteration_history:
            for item in iteration["selected_items"]:
                all_selected_ids.add(item.id)

        assert len(all_selected_ids) == 6

    def test_run_invalid_stopping_criterion(
        self,
        mock_items: list[Item],
        mock_trainer: Mock,
        simple_predict_fn: Callable[[Any, Item], np.ndarray],
    ) -> None:
        """Test that invalid stopping criterion raises error."""
        selector = UncertaintySampler()
        loop = ActiveLearningLoop(
            item_selector=selector,
            trainer=mock_trainer,
            predict_fn=simple_predict_fn,
        )

        with pytest.raises(ValueError, match="Unknown stopping criterion"):
            loop.run(
                initial_items=[],
                initial_model=None,
                unlabeled_pool=mock_items,
                stopping_criterion="invalid",
            )

    def test_run_performance_threshold_missing(
        self,
        mock_items: list[Item],
        mock_trainer: Mock,
        simple_predict_fn: Callable[[Any, Item], np.ndarray],
    ) -> None:
        """Test that performance_threshold criterion requires threshold value."""
        selector = UncertaintySampler()
        loop = ActiveLearningLoop(
            item_selector=selector,
            trainer=mock_trainer,
            predict_fn=simple_predict_fn,
        )

        with pytest.raises(ValueError, match="performance_threshold must be provided"):
            loop.run(
                initial_items=[],
                initial_model=None,
                unlabeled_pool=mock_items,
                stopping_criterion="performance_threshold",
                performance_threshold=None,
            )

    def test_check_convergence_not_converged(self) -> None:
        """Test convergence detection when not converged."""
        selector = UncertaintySampler()
        loop = ActiveLearningLoop(
            item_selector=selector,
            trainer=None,  # type: ignore
            predict_fn=lambda m, i: np.array([0.5, 0.5]),
        )

        # Improving performance
        metrics = [
            {"accuracy": 0.70},
            {"accuracy": 0.75},
            {"accuracy": 0.80},
            {"accuracy": 0.85},
        ]

        converged = loop.check_convergence(
            metrics, metric_name="accuracy", patience=2, min_delta=0.01
        )

        assert not converged

    def test_check_convergence_converged(self) -> None:
        """Test convergence detection when converged."""
        selector = UncertaintySampler()
        loop = ActiveLearningLoop(
            item_selector=selector,
            trainer=None,  # type: ignore
            predict_fn=lambda m, i: np.array([0.5, 0.5]),
        )

        # Performance plateau
        metrics = [
            {"accuracy": 0.70},
            {"accuracy": 0.80},
            {"accuracy": 0.81},
            {"accuracy": 0.81},
            {"accuracy": 0.80},
        ]

        converged = loop.check_convergence(
            metrics, metric_name="accuracy", patience=2, min_delta=0.02
        )

        assert converged

    def test_check_convergence_insufficient_history(self) -> None:
        """Test convergence with insufficient history."""
        selector = UncertaintySampler()
        loop = ActiveLearningLoop(
            item_selector=selector,
            trainer=None,  # type: ignore
            predict_fn=lambda m, i: np.array([0.5, 0.5]),
        )

        # Only 2 iterations
        metrics = [
            {"accuracy": 0.70},
            {"accuracy": 0.75},
        ]

        converged = loop.check_convergence(
            metrics, metric_name="accuracy", patience=3, min_delta=0.01
        )

        # Need at least patience + 1 iterations
        assert not converged

    def test_check_convergence_custom_metric(
        self, sample_metrics_history: list[dict[str, float]]
    ) -> None:
        """Test convergence with custom metric."""
        selector = UncertaintySampler()
        loop = ActiveLearningLoop(
            item_selector=selector,
            trainer=None,  # type: ignore
            predict_fn=lambda m, i: np.array([0.5, 0.5]),
        )

        # Check loss instead of accuracy
        converged = loop.check_convergence(
            sample_metrics_history, metric_name="loss", patience=2, min_delta=0.01
        )

        # Loss has plateaued
        assert converged

    def test_get_summary_empty(
        self, mock_trainer: Mock, simple_predict_fn: Callable[[Any, Item], np.ndarray]
    ) -> None:
        """Test summary with no iterations run."""
        selector = UncertaintySampler()
        loop = ActiveLearningLoop(
            item_selector=selector,
            trainer=mock_trainer,
            predict_fn=simple_predict_fn,
        )

        summary = loop.get_summary()

        assert summary["total_iterations"] == 0
        assert summary["total_items_selected"] == 0
        assert "convergence_info" in summary

    def test_get_summary_after_iterations(
        self,
        mock_items: list[Item],
        mock_trainer: Mock,
        simple_predict_fn: Callable[[Any, Item], np.ndarray],
    ) -> None:
        """Test summary after running iterations."""
        selector = UncertaintySampler()
        loop = ActiveLearningLoop(
            item_selector=selector,
            trainer=mock_trainer,
            predict_fn=simple_predict_fn,
            max_iterations=2,
            budget_per_iteration=3,
        )

        loop.run(
            initial_items=[],
            initial_model=None,
            unlabeled_pool=mock_items,
            stopping_criterion="max_iterations",
        )

        summary = loop.get_summary()

        assert summary["total_iterations"] == 2
        assert summary["total_items_selected"] == 6
        assert summary["convergence_info"]["max_iterations"] == 2
        assert summary["convergence_info"]["budget_per_iteration"] == 3

    def test_iteration_history_structure(
        self,
        mock_items: list[Item],
        mock_trainer: Mock,
        simple_predict_fn: Callable[[Any, Item], np.ndarray],
    ) -> None:
        """Test that iteration history has correct structure."""
        selector = UncertaintySampler()
        loop = ActiveLearningLoop(
            item_selector=selector,
            trainer=mock_trainer,
            predict_fn=simple_predict_fn,
            max_iterations=2,
            budget_per_iteration=3,
        )

        loop.run(
            initial_items=[],
            initial_model=None,
            unlabeled_pool=mock_items,
            stopping_criterion="max_iterations",
        )

        assert len(loop.iteration_history) == 2

        for i, iteration in enumerate(loop.iteration_history):
            assert iteration["iteration"] == i
            assert "selected_items" in iteration
            assert "model" in iteration
            assert "metadata" in iteration
            assert len(iteration["selected_items"]) == 3

    def test_with_different_selectors(
        self,
        mock_items: list[Item],
        mock_trainer: Mock,
        simple_predict_fn: Callable[[Any, Item], np.ndarray],
    ) -> None:
        """Test loop with different item selectors."""
        # Test with uncertainty sampler
        uncertainty_loop = ActiveLearningLoop(
            item_selector=UncertaintySampler(method="margin"),
            trainer=mock_trainer,
            predict_fn=simple_predict_fn,
            max_iterations=2,
            budget_per_iteration=3,
        )

        uncertainty_loop.run(
            initial_items=[],
            initial_model=None,
            unlabeled_pool=mock_items.copy(),
            stopping_criterion="max_iterations",
        )

        # Test with random selector
        random_loop = ActiveLearningLoop(
            item_selector=RandomSelector(seed=42),
            trainer=mock_trainer,
            predict_fn=simple_predict_fn,
            max_iterations=2,
            budget_per_iteration=3,
        )

        random_loop.run(
            initial_items=[],
            initial_model=None,
            unlabeled_pool=mock_items.copy(),
            stopping_criterion="max_iterations",
        )

        # Both should complete successfully
        assert len(uncertainty_loop.iteration_history) == 2
        assert len(random_loop.iteration_history) == 2

    def test_with_varying_predictions(
        self,
        mock_items: list[Item],
        mock_trainer: Mock,
        varying_predict_fn: Callable[[Any, Item], np.ndarray],
    ) -> None:
        """Test loop with varying prediction function."""
        selector = UncertaintySampler(method="entropy")
        loop = ActiveLearningLoop(
            item_selector=selector,
            trainer=mock_trainer,
            predict_fn=varying_predict_fn,
            max_iterations=2,
            budget_per_iteration=3,
        )

        loop.run(
            initial_items=[],
            initial_model=None,
            unlabeled_pool=mock_items,
            stopping_criterion="max_iterations",
        )

        # Should prioritize uncertain items
        first_iteration = loop.iteration_history[0]
        selected_indices = [
            int(item.rendered_elements["index"])
            for item in first_iteration["selected_items"]
        ]

        # Check that uncertain items are selected
        # (index % 3 == 1 have uncertain predictions)
        uncertain_count = sum(1 for idx in selected_indices if idx % 3 == 1)

        # Should have at least one uncertain item
        assert uncertain_count >= 1


class TestActiveLearningLoopEdgeCases:
    """Test edge cases for active learning loop."""

    def test_single_item_pool(
        self, mock_trainer: Mock, simple_predict_fn: Callable[[Any, Item], np.ndarray]
    ) -> None:
        """Test with single item in pool."""
        item = Item(item_template_id=uuid4(), rendered_elements={"text": "item"})

        selector = UncertaintySampler()
        loop = ActiveLearningLoop(
            item_selector=selector,
            trainer=mock_trainer,
            predict_fn=simple_predict_fn,
            max_iterations=5,
            budget_per_iteration=10,
        )

        loop.run(
            initial_items=[],
            initial_model=None,
            unlabeled_pool=[item],
            stopping_criterion="max_iterations",
        )

        # Should stop after 1 iteration (pool exhausted)
        assert len(loop.iteration_history) == 1
        assert len(loop.iteration_history[0]["selected_items"]) == 1

    def test_empty_initial_pool(
        self, mock_trainer: Mock, simple_predict_fn: Callable[[Any, Item], np.ndarray]
    ) -> None:
        """Test with empty unlabeled pool."""
        selector = UncertaintySampler()
        loop = ActiveLearningLoop(
            item_selector=selector,
            trainer=mock_trainer,
            predict_fn=simple_predict_fn,
            max_iterations=5,
            budget_per_iteration=10,
        )

        model_history = loop.run(
            initial_items=[],
            initial_model=None,
            unlabeled_pool=[],
            stopping_criterion="max_iterations",
        )

        # Should not run any iterations
        assert len(loop.iteration_history) == 0
        assert len(model_history) == 0

    def test_budget_one(
        self,
        mock_items: list[Item],
        mock_trainer: Mock,
        simple_predict_fn: Callable[[Any, Item], np.ndarray],
    ) -> None:
        """Test with budget of 1 item per iteration."""
        selector = UncertaintySampler()
        loop = ActiveLearningLoop(
            item_selector=selector,
            trainer=mock_trainer,
            predict_fn=simple_predict_fn,
            max_iterations=3,
            budget_per_iteration=1,
        )

        loop.run(
            initial_items=[],
            initial_model=None,
            unlabeled_pool=mock_items,
            stopping_criterion="max_iterations",
        )

        # Should run 3 iterations, selecting 1 item each
        assert len(loop.iteration_history) == 3

        for iteration in loop.iteration_history:
            assert len(iteration["selected_items"]) == 1

    def test_large_pool(
        self,
        large_mock_items: list[Item],
        mock_trainer: Mock,
        simple_predict_fn: Callable[[Any, Item], np.ndarray],
    ) -> None:
        """Test with large item pool."""
        selector = UncertaintySampler(batch_size=32)
        loop = ActiveLearningLoop(
            item_selector=selector,
            trainer=mock_trainer,
            predict_fn=simple_predict_fn,
            max_iterations=2,
            budget_per_iteration=20,
        )

        loop.run(
            initial_items=[],
            initial_model=None,
            unlabeled_pool=large_mock_items,
            stopping_criterion="max_iterations",
        )

        # Should complete successfully
        assert len(loop.iteration_history) == 2

        # Should select 40 items total
        total_selected = sum(len(it["selected_items"]) for it in loop.iteration_history)
        assert total_selected == 40
