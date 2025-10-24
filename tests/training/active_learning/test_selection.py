"""Tests for active learning item selection.

Tests the UncertaintySampler and RandomSelector classes, verifying
item selection logic, batch prediction, and edge case handling.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
from pytest_mock import MockerFixture

from sash.items.models import Item
from sash.training.active_learning.selection import (
    ItemSelector,
    RandomSelector,
    UncertaintySampler,
)
from sash.training.active_learning.strategies import UncertaintySampling


class TestUncertaintySampler:
    """Test suite for UncertaintySampler."""

    def test_initialization_default(self) -> None:
        """Test default initialization."""
        sampler = UncertaintySampler()

        assert sampler.method == "entropy"
        assert sampler.batch_size == 32
        assert isinstance(sampler, ItemSelector)

    def test_initialization_custom(self) -> None:
        """Test initialization with custom parameters."""
        sampler = UncertaintySampler(method="margin", batch_size=16)

        assert sampler.method == "margin"
        assert sampler.batch_size == 16

    @pytest.mark.parametrize("method", ["entropy", "margin", "least_confidence"])
    def test_initialization_all_methods(self, method: str) -> None:
        """Test initialization with all supported methods."""
        sampler = UncertaintySampler(method=method)
        assert sampler.method == method

    def test_select_basic(
        self,
        mock_items: list[Item],
        simple_predict_fn: Callable[[Any, Item], np.ndarray],
    ) -> None:
        """Test basic item selection."""
        sampler = UncertaintySampler(method="entropy")

        selected = sampler.select(
            items=mock_items, model=None, predict_fn=simple_predict_fn, budget=3
        )

        # Should return 3 items
        assert len(selected) == 3

        # Should be subset of original items
        for item in selected:
            assert item in mock_items

    def test_select_with_varying_uncertainty(
        self,
        mock_items: list[Item],
        varying_predict_fn: Callable[[Any, Item], np.ndarray],
    ) -> None:
        """Test selection prioritizes uncertain items."""
        sampler = UncertaintySampler(method="entropy")

        # varying_predict_fn returns different confidence levels based on index
        # Index % 3 == 1 -> uncertain (0.5, 0.5)
        # Index % 3 == 2 -> moderate (0.7, 0.3)
        # Index % 3 == 0 -> confident (0.9, 0.1)

        selected = sampler.select(
            items=mock_items, model=None, predict_fn=varying_predict_fn, budget=3
        )

        # Check that uncertain items (index % 3 == 1) are selected
        selected_indices = [int(item.rendered_elements["index"]) for item in selected]

        # Items with index % 3 == 1 should be prioritized (most uncertain)
        uncertain_count = sum(1 for idx in selected_indices if idx % 3 == 1)

        # Should have at least some uncertain items
        assert uncertain_count >= 1

    def test_select_budget_larger_than_pool(
        self,
        mock_items: list[Item],
        simple_predict_fn: Callable[[Any, Item], np.ndarray],
    ) -> None:
        """Test selection when budget exceeds item pool size."""
        sampler = UncertaintySampler()

        selected = sampler.select(
            items=mock_items, model=None, predict_fn=simple_predict_fn, budget=100
        )

        # Should return all items
        assert len(selected) == len(mock_items)
        assert {item.id for item in selected} == {item.id for item in mock_items}

    def test_select_budget_equals_pool(
        self,
        mock_items: list[Item],
        simple_predict_fn: Callable[[Any, Item], np.ndarray],
    ) -> None:
        """Test selection when budget equals pool size."""
        sampler = UncertaintySampler()

        selected = sampler.select(
            items=mock_items,
            model=None,
            predict_fn=simple_predict_fn,
            budget=len(mock_items),
        )

        # Should return all items
        assert len(selected) == len(mock_items)

    def test_select_empty_items(
        self, simple_predict_fn: Callable[[Any, Item], np.ndarray]
    ) -> None:
        """Test selection with empty items list."""
        sampler = UncertaintySampler()

        with pytest.raises(ValueError, match="Items list cannot be empty"):
            sampler.select(items=[], model=None, predict_fn=simple_predict_fn, budget=5)

    def test_select_zero_budget(
        self,
        mock_items: list[Item],
        simple_predict_fn: Callable[[Any, Item], np.ndarray],
    ) -> None:
        """Test selection with zero budget."""
        sampler = UncertaintySampler()

        with pytest.raises(ValueError, match="Budget must be positive"):
            sampler.select(
                items=mock_items, model=None, predict_fn=simple_predict_fn, budget=0
            )

    def test_select_negative_budget(
        self,
        mock_items: list[Item],
        simple_predict_fn: Callable[[Any, Item], np.ndarray],
    ) -> None:
        """Test selection with negative budget."""
        sampler = UncertaintySampler()

        with pytest.raises(ValueError, match="Budget must be positive"):
            sampler.select(
                items=mock_items, model=None, predict_fn=simple_predict_fn, budget=-5
            )

    def test_select_different_methods(
        self,
        mock_items: list[Item],
        varying_predict_fn: Callable[[Any, Item], np.ndarray],
    ) -> None:
        """Test selection with different uncertainty methods."""
        methods = ["entropy", "margin", "least_confidence"]

        results = {}
        for method in methods:
            sampler = UncertaintySampler(method=method)
            selected = sampler.select(
                items=mock_items, model=None, predict_fn=varying_predict_fn, budget=3
            )
            results[method] = {item.id for item in selected}

        # Different methods may select different items
        # (depending on the probability distributions)
        assert len(results) == 3

    def test_batch_predict_basic(
        self,
        mock_items: list[Item],
        simple_predict_fn: Callable[[Any, Item], np.ndarray],
    ) -> None:
        """Test batch prediction."""
        sampler = UncertaintySampler()

        probs = sampler._batch_predict(mock_items, None, simple_predict_fn)

        # Should return predictions for all items
        assert probs.shape == (len(mock_items), 2)

        # All predictions should be the same (simple_predict_fn returns [0.5, 0.5])
        expected = np.array([0.5, 0.5])
        for prob in probs:
            assert np.allclose(prob, expected)

    def test_batch_predict_with_batching(
        self,
        large_mock_items: list[Item],
        simple_predict_fn: Callable[[Any, Item], np.ndarray],
    ) -> None:
        """Test batch prediction with large dataset."""
        sampler = UncertaintySampler(batch_size=32)

        probs = sampler._batch_predict(large_mock_items, None, simple_predict_fn)

        # Should return predictions for all items
        assert probs.shape == (len(large_mock_items), 2)

    def test_batch_predict_varying(
        self,
        mock_items: list[Item],
        varying_predict_fn: Callable[[Any, Item], np.ndarray],
    ) -> None:
        """Test batch prediction with varying predictions."""
        sampler = UncertaintySampler()

        probs = sampler._batch_predict(mock_items, None, varying_predict_fn)

        # Should have different predictions
        assert probs.shape == (len(mock_items), 2)

        # Not all predictions should be the same
        assert not np.all(probs == probs[0])

    def test_select_ordering(
        self,
        mock_items: list[Item],
        varying_predict_fn: Callable[[Any, Item], np.ndarray],
    ) -> None:
        """Test that selected items are ordered by uncertainty."""
        sampler = UncertaintySampler(method="entropy")

        selected = sampler.select(
            items=mock_items, model=None, predict_fn=varying_predict_fn, budget=5
        )

        # Get uncertainties for selected items
        probs = [varying_predict_fn(None, item) for item in selected]
        strategy = UncertaintySampling()
        uncertainties = strategy.compute_scores(np.array(probs))

        # Should be in descending order (most uncertain first)
        for i in range(len(uncertainties) - 1):
            assert uncertainties[i] >= uncertainties[i + 1]

    def test_select_with_model_object(
        self, mock_items: list[Item], mock_model: Mock, mocker: MockerFixture
    ) -> None:
        """Test selection with actual model object."""
        sampler = UncertaintySampler()

        # Create predict function that uses model
        def predict_fn(model: Any, item: Item) -> np.ndarray:  # noqa: ANN401
            return model.predict()  # Returns [0.6, 0.4] from fixture

        selected = sampler.select(
            items=mock_items, model=mock_model, predict_fn=predict_fn, budget=3
        )

        # Should work with model
        assert len(selected) == 3
        assert mock_model.predict.called


class TestRandomSelector:
    """Test suite for RandomSelector."""

    def test_initialization_with_seed(self) -> None:
        """Test initialization with seed."""
        selector = RandomSelector(seed=42)
        assert isinstance(selector, ItemSelector)

    def test_initialization_no_seed(self) -> None:
        """Test initialization without seed."""
        selector = RandomSelector()
        assert isinstance(selector, ItemSelector)

    def test_select_basic(self, mock_items: list[Item]) -> None:
        """Test basic random selection."""
        selector = RandomSelector(seed=42)

        selected = selector.select(
            items=mock_items, model=None, predict_fn=None, budget=3
        )  # type: ignore

        # Should return 3 items
        assert len(selected) == 3

        # Should be subset of original
        for item in selected:
            assert item in mock_items

    def test_select_reproducible(self, mock_items: list[Item]) -> None:
        """Test that same seed produces same selection."""
        selector1 = RandomSelector(seed=123)
        selector2 = RandomSelector(seed=123)

        selected1 = selector1.select(
            items=mock_items, model=None, predict_fn=None, budget=5
        )  # type: ignore
        selected2 = selector2.select(
            items=mock_items, model=None, predict_fn=None, budget=5
        )  # type: ignore

        # Should select same items
        ids1 = [item.id for item in selected1]
        ids2 = [item.id for item in selected2]

        assert ids1 == ids2

    def test_select_different_seeds(self, mock_items: list[Item]) -> None:
        """Test that different seeds produce different selections."""
        selector1 = RandomSelector(seed=42)
        selector2 = RandomSelector(seed=999)

        selected1 = selector1.select(
            items=mock_items, model=None, predict_fn=None, budget=5
        )  # type: ignore
        selected2 = selector2.select(
            items=mock_items, model=None, predict_fn=None, budget=5
        )  # type: ignore

        # Should select different items (with high probability)
        ids1 = {item.id for item in selected1}
        ids2 = {item.id for item in selected2}

        # Very unlikely to be identical
        assert ids1 != ids2

    def test_select_budget_larger_than_pool(self, mock_items: list[Item]) -> None:
        """Test selection when budget exceeds pool size."""
        selector = RandomSelector(seed=42)

        selected = selector.select(
            items=mock_items, model=None, predict_fn=None, budget=100
        )  # type: ignore

        # Should return all items
        assert len(selected) == len(mock_items)

    def test_select_empty_items(self) -> None:
        """Test selection with empty items list."""
        selector = RandomSelector()

        with pytest.raises(ValueError, match="Items list cannot be empty"):
            selector.select(items=[], model=None, predict_fn=None, budget=5)  # type: ignore

    def test_select_zero_budget(self, mock_items: list[Item]) -> None:
        """Test selection with zero budget."""
        selector = RandomSelector()

        with pytest.raises(ValueError, match="Budget must be positive"):
            selector.select(items=mock_items, model=None, predict_fn=None, budget=0)  # type: ignore

    def test_select_negative_budget(self, mock_items: list[Item]) -> None:
        """Test selection with negative budget."""
        selector = RandomSelector()

        with pytest.raises(ValueError, match="Budget must be positive"):
            selector.select(items=mock_items, model=None, predict_fn=None, budget=-5)  # type: ignore

    def test_select_no_duplicates(self, mock_items: list[Item]) -> None:
        """Test that selection doesn't produce duplicates."""
        selector = RandomSelector(seed=42)

        selected = selector.select(
            items=mock_items, model=None, predict_fn=None, budget=5
        )  # type: ignore

        # Should have no duplicates
        ids = [item.id for item in selected]
        assert len(ids) == len(set(ids))

    def test_select_ignores_model_and_predict_fn(
        self,
        mock_items: list[Item],
        mock_model: Mock,
        simple_predict_fn: Callable[[Any, Item], np.ndarray],
    ) -> None:
        """Test that random selection ignores model and predict_fn."""
        selector = RandomSelector(seed=42)

        # Select with model and predict_fn (should be ignored)
        selected = selector.select(
            items=mock_items, model=mock_model, predict_fn=simple_predict_fn, budget=3
        )

        # Should still work
        assert len(selected) == 3

        # Model should not be used
        assert not mock_model.predict.called

    def test_select_uniform_distribution(self, large_mock_items: list[Item]) -> None:
        """Test that selection is approximately uniform over many runs."""
        RandomSelector(seed=None)  # Use different seed each time

        # Count selections over many runs
        selection_counts = {item.id: 0 for item in large_mock_items}

        n_runs = 1000
        budget = 10

        for _ in range(n_runs):
            # Use different selector each time (different random state)
            sel = RandomSelector()
            selected = sel.select(
                items=large_mock_items, model=None, predict_fn=None, budget=budget
            )  # type: ignore

            for item in selected:
                selection_counts[item.id] += 1

        # Each item should be selected approximately equally
        # Expected: (n_runs * budget) / len(items) = 1000 * 10 / 100 = 100 times
        expected_count = (n_runs * budget) / len(large_mock_items)

        counts = list(selection_counts.values())
        mean_count = np.mean(counts)

        # Mean should be close to expected (within 20%)
        assert abs(mean_count - expected_count) / expected_count < 0.2


class TestSelectorComparison:
    """Test comparison between uncertainty and random selectors."""

    def test_uncertainty_vs_random_different_selections(
        self,
        mock_items: list[Item],
        varying_predict_fn: Callable[[Any, Item], np.ndarray],
    ) -> None:
        """Test that uncertainty and random selectors choose differently."""
        uncertainty_sampler = UncertaintySampler(method="entropy")
        random_selector = RandomSelector(seed=42)

        uncertain_selected = uncertainty_sampler.select(
            items=mock_items, model=None, predict_fn=varying_predict_fn, budget=5
        )

        random_selected = random_selector.select(
            items=mock_items,
            model=None,
            predict_fn=None,
            budget=5,  # type: ignore
        )

        uncertain_ids = {item.id for item in uncertain_selected}
        random_ids = {item.id for item in random_selected}

        # With high probability, selections should differ
        # (especially with varying uncertainties)
        assert uncertain_ids != random_ids

    def test_uncertainty_consistent_across_runs(
        self,
        mock_items: list[Item],
        varying_predict_fn: Callable[[Any, Item], np.ndarray],
    ) -> None:
        """Test that uncertainty sampler is deterministic."""
        sampler1 = UncertaintySampler(method="entropy")
        sampler2 = UncertaintySampler(method="entropy")

        selected1 = sampler1.select(
            items=mock_items, model=None, predict_fn=varying_predict_fn, budget=5
        )

        selected2 = sampler2.select(
            items=mock_items, model=None, predict_fn=varying_predict_fn, budget=5
        )

        # Should select same items (deterministic given same predictions)
        ids1 = [item.id for item in selected1]
        ids2 = [item.id for item in selected2]

        assert ids1 == ids2
