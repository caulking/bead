"""Tests for active learning sampling strategies.

Tests all uncertainty quantification methods including entropy, margin,
least confidence, and random sampling, verifying mathematical correctness
and edge case handling.
"""

from __future__ import annotations

import numpy as np
import pytest

from sash.training.active_learning.strategies import (
    LeastConfidenceSampling,
    MarginSampling,
    RandomSampling,
    SamplingStrategy,
    UncertaintySampling,
    create_strategy,
)


class TestUncertaintySampling:
    """Test suite for entropy-based uncertainty sampling."""

    def test_initialization(self) -> None:
        """Test strategy initialization."""
        strategy = UncertaintySampling()
        assert isinstance(strategy, SamplingStrategy)

    def test_compute_scores_uniform_distribution(self) -> None:
        """Test entropy for uniform distribution (maximum uncertainty)."""
        strategy = UncertaintySampling()

        # Uniform distribution over 3 classes
        probs = np.array([[1 / 3, 1 / 3, 1 / 3]])

        scores = strategy.compute_scores(probs)

        # Entropy of uniform distribution: -log(1/3) ≈ 1.0986
        expected_entropy = -np.log(1 / 3)
        assert np.allclose(scores[0], expected_entropy, rtol=1e-3)

    def test_compute_scores_confident_prediction(self) -> None:
        """Test entropy for confident prediction (minimum uncertainty)."""
        strategy = UncertaintySampling()

        # Very confident prediction
        probs = np.array([[0.99, 0.005, 0.005]])

        scores = strategy.compute_scores(probs)

        # Entropy should be low
        assert scores[0] < 0.1

    def test_compute_scores_batch(self) -> None:
        """Test entropy computation for batch of predictions."""
        strategy = UncertaintySampling()

        probs = np.array(
            [
                [0.9, 0.05, 0.05],  # Confident (low entropy)
                [1 / 3, 1 / 3, 1 / 3],  # Uniform (high entropy)
                [0.5, 0.5, 0.0],  # Binary uncertain
            ]
        )

        scores = strategy.compute_scores(probs)

        # Check shape
        assert scores.shape == (3,)

        # Check ordering: uniform > binary > confident
        assert scores[1] > scores[2] > scores[0]

    def test_compute_scores_numerical_stability(self) -> None:
        """Test that computation handles probabilities near 0 and 1."""
        strategy = UncertaintySampling()

        # Edge case: probabilities very close to 0
        probs = np.array([[1.0 - 1e-15, 1e-16, 1e-16]])

        scores = strategy.compute_scores(probs)

        # Should not produce NaN or inf
        assert np.isfinite(scores[0])
        assert scores[0] >= 0

    def test_select_top_k_basic(self) -> None:
        """Test selecting top k items by entropy."""
        strategy = UncertaintySampling()

        scores = np.array([0.1, 0.5, 0.3, 0.9, 0.2])
        indices = strategy.select_top_k(scores, k=3)

        # Should return top 3: indices 3, 1, 2 (scores 0.9, 0.5, 0.3)
        assert len(indices) == 3
        assert indices[0] == 3  # Highest score
        assert indices[1] == 1
        assert indices[2] == 2

    def test_select_top_k_all_items(self) -> None:
        """Test selecting when k >= number of items."""
        strategy = UncertaintySampling()

        scores = np.array([0.1, 0.5, 0.3])
        indices = strategy.select_top_k(scores, k=5)

        # Should return all indices
        assert len(indices) == 3
        assert set(indices) == {0, 1, 2}


class TestMarginSampling:
    """Test suite for margin-based uncertainty sampling."""

    def test_initialization(self) -> None:
        """Test strategy initialization."""
        strategy = MarginSampling()
        assert isinstance(strategy, SamplingStrategy)

    def test_compute_scores_small_margin(self) -> None:
        """Test margin for close top-2 predictions (high uncertainty)."""
        strategy = MarginSampling()

        # Top 2 are very close: 0.51 and 0.49
        probs = np.array([[0.51, 0.49, 0.0]])

        scores = strategy.compute_scores(probs)

        # Margin = 1 - (0.51 - 0.49) = 1 - 0.02 = 0.98
        expected = 1.0 - (0.51 - 0.49)
        assert np.allclose(scores[0], expected, rtol=1e-3)

    def test_compute_scores_large_margin(self) -> None:
        """Test margin for confident prediction (low uncertainty)."""
        strategy = MarginSampling()

        # Large margin between top 2
        probs = np.array([[0.9, 0.05, 0.05]])

        scores = strategy.compute_scores(probs)

        # Margin = 1 - (0.9 - 0.05) = 1 - 0.85 = 0.15
        expected = 1.0 - (0.9 - 0.05)
        assert np.allclose(scores[0], expected, rtol=1e-3)

    def test_compute_scores_batch(self) -> None:
        """Test margin computation for batch of predictions."""
        strategy = MarginSampling()

        probs = np.array(
            [
                [0.9, 0.05, 0.05],  # Large margin (low uncertainty)
                [0.51, 0.49, 0.0],  # Small margin (high uncertainty)
                [0.6, 0.3, 0.1],  # Medium margin
            ]
        )

        scores = strategy.compute_scores(probs)

        # Check shape
        assert scores.shape == (3,)

        # Check ordering: small margin > medium > large
        assert scores[1] > scores[2] > scores[0]

    def test_compute_scores_three_way_tie(self) -> None:
        """Test margin with near-uniform distribution."""
        strategy = MarginSampling()

        # Three-way near-tie
        probs = np.array([[0.34, 0.33, 0.33]])

        scores = strategy.compute_scores(probs)

        # Margin between top 2 should be very small (high uncertainty)
        assert scores[0] > 0.95


class TestLeastConfidenceSampling:
    """Test suite for least confidence sampling."""

    def test_initialization(self) -> None:
        """Test strategy initialization."""
        strategy = LeastConfidenceSampling()
        assert isinstance(strategy, SamplingStrategy)

    def test_compute_scores_low_confidence(self) -> None:
        """Test least confidence for uncertain prediction."""
        strategy = LeastConfidenceSampling()

        # Low confidence: max = 0.4
        probs = np.array([[0.4, 0.3, 0.3]])

        scores = strategy.compute_scores(probs)

        # Least confidence = 1 - 0.4 = 0.6
        assert np.allclose(scores[0], 0.6)

    def test_compute_scores_high_confidence(self) -> None:
        """Test least confidence for confident prediction."""
        strategy = LeastConfidenceSampling()

        # High confidence: max = 0.95
        probs = np.array([[0.95, 0.03, 0.02]])

        scores = strategy.compute_scores(probs)

        # Least confidence = 1 - 0.95 = 0.05
        assert np.allclose(scores[0], 0.05)

    def test_compute_scores_batch(self) -> None:
        """Test least confidence for batch of predictions."""
        strategy = LeastConfidenceSampling()

        probs = np.array(
            [
                [0.95, 0.03, 0.02],  # High confidence (low score)
                [0.4, 0.3, 0.3],  # Low confidence (high score)
                [0.7, 0.2, 0.1],  # Medium confidence
            ]
        )

        scores = strategy.compute_scores(probs)

        # Check shape
        assert scores.shape == (3,)

        # Check ordering: low confidence > medium > high
        assert scores[1] > scores[2] > scores[0]

    def test_compute_scores_uniform(self) -> None:
        """Test least confidence for uniform distribution."""
        strategy = LeastConfidenceSampling()

        # Uniform over 3 classes: max = 1/3
        probs = np.array([[1 / 3, 1 / 3, 1 / 3]])

        scores = strategy.compute_scores(probs)

        # Least confidence = 1 - 1/3 ≈ 0.667
        assert np.allclose(scores[0], 2 / 3, rtol=1e-3)


class TestRandomSampling:
    """Test suite for random sampling baseline."""

    def test_initialization_with_seed(self) -> None:
        """Test initialization with seed for reproducibility."""
        strategy1 = RandomSampling(seed=42)
        strategy2 = RandomSampling(seed=42)

        assert isinstance(strategy1, SamplingStrategy)
        assert isinstance(strategy2, SamplingStrategy)

    def test_compute_scores_reproducible(self) -> None:
        """Test that same seed produces same random scores."""
        strategy1 = RandomSampling(seed=123)
        strategy2 = RandomSampling(seed=123)

        probs = np.array([[0.5, 0.5], [0.9, 0.1]])

        scores1 = strategy1.compute_scores(probs)
        scores2 = strategy2.compute_scores(probs)

        # Same seed should give same scores
        assert np.allclose(scores1, scores2)

    def test_compute_scores_different_seeds(self) -> None:
        """Test that different seeds produce different scores."""
        strategy1 = RandomSampling(seed=42)
        strategy2 = RandomSampling(seed=999)

        probs = np.array([[0.5, 0.5], [0.9, 0.1]])

        scores1 = strategy1.compute_scores(probs)
        scores2 = strategy2.compute_scores(probs)

        # Different seeds should give different scores
        assert not np.allclose(scores1, scores2)

    def test_compute_scores_range(self) -> None:
        """Test that random scores are in valid range [0, 1]."""
        strategy = RandomSampling(seed=42)

        probs = np.random.dirichlet(np.ones(3), size=100)

        scores = strategy.compute_scores(probs)

        # All scores should be in [0, 1]
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)

    def test_compute_scores_ignores_probabilities(self) -> None:
        """Test that scores are independent of input probabilities."""
        RandomSampling(seed=42)

        # Very different probability distributions
        probs1 = np.array([[0.9, 0.1], [0.8, 0.2]])
        probs2 = np.array([[0.5, 0.5], [0.5, 0.5]])

        # Reset RNG state
        strategy1 = RandomSampling(seed=42)
        strategy2 = RandomSampling(seed=42)

        scores1 = strategy1.compute_scores(probs1)
        scores2 = strategy2.compute_scores(probs2)

        # Should produce same random scores regardless of input
        assert np.allclose(scores1, scores2)


class TestCreateStrategy:
    """Test suite for strategy factory function."""

    def test_create_entropy_strategy(self) -> None:
        """Test creating entropy strategy."""
        strategy = create_strategy("entropy")
        assert isinstance(strategy, UncertaintySampling)

    def test_create_margin_strategy(self) -> None:
        """Test creating margin strategy."""
        strategy = create_strategy("margin")
        assert isinstance(strategy, MarginSampling)

    def test_create_least_confidence_strategy(self) -> None:
        """Test creating least confidence strategy."""
        strategy = create_strategy("least_confidence")
        assert isinstance(strategy, LeastConfidenceSampling)

    def test_create_random_strategy(self) -> None:
        """Test creating random strategy."""
        strategy = create_strategy("random", seed=42)
        assert isinstance(strategy, RandomSampling)

    def test_create_invalid_strategy(self) -> None:
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown sampling method"):
            create_strategy("invalid_method")  # type: ignore

    def test_random_strategy_with_seed(self) -> None:
        """Test that seed is passed to random strategy."""
        strategy1 = create_strategy("random", seed=42)
        strategy2 = create_strategy("random", seed=42)

        probs = np.array([[0.5, 0.5]])

        scores1 = strategy1.compute_scores(probs)
        scores2 = strategy2.compute_scores(probs)

        # Same seed should produce same results
        assert np.allclose(scores1, scores2)


class TestEdgeCases:
    """Test edge cases across all strategies."""

    @pytest.mark.parametrize(
        "method",
        ["entropy", "margin", "least_confidence", "random"],
    )
    def test_single_item(self, method: str) -> None:
        """Test strategies with single item."""
        strategy = create_strategy(method, seed=42)

        probs = np.array([[0.7, 0.2, 0.1]])

        scores = strategy.compute_scores(probs)

        assert scores.shape == (1,)
        assert np.isfinite(scores[0])

    @pytest.mark.parametrize(
        "method",
        ["entropy", "margin", "least_confidence", "random"],
    )
    def test_binary_classification(self, method: str) -> None:
        """Test strategies with binary classification."""
        strategy = create_strategy(method, seed=42)

        probs = np.array([[0.6, 0.4], [0.9, 0.1], [0.5, 0.5]])

        scores = strategy.compute_scores(probs)

        assert scores.shape == (3,)
        assert np.all(np.isfinite(scores))

    @pytest.mark.parametrize(
        "method",
        ["entropy", "margin", "least_confidence"],
    )
    def test_many_classes(self, method: str) -> None:
        """Test strategies with many classes."""
        strategy = create_strategy(method)

        # 10 classes
        probs = np.random.dirichlet(np.ones(10), size=5)

        scores = strategy.compute_scores(probs)

        assert scores.shape == (5,)
        assert np.all(np.isfinite(scores))

    def test_select_top_k_zero(self) -> None:
        """Test selecting zero items."""
        strategy = UncertaintySampling()

        scores = np.array([0.1, 0.5, 0.3])
        indices = strategy.select_top_k(scores, k=0)

        # Should handle gracefully (implementation dependent)
        assert len(indices) == 0

    def test_select_top_k_single(self) -> None:
        """Test selecting single item."""
        strategy = UncertaintySampling()

        scores = np.array([0.1, 0.9, 0.3])
        indices = strategy.select_top_k(scores, k=1)

        assert len(indices) == 1
        assert indices[0] == 1  # Highest score


class TestMathematicalCorrectness:
    """Test mathematical correctness of uncertainty measures."""

    def test_entropy_formula(self) -> None:
        """Test entropy matches mathematical definition."""
        strategy = UncertaintySampling()

        probs = np.array([[0.5, 0.3, 0.2]])

        scores = strategy.compute_scores(probs)

        # Manual calculation: -∑(p * log(p))
        expected = -(0.5 * np.log(0.5) + 0.3 * np.log(0.3) + 0.2 * np.log(0.2))

        assert np.allclose(scores[0], expected, rtol=1e-5)

    def test_margin_formula(self) -> None:
        """Test margin matches mathematical definition."""
        strategy = MarginSampling()

        probs = np.array([[0.6, 0.3, 0.1]])

        scores = strategy.compute_scores(probs)

        # Manual calculation: 1 - (p1 - p2) where p1=0.6, p2=0.3
        expected = 1.0 - (0.6 - 0.3)

        assert np.allclose(scores[0], expected, rtol=1e-5)

    def test_least_confidence_formula(self) -> None:
        """Test least confidence matches mathematical definition."""
        strategy = LeastConfidenceSampling()

        probs = np.array([[0.7, 0.2, 0.1]])

        scores = strategy.compute_scores(probs)

        # Manual calculation: 1 - max(p) where max=0.7
        expected = 1.0 - 0.7

        assert np.allclose(scores[0], expected, rtol=1e-5)

    def test_entropy_symmetry(self) -> None:
        """Test that entropy is symmetric under permutation."""
        strategy = UncertaintySampling()

        # Same distribution, different order
        probs1 = np.array([[0.5, 0.3, 0.2]])
        probs2 = np.array([[0.2, 0.5, 0.3]])

        scores1 = strategy.compute_scores(probs1)
        scores2 = strategy.compute_scores(probs2)

        # Entropy should be the same
        assert np.allclose(scores1, scores2)

    def test_margin_not_symmetric(self) -> None:
        """Test that margin depends on top 2 classes only."""
        strategy = MarginSampling()

        # Different distributions but same top 2
        probs1 = np.array([[0.6, 0.3, 0.1]])
        probs2 = np.array([[0.6, 0.3, 0.05, 0.05]])

        scores1 = strategy.compute_scores(probs1)
        scores2 = strategy.compute_scores(probs2)

        # Margin should be the same (depends only on top 2)
        assert np.allclose(scores1, scores2)
