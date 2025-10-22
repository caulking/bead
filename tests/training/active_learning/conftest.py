"""Test fixtures for active learning tests.

Provides shared fixtures for testing active learning components including
mock items, models, predictions, and sampling strategies.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock
from uuid import uuid4

import numpy as np
import pytest
from pytest_mock import MockerFixture

from sash.items.models import Item

if TYPE_CHECKING:
    from sash.training.trainers.base import ModelMetadata


@pytest.fixture
def mock_items(mocker: MockerFixture) -> list[Item]:
    """Create a list of mock Item objects.

    Returns
    -------
    list[Item]
        List of 10 mock items with unique IDs and rendered elements.

    Examples
    --------
    >>> def test_with_items(mock_items):  # doctest: +SKIP
    ...     assert len(mock_items) == 10
    """
    items = []
    for i in range(10):
        item = Item(
            item_template_id=uuid4(),
            rendered_elements={"text": f"Item {i}", "index": str(i)},
        )
        items.append(item)
    return items


@pytest.fixture
def large_mock_items(mocker: MockerFixture) -> list[Item]:
    """Create a large list of mock Item objects.

    Returns
    -------
    list[Item]
        List of 100 mock items for testing batch operations.

    Examples
    --------
    >>> def test_with_large_items(large_mock_items):  # doctest: +SKIP
    ...     assert len(large_mock_items) == 100
    """
    items = []
    for i in range(100):
        item = Item(
            item_template_id=uuid4(),
            rendered_elements={"text": f"Item {i}"},
        )
        items.append(item)
    return items


@pytest.fixture
def mock_model(mocker: MockerFixture) -> Mock:
    """Create a mock trained model.

    Returns
    -------
    Mock
        Mock model object with predict method.

    Examples
    --------
    >>> def test_with_model(mock_model):  # doctest: +SKIP
    ...     assert mock_model is not None
    """
    model = mocker.Mock()
    model.predict = mocker.Mock(return_value=np.array([0.6, 0.4]))
    return model


@pytest.fixture
def mock_predictions_uniform() -> np.ndarray:
    """Create uniform prediction probabilities.

    Returns
    -------
    np.ndarray
        Array of shape (10, 3) with uniform probabilities (high entropy).

    Examples
    --------
    >>> def test_uniform(mock_predictions_uniform):  # doctest: +SKIP
    ...     assert mock_predictions_uniform.shape == (10, 3)
    """
    return np.ones((10, 3)) / 3.0


@pytest.fixture
def mock_predictions_confident() -> np.ndarray:
    """Create confident prediction probabilities.

    Returns
    -------
    np.ndarray
        Array of shape (10, 3) with confident predictions (low entropy).

    Examples
    --------
    >>> def test_confident(mock_predictions_confident):  # doctest: +SKIP
    ...     assert mock_predictions_confident.shape == (10, 3)
    """
    probs = np.zeros((10, 3))
    probs[:, 0] = 0.9
    probs[:, 1] = 0.05
    probs[:, 2] = 0.05
    return probs


@pytest.fixture
def mock_predictions_mixed() -> np.ndarray:
    """Create mixed prediction probabilities.

    Returns
    -------
    np.ndarray
        Array of shape (10, 3) with varying confidence levels.

    Examples
    --------
    >>> def test_mixed(mock_predictions_mixed):  # doctest: +SKIP
    ...     assert mock_predictions_mixed.shape == (10, 3)
    """
    probs = np.array(
        [
            [0.9, 0.05, 0.05],  # Confident
            [0.33, 0.33, 0.34],  # Uncertain (uniform)
            [0.7, 0.2, 0.1],  # Fairly confident
            [0.5, 0.3, 0.2],  # Moderately uncertain
            [0.95, 0.03, 0.02],  # Very confident
            [0.4, 0.35, 0.25],  # Uncertain
            [0.8, 0.15, 0.05],  # Confident
            [0.45, 0.45, 0.1],  # Uncertain (close margin)
            [0.6, 0.25, 0.15],  # Moderately confident
            [0.5, 0.5, 0.0],  # Uncertain (binary tie)
        ]
    )
    return probs


@pytest.fixture
def simple_predict_fn(mocker: MockerFixture) -> Callable[[Any, Item], np.ndarray]:
    """Create a simple prediction function.

    Returns a callable that returns uniform probabilities for all items.

    Returns
    -------
    Callable[[Any, Item], np.ndarray]
        Prediction function returning [0.5, 0.5].

    Examples
    --------
    >>> def test_predict(simple_predict_fn, mock_items):  # doctest: +SKIP
    ...     probs = simple_predict_fn(None, mock_items[0])
    ...     assert probs.shape == (2,)
    """

    def predict_fn(model: Any, item: Item) -> np.ndarray:  # noqa: ANN401
        return np.array([0.5, 0.5])

    return predict_fn


@pytest.fixture
def varying_predict_fn(
    mocker: MockerFixture,
) -> Callable[[Any, Item], np.ndarray]:
    """Create a prediction function with varying confidence.

    Returns different probabilities based on item index, allowing
    testing of selection based on uncertainty.

    Returns
    -------
    Callable[[Any, Item], np.ndarray]
        Prediction function with varying confidence levels.

    Examples
    --------
    >>> def test_varying(varying_predict_fn, mock_items):  # doctest: +SKIP
    ...     probs0 = varying_predict_fn(None, mock_items[0])
    ...     probs1 = varying_predict_fn(None, mock_items[1])
    ...     assert not np.allclose(probs0, probs1)
    """

    def predict_fn(model: Any, item: Item) -> np.ndarray:  # noqa: ANN401
        # Get item index from rendered elements
        index = int(item.rendered_elements.get("index", 0))

        # Return varying confidence based on index
        if index % 3 == 0:
            # Confident
            return np.array([0.9, 0.1])
        elif index % 3 == 1:
            # Uncertain
            return np.array([0.5, 0.5])
        else:
            # Moderately confident
            return np.array([0.7, 0.3])

    return predict_fn


@pytest.fixture
def mock_trainer(mocker: MockerFixture) -> Mock:
    """Create a mock trainer.

    Returns
    -------
    Mock
        Mock BaseTrainer with train method.

    Examples
    --------
    >>> def test_trainer(mock_trainer):  # doctest: +SKIP
    ...     assert mock_trainer.train is not None
    """
    from pathlib import Path

    from sash.training.trainers.base import ModelMetadata

    trainer = mocker.Mock()

    # Mock train method to return metadata
    def mock_train(train_data: Any, eval_data: Any = None) -> ModelMetadata:  # noqa: ANN401
        return ModelMetadata(
            model_name="test-model",
            framework="test",
            training_config={},
            training_data_path=Path("train.json"),
            metrics={"accuracy": 0.85},
            training_time=10.0,
            training_timestamp="2025-01-17T00:00:00+00:00",
        )

    trainer.train = mocker.Mock(side_effect=mock_train)

    return trainer


@pytest.fixture
def sample_metrics_history() -> list[dict[str, float]]:
    """Create sample metrics history for convergence testing.

    Returns
    -------
    list[dict[str, float]]
        List of metrics from multiple iterations.

    Examples
    --------
    >>> def test_convergence(sample_metrics_history):  # doctest: +SKIP
    ...     assert len(sample_metrics_history) > 0
    """
    return [
        {"accuracy": 0.70, "loss": 0.5},
        {"accuracy": 0.75, "loss": 0.45},
        {"accuracy": 0.78, "loss": 0.42},
        {"accuracy": 0.80, "loss": 0.40},
        {"accuracy": 0.81, "loss": 0.39},
        {"accuracy": 0.81, "loss": 0.39},
        {"accuracy": 0.80, "loss": 0.40},
    ]
