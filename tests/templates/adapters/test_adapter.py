"""Test template filling model adapter base class."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.templates.adapters.adapter_helpers import MockMLMAdapter


def test_adapter_init(mock_adapter: MockMLMAdapter) -> None:
    """Test adapter initialization."""
    assert mock_adapter.model_name == "mock-model"
    assert mock_adapter.device == "cpu"
    assert not mock_adapter.is_loaded()


def test_adapter_load_unload(mock_adapter: MockMLMAdapter) -> None:
    """Test adapter load and unload."""
    assert not mock_adapter.is_loaded()

    mock_adapter.load_model()
    assert mock_adapter.is_loaded()
    assert mock_adapter.load_calls == 1

    mock_adapter.unload_model()
    assert not mock_adapter.is_loaded()
    assert mock_adapter.unload_calls == 1


def test_adapter_predict_not_loaded(mock_adapter: MockMLMAdapter) -> None:
    """Test predict fails when model not loaded."""
    with pytest.raises(RuntimeError, match="Model not loaded"):
        mock_adapter.predict_masked_token("test [MASK]", 0)


def test_adapter_predict_loaded(loaded_mock_adapter: MockMLMAdapter) -> None:
    """Test predict succeeds when loaded."""
    predictions = loaded_mock_adapter.predict_masked_token("test [MASK]", 0, top_k=2)

    assert len(predictions) == 2
    assert all(isinstance(p, tuple) for p in predictions)
    assert all(len(p) == 2 for p in predictions)
    assert all(isinstance(p[0], str) and isinstance(p[1], float) for p in predictions)


def test_adapter_context_manager(mock_adapter: MockMLMAdapter) -> None:
    """Test adapter as context manager."""
    assert not mock_adapter.is_loaded()

    with mock_adapter as adapter:
        assert adapter.is_loaded()
        assert adapter.load_calls == 1

    assert not mock_adapter.is_loaded()
    assert mock_adapter.unload_calls == 1


def test_adapter_get_mask_token(loaded_mock_adapter: MockMLMAdapter) -> None:
    """Test get_mask_token method."""
    mask = loaded_mock_adapter.get_mask_token()
    assert mask == "[MASK]"


def test_adapter_with_cache_dir() -> None:
    """Test adapter with cache directory."""
    cache_dir = Path("/tmp/test_cache")
    adapter = MockMLMAdapter(cache_dir=cache_dir)

    assert adapter.cache_dir == cache_dir


def test_adapter_predict_calls_tracked(
    loaded_mock_adapter: MockMLMAdapter,
) -> None:
    """Test that predict calls are tracked."""
    loaded_mock_adapter.predict_masked_token("text1 [MASK]", 0, top_k=5)
    loaded_mock_adapter.predict_masked_token("text2 [MASK]", 1, top_k=10)

    assert len(loaded_mock_adapter.predict_calls) == 2
    assert loaded_mock_adapter.predict_calls[0] == ("text1 [MASK]", 0, 5)
    assert loaded_mock_adapter.predict_calls[1] == ("text2 [MASK]", 1, 10)


def test_adapter_custom_predictions() -> None:
    """Test adapter with custom predictions."""
    custom_preds = {
        "The cat [MASK] on the mat": [
            ("sat", -0.3),
            ("slept", -0.8),
            ("lay", -1.2),
        ]
    }

    adapter = MockMLMAdapter(predictions=custom_preds)
    adapter.load_model()

    predictions = adapter.predict_masked_token("The cat [MASK] on the mat", 0, top_k=2)

    assert len(predictions) == 2
    assert predictions[0] == ("sat", -0.3)
    assert predictions[1] == ("slept", -0.8)
