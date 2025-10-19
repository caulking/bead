"""Test HuggingFace MLM adapter."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

from sash.templates.models.huggingface_mlm import HuggingFaceMLMAdapter


def test_adapter_init() -> None:
    """Test HuggingFace adapter initialization."""
    adapter = HuggingFaceMLMAdapter("bert-base-uncased", device="cpu")

    assert adapter.model_name == "bert-base-uncased"
    assert adapter.device == "cpu"
    assert adapter.model is None
    assert adapter.tokenizer is None
    assert not adapter.is_loaded()


def test_adapter_load_missing_transformers(mocker: MockerFixture) -> None:
    """Test load fails when transformers is not installed."""

    def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "transformers":
            raise ImportError("No module named 'transformers'")
        return __import__(name, *args, **kwargs)

    mocker.patch("builtins.__import__", side_effect=mock_import)

    adapter = HuggingFaceMLMAdapter("bert-base-uncased")

    with pytest.raises(ImportError, match="transformers package required"):
        adapter.load_model()


def test_adapter_load_success(mocker: MockerFixture) -> None:
    """Test successful model loading."""
    mock_tokenizer_cls = mocker.patch("transformers.AutoTokenizer")
    mock_model_cls = mocker.patch("transformers.AutoModelForMaskedLM")

    mock_tokenizer = mocker.MagicMock()
    mock_model = mocker.MagicMock()

    mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
    mock_model_cls.from_pretrained.return_value = mock_model

    adapter = HuggingFaceMLMAdapter("bert-base-uncased", device="cpu")
    adapter.load_model()

    assert adapter.is_loaded()
    assert adapter.tokenizer is mock_tokenizer
    assert adapter.model is mock_model

    # Verify loading calls
    mock_tokenizer_cls.from_pretrained.assert_called_once_with(
        "bert-base-uncased", cache_dir=None
    )
    mock_model_cls.from_pretrained.assert_called_once_with(
        "bert-base-uncased", cache_dir=None
    )

    # Verify model was moved to device and set to eval
    mock_model.to.assert_called_once_with("cpu")
    mock_model.eval.assert_called_once()


def test_adapter_unload(mocker: MockerFixture) -> None:
    """Test model unloading."""
    mock_tokenizer_cls = mocker.patch("transformers.AutoTokenizer")
    mock_model_cls = mocker.patch("transformers.AutoModelForMaskedLM")

    mock_tokenizer = mocker.MagicMock()
    mock_model = mocker.MagicMock()

    mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
    mock_model_cls.from_pretrained.return_value = mock_model

    adapter = HuggingFaceMLMAdapter("bert-base-uncased")
    adapter.load_model()

    assert adapter.is_loaded()

    adapter.unload_model()

    assert not adapter.is_loaded()
    assert adapter.model is None
    assert adapter.tokenizer is None

    # Verify model was moved to CPU before deletion
    assert mock_model.to.call_count == 2  # Once to device, once to CPU


def test_adapter_predict_not_loaded() -> None:
    """Test predict fails when model not loaded."""
    adapter = HuggingFaceMLMAdapter("bert-base-uncased")

    with pytest.raises(RuntimeError, match="Model not loaded"):
        adapter.predict_masked_token("The cat [MASK]", 0)


def test_adapter_predict_no_mask_token(mocker: MockerFixture) -> None:
    """Test predict fails when text has no mask token."""
    mock_tokenizer_cls = mocker.patch("transformers.AutoTokenizer")
    mock_model_cls = mocker.patch("transformers.AutoModelForMaskedLM")

    mock_tokenizer = mocker.MagicMock()
    mock_model = mocker.MagicMock()

    mock_tokenizer.mask_token_id = 103
    mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
    mock_model_cls.from_pretrained.return_value = mock_model

    # Mock tokenizer to return input without mask token
    mock_input_ids = mocker.MagicMock()
    mock_input_ids.to.return_value = mock_input_ids

    # Mock comparison result with nonzero method
    mock_comparison = mocker.MagicMock()
    empty_tensor = mocker.MagicMock()
    empty_tensor.__len__ = mocker.MagicMock(return_value=0)
    mock_comparison.nonzero.return_value = (empty_tensor,)

    mock_input_ids.__eq__.return_value = mock_comparison
    mock_input_ids.__getitem__.return_value = mock_input_ids

    mock_tokenizer.return_value = {"input_ids": mock_input_ids}

    adapter = HuggingFaceMLMAdapter("bert-base-uncased")
    adapter.load_model()

    with pytest.raises(ValueError, match="No mask token found"):
        adapter.predict_masked_token("The cat sat", 0)


def test_adapter_get_mask_token(mocker: MockerFixture) -> None:
    """Test get_mask_token method."""
    mock_tokenizer_cls = mocker.patch("transformers.AutoTokenizer")
    mock_model_cls = mocker.patch("transformers.AutoModelForMaskedLM")

    mock_tokenizer = mocker.MagicMock()
    mock_tokenizer.mask_token = "[MASK]"

    mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
    mock_model_cls.from_pretrained.return_value = mocker.MagicMock()

    adapter = HuggingFaceMLMAdapter("bert-base-uncased")
    adapter.load_model()

    mask_token = adapter.get_mask_token()

    assert mask_token == "[MASK]"


def test_adapter_get_mask_token_not_loaded() -> None:
    """Test get_mask_token fails when not loaded."""
    adapter = HuggingFaceMLMAdapter("bert-base-uncased")

    with pytest.raises(RuntimeError, match="Model not loaded"):
        adapter.get_mask_token()


def test_adapter_with_cache_dir(mocker: MockerFixture) -> None:
    """Test adapter with custom cache directory."""
    mock_tokenizer_cls = mocker.patch("transformers.AutoTokenizer")
    mock_model_cls = mocker.patch("transformers.AutoModelForMaskedLM")

    mock_tokenizer_cls.from_pretrained.return_value = mocker.MagicMock()
    mock_model_cls.from_pretrained.return_value = mocker.MagicMock()

    cache_dir = Path("/tmp/test_cache")
    adapter = HuggingFaceMLMAdapter("bert-base-uncased", cache_dir=cache_dir)
    adapter.load_model()

    # Verify cache_dir was passed to from_pretrained
    mock_tokenizer_cls.from_pretrained.assert_called_once_with(
        "bert-base-uncased", cache_dir=cache_dir
    )
    mock_model_cls.from_pretrained.assert_called_once_with(
        "bert-base-uncased", cache_dir=cache_dir
    )


def test_adapter_context_manager(mocker: MockerFixture) -> None:
    """Test adapter as context manager."""
    mock_tokenizer_cls = mocker.patch("transformers.AutoTokenizer")
    mock_model_cls = mocker.patch("transformers.AutoModelForMaskedLM")

    mock_tokenizer_cls.from_pretrained.return_value = mocker.MagicMock()
    mock_model_cls.from_pretrained.return_value = mocker.MagicMock()

    adapter = HuggingFaceMLMAdapter("bert-base-uncased")

    assert not adapter.is_loaded()

    with adapter:
        assert adapter.is_loaded()

    assert not adapter.is_loaded()
