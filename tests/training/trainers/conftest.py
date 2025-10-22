"""Pytest fixtures for trainer tests.

This module provides mocked fixtures for testing trainers without requiring
actual model downloads or GPU training.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def training_config(tmp_path: Path) -> dict:
    """Training configuration for tests.

    Parameters
    ----------
    tmp_path : Path
        Pytest tmp_path fixture.

    Returns
    -------
    dict
        Training configuration.
    """
    return {
        "model_name": "bert-base-uncased",
        "task_type": "classification",
        "num_labels": 2,
        "output_dir": tmp_path / "output",
        "num_epochs": 1,
        "batch_size": 4,
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "warmup_steps": 0,
        "evaluation_strategy": "epoch",
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "logging_dir": None,
        "fp16": False,
    }


@pytest.fixture
def mock_transformers(mocker):
    """Mock transformers to avoid model downloads.

    Parameters
    ----------
    mocker : pytest_mock.MockerFixture
        Pytest mocker fixture.

    Returns
    -------
    dict
        Dictionary of mocked transformers components.
    """
    mock_model = mocker.Mock()
    mock_tokenizer = mocker.Mock()

    mock_trainer_instance = mocker.Mock()
    mock_trainer_instance.train.return_value = None
    mock_trainer_instance.evaluate.return_value = {"eval_loss": 0.5}
    mock_trainer_instance.state.best_model_checkpoint = None

    mock_trainer_cls = mocker.Mock(return_value=mock_trainer_instance)

    mock_auto_model = mocker.Mock()
    mock_auto_model.from_pretrained = mocker.Mock(return_value=mock_model)

    mock_auto_tokenizer = mocker.Mock()
    mock_auto_tokenizer.from_pretrained = mocker.Mock(return_value=mock_tokenizer)

    mock_training_args = mocker.Mock()
    mock_data_collator_cls = mocker.Mock()

    # Create a comprehensive transformers mock module
    mock_transformers_module = mocker.Mock()
    mock_transformers_module.AutoModelForSequenceClassification = mock_auto_model
    mock_transformers_module.AutoTokenizer = mock_auto_tokenizer
    mock_transformers_module.Trainer = mock_trainer_cls
    mock_transformers_module.TrainingArguments = mocker.Mock(return_value=mock_training_args)
    mock_transformers_module.DataCollatorWithPadding = mock_data_collator_cls

    # Patch sys.modules so imports inside functions work
    mocker.patch.dict("sys.modules", {"transformers": mock_transformers_module})

    return {
        "model": mock_model,
        "tokenizer": mock_tokenizer,
        "trainer_cls": mock_trainer_cls,
        "trainer_instance": mock_trainer_instance,
        "auto_model": mock_auto_model,
        "auto_tokenizer": mock_auto_tokenizer,
        "training_args": mock_training_args,
        "data_collator": mock_data_collator_cls,
    }


@pytest.fixture
def mock_lightning(mocker):
    """Mock PyTorch Lightning to avoid actual training.

    Parameters
    ----------
    mocker : pytest_mock.MockerFixture
        Pytest mocker fixture.

    Returns
    -------
    dict
        Dictionary of mocked Lightning components.
    """
    mock_trainer_instance = mocker.Mock()
    mock_trainer_instance.checkpoint_callback.best_model_path = "/tmp/best.ckpt"
    mock_trainer_instance.validate.return_value = [{"val_loss": 0.5}]

    mock_trainer_cls = mocker.Mock(return_value=mock_trainer_instance)

    mock_checkpoint_cls = mocker.Mock()
    mock_early_stopping_cls = mocker.Mock()
    mock_tensorboard_logger_cls = mocker.Mock()

    # Create mock Lightning module
    mock_pl = mocker.Mock()
    mock_pl.Trainer = mock_trainer_cls
    mock_pl.callbacks = mocker.Mock()
    mock_pl.callbacks.ModelCheckpoint = mock_checkpoint_cls
    mock_pl.callbacks.EarlyStopping = mock_early_stopping_cls
    mock_pl.loggers = mocker.Mock()
    mock_pl.loggers.TensorBoardLogger = mock_tensorboard_logger_cls
    mock_pl.LightningModule = mocker.Mock

    mocker.patch.dict("sys.modules", {"pytorch_lightning": mock_pl})

    # Mock transformers for model loading
    mock_model = mocker.Mock()
    mock_auto_model = mocker.Mock()
    mock_auto_model.from_pretrained = mocker.Mock(return_value=mock_model)

    mock_transformers_module = mocker.Mock()
    mock_transformers_module.AutoModelForSequenceClassification = mock_auto_model
    mocker.patch.dict("sys.modules", {"transformers": mock_transformers_module})

    # Mock torch
    mock_torch = mocker.Mock()
    mock_torch.optim = mocker.Mock()
    mock_torch.optim.AdamW = mocker.Mock()
    mocker.patch.dict("sys.modules", {"torch": mock_torch})

    return {
        "trainer_cls": mock_trainer_cls,
        "trainer_instance": mock_trainer_instance,
        "checkpoint": mock_checkpoint_cls,
        "early_stopping": mock_early_stopping_cls,
        "logger": mock_tensorboard_logger_cls,
        "model": mock_model,
        "auto_model": mock_auto_model,
        "torch": mock_torch,
        "pl": mock_pl,
    }


@pytest.fixture
def mock_dataset(mocker):
    """Mock dataset for training.

    Parameters
    ----------
    mocker : pytest_mock.MockerFixture
        Pytest mocker fixture.

    Returns
    -------
    Mock
        Mock dataset.
    """
    return mocker.Mock()
