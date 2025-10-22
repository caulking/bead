"""Training framework adapters for sash.

This module provides trainer implementations for different ML frameworks:
- HuggingFace Transformers
- PyTorch Lightning

All trainers implement the BaseTrainer interface and return ModelMetadata
for tracking training results.
"""

from __future__ import annotations

from sash.training.trainers.base import BaseTrainer, ModelMetadata
from sash.training.trainers.huggingface import HuggingFaceTrainer
from sash.training.trainers.lightning import PyTorchLightningTrainer
from sash.training.trainers.registry import (
    get_trainer,
    list_trainers,
    register_trainer,
)

__all__ = [
    "BaseTrainer",
    "ModelMetadata",
    "HuggingFaceTrainer",
    "PyTorchLightningTrainer",
    "register_trainer",
    "get_trainer",
    "list_trainers",
]
