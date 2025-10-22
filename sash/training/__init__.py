"""Model training and active learning for SASH.

This module provides infrastructure for:
- Data collection from experimental platforms (JATOS, Prolific)
- Model training with HuggingFace and PyTorch Lightning
- Active learning strategies for item selection
- Iterative model improvement loops
"""

from __future__ import annotations

from sash.training.trainers import (
    BaseTrainer,
    HuggingFaceTrainer,
    ModelMetadata,
    PyTorchLightningTrainer,
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
