"""Test helpers for data module tests."""

from __future__ import annotations

from sash.data.base import SashBaseModel


class SimpleTestModel(SashBaseModel):
    """Simple test model for serialization tests."""

    name: str
    value: int
