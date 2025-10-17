"""Pytest fixtures for data module tests."""

from __future__ import annotations

import pytest

from sash.data.base import SashBaseModel


class SimpleTestModel(SashBaseModel):
    """Simple test model for serialization tests."""

    name: str
    value: int


@pytest.fixture
def simple_test_model() -> type[SimpleTestModel]:
    """Provide SimpleTestModel class.

    Returns
    -------
    type[SimpleTestModel]
        Test model class
    """
    return SimpleTestModel


@pytest.fixture
def sample_test_objects() -> list[SimpleTestModel]:
    """Create sample test objects.

    Returns
    -------
    list[SimpleTestModel]
        List of test objects
    """
    return [
        SimpleTestModel(name="test1", value=1),
        SimpleTestModel(name="test2", value=2),
        SimpleTestModel(name="test3", value=3),
    ]
