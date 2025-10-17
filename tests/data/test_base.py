"""Tests for SashBaseModel."""

from __future__ import annotations

import time

import pytest
from pydantic import ValidationError

from sash.data.base import SashBaseModel
from sash.data.identifiers import is_valid_uuid7


class TestModel(SashBaseModel):
    """Test model for base model tests."""

    name: str
    value: int


def test_sashbasemodel_creates_uuid() -> None:
    """Test that SashBaseModel automatically generates UUID."""
    obj = TestModel(name="test", value=42)
    assert obj.id is not None
    assert is_valid_uuid7(obj.id)


def test_sashbasemodel_creates_timestamps() -> None:
    """Test that SashBaseModel automatically creates timestamps."""
    obj = TestModel(name="test", value=42)
    assert obj.created_at is not None
    assert obj.modified_at is not None
    assert obj.created_at.tzinfo is not None
    assert obj.modified_at.tzinfo is not None


def test_sashbasemodel_default_version() -> None:
    """Test that SashBaseModel has default version."""
    obj = TestModel(name="test", value=42)
    assert obj.version == "1.0.0"


def test_sashbasemodel_default_metadata() -> None:
    """Test that SashBaseModel has default empty metadata."""
    obj = TestModel(name="test", value=42)
    assert obj.metadata == {}


def test_sashbasemodel_update_modified_time() -> None:
    """Test that update_modified_time updates timestamp."""
    obj = TestModel(name="test", value=42)
    original_modified = obj.modified_at

    time.sleep(0.01)  # Small delay to ensure different timestamp
    obj.update_modified_time()

    assert obj.modified_at > original_modified


def test_sashbasemodel_forbids_extra_fields() -> None:
    """Test that SashBaseModel forbids extra fields."""
    with pytest.raises(ValidationError):
        TestModel(name="test", value=42, extra_field="not allowed")  # type: ignore[call-arg]


def test_sashbasemodel_validates_on_assignment() -> None:
    """Test that SashBaseModel validates on assignment."""
    obj = TestModel(name="test", value=42)

    with pytest.raises(ValidationError):
        obj.value = "not an int"  # type: ignore[assignment]


def test_sashbasemodel_timestamps_ordered() -> None:
    """Test that created_at <= modified_at."""
    obj = TestModel(name="test", value=42)
    assert obj.created_at <= obj.modified_at


def test_sashbasemodel_custom_metadata() -> None:
    """Test that custom metadata can be provided."""
    metadata = {"key": "value", "number": 42}
    obj = TestModel(name="test", value=42, metadata=metadata)
    assert obj.metadata == metadata


def test_sashbasemodel_custom_version() -> None:
    """Test that custom version can be provided."""
    obj = TestModel(name="test", value=42, version="2.0.0")
    assert obj.version == "2.0.0"
