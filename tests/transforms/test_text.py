"""Tests for text transforms."""

from __future__ import annotations

from bead.transforms.base import TransformContext
from bead.transforms.text import (
    CapitalizeTransform,
    LowerTransform,
    TitleTransform,
    UpperTransform,
)


class TestLowerTransform:
    """Tests for LowerTransform."""

    def test_basic(self) -> None:
        assert LowerTransform()("HELLO", TransformContext()) == "hello"

    def test_mixed_case(self) -> None:
        assert LowerTransform()("HeLLo WoRLd", TransformContext()) == "hello world"

    def test_already_lower(self) -> None:
        assert LowerTransform()("hello", TransformContext()) == "hello"


class TestUpperTransform:
    """Tests for UpperTransform."""

    def test_basic(self) -> None:
        assert UpperTransform()("hello", TransformContext()) == "HELLO"

    def test_already_upper(self) -> None:
        assert UpperTransform()("HELLO", TransformContext()) == "HELLO"


class TestCapitalizeTransform:
    """Tests for CapitalizeTransform."""

    def test_basic(self) -> None:
        assert CapitalizeTransform()("hello world", TransformContext()) == "Hello world"

    def test_all_caps_input(self) -> None:
        assert CapitalizeTransform()("HELLO WORLD", TransformContext()) == "Hello world"


class TestTitleTransform:
    """Tests for TitleTransform."""

    def test_basic(self) -> None:
        assert TitleTransform()("hello world", TransformContext()) == "Hello World"

    def test_already_title(self) -> None:
        assert TitleTransform()("Hello World", TransformContext()) == "Hello World"
