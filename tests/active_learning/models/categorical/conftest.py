"""Shared fixtures for categorical model tests."""

from uuid import uuid4

import pytest

from bead.items.item import Item


@pytest.fixture
def sample_items() -> list[Item]:
    """Create sample categorical items.

    Creates items with premise/hypothesis pairs suitable for NLI-style
    categorical classification.
    """
    items = []
    for i in range(20):
        item = Item(
            item_template_id=uuid4(),
            rendered_elements={
                "premise": f"The cat sat on the mat variant {i}",
                "hypothesis": f"An animal is on the mat variant {i}",
            },
        )
        items.append(item)
    return items


@pytest.fixture
def sample_labels() -> list[str]:
    """Create sample labels for 3-class NLI.

    Returns balanced labels: entailment, neutral, contradiction.
    """
    labels = []
    categories = ["entailment", "neutral", "contradiction"]
    for i in range(20):
        labels.append(categories[i % 3])
    return labels


@pytest.fixture
def binary_labels() -> list[str]:
    """Create binary labels for testing 2-class categorical."""
    return ["yes" if i % 2 == 0 else "no" for i in range(20)]
