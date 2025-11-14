"""Fixtures for binary model tests."""

from __future__ import annotations

from uuid import uuid4

import pytest

from bead.items.item import Item


@pytest.fixture
def sample_items() -> list[Item]:
    """Create sample binary items.

    Returns
    -------
    list[Item]
        List of 20 binary items with text rendered elements.
    """
    items = []
    for i in range(20):
        item = Item(
            item_template_id=uuid4(),
            rendered_elements={
                "text": f"This is sentence number {i} for binary classification."
            },
        )
        items.append(item)
    return items


@pytest.fixture
def sample_labels() -> list[str]:
    """Create sample binary labels.

    Returns
    -------
    list[str]
        List of 20 labels alternating "yes" and "no".
    """
    labels = []
    for i in range(20):
        labels.append("yes" if i % 2 == 0 else "no")
    return labels
