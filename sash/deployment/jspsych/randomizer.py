"""JavaScript code generator for constraint-aware trial randomization.

This module generates JavaScript functions that enforce OrderingConstraints
at jsPsych runtime. The generated code uses rejection sampling to find
valid randomized trial orders that satisfy all constraints.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from jinja2 import Environment, PackageLoader

from sash.lists.constraints import OrderingConstraint


def generate_randomizer_function(
    item_ids: list[UUID],
    constraints: list[OrderingConstraint],
    metadata: dict[UUID, dict[str, Any]],
) -> str:
    """Generate JavaScript code for constraint-aware trial randomization.

    Creates JavaScript functions that:
    1. Separate practice from main trials
    2. Apply blocking if specified
    3. Randomize trials (within blocks if applicable)
    4. Check all constraints
    5. Retry if constraints violated (rejection sampling, max 1000 attempts)

    Parameters
    ----------
    item_ids : list[UUID]
        UUIDs of all items in the list.
    constraints : list[OrderingConstraint]
        Ordering constraints to enforce.
    metadata : dict[UUID, dict[str, Any]]
        Item metadata needed for constraint checking.
        Keys are item UUIDs, values are metadata dicts.

    Returns
    -------
    str
        JavaScript code as a string.

    Raises
    ------
    ValueError
        If constraints reference undefined properties in metadata.

    Examples
    --------
    >>> from uuid import uuid4
    >>> item_id = uuid4()
    >>> constraint = OrderingConstraint(
    ...     practice_item_property="is_practice"
    ... )
    >>> metadata = {item_id: {"is_practice": True}}
    >>> js_code = generate_randomizer_function([item_id], [constraint], metadata)
    >>> "function randomizeTrials" in js_code
    True

    Notes
    -----
    The generated JavaScript uses:
    - Math.seedrandom for reproducible randomization (seeded by participant ID)
    - Rejection sampling with max 1000 attempts
    - Fallback to last attempt if no valid order found (with warning)

    Generated functions:
    - randomizeTrials(trials, seed): Main entry point
    - shuffleWithConstraints(trials, rng, metadata): Rejection sampling
    - checkAllConstraints(trials, metadata): Validate all constraints
    - checkPrecedence(trials, pairs): Check precedence constraints
    - checkNoAdjacent(trials, property, metadata): Check adjacency constraints
    - checkMinDistance(trials, property, minDist, metadata): Check distance
    - checkPracticeFirst(trials, property, metadata): Check practice items first
    """
    # Validate constraints reference properties that exist in metadata
    _validate_constraints(constraints, metadata)

    # Load Jinja2 template
    env = Environment(loader=PackageLoader("sash.deployment.jspsych", "templates"))
    template = env.get_template("randomizer.js")

    # Prepare template context
    context: dict[str, Any] = {
        "item_ids": [str(uuid) for uuid in item_ids],
        "constraints": _serialize_constraints(constraints),
        "metadata": _serialize_metadata(metadata),
        "has_precedence": any(c.precedence_pairs for c in constraints),
        "has_no_adjacent": any(c.no_adjacent_property for c in constraints),
        "has_min_distance": any(c.min_distance for c in constraints),
        "has_blocking": any(c.block_by_property for c in constraints),
        "has_practice": any(c.practice_item_property for c in constraints),
    }

    # Extract specific constraint values for template
    for constraint in constraints:
        if constraint.no_adjacent_property:
            context["no_adjacent_property"] = constraint.no_adjacent_property
        if constraint.min_distance:
            context["min_distance"] = constraint.min_distance
        if constraint.block_by_property:
            context["block_property"] = constraint.block_by_property
            context["randomize_within_blocks"] = constraint.randomize_within_blocks
        if constraint.practice_item_property:
            context["practice_property"] = constraint.practice_item_property

    return template.render(**context)


def _validate_constraints(
    constraints: list[OrderingConstraint], metadata: dict[UUID, dict[str, Any]]
) -> None:
    """Validate constraints reference properties that exist in metadata.

    Parameters
    ----------
    constraints : list[OrderingConstraint]
        Constraints to validate.
    metadata : dict[UUID, dict[str, Any]]
        Item metadata.

    Raises
    ------
    ValueError
        If a constraint references a property not found in metadata.
    """
    if not metadata:
        return

    # Get a sample metadata dict to check property paths
    sample_meta = next(iter(metadata.values()))

    for constraint in constraints:
        # Check no_adjacent_property
        if constraint.no_adjacent_property:
            _validate_property_path(
                constraint.no_adjacent_property, sample_meta, "no_adjacent_property"
            )

        # Check block_by_property
        if constraint.block_by_property:
            _validate_property_path(
                constraint.block_by_property, sample_meta, "block_by_property"
            )

        # Check practice_item_property
        if constraint.practice_item_property:
            _validate_property_path(
                constraint.practice_item_property, sample_meta, "practice_item_property"
            )


def _validate_property_path(
    property_path: str, metadata: dict[str, Any], constraint_field: str
) -> None:
    """Validate a property path exists in metadata.

    Parameters
    ----------
    property_path : str
        Dot-notation property path (e.g., "item_metadata.condition").
    metadata : dict[str, Any]
        Metadata dict to check.
    constraint_field : str
        Name of constraint field (for error messages).

    Raises
    ------
    ValueError
        If property path is not found in metadata.
    """
    parts = property_path.split(".")
    current = metadata

    for part in parts:
        if not isinstance(current, dict) or part not in current:
            raise ValueError(
                f"Property path '{property_path}' in {constraint_field} "
                f"not found in metadata. Available keys: {list(metadata.keys())}"
            )
        current = current[part]


def _serialize_constraints(
    constraints: list[OrderingConstraint],
) -> list[dict[str, Any]]:
    """Serialize constraints for JavaScript template.

    Parameters
    ----------
    constraints : list[OrderingConstraint]
        Constraints to serialize.

    Returns
    -------
    list[dict[str, Any]]
        Serialized constraints.
    """
    serialized: list[dict[str, Any]] = []
    for constraint in constraints:
        data: dict[str, Any] = {
            "constraint_type": constraint.constraint_type,
            "precedence_pairs": [
                (str(a), str(b)) for a, b in constraint.precedence_pairs
            ],
            "no_adjacent_property": constraint.no_adjacent_property,
            "block_by_property": constraint.block_by_property,
            "min_distance": constraint.min_distance,
            "max_distance": constraint.max_distance,
            "practice_item_property": constraint.practice_item_property,
            "randomize_within_blocks": constraint.randomize_within_blocks,
        }
        serialized.append(data)
    return serialized


def _serialize_metadata(
    metadata: dict[UUID, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Serialize metadata for JavaScript template (UUID keys → strings).

    Parameters
    ----------
    metadata : dict[UUID, dict[str, Any]]
        Metadata with UUID keys.

    Returns
    -------
    dict[str, dict[str, Any]]
        Metadata with string keys.
    """
    return {str(uuid): meta for uuid, meta in metadata.items()}
