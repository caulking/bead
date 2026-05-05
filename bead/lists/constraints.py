"""Constraint models for experimental list composition.

List-level constraints govern composition of a single list (uniqueness,
balance, quantile distribution, size, ordering, etc.). Batch-level
constraints govern composition across a collection of lists (coverage,
balance, diversity, minimum occurrence). Each family is a discriminated
union rooted at ``ListConstraint`` / ``BatchConstraint``; subclass
construction takes the matching ``constraint_type`` value.
"""

from __future__ import annotations

import typing
from uuid import UUID

import didactic.api as dx

from bead.data.base import BeadBaseModel
from bead.resources.constraints import ContextValue

ListConstraintType = typing.Literal[
    "uniqueness",
    "conditional_uniqueness",
    "balance",
    "quantile",
    "grouped_quantile",
    "diversity",
    "size",
    "ordering",
]

BatchConstraintType = typing.Literal[
    "coverage",
    "balance",
    "diversity",
    "min_occurrence",
]


class ListConstraint(BeadBaseModel, dx.TaggedUnion, discriminator="constraint_type"):
    """Discriminated union root for list-level constraints."""


class BatchConstraint(BeadBaseModel, dx.TaggedUnion, discriminator="constraint_type"):
    """Discriminated union root for batch-level constraints."""


def _check_non_empty(_cls: type, value: str) -> str:
    if not value or not value.strip():
        raise ValueError("expression must be non-empty")
    return value.strip()


# ---------------------------------------------------------------------------
# list-level constraints
# ---------------------------------------------------------------------------


class UniquenessConstraint(ListConstraint):
    """No two items in a list share the same value of ``property_expression``.

    Attributes
    ----------
    property_expression : str
        DSL expression returning the value that must be unique across the
        list.
    context : dict[str, ContextValue]
        Extra DSL evaluation context.
    allow_null : bool
        Allow multiple items with a ``None`` value.
    priority : int
        Higher values are weighted more heavily during partitioning.
    """

    constraint_type: typing.Literal["uniqueness"]
    property_expression: str
    context: dict[str, ContextValue] = dx.field(default_factory=dict)
    allow_null: bool = False
    priority: int = 1

    @dx.validates("property_expression")
    def _check_property_expression(self, value: str) -> str:
        return _check_non_empty(type(self), value)


class ConditionalUniquenessConstraint(ListConstraint):
    """Uniqueness applied only when ``condition_expression`` evaluates true.

    Attributes
    ----------
    property_expression : str
        DSL expression returning the value that must be unique.
    condition_expression : str
        DSL boolean expression gating constraint application.
    context : dict[str, ContextValue]
        Extra DSL evaluation context.
    allow_null : bool
        Allow multiple items with a ``None`` value.
    priority : int
        Constraint priority.
    """

    constraint_type: typing.Literal["conditional_uniqueness"]
    property_expression: str
    condition_expression: str
    context: dict[str, ContextValue] = dx.field(default_factory=dict)
    allow_null: bool = False
    priority: int = 1

    @dx.validates("property_expression", "condition_expression")
    def _check_expr(self, value: str) -> str:
        return _check_non_empty(type(self), value)


class BalanceConstraint(ListConstraint):
    """Balanced distribution of a categorical property within a list.

    Attributes
    ----------
    property_expression : str
        DSL expression returning the category value.
    context : dict[str, ContextValue]
        Extra DSL evaluation context.
    target_counts : dict[str, int] | None
        Target counts per category. ``None`` means equal distribution.
    tolerance : float
        Allowed deviation from target as a proportion (0.0-1.0).
    priority : int
        Constraint priority.
    """

    constraint_type: typing.Literal["balance"]
    property_expression: str
    context: dict[str, ContextValue] = dx.field(default_factory=dict)
    target_counts: dict[str, int] | None = None
    tolerance: float = 0.1
    priority: int = 1

    @dx.validates("property_expression")
    def _check_property_expression(self, value: str) -> str:
        return _check_non_empty(type(self), value)

    @dx.validates("target_counts")
    def _check_target_counts(
        self, value: dict[str, int] | None
    ) -> dict[str, int] | None:
        if value is None:
            return value
        for category, count in value.items():
            if count < 0:
                raise ValueError(
                    f"target_counts values must be non-negative, "
                    f"got {count} for '{category}'"
                )
        return value


class QuantileConstraint(ListConstraint):
    """Uniform distribution of items across quantiles of a numeric property.

    Attributes
    ----------
    property_expression : str
        DSL expression returning the numeric value to quantile.
    context : dict[str, ContextValue]
        Extra DSL evaluation context.
    n_quantiles : int
        Number of quantiles to create (>= 2).
    items_per_quantile : int
        Target items per quantile (>= 1).
    priority : int
        Constraint priority.
    """

    constraint_type: typing.Literal["quantile"]
    property_expression: str
    context: dict[str, ContextValue] = dx.field(default_factory=dict)
    n_quantiles: int = 5
    items_per_quantile: int = 2
    priority: int = 1

    @dx.validates("property_expression")
    def _check_property_expression(self, value: str) -> str:
        return _check_non_empty(type(self), value)


class GroupedQuantileConstraint(ListConstraint):
    """Quantile uniformity applied within groups defined by another expression.

    Attributes
    ----------
    property_expression : str
        DSL expression returning the numeric value to quantile.
    group_by_expression : str
        DSL expression returning the grouping key.
    context : dict[str, ContextValue]
        Extra DSL evaluation context.
    n_quantiles : int
        Quantiles per group.
    items_per_quantile : int
        Target items per quantile per group.
    priority : int
        Constraint priority.
    """

    constraint_type: typing.Literal["grouped_quantile"]
    property_expression: str
    group_by_expression: str
    context: dict[str, ContextValue] = dx.field(default_factory=dict)
    n_quantiles: int = 5
    items_per_quantile: int = 2
    priority: int = 1

    @dx.validates("property_expression", "group_by_expression")
    def _check_expr(self, value: str) -> str:
        return _check_non_empty(type(self), value)


class DiversityConstraint(ListConstraint):
    """Minimum number of unique values for a property within a list.

    Attributes
    ----------
    property_expression : str
        DSL expression returning the value to count for diversity.
    min_unique_values : int
        Minimum number of unique values required (>= 1).
    context : dict[str, ContextValue]
        Extra DSL evaluation context.
    priority : int
        Constraint priority.
    """

    constraint_type: typing.Literal["diversity"]
    property_expression: str
    min_unique_values: int
    context: dict[str, ContextValue] = dx.field(default_factory=dict)
    priority: int = 1

    @dx.validates("property_expression")
    def _check_property_expression(self, value: str) -> str:
        return _check_non_empty(type(self), value)


class SizeConstraint(ListConstraint):
    """Size requirements for a list.

    Specify ``exact_size``, or ``min_size`` and/or ``max_size``. Use
    ``validate_size_constraint`` to enforce that at least one is set and
    that ``min_size <= max_size``.

    Attributes
    ----------
    min_size : int | None
        Minimum list size.
    max_size : int | None
        Maximum list size.
    exact_size : int | None
        Exact required size; mutually exclusive with ``min_size`` /
        ``max_size``.
    priority : int
        Constraint priority.
    """

    constraint_type: typing.Literal["size"]
    min_size: int | None = None
    max_size: int | None = None
    exact_size: int | None = None
    priority: int = 1


def validate_size_constraint(constraint: SizeConstraint) -> None:
    """Raise ``ValueError`` if *constraint*'s size parameters are inconsistent."""
    if (
        constraint.exact_size is None
        and constraint.min_size is None
        and constraint.max_size is None
    ):
        raise ValueError(
            "Must specify at least one of: min_size, max_size, exact_size"
        )
    if constraint.exact_size is not None and (
        constraint.min_size is not None or constraint.max_size is not None
    ):
        raise ValueError("exact_size cannot be used with min_size or max_size")
    if (
        constraint.min_size is not None
        and constraint.max_size is not None
        and constraint.min_size > constraint.max_size
    ):
        raise ValueError("min_size must be <= max_size")


class OrderingPair(BeadBaseModel):
    """Precedence relation between two items in a list.

    Attributes
    ----------
    before : UUID
        Item that must appear earlier in the list.
    after : UUID
        Item that must appear later in the list.
    """

    before: UUID
    after: UUID


class OrderingConstraint(ListConstraint):
    """Item presentation order requirements.

    Enforced primarily at jsPsych runtime; the Python model stores the
    specification.

    Attributes
    ----------
    precedence_pairs : tuple[OrderingPair, ...]
        Pairs ``(before, after)`` requiring ``before`` to precede ``after``.
    no_adjacent_property : str | None
        Property path; items sharing a value cannot be adjacent.
    block_by_property : str | None
        Property path used to group items into contiguous blocks.
    min_distance : int | None
        Minimum item separation between equal-property neighbours.
    max_distance : int | None
        Maximum span between start and end of a property block.
    practice_item_property : str | None
        Property identifying practice items, which precede main items.
    randomize_within_blocks : bool
        Randomize order within property blocks.
    priority : int
        Constraint priority (unused for static partitioning).
    """

    constraint_type: typing.Literal["ordering"]
    precedence_pairs: tuple[dx.Embed[OrderingPair], ...] = ()
    no_adjacent_property: str | None = None
    block_by_property: str | None = None
    min_distance: int | None = None
    max_distance: int | None = None
    practice_item_property: str | None = None
    randomize_within_blocks: bool = True
    priority: int = 1


def validate_ordering_constraint(constraint: OrderingConstraint) -> None:
    """Raise ``ValueError`` if *constraint*'s distance options are inconsistent."""
    if (
        constraint.min_distance is not None
        and constraint.no_adjacent_property is None
    ):
        raise ValueError("min_distance requires no_adjacent_property to be set")
    if (
        constraint.max_distance is not None
        and constraint.block_by_property is None
    ):
        raise ValueError("max_distance requires block_by_property to be set")
    if (
        constraint.min_distance is not None
        and constraint.max_distance is not None
        and constraint.min_distance > constraint.max_distance
    ):
        raise ValueError("min_distance cannot be greater than max_distance")


# ---------------------------------------------------------------------------
# batch-level constraints
# ---------------------------------------------------------------------------


class BatchCoverageConstraint(BatchConstraint):
    """All values of *property_expression* appear somewhere in the batch.

    Attributes
    ----------
    property_expression : str
        DSL expression returning the property value to cover.
    context : dict[str, ContextValue]
        Extra DSL evaluation context.
    target_values : tuple[str | int | float, ...] | None
        Values that must be covered. ``None`` uses every observed value.
    min_coverage : float
        Minimum fraction of target values that must appear (0.0-1.0).
    priority : int
        Constraint priority.
    """

    constraint_type: typing.Literal["coverage"]
    property_expression: str
    context: dict[str, ContextValue] = dx.field(default_factory=dict)
    target_values: tuple[str | int | float, ...] | None = None
    min_coverage: float = 1.0
    priority: int = 1

    @dx.validates("property_expression")
    def _check_property_expression(self, value: str) -> str:
        return _check_non_empty(type(self), value)


class BatchBalanceConstraint(BatchConstraint):
    """Balanced distribution of a categorical property across the entire batch.

    Attributes
    ----------
    property_expression : str
        DSL expression returning the category value.
    target_distribution : dict[str, float]
        Target proportions per category (values sum to ~1.0).
    context : dict[str, ContextValue]
        Extra DSL evaluation context.
    tolerance : float
        Allowed deviation from target.
    priority : int
        Constraint priority.
    """

    constraint_type: typing.Literal["balance"]
    property_expression: str
    target_distribution: dict[str, float]
    context: dict[str, ContextValue] = dx.field(default_factory=dict)
    tolerance: float = 0.1
    priority: int = 1

    @dx.validates("property_expression")
    def _check_property_expression(self, value: str) -> str:
        return _check_non_empty(type(self), value)

    @dx.validates("target_distribution")
    def _check_distribution(
        self, value: dict[str, float]
    ) -> dict[str, float]:
        if not value:
            raise ValueError("target_distribution must not be empty")
        for category, prob in value.items():
            if not 0.0 <= prob <= 1.0:
                raise ValueError(
                    f"target_distribution values must be in [0, 1], "
                    f"got {prob} for '{category}'"
                )
        total = sum(value.values())
        if not 0.99 <= total <= 1.01:
            raise ValueError(
                f"target_distribution values must sum to ~1.0, got {total}"
            )
        return value


class BatchDiversityConstraint(BatchConstraint):
    """No single value appears in too many lists.

    Attributes
    ----------
    property_expression : str
        DSL expression returning the property value.
    max_lists_per_value : int
        Maximum lists any value may appear in (>= 1).
    context : dict[str, ContextValue]
        Extra DSL evaluation context.
    priority : int
        Constraint priority.
    """

    constraint_type: typing.Literal["diversity"]
    property_expression: str
    max_lists_per_value: int
    context: dict[str, ContextValue] = dx.field(default_factory=dict)
    priority: int = 1

    @dx.validates("property_expression")
    def _check_property_expression(self, value: str) -> str:
        return _check_non_empty(type(self), value)


class BatchMinOccurrenceConstraint(BatchConstraint):
    """Each value of *property_expression* appears at least *min_occurrences* times.

    Attributes
    ----------
    property_expression : str
        DSL expression returning the property value.
    min_occurrences : int
        Minimum total occurrences across all lists (>= 1).
    context : dict[str, ContextValue]
        Extra DSL evaluation context.
    priority : int
        Constraint priority.
    """

    constraint_type: typing.Literal["min_occurrence"]
    property_expression: str
    min_occurrences: int
    context: dict[str, ContextValue] = dx.field(default_factory=dict)
    priority: int = 1

    @dx.validates("property_expression")
    def _check_property_expression(self, value: str) -> str:
        return _check_non_empty(type(self), value)


# Public aliases preserved for callers that previously imported the union types.
type ListConstraintUnion = (
    UniquenessConstraint
    | ConditionalUniquenessConstraint
    | BalanceConstraint
    | QuantileConstraint
    | GroupedQuantileConstraint
    | DiversityConstraint
    | SizeConstraint
    | OrderingConstraint
)

type BatchConstraintUnion = (
    BatchCoverageConstraint
    | BatchBalanceConstraint
    | BatchDiversityConstraint
    | BatchMinOccurrenceConstraint
)
