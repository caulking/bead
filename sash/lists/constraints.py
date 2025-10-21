"""Constraint models for experimental list composition.

This module defines constraints that can be applied to experimental lists
to ensure balanced, well-distributed item selections. Constraints can specify:
- Uniqueness: No duplicate property values
- Balance: Balanced distribution across categories
- Quantile: Uniform distribution across quantiles
- Size: List size requirements
- Ordering: Item presentation order constraints (runtime enforcement)

All constraints inherit from SashBaseModel and use Pydantic discriminated unions
for type-safe deserialization.
"""

from __future__ import annotations

from typing import Annotated, Literal
from uuid import UUID

from pydantic import Field, field_validator, model_validator

from sash.data.base import SashBaseModel

# Type alias for list constraint types
ListConstraintType = Literal[
    "uniqueness",  # No duplicate property values
    "balance",  # Balanced distribution of property
    "quantile",  # Uniform across quantiles
    "size",  # List size constraints
    "ordering",  # Presentation order constraints (runtime enforcement)
]


class UniquenessConstraint(SashBaseModel):
    """Constraint requiring unique values for a property.

    Ensures that no two items in a list have the same value for the
    specified property. Useful for preventing duplicate target verbs,
    sentence structures, or other experimental materials.

    Attributes
    ----------
    constraint_type : Literal["uniqueness"]
        Discriminator field for constraint type (always "uniqueness").
    property_path : str
        Dot-notation path to property that must be unique
        (e.g., "item_metadata.target_verb", "rendered_elements.sentence").
    allow_null : bool, default=False
        Whether to allow null/None values. If False, None values count
        as duplicates. If True, multiple None values are allowed.

    Examples
    --------
    >>> # No two items with same target verb
    >>> constraint = UniquenessConstraint(
    ...     property_path="item_metadata.target_verb",
    ...     allow_null=False
    ... )
    >>> constraint.property_path
    'item_metadata.target_verb'
    """

    constraint_type: Literal["uniqueness"] = "uniqueness"
    property_path: str = Field(..., description="Property path that must be unique")
    allow_null: bool = Field(
        default=False, description="Whether to allow multiple null values"
    )

    @field_validator("property_path")
    @classmethod
    def validate_property_path(cls, v: str) -> str:
        """Validate property path is non-empty.

        Parameters
        ----------
        v : str
            Property path to validate.

        Returns
        -------
        str
            Validated property path.

        Raises
        ------
        ValueError
            If property path is empty or contains only whitespace.
        """
        if not v or not v.strip():
            raise ValueError("property_path must be non-empty")
        return v.strip()


class BalanceConstraint(SashBaseModel):
    """Constraint requiring balanced distribution.

    Ensures balanced distribution of a categorical property across items
    in a list. Can specify target counts for each category or request
    equal distribution.

    Attributes
    ----------
    constraint_type : Literal["balance"]
        Discriminator field for constraint type (always "balance").
    property_path : str
        Dot-notation path to property to balance.
    target_counts : dict[str, int] | None, default=None
        Target counts for each category value. If None, equal distribution
        is assumed. Keys are category values, values are target counts.
    tolerance : float, default=0.1
        Allowed deviation from target as a proportion (0.0-1.0).
        For example, 0.1 means up to 10% deviation is acceptable.

    Examples
    --------
    >>> # Equal number of transitive and intransitive verbs
    >>> constraint = BalanceConstraint(
    ...     property_path="item_metadata.transitivity",
    ...     tolerance=0.1
    ... )
    >>> # 2:1 ratio of grammatical to ungrammatical
    >>> constraint2 = BalanceConstraint(
    ...     property_path="item_metadata.grammatical",
    ...     target_counts={"true": 20, "false": 10},
    ...     tolerance=0.05
    ... )
    """

    constraint_type: Literal["balance"] = "balance"
    property_path: str = Field(..., description="Property path to balance")
    target_counts: dict[str, int] | None = Field(
        default=None, description="Target counts per category (None = equal)"
    )
    tolerance: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Allowed deviation from target"
    )

    @field_validator("property_path")
    @classmethod
    def validate_property_path(cls, v: str) -> str:
        """Validate property path is non-empty.

        Parameters
        ----------
        v : str
            Property path to validate.

        Returns
        -------
        str
            Validated property path.

        Raises
        ------
        ValueError
            If property path is empty or contains only whitespace.
        """
        if not v or not v.strip():
            raise ValueError("property_path must be non-empty")
        return v.strip()

    @field_validator("target_counts")
    @classmethod
    def validate_target_counts(cls, v: dict[str, int] | None) -> dict[str, int] | None:
        """Validate target counts are non-negative.

        Parameters
        ----------
        v : dict[str, int] | None
            Target counts to validate.

        Returns
        -------
        dict[str, int] | None
            Validated target counts.

        Raises
        ------
        ValueError
            If any count is negative.
        """
        if v is not None:
            for category, count in v.items():
                if count < 0:
                    raise ValueError(
                        f"target_counts values must be non-negative, "
                        f"got {count} for '{category}'"
                    )
        return v


class QuantileConstraint(SashBaseModel):
    """Constraint requiring uniform distribution across quantiles.

    Ensures uniform distribution of items across quantiles of a numeric
    property. Useful for balancing language model probabilities, word
    frequencies, or other continuous variables.

    Attributes
    ----------
    constraint_type : Literal["quantile"]
        Discriminator field for constraint type (always "quantile").
    property_path : str
        Dot-notation path to numeric property.
    n_quantiles : int, default=5
        Number of quantiles to create (must be >= 2).
    items_per_quantile : int, default=2
        Target number of items per quantile (must be >= 1).

    Examples
    --------
    >>> # Uniform distribution of LM probabilities across 5 quantiles
    >>> constraint = QuantileConstraint(
    ...     property_path="item_metadata.lm_prob",
    ...     n_quantiles=5,
    ...     items_per_quantile=2
    ... )
    >>> # 10 deciles of word frequency
    >>> constraint2 = QuantileConstraint(
    ...     property_path="item_metadata.frequency",
    ...     n_quantiles=10,
    ...     items_per_quantile=3
    ... )
    """

    constraint_type: Literal["quantile"] = "quantile"
    property_path: str = Field(..., description="Property path to numeric property")
    n_quantiles: int = Field(default=5, ge=2, description="Number of quantiles")
    items_per_quantile: int = Field(default=2, ge=1, description="Items per quantile")

    @field_validator("property_path")
    @classmethod
    def validate_property_path(cls, v: str) -> str:
        """Validate property path is non-empty.

        Parameters
        ----------
        v : str
            Property path to validate.

        Returns
        -------
        str
            Validated property path.

        Raises
        ------
        ValueError
            If property path is empty or contains only whitespace.
        """
        if not v or not v.strip():
            raise ValueError("property_path must be non-empty")
        return v.strip()


class SizeConstraint(SashBaseModel):
    """Constraint on list size.

    Specifies size requirements for a list. Can specify exact size,
    minimum size, maximum size, or a range (min and max).

    Attributes
    ----------
    constraint_type : Literal["size"]
        Discriminator field for constraint type (always "size").
    min_size : int | None, default=None
        Minimum list size (must be >= 0 if set).
    max_size : int | None, default=None
        Maximum list size (must be >= 0 if set).
    exact_size : int | None, default=None
        Exact required size (must be >= 0 if set).
        Cannot be used with min_size or max_size.

    Examples
    --------
    >>> # Exactly 40 items per list
    >>> constraint = SizeConstraint(exact_size=40)
    >>> # Between 30-50 items per list
    >>> constraint2 = SizeConstraint(min_size=30, max_size=50)
    >>> # At least 20 items
    >>> constraint3 = SizeConstraint(min_size=20)
    >>> # At most 100 items
    >>> constraint4 = SizeConstraint(max_size=100)
    """

    constraint_type: Literal["size"] = "size"
    min_size: int | None = Field(default=None, ge=0, description="Minimum list size")
    max_size: int | None = Field(default=None, ge=0, description="Maximum list size")
    exact_size: int | None = Field(
        default=None, ge=0, description="Exact required size"
    )

    @model_validator(mode="after")
    def validate_size_params(self) -> SizeConstraint:
        """Validate size parameter combinations.

        Ensures that:
        - At least one size parameter is set
        - exact_size is not used with min_size or max_size
        - min_size <= max_size if both are set

        Returns
        -------
        SizeConstraint
            Validated constraint.

        Raises
        ------
        ValueError
            If validation fails.
        """
        # Check that at least one parameter is set
        if self.exact_size is None and self.min_size is None and self.max_size is None:
            raise ValueError(
                "Must specify at least one of: min_size, max_size, exact_size"
            )

        # Check that exact_size is not used with min/max
        if self.exact_size is not None:
            if self.min_size is not None or self.max_size is not None:
                raise ValueError("exact_size cannot be used with min_size or max_size")

        # Check that min <= max if both are set
        if self.min_size is not None and self.max_size is not None:
            if self.min_size > self.max_size:
                raise ValueError("min_size must be <= max_size")

        return self


class OrderingConstraint(SashBaseModel):
    """Constraint on item presentation order.

    **CRITICAL**: This constraint is primarily enforced at **jsPsych runtime**,
    not during static list construction. The Python data model stores the
    constraint specification, which is then translated to JavaScript code
    for runtime enforcement during per-participant randomization.

    Attributes
    ----------
    constraint_type : Literal["ordering"]
        Discriminator for constraint type.
    precedence_pairs : list[tuple[UUID, UUID]]
        Pairs of (item_a_id, item_b_id) where item_a must appear before item_b.
    no_adjacent_property : str | None
        Property path; items with same value cannot be adjacent.
        Example: "item_metadata.condition" prevents AA, BB patterns.
    block_by_property : str | None
        Property path to group items into contiguous blocks.
        Example: "item_metadata.block_type" creates blocked design.
    min_distance : int | None
        Minimum number of items between items with same no_adjacent_property value.
    max_distance : int | None
        Maximum number of items between start and end of items with same
        block_by_property value (enforces tight blocking).
    practice_item_property : str | None
        Property path identifying practice items (should appear first).
        Example: "item_metadata.is_practice" with value True.
    randomize_within_blocks : bool
        Whether to randomize order within blocks (default True).
        Only applies when block_by_property is set.

    Examples
    --------
    >>> # No adjacent items with same condition
    >>> constraint = OrderingConstraint(
    ...     no_adjacent_property="item_metadata.condition"
    ... )

    >>> # Practice items first, then main items
    >>> constraint = OrderingConstraint(
    ...     practice_item_property="item_metadata.is_practice"
    ... )

    >>> # Blocked by condition, randomized within blocks
    >>> constraint = OrderingConstraint(
    ...     block_by_property="item_metadata.condition",
    ...     randomize_within_blocks=True
    ... )

    >>> # Item A before Item B
    >>> from uuid import uuid4
    >>> item_a, item_b = uuid4(), uuid4()
    >>> constraint = OrderingConstraint(
    ...     precedence_pairs=[(item_a, item_b)]
    ... )
    """

    constraint_type: Literal["ordering"] = "ordering"
    precedence_pairs: list[tuple[UUID, UUID]] = Field(
        default_factory=lambda: [], description="Pairs (a,b) where a must precede b"
    )
    no_adjacent_property: str | None = Field(
        default=None,
        description="Property that cannot have same value in adjacent items",
    )
    block_by_property: str | None = Field(
        default=None, description="Property to group into contiguous blocks"
    )
    min_distance: int | None = Field(
        default=None,
        ge=1,
        description="Minimum items between same no_adjacent_property values",
    )
    max_distance: int | None = Field(
        default=None, ge=1, description="Maximum distance for blocked items"
    )
    practice_item_property: str | None = Field(
        default=None, description="Property identifying practice items (shown first)"
    )
    randomize_within_blocks: bool = Field(
        default=True, description="Whether to randomize within blocks"
    )

    @model_validator(mode="after")
    def validate_distance_constraints(self) -> OrderingConstraint:
        """Validate distance constraint combinations.

        Returns
        -------
        OrderingConstraint
            Validated constraint.

        Raises
        ------
        ValueError
            If validation fails.
        """
        if self.min_distance is not None and self.no_adjacent_property is None:
            raise ValueError("min_distance requires no_adjacent_property to be set")
        if self.max_distance is not None and self.block_by_property is None:
            raise ValueError("max_distance requires block_by_property to be set")
        if (
            self.min_distance
            and self.max_distance
            and self.min_distance > self.max_distance
        ):
            raise ValueError("min_distance cannot be greater than max_distance")
        return self


# Discriminated union for all list constraints
ListConstraint = Annotated[
    UniquenessConstraint
    | BalanceConstraint
    | QuantileConstraint
    | SizeConstraint
    | OrderingConstraint,
    Field(discriminator="constraint_type"),
]
