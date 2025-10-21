"""List partitioning for experimental item distribution.

This module provides the ListPartitioner class for partitioning items into
balanced experimental lists. Implements three strategies: random, balanced,
and stratified. Uses stand-off annotation (works with UUIDs only).
"""

from __future__ import annotations

from collections import Counter
from typing import Any
from uuid import UUID

import numpy as np

from sash.lists.balancer import QuantileBalancer
from sash.lists.constraints import (
    BalanceConstraint,
    ListConstraint,
    QuantileConstraint,
    SizeConstraint,
    UniquenessConstraint,
)
from sash.lists.models import ExperimentList

# Type aliases for clarity
type ItemMetadata = dict[str, Any]  # Arbitrary item properties
type MetadataDict = dict[UUID, ItemMetadata]  # Metadata indexed by UUID
type BalanceMetrics = dict[str, int | float | list[float] | dict[str, int]]


class ListPartitioner:
    """Partitions items into balanced experimental lists.

    Uses stand-off annotation: only stores UUIDs, not full item objects.
    Requires item metadata dict for constraint checking and balancing.

    Implements three partitioning strategies:
    - Random: Simple round-robin after shuffling
    - Balanced: Greedy algorithm to minimize constraint violations
    - Stratified: Quantile-based stratification with balanced distribution

    Parameters
    ----------
    random_seed : int | None, default=None
        Random seed for reproducibility.

    Attributes
    ----------
    random_seed : int | None
        Random seed for reproducibility.

    Examples
    --------
    >>> from uuid import uuid4
    >>> partitioner = ListPartitioner(random_seed=42)
    >>> items = [uuid4() for _ in range(100)]
    >>> metadata = {uid: {"property": i} for i, uid in enumerate(items)}
    >>> lists = partitioner.partition(items, n_lists=5, metadata=metadata)
    >>> len(lists)
    5
    """

    def __init__(self, random_seed: int | None = None) -> None:
        self.random_seed = random_seed
        self._rng = np.random.default_rng(random_seed)

    def partition(
        self,
        items: list[UUID],
        n_lists: int,
        constraints: list[ListConstraint] | None = None,
        strategy: str = "balanced",
        metadata: MetadataDict | None = None,
    ) -> list[ExperimentList]:
        """Partition items into lists.

        Parameters
        ----------
        items : list[UUID]
            Item UUIDs to partition.
        n_lists : int
            Number of lists to create.
        constraints : list[ListConstraint] | None, default=None
            Constraints to satisfy.
        strategy : str, default="balanced"
            Partitioning strategy ("balanced", "random", "stratified").
        metadata : dict[UUID, dict[str, Any]] | None, default=None
            Metadata for each item UUID. Required for constraint checking.

        Returns
        -------
        list[ExperimentList]
            The partitioned lists.

        Raises
        ------
        ValueError
            If strategy is unknown or n_lists < 1.
        """
        if n_lists < 1:
            raise ValueError(f"n_lists must be >= 1, got {n_lists}")

        constraints = constraints or []
        metadata = metadata or {}

        # Select partitioning method based on strategy
        match strategy:
            case "balanced":
                return self._partition_balanced(items, n_lists, constraints, metadata)
            case "random":
                return self._partition_random(items, n_lists, constraints, metadata)
            case "stratified":
                return self._partition_stratified(items, n_lists, constraints, metadata)
            case _:
                raise ValueError(f"Unknown strategy: {strategy}")

    def _partition_random(
        self,
        items: list[UUID],
        n_lists: int,
        constraints: list[ListConstraint],
        metadata: MetadataDict,
    ) -> list[ExperimentList]:
        """Partition items randomly.

        Parameters
        ----------
        items : list[UUID]
            Items to partition.
        n_lists : int
            Number of lists.
        constraints : list[ListConstraint]
            Constraints to attach to lists.
        metadata : dict[UUID, dict[str, Any]]
            Item metadata.

        Returns
        -------
        list[ExperimentList]
            Partitioned lists.
        """
        # Initialize lists
        lists = [
            ExperimentList(
                name=f"list_{i}",
                list_number=i,
                list_constraints=constraints,
            )
            for i in range(n_lists)
        ]

        # Shuffle and distribute round-robin
        items_shuffled = np.array(items)
        self._rng.shuffle(items_shuffled)

        for i, item_id in enumerate(items_shuffled):
            list_idx = i % n_lists
            lists[list_idx].add_item(item_id)

        # Compute balance metrics for each list
        for exp_list in lists:
            exp_list.balance_metrics = self._compute_balance_metrics(
                exp_list, constraints, metadata
            )

        return lists

    def _partition_balanced(
        self,
        items: list[UUID],
        n_lists: int,
        constraints: list[ListConstraint],
        metadata: MetadataDict,
    ) -> list[ExperimentList]:
        """Partition items with balanced distribution.

        Uses greedy algorithm to distribute items to minimize imbalance.

        Parameters
        ----------
        items : list[UUID]
            Items to partition.
        n_lists : int
            Number of lists.
        constraints : list[ListConstraint]
            Constraints to satisfy.
        metadata : dict[UUID, dict[str, Any]]
            Item metadata.

        Returns
        -------
        list[ExperimentList]
            Partitioned lists.
        """
        # Initialize lists
        lists = [
            ExperimentList(
                name=f"list_{i}",
                list_number=i,
                list_constraints=constraints,
            )
            for i in range(n_lists)
        ]

        # Shuffle items
        items_shuffled = np.array(items)
        self._rng.shuffle(items_shuffled)

        # For each item, assign to list that best maintains balance
        for item_id in items_shuffled:
            best_list = self._find_best_list(item_id, lists, constraints, metadata)
            best_list.add_item(item_id)

        # Compute balance metrics for each list
        for exp_list in lists:
            exp_list.balance_metrics = self._compute_balance_metrics(
                exp_list, constraints, metadata
            )

        return lists

    def _partition_stratified(
        self,
        items: list[UUID],
        n_lists: int,
        constraints: list[ListConstraint],
        metadata: MetadataDict,
    ) -> list[ExperimentList]:
        """Partition items with stratification.

        Creates strata based on quantile constraints and distributes
        items from each stratum across lists.

        Parameters
        ----------
        items : list[UUID]
            Items to partition.
        n_lists : int
            Number of lists.
        constraints : list[ListConstraint]
            Constraints to satisfy (must include quantile constraints).
        metadata : dict[UUID, dict[str, Any]]
            Item metadata.

        Returns
        -------
        list[ExperimentList]
            Partitioned lists.
        """
        # Find quantile constraints
        quantile_constraints = [
            c for c in constraints if isinstance(c, QuantileConstraint)
        ]

        if not quantile_constraints:
            # Fall back to balanced
            return self._partition_balanced(items, n_lists, constraints, metadata)

        # Use first quantile constraint for stratification
        qc = quantile_constraints[0]

        # Create balancer
        balancer = QuantileBalancer(
            n_quantiles=qc.n_quantiles, random_seed=self.random_seed
        )

        # Create value function
        def value_func(item_id: UUID) -> float:
            return float(self._get_property_value(item_id, qc.property_path, metadata))

        # Balance items across lists
        balanced_lists = balancer.balance(
            items, value_func, n_lists, qc.items_per_quantile
        )

        # Convert to ExperimentList objects
        lists: list[ExperimentList] = []
        for i, item_ids in enumerate(balanced_lists):
            exp_list = ExperimentList(
                name=f"list_{i}",
                list_number=i,
                list_constraints=constraints,
            )
            for item_id in item_ids:
                exp_list.add_item(item_id)

            exp_list.balance_metrics = self._compute_balance_metrics(
                exp_list, constraints, metadata
            )
            lists.append(exp_list)

        return lists

    def _find_best_list(
        self,
        item_id: UUID,
        lists: list[ExperimentList],
        constraints: list[ListConstraint],
        metadata: MetadataDict,
    ) -> ExperimentList:
        """Find the list that best maintains balance after adding item.

        Parameters
        ----------
        item_id : UUID
            Item to add.
        lists : list[ExperimentList]
            Available lists.
        constraints : list[ListConstraint]
            Constraints to consider.
        metadata : MetadataDict
            Item metadata.

        Returns
        -------
        ExperimentList
            Best list for this item.
        """
        # Compute score for each list (violations + size as tie-breaker)
        scores: list[tuple[int, int]] = []
        for exp_list in lists:
            # Temporarily add item
            exp_list.add_item(item_id)

            # Compute constraint violations
            violations = self._count_violations(exp_list, constraints, metadata)

            # Remove item
            exp_list.remove_item(item_id)

            # Use (violations, current_size) as score
            # Prefer lists with fewer violations, then smaller lists
            scores.append((violations, len(exp_list.item_refs)))

        # Return list with lowest score
        best_idx = int(np.argmin([s[0] * 1000 + s[1] for s in scores]))
        return lists[best_idx]

    def _count_violations(
        self,
        exp_list: ExperimentList,
        constraints: list[ListConstraint],
        metadata: MetadataDict,
    ) -> int:
        """Count constraint violations for a list.

        Violations are weighted by constraint priority. Higher priority
        constraints contribute more to the total violation score.

        Parameters
        ----------
        exp_list : ExperimentList
            The list to check.
        constraints : list[ListConstraint]
            Constraints to check.
        metadata : MetadataDict
            Item metadata.

        Returns
        -------
        int
            Weighted violation score (sum of priorities of violated constraints).
        """
        violations = 0

        for constraint in constraints:
            is_violated = False

            if isinstance(constraint, UniquenessConstraint):
                if not self._check_uniqueness(exp_list, constraint, metadata):
                    is_violated = True
            elif isinstance(constraint, BalanceConstraint):
                if not self._check_balance(exp_list, constraint, metadata):
                    is_violated = True
            elif isinstance(constraint, SizeConstraint):
                if not self._check_size(exp_list, constraint):
                    is_violated = True

            # Add constraint priority if violated
            if is_violated:
                violations += constraint.priority

        return violations

    def _check_uniqueness(
        self,
        exp_list: ExperimentList,
        constraint: UniquenessConstraint,
        metadata: MetadataDict,
    ) -> bool:
        """Check uniqueness constraint.

        Parameters
        ----------
        exp_list : ExperimentList
            List to check.
        constraint : UniquenessConstraint
            Uniqueness constraint.
        metadata : dict[UUID, dict[str, Any]]
            Item metadata.

        Returns
        -------
        bool
            True if constraint is satisfied.
        """
        # Get values for property
        values: list[Any] = []
        for item_id in exp_list.item_refs:
            value = self._get_property_value(
                item_id, constraint.property_path, metadata
            )
            values.append(value)

        # Check for duplicates
        if constraint.allow_null:
            values = [v for v in values if v is not None]

        return bool(len(values) == len(set(values)))

    def _check_balance(
        self,
        exp_list: ExperimentList,
        constraint: BalanceConstraint,
        metadata: MetadataDict,
    ) -> bool:
        """Check balance constraint.

        Parameters
        ----------
        exp_list : ExperimentList
            List to check.
        constraint : BalanceConstraint
            Balance constraint.
        metadata : dict[UUID, dict[str, Any]]
            Item metadata.

        Returns
        -------
        bool
            True if constraint is satisfied.
        """
        # Get values for property
        values: list[Any] = []
        for item_id in exp_list.item_refs:
            value = self._get_property_value(
                item_id, constraint.property_path, metadata
            )
            values.append(value)

        # Count occurrences
        counts = Counter(values)

        # Check against target counts if specified
        if constraint.target_counts is not None:
            for category, target_count in constraint.target_counts.items():
                actual_count = counts.get(category, 0)
                deviation = abs(actual_count - target_count) / max(target_count, 1)
                if deviation > constraint.tolerance:
                    return False
            return True

        # Otherwise check for balanced distribution
        if len(counts) == 0:
            return True

        count_values = list(counts.values())
        mean_count = np.mean(count_values)
        max_deviation = max(
            abs(c - mean_count) / max(mean_count, 1) for c in count_values
        )

        return bool(max_deviation <= constraint.tolerance)

    def _check_size(self, exp_list: ExperimentList, constraint: SizeConstraint) -> bool:
        """Check size constraint.

        Parameters
        ----------
        exp_list : ExperimentList
            List to check.
        constraint : SizeConstraint
            Size constraint.

        Returns
        -------
        bool
            True if constraint is satisfied.
        """
        size = len(exp_list.item_refs)

        if constraint.exact_size is not None:
            return size == constraint.exact_size

        if constraint.min_size is not None and size < constraint.min_size:
            return False

        if constraint.max_size is not None and size > constraint.max_size:
            return False

        return True

    def _get_property_value(
        self,
        item_id: UUID,
        property_path: str,
        metadata: MetadataDict,
    ) -> Any:
        """Extract property value using dot notation.

        Parameters
        ----------
        item_id : UUID
            Item UUID.
        property_path : str
            Dot-notation path (e.g., "item_metadata.lm_prob").
        metadata : dict[UUID, dict[str, Any]]
            Metadata dict.

        Returns
        -------
        Any
            Property value.

        Raises
        ------
        KeyError
            If item_id not in metadata or property path not found.
        """
        if item_id not in metadata:
            raise KeyError(f"Item {item_id} not found in metadata")

        parts = property_path.split(".")
        value: Any = metadata[item_id]

        for part in parts:
            if isinstance(value, dict):
                value = value[part]  # type: ignore[assignment]
            else:
                value = getattr(value, part)  # type: ignore[assignment]

        return value  # type: ignore[return-value]

    def _compute_balance_metrics(
        self,
        exp_list: ExperimentList,
        constraints: list[ListConstraint],
        metadata: MetadataDict,
    ) -> BalanceMetrics:
        """Compute balance metrics for a list.

        Parameters
        ----------
        exp_list : ExperimentList
            The list.
        constraints : list[ListConstraint]
            Constraints to compute metrics for.
        metadata : dict[UUID, dict[str, Any]]
            Item metadata.

        Returns
        -------
        dict[str, Any]
            Balance metrics.
        """
        metrics: dict[str, Any] = {}

        # Compute metrics for each constraint
        for constraint in constraints:
            if isinstance(constraint, QuantileConstraint):
                metrics[f"quantile_{constraint.property_path}"] = (
                    self._compute_quantile_distribution(exp_list, constraint, metadata)
                )
            elif isinstance(constraint, BalanceConstraint):
                metrics[f"balance_{constraint.property_path}"] = (
                    self._compute_category_distribution(exp_list, constraint, metadata)
                )

        # Overall size
        metrics["size"] = len(exp_list.item_refs)

        return metrics

    def _compute_quantile_distribution(
        self,
        exp_list: ExperimentList,
        constraint: QuantileConstraint,
        metadata: MetadataDict,
    ) -> dict[str, float | list[float]]:
        """Compute distribution across quantiles.

        Parameters
        ----------
        exp_list : ExperimentList
            The list.
        constraint : QuantileConstraint
            Quantile constraint.
        metadata : dict[UUID, dict[str, Any]]
            Item metadata.

        Returns
        -------
        dict[str, Any]
            Distribution metrics.
        """
        if not exp_list.item_refs:
            return {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "quantiles": [],
            }

        values = [
            float(self._get_property_value(item_id, constraint.property_path, metadata))
            for item_id in exp_list.item_refs
        ]

        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "quantiles": [float(q) for q in np.percentile(values, [25, 50, 75])],
        }

    def _compute_category_distribution(
        self,
        exp_list: ExperimentList,
        constraint: BalanceConstraint,
        metadata: MetadataDict,
    ) -> dict[str, dict[str, int] | int | tuple[Any, int] | None]:
        """Compute distribution across categories.

        Parameters
        ----------
        exp_list : ExperimentList
            The list.
        constraint : BalanceConstraint
            Balance constraint.
        metadata : dict[UUID, dict[str, Any]]
            Item metadata.

        Returns
        -------
        dict[str, Any]
            Distribution metrics.
        """
        if not exp_list.item_refs:
            return {"counts": {}, "n_categories": 0, "most_common": None}

        values = [
            self._get_property_value(item_id, constraint.property_path, metadata)
            for item_id in exp_list.item_refs
        ]
        counts = Counter(values)

        return {
            "counts": dict(counts),
            "n_categories": len(counts),
            "most_common": counts.most_common(1)[0] if counts else None,
        }
