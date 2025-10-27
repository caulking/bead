"""Filling strategies for template population."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Literal

from sash.data.language_codes import LanguageCode, validate_iso639_code
from sash.dsl.evaluator import DSLEvaluator
from sash.items.models import Item
from sash.resources.constraints import ContextValue
from sash.resources.lexicon import Lexicon
from sash.resources.models import LexicalItem
from sash.resources.structures import Slot, Template
from sash.templates.adapters import HuggingFaceMLMAdapter, ModelOutputCache
from sash.templates.combinatorics import cartesian_product
from sash.templates.filler import FilledTemplate, TemplateFiller
from sash.templates.resolver import ConstraintResolver


class FillingStrategy(ABC):
    """Abstract base class for template filling strategies.

    A filling strategy determines how to combine lexical items
    to fill template slots. Strategies differ in:
    - Selection criteria (all vs. sample)
    - Ordering (deterministic vs. random)
    - Grouping (balanced vs. unbalanced)

    Examples
    --------
    >>> strategy = ExhaustiveStrategy()
    >>> combinations = strategy.generate_combinations(slot_items)
    >>> len(list(combinations))
    12
    """

    @abstractmethod
    def generate_combinations(
        self,
        slot_items: dict[str, list[LexicalItem]],
    ) -> list[dict[str, LexicalItem]]:
        """Generate combinations of items for template slots.

        Parameters
        ----------
        slot_items : dict[str, list[LexicalItem]]
            Mapping of slot names to lists of valid items.

        Returns
        -------
        list[dict[str, LexicalItem]]
            List of slot-to-item mappings representing filled templates.

        Examples
        --------
        >>> slot_items = {
        ...     "subject": [item1, item2],
        ...     "verb": [item3, item4],
        ... }
        >>> combinations = strategy.generate_combinations(slot_items)
        >>> len(combinations)
        4
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Get strategy name for metadata.

        Returns
        -------
        str
            Strategy name.
        """
        pass


class ExhaustiveStrategy(FillingStrategy):
    """Generate all possible combinations of slot fillers.

    This strategy produces the complete Cartesian product of all
    valid items for each slot. Use for small combinatorial spaces.

    **Warning**: Combinatorial explosion! With N slots and M items
    per slot, generates M^N combinations.

    Examples
    --------
    >>> strategy = ExhaustiveStrategy()
    >>> slot_items = {"a": [1, 2], "b": [3, 4]}
    >>> combinations = strategy.generate_combinations(slot_items)
    >>> len(combinations)
    4
    >>> combinations[0]
    {"a": 1, "b": 3}
    """

    @property
    def name(self) -> str:
        """Get strategy name."""
        return "exhaustive"

    def generate_combinations(
        self,
        slot_items: dict[str, list[LexicalItem]],
    ) -> list[dict[str, LexicalItem]]:
        """Generate all combinations.

        Parameters
        ----------
        slot_items : dict[str, list[LexicalItem]]
            Mapping of slot names to valid items.

        Returns
        -------
        list[dict[str, LexicalItem]]
            All possible slot-to-item combinations.
        """
        if not slot_items:
            return []

        # Get ordered slot names and item lists
        slot_names = list(slot_items.keys())
        item_lists = [slot_items[name] for name in slot_names]

        # Generate all combinations
        combinations: list[dict[str, LexicalItem]] = []
        for combo_tuple in cartesian_product(*item_lists):
            combo_dict = dict(zip(slot_names, combo_tuple, strict=True))
            combinations.append(combo_dict)

        return combinations


class RandomStrategy(FillingStrategy):
    """Generate random sample of combinations.

    Sample combinations randomly with optional seeding for
    reproducibility. Use for large combinatorial spaces.

    Parameters
    ----------
    n_samples : int
        Number of combinations to generate.
    seed : int | None
        Random seed for reproducibility. Default: None.

    Examples
    --------
    >>> strategy = RandomStrategy(n_samples=10, seed=42)
    >>> combinations = strategy.generate_combinations(slot_items)
    >>> len(combinations)
    10
    """

    def __init__(self, n_samples: int, seed: int | None = None) -> None:
        """Initialize random strategy.

        Parameters
        ----------
        n_samples : int
            Number of combinations to generate.
        seed : int | None
            Random seed for reproducibility.
        """
        self.n_samples = n_samples
        self.seed = seed

    @property
    def name(self) -> str:
        """Get strategy name."""
        return "random"

    def generate_combinations(
        self,
        slot_items: dict[str, list[LexicalItem]],
    ) -> list[dict[str, LexicalItem]]:
        """Generate random combinations.

        Parameters
        ----------
        slot_items : dict[str, list[LexicalItem]]
            Mapping of slot names to valid items.

        Returns
        -------
        list[dict[str, LexicalItem]]
            Randomly sampled combinations.
        """
        if not slot_items:
            return []

        # Set random seed if provided
        if self.seed is not None:
            random.seed(self.seed)

        # Get ordered slot names and item lists
        slot_names = list(slot_items.keys())
        item_lists = [slot_items[name] for name in slot_names]

        # Generate random combinations
        combinations: list[dict[str, LexicalItem]] = []
        for _ in range(self.n_samples):
            combo_tuple = tuple(random.choice(items) for items in item_lists)
            combo_dict = dict(zip(slot_names, combo_tuple, strict=True))
            combinations.append(combo_dict)

        return combinations


class StratifiedStrategy(FillingStrategy):
    """Generate balanced sample across item groups.

    Ensure each group of items (e.g., by POS, features) is
    represented proportionally in the sample.

    Parameters
    ----------
    n_samples : int
        Total number of combinations to generate.
    grouping_property : str
        Property to group items by (e.g., "pos", "features.transitivity").
    seed : int | None
        Random seed for reproducibility. Default: None.

    Examples
    --------
    >>> strategy = StratifiedStrategy(
    ...     n_samples=20,
    ...     grouping_property="pos",
    ...     seed=42
    ... )
    >>> combinations = strategy.generate_combinations(slot_items)
    >>> # Ensures balanced representation of different POS values
    """

    def __init__(
        self,
        n_samples: int,
        grouping_property: str,
        seed: int | None = None,
    ) -> None:
        """Initialize stratified strategy.

        Parameters
        ----------
        n_samples : int
            Total number of combinations to generate.
        grouping_property : str
            Property to group items by.
        seed : int | None
            Random seed for reproducibility.
        """
        self.n_samples = n_samples
        self.grouping_property = grouping_property
        self.seed = seed

    @property
    def name(self) -> str:
        """Get strategy name."""
        return "stratified"

    def generate_combinations(
        self,
        slot_items: dict[str, list[LexicalItem]],
    ) -> list[dict[str, LexicalItem]]:
        """Generate stratified combinations.

        Parameters
        ----------
        slot_items : dict[str, list[LexicalItem]]
            Mapping of slot names to valid items.

        Returns
        -------
        list[dict[str, LexicalItem]]
            Balanced combinations across groups.
        """
        if not slot_items:
            return []

        # Set random seed if provided
        if self.seed is not None:
            random.seed(self.seed)

        # Group items by the specified property
        grouped_items: dict[str, dict[str, list[LexicalItem]]] = {}
        for slot_name, items in slot_items.items():
            slot_groups: dict[str, list[LexicalItem]] = {}
            for item in items:
                # Get property value (handle nested properties)
                value = self._get_property_value(item, self.grouping_property)
                if value not in slot_groups:
                    slot_groups[value] = []
                slot_groups[value].append(item)
            grouped_items[slot_name] = slot_groups

        # Sample proportionally from each group
        combinations: list[dict[str, LexicalItem]] = []
        slot_names = list(slot_items.keys())

        # Calculate samples per group
        # For simplicity, sample equally from all groups
        for _ in range(self.n_samples):
            combo_dict: dict[str, LexicalItem] = {}
            for slot_name in slot_names:
                slot_groups = grouped_items[slot_name]
                # Choose a random group, then a random item from that group
                if slot_groups:
                    group_key = random.choice(list(slot_groups.keys()))
                    item = random.choice(slot_groups[group_key])
                    combo_dict[slot_name] = item
            combinations.append(combo_dict)

        return combinations

    def _get_property_value(self, item: LexicalItem, property_path: str) -> str:
        """Get property value from item, handling nested properties.

        Parameters
        ----------
        item : LexicalItem
            Item to get property from.
        property_path : str
            Property path (e.g., "pos" or "features.transitivity").

        Returns
        -------
        str
            Property value as string.
        """
        parts = property_path.split(".")
        value = item
        for part in parts:
            if hasattr(value, part):
                value = getattr(value, part)
            else:
                return "unknown"

        # Convert to string for grouping
        if value is None:
            return "none"
        return str(value)


class MLMFillingStrategy(FillingStrategy):
    """Fill templates using masked language models with beam search.

    Uses pre-trained MLMs (BERT, RoBERTa, etc.) to propose linguistically
    plausible slot fillers. Supports beam search for multiple slots with
    configurable fill directions.

    Parameters
    ----------
    resolver : ConstraintResolver
        Constraint resolver for filtering candidates
    model_adapter : HuggingFaceMLMAdapter
        Loaded MLM adapter
    beam_size : int
        Beam search width (K best hypotheses)
    fill_direction : Literal
        Direction for filling slots. One of: "left_to_right", "right_to_left",
        "inside_out", "outside_in", "custom"
    custom_order : list[int] | None
        Custom slot fill order (slot indices)
    top_k : int
        Top-K candidates per slot from MLM
    cache : ModelOutputCache | None
        Cache for model predictions
    budget : int | None
        Maximum combinations to generate

    Examples
    --------
    >>> from sash.templates.adapters import HuggingFaceMLMAdapter, ModelOutputCache
    >>> adapter = HuggingFaceMLMAdapter("bert-base-uncased")
    >>> adapter.load_model()
    >>> cache = ModelOutputCache(Path("/tmp/cache"))
    >>> strategy = MLMFillingStrategy(
    ...     resolver=resolver,
    ...     model_adapter=adapter,
    ...     beam_size=5,
    ...     fill_direction="left_to_right",
    ...     cache=cache
    ... )
    >>> combinations = strategy.generate_combinations(slot_items)
    """

    def __init__(
        self,
        resolver: ConstraintResolver,
        model_adapter: HuggingFaceMLMAdapter,
        beam_size: int = 5,
        fill_direction: Literal[
            "left_to_right", "right_to_left", "inside_out", "outside_in", "custom"
        ] = "left_to_right",
        custom_order: list[int] | None = None,
        top_k: int = 20,
        cache: ModelOutputCache | None = None,
        budget: int | None = None,
    ) -> None:
        """Initialize MLM strategy.

        Parameters
        ----------
        resolver : ConstraintResolver
            Constraint resolver
        model_adapter : HuggingFaceMLMAdapter
            MLM adapter (must be loaded)
        beam_size : int
            Beam width
        fill_direction : str
            Fill direction
        custom_order : list[int] | None
            Custom fill order
        top_k : int
            Top-K from MLM
        cache : ModelOutputCache | None
            Prediction cache
        budget : int | None
            Max combinations
        """
        self.resolver = resolver
        self.model_adapter = model_adapter
        self.beam_size = beam_size
        self.fill_direction = fill_direction
        self.custom_order = custom_order
        self.top_k = top_k
        self.cache = cache
        self.budget = budget

        if not model_adapter.is_loaded():
            raise ValueError("Model adapter must be loaded before use")

        if fill_direction == "custom" and custom_order is None:
            raise ValueError("custom_order required when fill_direction is 'custom'")

    @property
    def name(self) -> str:
        """Get strategy name."""
        return "mlm"

    def generate_combinations(
        self,
        slot_items: dict[str, list[LexicalItem]],
    ) -> list[dict[str, LexicalItem]]:
        """Generate combinations using MLM beam search.

        Note: This method adapts slot_items to template-based generation.
        The actual beam search is implemented in generate_from_template.

        Parameters
        ----------
        slot_items : dict[str, list[LexicalItem]]
            Mapping of slot names to valid items (for constraint filtering)

        Returns
        -------
        list[dict[str, LexicalItem]]
            Combinations generated via beam search

        Raises
        ------
        NotImplementedError
            This method requires template context. Use generate_from_template instead.
        """
        raise NotImplementedError(
            "MLMFillingStrategy requires template context. "
            "Use TemplateFiller with MLMFillingStrategy, which calls "
            "generate_from_template internally."
        )

    def generate_from_template(
        self,
        template: Template,
        lexicons: list[Lexicon],
        language_code: LanguageCode | None = None,
    ) -> Iterator[dict[str, LexicalItem]]:
        """Generate combinations from template using beam search.

        Parameters
        ----------
        template : Template
            Template to fill
        lexicons : list[Lexicon]
            Lexicons for constraint resolution
        language_code : LanguageCode | None
            Language filter

        Yields
        ------
        dict[str, LexicalItem]
            Slot-to-item mappings
        """
        # Get slot names and order
        slot_names = list(template.slots.keys())
        if not slot_names:
            return

        fill_order = self._get_fill_order(len(slot_names))

        # Initialize beam with empty hypothesis
        # Each beam item: (filled_slots_dict, cumulative_log_prob)
        beam: list[tuple[dict[str, LexicalItem], float]] = [({}, 0.0)]

        # Fill slots in order
        for slot_idx in fill_order:
            slot_name = slot_names[slot_idx]
            slot = template.slots[slot_name]

            new_beam: list[tuple[dict[str, LexicalItem], float]] = []

            # Expand each hypothesis in current beam
            for filled_slots, cum_log_prob in beam:
                # Get MLM candidates for this slot
                candidates = self._get_mlm_candidates(
                    template,
                    slot_names,
                    slot_idx,
                    filled_slots,
                    slot,
                    lexicons,
                    language_code,
                )

                # Add each candidate to beam
                for item, log_prob in candidates:
                    new_filled = filled_slots.copy()
                    new_filled[slot_name] = item
                    new_log_prob = cum_log_prob + log_prob
                    new_beam.append((new_filled, new_log_prob))

            # Prune beam to top-K by score (length-normalized)
            if new_beam:
                # Length-normalize scores
                num_filled = len(new_beam[0][0])
                scored_beam = [
                    (filled, log_prob / num_filled, log_prob)
                    for filled, log_prob in new_beam
                ]
                scored_beam.sort(key=lambda x: x[1], reverse=True)

                # Keep top beam_size
                beam = [
                    (filled, cum_log_prob)
                    for filled, _, cum_log_prob in scored_beam[: self.beam_size]
                ]
            else:
                # No valid candidates - empty beam
                beam = []
                break

        # Yield final hypotheses
        count = 0
        for filled_slots, _ in beam:
            if self.budget and count >= self.budget:
                break
            yield filled_slots
            count += 1

    def _get_fill_order(self, num_slots: int) -> list[int]:
        """Get slot fill order based on fill_direction.

        Parameters
        ----------
        num_slots : int
            Number of slots

        Returns
        -------
        list[int]
            Slot indices in fill order
        """
        if self.fill_direction == "custom":
            if self.custom_order is None:
                raise ValueError("custom_order not set")
            return self.custom_order

        indices = list(range(num_slots))

        if self.fill_direction == "left_to_right":
            return indices
        elif self.fill_direction == "right_to_left":
            return list(reversed(indices))
        elif self.fill_direction == "inside_out":
            # Alternate from center outward
            mid = num_slots // 2
            order: list[int] = []
            for i in range(num_slots):
                if i % 2 == 0:
                    order.append(mid + i // 2)
                else:
                    order.append(mid - (i + 1) // 2)
            return [idx for idx in order if 0 <= idx < num_slots]
        elif self.fill_direction == "outside_in":
            # Alternate from edges inward
            order: list[int] = []
            left, right = 0, num_slots - 1
            while left <= right:
                order.append(left)
                if left != right:
                    order.append(right)
                left += 1
                right -= 1
            return order
        else:
            raise ValueError(f"Unknown fill_direction: {self.fill_direction}")

    def _get_mlm_candidates(
        self,
        template: Template,
        slot_names: list[str],
        slot_idx: int,
        filled_slots: dict[str, LexicalItem],
        slot: object,
        lexicons: list[Lexicon],
        language_code: LanguageCode | None,
    ) -> list[tuple[LexicalItem, float]]:
        """Get MLM candidates for a slot.

        Parameters
        ----------
        template : Template
            Template being filled
        slot_names : list[str]
            Ordered slot names
        slot_idx : int
            Index of slot to fill
        filled_slots : dict[str, LexicalItem]
            Already-filled slots
        slot : object
            Slot object
        lexicons : list[Lexicon]
            Lexicons for lookup
        language_code : LanguageCode | None
            Language filter

        Returns
        -------
        list[tuple[LexicalItem, float]]
            (item, log_prob) pairs
        """
        # Normalize language code to ISO 639-3
        if language_code is not None:
            language_code = validate_iso639_code(language_code)

        # Create masked text
        masked_text = self._create_masked_text(
            template, slot_names, filled_slots, slot_idx
        )

        # Get predictions from MLM (with cache)
        if self.cache:
            predictions = self.cache.get(
                self.model_adapter.model_name,
                masked_text,
                0,  # First mask position
                self.top_k,
            )
        else:
            predictions = None

        if predictions is None:
            predictions = self.model_adapter.predict_masked_token(
                masked_text,
                mask_position=0,
                top_k=self.top_k,
            )
            if self.cache:
                self.cache.set(
                    self.model_adapter.model_name,
                    masked_text,
                    0,
                    self.top_k,
                    predictions,
                )

        # Filter by constraints and find matching lexical items
        candidates: list[tuple[LexicalItem, float]] = []
        for token, log_prob in predictions:
            # Find matching items in lexicons
            for lexicon in lexicons:
                for item in lexicon.items.values():
                    # Match lemma and language
                    if item.lemma.lower() == token.lower():
                        if language_code is None or item.language_code == language_code:
                            # Check slot constraints
                            if isinstance(slot, Slot) and slot.constraints:
                                # Evaluate constraints using resolver
                                if self.resolver.evaluate_slot_constraints(
                                    item, slot.constraints
                                ):
                                    candidates.append((item, log_prob))
                            else:
                                candidates.append((item, log_prob))

        return candidates

    def _create_masked_text(
        self,
        template: Template,
        slot_names: list[str],
        filled_slots: dict[str, LexicalItem],
        current_slot_idx: int,
    ) -> str:
        """Create text with mask token for current slot.

        Parameters
        ----------
        template : Template
            Template
        slot_names : list[str]
            Slot names
        filled_slots : dict[str, LexicalItem]
            Filled slots
        current_slot_idx : int
            Current slot index

        Returns
        -------
        str
            Text with [MASK] token
        """
        mask_token = self.model_adapter.get_mask_token()
        text = template.template_string

        # Replace filled slots with lemmas
        for slot_name, item in filled_slots.items():
            placeholder = f"{{{slot_name}}}"
            text = text.replace(placeholder, item.lemma)

        # Replace current slot with mask
        current_slot_name = slot_names[current_slot_idx]
        current_placeholder = f"{{{current_slot_name}}}"
        text = text.replace(current_placeholder, mask_token)

        # Replace remaining unfilled slots with mask for context
        for slot_name in slot_names:
            placeholder = f"{{{slot_name}}}"
            if placeholder in text:
                text = text.replace(placeholder, mask_token)

        return text


class StrategyFiller(TemplateFiller):
    """Strategy-based template filling for simple templates.

    Fast filling using enumeration strategies (exhaustive, random, stratified).
    Does NOT handle template-level multi-slot constraints (Template.constraints).

    For templates with multi-slot constraints requiring agreement or
    relational checks, use CSPFiller instead.

    Parameters
    ----------
    lexicon : Lexicon
        Lexicon containing candidate items.
    strategy : FillingStrategy
        Strategy for generating combinations.

    Examples
    --------
    >>> from sash.templates.strategies import StrategyFiller, ExhaustiveStrategy
    >>> filler = StrategyFiller(lexicon, ExhaustiveStrategy())
    >>> filled = filler.fill(template)
    >>> len(filled)
    12
    """

    def __init__(self, lexicon: Lexicon, strategy: FillingStrategy) -> None:
        self.lexicon = lexicon
        self.strategy = strategy
        self.resolver = ConstraintResolver()

    def fill(
        self,
        template: Template,
        language_code: LanguageCode | None = None,
    ) -> list[FilledTemplate]:
        """Fill template with lexical items using strategy.

        Parameters
        ----------
        template : Template
            Template to fill.
        language_code : LanguageCode | None
            Optional language code to filter items.

        Returns
        -------
        list[FilledTemplate]
            List of all filled template instances.

        Raises
        ------
        ValueError
            If any slot has no valid items.
        """
        # 1. Resolve slot constraints
        slot_items = self._resolve_slot_constraints(template, language_code)

        # 2. Check for empty slots
        empty_slots = [name for name, items in slot_items.items() if not items]
        if empty_slots:
            raise ValueError(f"No valid items for slots: {empty_slots}")

        # 3. Generate combinations using strategy
        combinations = self.strategy.generate_combinations(slot_items)

        # 4. Create FilledTemplate instances
        filled_templates: list[FilledTemplate] = []
        for combo in combinations:
            rendered = self._render_template(template, combo)
            filled = FilledTemplate(
                template_id=str(template.id),
                template_name=template.name,
                slot_fillers=combo,
                rendered_text=rendered,
                strategy_name=self.strategy.name,
            )
            filled_templates.append(filled)

        return filled_templates

    def _resolve_slot_constraints(
        self,
        template: Template,
        language_code: LanguageCode | None,
    ) -> dict[str, list[LexicalItem]]:
        """Resolve constraints for each slot.

        Parameters
        ----------
        template : Template
            Template with slots and constraints.
        language_code : LanguageCode | None
            Optional language filter.

        Returns
        -------
        dict[str, list[LexicalItem]]
            Mapping of slot names to valid items.
        """
        slot_items: dict[str, list[LexicalItem]] = {}

        # Normalize language code if provided
        normalized_lang = validate_iso639_code(language_code) if language_code else None

        for slot_name, slot in template.slots.items():
            candidates = list(self.lexicon.items.values())

            # Filter by language code
            if normalized_lang:
                candidates = [
                    item for item in candidates if item.language_code == normalized_lang
                ]

            # Apply slot constraints
            if slot.constraints:
                for constraint in slot.constraints:
                    filtered: list[LexicalItem] = []
                    for item in candidates:
                        eval_context: dict[
                            str, ContextValue | LexicalItem | FilledTemplate | Item
                        ] = {"self": item}
                        if constraint.context:
                            eval_context.update(constraint.context)

                        evaluator = DSLEvaluator()
                        try:
                            if evaluator.evaluate(constraint.expression, eval_context):
                                filtered.append(item)
                        except Exception:
                            continue
                    candidates = filtered

            slot_items[slot_name] = candidates

        return slot_items

    def _render_template(
        self, template: Template, slot_fillers: dict[str, LexicalItem]
    ) -> str:
        """Render template string with slot fillers.

        Parameters
        ----------
        template : Template
            Template with template_string.
        slot_fillers : dict[str, LexicalItem]
            Items filling each slot.

        Returns
        -------
        str
            Rendered template string.
        """
        rendered = template.template_string
        for slot_name, item in slot_fillers.items():
            placeholder = f"{{{slot_name}}}"
            rendered = rendered.replace(placeholder, item.lemma)
        return rendered

    def count_combinations(self, template: Template) -> int:
        """Count total possible combinations for template.

        Parameters
        ----------
        template : Template
            Template to count combinations for.

        Returns
        -------
        int
            Total number of possible combinations.
        """
        slot_items = self._resolve_slot_constraints(template, None)

        if not slot_items:
            return 0

        count = 1
        for items in slot_items.values():
            count *= len(items)

        return count
