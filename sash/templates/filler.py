"""Template filling orchestration."""

from __future__ import annotations

from sash.data.base import SashBaseModel
from sash.data.language_codes import LanguageCode, validate_iso639_code
from sash.resources.adapters.registry import AdapterRegistry
from sash.resources.lexicon import Lexicon
from sash.resources.models import LexicalItem
from sash.resources.structures import Template
from sash.templates.combinatorics import count_combinations
from sash.templates.resolver import ConstraintResolver
from sash.templates.strategies import FillingStrategy


class FilledTemplate(SashBaseModel):
    """A template populated with lexical items.

    Represents a single instance of a template with specific
    items filling each slot.

    Attributes
    ----------
    template_id : str
        ID of the source template.
    template_name : str
        Name of the source template.
    slot_fillers : dict[str, LexicalItem]
        Mapping of slot names to items that fill them.
    rendered_text : str
        Template string with slots replaced by item lemmas.
    strategy_name : str
        Name of strategy used to generate this filled template.

    Examples
    --------
    >>> filled = FilledTemplate(
    ...     template_id="t1",
    ...     template_name="transitive",
    ...     slot_fillers={"subject": noun_item, "verb": verb_item},
    ...     rendered_text="cat broke the object",
    ...     strategy_name="exhaustive"
    ... )
    """

    template_id: str
    template_name: str
    slot_fillers: dict[str, LexicalItem]
    rendered_text: str
    strategy_name: str


class TemplateFiller:
    """Fill templates with lexical items using constraint resolution.

    Orchestrates the template filling process by:
    1. Using ConstraintResolver to find valid items for each slot
    2. Applying a FillingStrategy to generate combinations
    3. Creating FilledTemplate instances with metadata

    Parameters
    ----------
    lexicon : Lexicon
        Lexicon containing candidate items.
    strategy : FillingStrategy
        Strategy for generating combinations.
    adapter_registry : AdapterRegistry | None
        Registry for external resource adapters (for relational constraints).

    Examples
    --------
    >>> from sash.templates.strategies import ExhaustiveStrategy
    >>> filler = TemplateFiller(lexicon, strategy=ExhaustiveStrategy())
    >>> filled_templates = filler.fill(template)
    >>> len(filled_templates)
    12
    """

    def __init__(
        self,
        lexicon: Lexicon,
        strategy: FillingStrategy,
        adapter_registry: AdapterRegistry | None = None,
    ) -> None:
        self.lexicon = lexicon
        self.strategy = strategy
        self.adapter_registry = adapter_registry

        # Initialize constraint resolver
        self.resolver = ConstraintResolver(
            lexicon=lexicon,
            adapter_registry=adapter_registry,
        )

    def fill(
        self,
        template: Template,
        language_code: LanguageCode | None = None,
    ) -> list[FilledTemplate]:
        """Fill template with lexical items.

        Resolve constraints for each slot and generate combinations
        according to the strategy.

        Parameters
        ----------
        template : Template
            Template to fill.
        language_code : LanguageCode | None
            Optional language code to filter items.

        Returns
        -------
        list[FilledTemplate]
            List of filled template instances.

        Raises
        ------
        ValueError
            If any slot has no valid items.

        Examples
        --------
        >>> filled = filler.fill(template, language_code="en")
        >>> all(f.template_name == template.name for f in filled)
        True
        """
        # Normalize language code to ISO 639-3 format if provided
        normalized_language_code = validate_iso639_code(language_code)

        # 1. Resolve constraints for each slot
        slot_items = self._resolve_slot_constraints(template, normalized_language_code)

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
        for slot_name, slot in template.slots.items():
            if slot.constraints:
                # Resolve each constraint and find intersection
                # (items must satisfy ALL constraints)
                valid_items_list: list[LexicalItem] | None = None
                for constraint in slot.constraints:
                    items = self.resolver.resolve(constraint, language_code)
                    if valid_items_list is None:
                        valid_items_list = items
                    else:
                        # Find intersection using item IDs
                        item_ids = {item.id for item in items}
                        valid_items_list = [
                            item for item in valid_items_list if item.id in item_ids
                        ]

                slot_items[slot_name] = valid_items_list if valid_items_list else []
            else:
                # No constraints means all items are valid
                if language_code:
                    items = [
                        item
                        for item in self.lexicon.items.values()
                        if item.language_code == language_code
                    ]
                else:
                    items = list(self.lexicon.items.values())
                slot_items[slot_name] = items
        return slot_items

    def _render_template(
        self,
        template: Template,
        slot_fillers: dict[str, LexicalItem],
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

    def count_combinations(
        self,
        template: Template,
        language_code: LanguageCode | None = None,
    ) -> int:
        """Count total combinations without generating them.

        Useful for checking combinatorial explosion before filling.

        Parameters
        ----------
        template : Template
            Template to analyze.
        language_code : LanguageCode | None
            Optional language filter.

        Returns
        -------
        int
            Total number of possible combinations.

        Examples
        --------
        >>> count = filler.count_combinations(template)
        >>> count
        1000000
        >>> # Too many! Use RandomStrategy instead of ExhaustiveStrategy
        """
        slot_items = self._resolve_slot_constraints(template, language_code)
        item_lists = list(slot_items.values())
        return count_combinations(*item_lists)
