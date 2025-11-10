"""Constraint models for lexical item selection.

This module provides a universal constraint model based on DSL expressions.
Constraints are pure DSL expressions with optional context variables.

Scope is determined by storage location:
- Slot.constraints → single-slot constraints (self = slot filler)
- Template.constraints → multi-slot constraints (slot names as variables)
- TemplateSequence.constraints → cross-template constraints
"""

from __future__ import annotations

from uuid import UUID

from pydantic import Field

from bead.data.base import BeadBaseModel
from bead.dsl.ast import ASTNode

# Type aliases for constraint context values
type ContextValue = str | int | float | bool | list[str] | set[str] | set[UUID]
type MetadataValue = (
    str | int | float | bool | list[str | int | float] | dict[str, str | int | float]
)


class Constraint(BeadBaseModel):
    """Universal constraint expressed via DSL.

    All constraints are DSL expressions evaluated with a context dictionary.
    The scope of the constraint is determined by where it is stored:
    - Slot.constraints: single-slot constraints where 'self' refers to the slot filler
    - Template.constraints: multi-slot constraints where slot names are variables
    - TemplateSequence.constraints: cross-template constraints

    Attributes
    ----------
    expression : str
        DSL expression to evaluate (must return boolean).
    context : dict[str, ContextValue]
        Context variables available during evaluation (e.g., whitelists, constants).
    description : str | None
        Optional human-readable description of the constraint.
    compiled : ASTNode | None
        Cached compiled AST after first compilation (optimization).

    Examples
    --------
    Extensional (whitelist):
    >>> constraint = Constraint(
    ...     expression="self.lemma in motion_verbs",
    ...     context={"motion_verbs": {"walk", "run", "jump"}}
    ... )

    Intensional (feature-based):
    >>> constraint = Constraint(
    ...     expression="self.pos == 'VERB' and self.features.number == 'singular'"
    ... )

    Binary agreement:
    >>> constraint = Constraint(
    ...     expression="subject.features.number == verb.features.number"
    ... )

    IF-THEN conditional:
    >>> constraint = Constraint(
    ...     expression="det.lemma != 'a' or noun.features.number == 'singular'"
    ... )
    """

    expression: str
    context: dict[str, ContextValue] = Field(default_factory=dict)
    description: str | None = None
    compiled: ASTNode | None = None

    @classmethod
    def combine(
        cls,
        *constraints: Constraint,
        logic: str = "and",
    ) -> Constraint:
        """Combine multiple constraints with AND or OR logic.

        Merges all context dictionaries from input constraints and combines
        their expressions using the specified logical operator.

        Parameters
        ----------
        *constraints : Constraint
            Variable number of constraints to combine.
        logic : str
            Logical operator to use: "and" or "or" (default: "and").

        Returns
        -------
        Constraint
            New constraint with combined expressions and merged contexts.

        Raises
        ------
        ValueError
            If no constraints provided or invalid logic operator.

        Examples
        --------
        >>> c1 = Constraint(
        ...     expression="self.pos == 'VERB'",
        ...     description="Must be a verb"
        ... )
        >>> c2 = Constraint(
        ...     expression="self.features.tense == 'present'",
        ...     description="Must be present tense"
        ... )
        >>> combined = Constraint.combine(c1, c2)
        >>> "and" in combined.expression
        True
        >>> combined.description
        'Must be a verb; Must be present tense'

        With OR logic and contexts:
        >>> c1 = Constraint(
        ...     expression="self.lemma in verbs",
        ...     context={"verbs": {"walk", "run"}},
        ...     description="Motion verb"
        ... )
        >>> c2 = Constraint(
        ...     expression="self.lemma in actions",
        ...     context={"actions": {"jump", "hop"}},
        ...     description="Action verb"
        ... )
        >>> combined = Constraint.combine(c1, c2, logic="or")
        >>> " or " in combined.expression
        True
        >>> len(combined.context)
        2
        """
        if not constraints:
            raise ValueError("Must provide at least one constraint")

        if logic not in ("and", "or"):
            raise ValueError(f"Invalid logic operator '{logic}'. Must be 'and' or 'or'")

        if len(constraints) == 1:
            return constraints[0]

        # Combine expressions with specified logic operator
        expressions = [f"({c.expression})" for c in constraints]
        combined_expression = f" {logic} ".join(expressions)

        # Merge contexts
        combined_context: dict[str, ContextValue] = {}
        for constraint in constraints:
            if constraint.context:
                combined_context.update(constraint.context)

        # Combine descriptions
        descriptions = [c.description for c in constraints if c.description]
        combined_description = "; ".join(descriptions) if descriptions else None

        return cls(
            expression=combined_expression,
            context=combined_context,
            description=combined_description,
        )
