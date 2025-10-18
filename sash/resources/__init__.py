"""Resource models for sash.

This module provides data models for lexical items, templates, constraints,
and template structures.
"""

from sash.resources.constraints import (
    Constraint,
    DSLConstraint,
    ExtensionalConstraint,
    IntensionalConstraint,
    RelationalConstraint,
)
from sash.resources.lexicon import Lexicon
from sash.resources.models import LexicalItem
from sash.resources.structures import (
    Slot,
    Template,
    TemplateSequence,
    TemplateTree,
)
from sash.resources.template_collection import TemplateCollection

__all__ = [
    # Lexical items
    "LexicalItem",
    # Lexicon
    "Lexicon",
    # Constraints
    "Constraint",
    "ExtensionalConstraint",
    "IntensionalConstraint",
    "RelationalConstraint",
    "DSLConstraint",
    # Templates and structures
    "Slot",
    "Template",
    "TemplateSequence",
    "TemplateTree",
    # Template collection
    "TemplateCollection",
]
