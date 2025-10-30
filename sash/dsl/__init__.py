"""Constraint Domain-Specific Language (DSL).

This module provides a DSL for expressing constraints on lexical items
in templates. The DSL includes:

- Boolean operators: and, or, not
- Comparison operators: ==, !=, <, >, <=, >=
- Membership operators: in, not in
- Arithmetic operators: +, -, *, /, %
- Function calls
- Attribute access
- List literals
- Standard library functions

Examples
--------
>>> from sash.dsl import parse, evaluate
>>> node = parse("lemma == 'walk' and pos == 'VERB'")
>>> from sash.dsl import EvaluationContext
>>> ctx = EvaluationContext()
>>> ctx.set_variable("lemma", "walk")
>>> ctx.set_variable("pos", "VERB")
>>> evaluate(node, ctx)
True
"""

from typing import Any

from sash.dsl.ast import (
    ASTNode,
    AttributeAccess,
    BinaryOp,
    FunctionCall,
    ListLiteral,
    Literal,
    UnaryOp,
    Variable,
)
from sash.dsl.context import EvaluationContext
from sash.dsl.errors import DSLError, EvaluationError, ParseError
from sash.dsl.evaluator import Evaluator
from sash.dsl.parser import parse
from sash.dsl.stdlib import SIMULATION_FUNCTIONS, STDLIB_FUNCTIONS, register_stdlib

__all__ = [
    # AST nodes
    "ASTNode",
    "Literal",
    "Variable",
    "BinaryOp",
    "UnaryOp",
    "FunctionCall",
    "ListLiteral",
    "AttributeAccess",
    # Errors
    "DSLError",
    "ParseError",
    "EvaluationError",
    # Parser
    "parse",
    # Evaluation
    "Evaluator",
    "EvaluationContext",
    "evaluate",
    # Standard library
    "STDLIB_FUNCTIONS",
    "SIMULATION_FUNCTIONS",
    "register_stdlib",
]


# Convenience function
def evaluate(node: ASTNode, context: EvaluationContext, use_cache: bool = True) -> Any:
    """Evaluate a constraint expression.

    Parameters
    ----------
    node : ASTNode
        Parsed AST node to evaluate.
    context : EvaluationContext
        Evaluation context with variables and functions.
    use_cache : bool
        Whether to use caching.

    Returns
    -------
    Any
        Result of evaluation.

    Examples
    --------
    >>> from sash.dsl import parse, EvaluationContext
    >>> ctx = EvaluationContext()
    >>> ctx.set_variable("x", 10)
    >>> node = parse("x > 5")
    >>> evaluate(node, ctx)
    True
    """
    evaluator = Evaluator(use_cache=use_cache)
    return evaluator.evaluate(node, context)
