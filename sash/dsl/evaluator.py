"""Constraint evaluator for DSL.

This module provides the Evaluator class that executes AST nodes
against an evaluation context to produce boolean results.
"""

from __future__ import annotations

from typing import Any

from sash.dsl import ast
from sash.dsl.context import EvaluationContext
from sash.dsl.errors import EvaluationError


class Evaluator:
    """Evaluator for constraint AST nodes.

    The evaluator walks the AST and computes values based on the
    evaluation context. It supports:
    - All AST node types
    - Operator evaluation
    - Function calls
    - Attribute access
    - Caching for performance

    Parameters
    ----------
    use_cache : bool
        Whether to cache evaluation results.

    Examples
    --------
    >>> from sash.dsl.context import EvaluationContext
    >>> from sash.dsl.parser import parse
    >>> ctx = EvaluationContext()
    >>> ctx.set_variable("x", 10)
    >>> evaluator = Evaluator()
    >>> node = parse("x > 5")
    >>> evaluator.evaluate(node, ctx)
    True
    """

    def __init__(self, use_cache: bool = True) -> None:
        self._use_cache = use_cache
        self._cache: dict[tuple[str, ...], Any] = {}

    def evaluate(self, node: ast.ASTNode, context: EvaluationContext) -> Any:
        """Evaluate an AST node in the given context.

        Parameters
        ----------
        node : ast.ASTNode
            AST node to evaluate.
        context : EvaluationContext
            Evaluation context with variables and functions.

        Returns
        -------
        Any
            Result of evaluation.

        Raises
        ------
        EvaluationError
            If evaluation fails (undefined variable, type error, etc.).
        """
        # Dispatch to specific evaluation methods
        if isinstance(node, ast.Literal):
            return self._evaluate_literal(node, context)
        elif isinstance(node, ast.Variable):
            return self._evaluate_variable(node, context)
        elif isinstance(node, ast.BinaryOp):
            return self._evaluate_binary_op(node, context)
        elif isinstance(node, ast.UnaryOp):
            return self._evaluate_unary_op(node, context)
        elif isinstance(node, ast.FunctionCall):
            return self._evaluate_function_call(node, context)
        elif isinstance(node, ast.AttributeAccess):
            return self._evaluate_attribute_access(node, context)
        elif isinstance(node, ast.ListLiteral):
            return self._evaluate_list_literal(node, context)
        else:
            raise EvaluationError(f"Unknown node type: {type(node).__name__}")

    def _evaluate_literal(self, node: ast.Literal, context: EvaluationContext) -> Any:
        """Evaluate literal node.

        Parameters
        ----------
        node : ast.Literal
            Literal node.
        context : EvaluationContext
            Evaluation context.

        Returns
        -------
        Any
            Literal value.
        """
        return node.value

    def _evaluate_variable(self, node: ast.Variable, context: EvaluationContext) -> Any:
        """Evaluate variable node.

        Parameters
        ----------
        node : ast.Variable
            Variable node.
        context : EvaluationContext
            Evaluation context.

        Returns
        -------
        Any
            Variable value from context.

        Raises
        ------
        EvaluationError
            If variable is not defined.
        """
        if not context.has_variable(node.name):
            raise EvaluationError(f"Undefined variable: {node.name}")
        return context.get_variable(node.name)

    def _evaluate_binary_op(
        self, node: ast.BinaryOp, context: EvaluationContext
    ) -> Any:
        """Evaluate binary operation node.

        Parameters
        ----------
        node : ast.BinaryOp
            Binary operation node.
        context : EvaluationContext
            Evaluation context.

        Returns
        -------
        Any
            Result of binary operation.

        Raises
        ------
        EvaluationError
            If operator is unknown or operation fails.
        """
        # Short-circuit evaluation for logical operators
        if node.operator == "and":
            left = self.evaluate(node.left, context)
            if not left:
                return False
            return bool(self.evaluate(node.right, context))
        elif node.operator == "or":
            left = self.evaluate(node.left, context)
            if left:
                return True
            return bool(self.evaluate(node.right, context))

        # Evaluate both operands for other operators
        left = self.evaluate(node.left, context)
        right = self.evaluate(node.right, context)

        try:
            # Comparison operators
            if node.operator == "==":
                return left == right
            elif node.operator == "!=":
                return left != right
            elif node.operator == "<":
                return left < right
            elif node.operator == ">":
                return left > right
            elif node.operator == "<=":
                return left <= right
            elif node.operator == ">=":
                return left >= right
            # Membership operators
            elif node.operator == "in":
                return left in right
            elif node.operator == "not in":
                return left not in right
            # Arithmetic operators
            elif node.operator == "+":
                return left + right
            elif node.operator == "-":
                return left - right
            elif node.operator == "*":
                return left * right
            elif node.operator == "/":
                if right == 0:
                    raise EvaluationError("Division by zero")
                return left / right
            elif node.operator == "%":
                if right == 0:
                    raise EvaluationError("Modulo by zero")
                return left % right
            else:
                raise EvaluationError(f"Unknown operator: {node.operator}")
        except TypeError as e:
            raise EvaluationError(
                f"Type error in operation '{node.operator}': "
                f"cannot operate on {type(left).__name__} and {type(right).__name__}"
            ) from e
        except ZeroDivisionError as e:
            raise EvaluationError("Division by zero") from e

    def _evaluate_unary_op(self, node: ast.UnaryOp, context: EvaluationContext) -> Any:
        """Evaluate unary operation node.

        Parameters
        ----------
        node : ast.UnaryOp
            Unary operation node.
        context : EvaluationContext
            Evaluation context.

        Returns
        -------
        Any
            Result of unary operation.

        Raises
        ------
        EvaluationError
            If operator is unknown or operation fails.
        """
        operand = self.evaluate(node.operand, context)

        try:
            if node.operator == "not":
                return not operand
            elif node.operator == "-":
                return -operand
            elif node.operator == "+":
                return +operand
            else:
                raise EvaluationError(f"Unknown unary operator: {node.operator}")
        except TypeError as e:
            raise EvaluationError(
                f"Type error in unary operation '{node.operator}': "
                f"cannot operate on {type(operand).__name__}"
            ) from e

    def _evaluate_function_call(
        self, node: ast.FunctionCall, context: EvaluationContext
    ) -> Any:
        """Evaluate function call node.

        Parameters
        ----------
        node : ast.FunctionCall
            Function call node.
        context : EvaluationContext
            Evaluation context.

        Returns
        -------
        Any
            Function return value.

        Raises
        ------
        EvaluationError
            If function is not defined or call fails.
        """
        # Get function name
        if not isinstance(node.function, ast.Variable):  # type: ignore[reportUnnecessaryIsInstance]
            raise EvaluationError("Function must be a variable")
        func_name = node.function.name

        # Evaluate arguments
        args = [self.evaluate(arg, context) for arg in node.arguments]

        # Call function
        return context.call_function(func_name, args)

    def _evaluate_attribute_access(
        self, node: ast.AttributeAccess, context: EvaluationContext
    ) -> Any:
        """Evaluate attribute access node.

        Parameters
        ----------
        node : ast.AttributeAccess
            Attribute access node.
        context : EvaluationContext
            Evaluation context.

        Returns
        -------
        Any
            Attribute value.

        Raises
        ------
        EvaluationError
            If attribute access fails.
        """
        obj = self.evaluate(node.object, context)

        # Try dictionary-style access first
        if isinstance(obj, dict):
            if node.attribute not in obj:
                raise EvaluationError(f"Dictionary does not have key: {node.attribute}")
            return obj[node.attribute]  # type: ignore[reportUnknownVariableType]

        # Try attribute access
        try:
            return getattr(obj, node.attribute)
        except AttributeError as e:
            raise EvaluationError(
                f"Object of type {type(obj).__name__} has no attribute: "
                f"{node.attribute}"
            ) from e

    def _evaluate_list_literal(
        self, node: ast.ListLiteral, context: EvaluationContext
    ) -> list[Any]:
        """Evaluate list literal node.

        Parameters
        ----------
        node : ast.ListLiteral
            List literal node.
        context : EvaluationContext
            Evaluation context.

        Returns
        -------
        list[Any]
            Evaluated list elements.
        """
        return [self.evaluate(element, context) for element in node.elements]

    def clear_cache(self) -> None:
        """Clear evaluation cache.

        Examples
        --------
        >>> evaluator = Evaluator()
        >>> evaluator.clear_cache()
        """
        self._cache.clear()
