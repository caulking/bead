"""DSL-specific exceptions."""

from __future__ import annotations


class DSLError(Exception):
    """Base exception for DSL errors."""

    pass


class ParseError(DSLError):
    """Exception raised when parsing fails.

    Parameters
    ----------
    message
        Error message describing what went wrong during parsing.
    line
        Line number where the error occurred (1-indexed). None if unknown.
    column
        Column number where the error occurred (1-indexed). None if unknown.
    text
        The text that caused the error. None if unavailable.

    Attributes
    ----------
    line : int | None
        Line number where error occurred.
    column : int | None
        Column number where error occurred.
    text : str | None
        Text that caused the error.

    Examples
    --------
    >>> try:
    ...     raise ParseError("Unexpected token", line=5, column=12, text="@invalid")
    ... except ParseError as e:
    ...     print(e.line, e.column)
    5 12
    """

    def __init__(
        self,
        message: str,
        line: int | None = None,
        column: int | None = None,
        text: str | None = None,
    ) -> None:
        self.line = line
        self.column = column
        self.text = text
        super().__init__(message)

    def __str__(self) -> str:
        """Return formatted error message."""
        parts = [super().__str__()]
        if self.line is not None:
            parts.append(f" at line {self.line}")
        if self.column is not None:
            parts.append(f", column {self.column}")
        if self.text is not None:
            parts.append(f"\n  {self.text}")
        return "".join(parts)


class EvaluationError(DSLError):
    """Exception raised when evaluation fails."""

    pass
