"""Standard library functions for constraint DSL.

This module provides built-in functions that can be used in constraint
expressions. Functions are organized by category and registered with
the evaluation context.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sash.dsl.context import EvaluationContext


# String functions
def len_(s: Any) -> int:
    """Return length of string or collection.

    Parameters
    ----------
    s : Any
        String or collection to measure.

    Returns
    -------
    int
        Length of string or collection.

    Examples
    --------
    >>> len_("hello")
    5
    >>> len_([1, 2, 3])
    3
    """
    return len(s)


def lower(s: str) -> str:
    """Convert string to lowercase.

    Parameters
    ----------
    s : str
        String to convert.

    Returns
    -------
    str
        Lowercase string.

    Examples
    --------
    >>> lower("HELLO")
    'hello'
    """
    return s.lower()


def upper(s: str) -> str:
    """Convert string to uppercase.

    Parameters
    ----------
    s : str
        String to convert.

    Returns
    -------
    str
        Uppercase string.

    Examples
    --------
    >>> upper("hello")
    'HELLO'
    """
    return s.upper()


def startswith(s: str, prefix: str) -> bool:
    """Check if string starts with prefix.

    Parameters
    ----------
    s : str
        String to check.
    prefix : str
        Prefix to look for.

    Returns
    -------
    bool
        True if string starts with prefix.

    Examples
    --------
    >>> startswith("hello", "hel")
    True
    >>> startswith("hello", "bye")
    False
    """
    return s.startswith(prefix)


def endswith(s: str, suffix: str) -> bool:
    """Check if string ends with suffix.

    Parameters
    ----------
    s : str
        String to check.
    suffix : str
        Suffix to look for.

    Returns
    -------
    bool
        True if string ends with suffix.

    Examples
    --------
    >>> endswith("hello", "lo")
    True
    >>> endswith("hello", "hi")
    False
    """
    return s.endswith(suffix)


def contains(s: str, substring: str) -> bool:
    """Check if string contains substring.

    Parameters
    ----------
    s : str
        String to check.
    substring : str
        Substring to look for.

    Returns
    -------
    bool
        True if string contains substring.

    Examples
    --------
    >>> contains("hello", "ell")
    True
    >>> contains("hello", "bye")
    False
    """
    return substring in s


def replace(s: str, old: str, new: str) -> str:
    """Replace occurrences of substring.

    Parameters
    ----------
    s : str
        String to modify.
    old : str
        Substring to replace.
    new : str
        Replacement substring.

    Returns
    -------
    str
        String with replacements.

    Examples
    --------
    >>> replace("hello world", "world", "there")
    'hello there'
    """
    return s.replace(old, new)


def split(s: str, sep: str = " ") -> list[str]:
    """Split string by separator.

    Parameters
    ----------
    s : str
        String to split.
    sep : str
        Separator string. Defaults to space.

    Returns
    -------
    list[str]
        List of substrings.

    Examples
    --------
    >>> split("a,b,c", ",")
    ['a', 'b', 'c']
    """
    return s.split(sep)


# Collection functions
def count(collection: Any, item: Any) -> int:
    """Count occurrences of item in collection.

    Parameters
    ----------
    collection : Any
        Collection to search.
    item : Any
        Item to count.

    Returns
    -------
    int
        Number of occurrences.

    Examples
    --------
    >>> count([1, 2, 2, 3], 2)
    2
    >>> count("hello", "l")
    2
    """
    return collection.count(item)


def sum_(collection: list[int | float]) -> int | float:
    """Sum numeric collection.

    Parameters
    ----------
    collection : list[int | float]
        Collection of numbers.

    Returns
    -------
    int | float
        Sum of all numbers.

    Examples
    --------
    >>> sum_([1, 2, 3])
    6
    >>> sum_([1.5, 2.5])
    4.0
    """
    return sum(collection)


def min_(collection: list[Any]) -> Any:
    """Return minimum value from collection.

    Parameters
    ----------
    collection : list[Any]
        Collection to search.

    Returns
    -------
    Any
        Minimum value.

    Examples
    --------
    >>> min_([3, 1, 2])
    1
    """
    return min(collection)


def max_(collection: list[Any]) -> Any:
    """Return maximum value from collection.

    Parameters
    ----------
    collection : list[Any]
        Collection to search.

    Returns
    -------
    Any
        Maximum value.

    Examples
    --------
    >>> max_([3, 1, 2])
    3
    """
    return max(collection)


def any_(collection: list[Any]) -> bool:
    """Check if any element is truthy.

    Parameters
    ----------
    collection : list[Any]
        Collection to check.

    Returns
    -------
    bool
        True if any element is truthy.

    Examples
    --------
    >>> any_([False, True, False])
    True
    >>> any_([False, False])
    False
    """
    return any(collection)


def all_(collection: list[Any]) -> bool:
    """Check if all elements are truthy.

    Parameters
    ----------
    collection : list[Any]
        Collection to check.

    Returns
    -------
    bool
        True if all elements are truthy.

    Examples
    --------
    >>> all_([True, True, True])
    True
    >>> all_([True, False, True])
    False
    """
    return all(collection)


# Type checking functions
def is_str(value: Any) -> bool:
    """Check if value is a string.

    Parameters
    ----------
    value : Any
        Value to check.

    Returns
    -------
    bool
        True if value is a string.

    Examples
    --------
    >>> is_str("hello")
    True
    >>> is_str(42)
    False
    """
    return isinstance(value, str)


def is_int(value: Any) -> bool:
    """Check if value is an integer.

    Parameters
    ----------
    value : Any
        Value to check.

    Returns
    -------
    bool
        True if value is an integer.

    Examples
    --------
    >>> is_int(42)
    True
    >>> is_int(42.0)
    False
    """
    return isinstance(value, int) and not isinstance(value, bool)


def is_float(value: Any) -> bool:
    """Check if value is a float.

    Parameters
    ----------
    value : Any
        Value to check.

    Returns
    -------
    bool
        True if value is a float.

    Examples
    --------
    >>> is_float(42.0)
    True
    >>> is_float(42)
    False
    """
    return isinstance(value, float)


def is_bool(value: Any) -> bool:
    """Check if value is a boolean.

    Parameters
    ----------
    value : Any
        Value to check.

    Returns
    -------
    bool
        True if value is a boolean.

    Examples
    --------
    >>> is_bool(True)
    True
    >>> is_bool(1)
    False
    """
    return isinstance(value, bool)


def is_list(value: Any) -> bool:
    """Check if value is a list.

    Parameters
    ----------
    value : Any
        Value to check.

    Returns
    -------
    bool
        True if value is a list.

    Examples
    --------
    >>> is_list([1, 2, 3])
    True
    >>> is_list((1, 2, 3))
    False
    """
    return isinstance(value, list)


# Math functions
def abs_(value: int | float) -> int | float:
    """Return absolute value.

    Parameters
    ----------
    value : int | float
        Numeric value.

    Returns
    -------
    int | float
        Absolute value.

    Examples
    --------
    >>> abs_(-5)
    5
    >>> abs_(5)
    5
    """
    return abs(value)


def round_(value: float, ndigits: int = 0) -> float:
    """Round numeric value.

    Parameters
    ----------
    value : float
        Value to round.
    ndigits : int
        Number of decimal places.

    Returns
    -------
    float
        Rounded value.

    Examples
    --------
    >>> round_(3.14159, 2)
    3.14
    """
    return round(value, ndigits)


def floor(value: float) -> int:
    """Return floor of value.

    Parameters
    ----------
    value : float
        Numeric value.

    Returns
    -------
    int
        Floor value.

    Examples
    --------
    >>> floor(3.7)
    3
    >>> floor(-3.7)
    -4
    """
    return math.floor(value)


def ceil(value: float) -> int:
    """Return ceiling of value.

    Parameters
    ----------
    value : float
        Numeric value.

    Returns
    -------
    int
        Ceiling value.

    Examples
    --------
    >>> ceil(3.2)
    4
    >>> ceil(-3.2)
    -3
    """
    return math.ceil(value)


# Logic functions
def not_(value: Any) -> bool:
    """Return logical negation of value.

    Parameters
    ----------
    value : Any
        Value to negate.

    Returns
    -------
    bool
        Logical negation.

    Examples
    --------
    >>> not_(True)
    False
    >>> not_(False)
    True
    >>> not_(0)
    True
    """
    return not value


# Registry
STDLIB_FUNCTIONS: dict[str, Any] = {
    # String functions
    "len": len_,
    "lower": lower,
    "upper": upper,
    "startswith": startswith,
    "endswith": endswith,
    "contains": contains,
    "replace": replace,
    "split": split,
    # Collection functions
    "count": count,
    "sum": sum_,
    "min": min_,
    "max": max_,
    "any": any_,
    "all": all_,
    # Type checking
    "is_str": is_str,
    "is_int": is_int,
    "is_float": is_float,
    "is_bool": is_bool,
    "is_list": is_list,
    # Math functions
    "abs": abs_,
    "round": round_,
    "floor": floor,
    "ceil": ceil,
    # Logic functions
    "not": not_,
}


def register_stdlib(context: EvaluationContext) -> None:
    """Register all standard library functions in context.

    Parameters
    ----------
    context : EvaluationContext
        Context to register functions in.

    Examples
    --------
    >>> from sash.dsl.context import EvaluationContext
    >>> ctx = EvaluationContext()
    >>> register_stdlib(ctx)
    >>> ctx.call_function("len", ["hello"])
    5
    """
    for name, func in STDLIB_FUNCTIONS.items():
        context.set_function(name, func)
