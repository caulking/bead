"""Pure text transforms that require no external resources.

These transforms operate on the surface string and ignore the
:class:`TransformContext`.  They are always safe to register
regardless of language.
"""

from __future__ import annotations

from bead.transforms.base import TransformContext


class LowerTransform:
    """Convert text to lowercase.

    Examples
    --------
    >>> LowerTransform()("Hello World", TransformContext())
    'hello world'
    """

    def __call__(self, text: str, context: TransformContext) -> str:
        return text.lower()


class UpperTransform:
    """Convert text to uppercase.

    Examples
    --------
    >>> UpperTransform()("Hello World", TransformContext())
    'HELLO WORLD'
    """

    def __call__(self, text: str, context: TransformContext) -> str:
        return text.upper()


class CapitalizeTransform:
    """Capitalize the first character, lowercase the rest.

    Examples
    --------
    >>> CapitalizeTransform()("hELLO WORLD", TransformContext())
    'Hello world'
    """

    def __call__(self, text: str, context: TransformContext) -> str:
        return text.capitalize()


class TitleTransform:
    """Title-case each word.

    Examples
    --------
    >>> TitleTransform()("hello world", TransformContext())
    'Hello World'
    """

    def __call__(self, text: str, context: TransformContext) -> str:
        return text.title()
