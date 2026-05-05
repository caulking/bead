"""Core abstractions for the span text transform system.

Defines the :class:`SpanTextTransform` protocol, :class:`TransformContext`
for passing metadata to transforms, :class:`TransformPipeline` for
composing transforms, and :class:`TransformRegistry` for name-based lookup.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class TransformContext:
    """Metadata available to transforms at resolution time.

    Carries information about the span being transformed so that
    transforms can make language- or syntax-aware decisions.  All
    fields are optional; a transform should degrade gracefully when
    a field is ``None``.

    Attributes
    ----------
    language_code : str | None
        ISO 639 code (e.g. ``"eng"``, ``"en"``).
    lemma : str | None
        Lemma of the span head, if known.
    pos : str | None
        Universal POS tag of the span head (e.g. ``"VERB"``).
    head_index : int | None
        Token index of the syntactic head within the span.
    tokens : list[str]
        Individual tokens of the span text.  Empty list when unknown.
    metadata : dict[str, object]
        Arbitrary extra data (e.g. morphological features already
        extracted by decomp).
    """

    language_code: str | None = None
    lemma: str | None = None
    pos: str | None = None
    head_index: int | None = None
    tokens: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)


@runtime_checkable
class SpanTextTransform(Protocol):
    """Protocol for a single text transform.

    Any callable ``(str, TransformContext) -> str`` satisfies this
    protocol.  Implementations may ignore the context when the
    transform is purely textual (e.g. lowercasing).
    """

    def __call__(self, text: str, context: TransformContext) -> str:
        """Apply the transform to *text*.

        Parameters
        ----------
        text : str
            The span text (possibly already transformed by an earlier
            stage in a pipeline).
        context : TransformContext
            Metadata about the span.

        Returns
        -------
        str
            Transformed text.
        """
        ...


class TransformPipeline:
    """An ordered chain of transforms applied left-to-right.

    Parameters
    ----------
    transforms : list[SpanTextTransform]
        Transforms to apply in order.

    Examples
    --------
    >>> from bead.transforms.text import LowerTransform, CapitalizeTransform
    >>> ctx = TransformContext()
    >>> pipe = TransformPipeline([LowerTransform(), CapitalizeTransform()])
    >>> pipe("HELLO WORLD", ctx)
    'Hello world'
    """

    def __init__(self, transforms: list[SpanTextTransform] | None = None) -> None:
        self._transforms: list[SpanTextTransform] = list(transforms or [])

    def __call__(self, text: str, context: TransformContext) -> str:
        """Apply each transform in sequence.

        Parameters
        ----------
        text : str
            Input text.
        context : TransformContext
            Shared context for all transforms in the pipeline.

        Returns
        -------
        str
            Fully transformed text.
        """
        for transform in self._transforms:
            text = transform(text, context)

        return text

    def __len__(self) -> int:
        return len(self._transforms)

    def __repr__(self) -> str:
        names = [type(t).__name__ for t in self._transforms]
        return f"TransformPipeline({names})"

    def append(self, transform: SpanTextTransform) -> None:
        """Append a transform to the end of the pipeline.

        Parameters
        ----------
        transform : SpanTextTransform
            Transform to append.
        """
        self._transforms.append(transform)

    def prepend(self, transform: SpanTextTransform) -> None:
        """Insert a transform at the beginning of the pipeline.

        Parameters
        ----------
        transform : SpanTextTransform
            Transform to prepend.
        """
        self._transforms.insert(0, transform)


class TransformRegistry:
    """Name-to-transform mapping with pipeline construction.

    Transforms are registered under short string names (e.g.
    ``"gerund"``, ``"lower"``) and looked up when resolving
    ``[[label|name1|name2]]`` prompt references.

    Examples
    --------
    >>> from bead.transforms.text import LowerTransform
    >>> reg = TransformRegistry()
    >>> reg.register("lower", LowerTransform())
    >>> t = reg.get("lower")
    >>> t("HELLO", TransformContext())
    'hello'
    """

    def __init__(self) -> None:
        self._transforms: dict[str, SpanTextTransform] = {}

    def register(
        self,
        name: str,
        transform: SpanTextTransform | Callable[[str, TransformContext], str],
    ) -> None:
        """Register a transform under *name*.

        Parameters
        ----------
        name : str
            Short identifier used in ``[[label|name]]`` syntax.
            Case-insensitive (stored lowered).
        transform : SpanTextTransform | Callable
            The transform callable.

        Raises
        ------
        ValueError
            If *name* is empty.
        """
        name = name.strip().lower()

        if not name:
            raise ValueError("Transform name must be non-empty")

        self._transforms[name] = transform

    def get(self, name: str) -> SpanTextTransform:
        """Look up a transform by name.

        Parameters
        ----------
        name : str
            Registered name (case-insensitive).

        Returns
        -------
        SpanTextTransform
            The registered transform.

        Raises
        ------
        KeyError
            If no transform with that name exists.
        """
        name = name.strip().lower()

        try:
            return self._transforms[name]

        except KeyError:
            available = sorted(self._transforms)
            raise KeyError(
                f"No transform registered as '{name}'. "
                f"Available: {available}"
            ) from None

    def resolve_pipeline(self, names: list[str]) -> TransformPipeline:
        """Build a pipeline from an ordered list of transform names.

        Parameters
        ----------
        names : list[str]
            Transform names in application order.

        Returns
        -------
        TransformPipeline
            A pipeline that applies the named transforms left-to-right.

        Raises
        ------
        KeyError
            If any name is unregistered.
        """
        return TransformPipeline([self.get(n) for n in names])

    def available(self) -> list[str]:
        """Return sorted list of registered transform names.

        Returns
        -------
        list[str]
            All registered names.
        """
        return sorted(self._transforms)

    def __contains__(self, name: str) -> bool:
        return name.strip().lower() in self._transforms

    def __len__(self) -> int:
        return len(self._transforms)

    def __repr__(self) -> str:
        return f"TransformRegistry(transforms={self.available()})"
