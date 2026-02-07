"""Core span annotation models.

Provides data models for labeled spans, span segments, span labels,
span relations, and span specifications. Supports discontiguous spans,
overlapping spans (nested and intersecting), static and interactive modes,
and two label sources (fixed sets and Wikidata entity search).
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator

from bead.data.base import BeadBaseModel

# Same recursive type as in item.py and item_template.py; duplicated here
# to avoid circular imports (item.py imports Span from this module).
type MetadataValue = (
    str | int | float | bool | None | dict[str, MetadataValue] | list[MetadataValue]
)

SpanIndexMode = Literal["token", "character"]
SpanInteractionMode = Literal["static", "interactive"]
LabelSourceType = Literal["fixed", "wikidata"]


# Factory functions for default values
def _empty_span_segment_list() -> list[SpanSegment]:
    """Return empty SpanSegment list."""
    return []


def _empty_span_metadata() -> dict[str, MetadataValue]:
    """Return empty metadata dict."""
    return {}


def _empty_relation_metadata() -> dict[str, MetadataValue]:
    """Return empty metadata dict."""
    return {}


class SpanSegment(BeadBaseModel):
    """Contiguous or discontiguous indices within a single element.

    Attributes
    ----------
    element_name : str
        Which rendered element this segment belongs to.
    indices : list[int]
        Token or character indices within the element.
    """

    element_name: str = Field(..., description="Rendered element name")
    indices: list[int] = Field(..., description="Token or character indices")

    @field_validator("element_name")
    @classmethod
    def validate_element_name(cls, v: str) -> str:
        """Validate element name is not empty.

        Parameters
        ----------
        v : str
            Element name to validate.

        Returns
        -------
        str
            Validated element name.

        Raises
        ------
        ValueError
            If element name is empty.
        """
        if not v or not v.strip():
            raise ValueError("element_name cannot be empty")
        return v.strip()

    @field_validator("indices")
    @classmethod
    def validate_indices(cls, v: list[int]) -> list[int]:
        """Validate indices are not empty and non-negative.

        Parameters
        ----------
        v : list[int]
            Indices to validate.

        Returns
        -------
        list[int]
            Validated indices.

        Raises
        ------
        ValueError
            If indices are empty or contain negative values.
        """
        if not v:
            raise ValueError("indices cannot be empty")
        if any(i < 0 for i in v):
            raise ValueError("indices must be non-negative")
        return v


class SpanLabel(BeadBaseModel):
    """Label applied to a span or relation.

    Attributes
    ----------
    label : str
        Human-readable label text.
    label_id : str | None
        External identifier (e.g. Wikidata QID "Q5").
    confidence : float | None
        Confidence score for model-assigned labels.
    """

    label: str = Field(..., description="Human-readable label text")
    label_id: str | None = Field(
        default=None, description="External ID (e.g. Wikidata QID)"
    )
    confidence: float | None = Field(
        default=None, description="Confidence for model-assigned labels"
    )

    @field_validator("label")
    @classmethod
    def validate_label(cls, v: str) -> str:
        """Validate label is not empty.

        Parameters
        ----------
        v : str
            Label to validate.

        Returns
        -------
        str
            Validated label.

        Raises
        ------
        ValueError
            If label is empty.
        """
        if not v or not v.strip():
            raise ValueError("label cannot be empty")
        return v.strip()


class Span(BeadBaseModel):
    """Labeled span across one or more elements.

    Supports discontiguous, overlapping, and nested spans.

    Attributes
    ----------
    span_id : str
        Unique identifier within the item.
    segments : list[SpanSegment]
        Index segments composing this span.
    head_index : int | None
        Syntactic head token index.
    label : SpanLabel | None
        Label applied to this span (None = to-be-labeled).
    span_type : str | None
        Semantic category (e.g. "entity", "event", "role").
    span_metadata : dict[str, MetadataValue]
        Additional span-specific metadata.
    """

    span_id: str = Field(..., description="Unique span ID within item")
    segments: list[SpanSegment] = Field(
        default_factory=_empty_span_segment_list, description="Index segments"
    )
    head_index: int | None = Field(
        default=None, description="Syntactic head token index"
    )
    label: SpanLabel | None = Field(
        default=None, description="Span label (None = to-be-labeled)"
    )
    span_type: str | None = Field(
        default=None, description="Semantic category"
    )
    span_metadata: dict[str, MetadataValue] = Field(
        default_factory=_empty_span_metadata, description="Span metadata"
    )

    @field_validator("span_id")
    @classmethod
    def validate_span_id(cls, v: str) -> str:
        """Validate span_id is not empty.

        Parameters
        ----------
        v : str
            Span ID to validate.

        Returns
        -------
        str
            Validated span ID.

        Raises
        ------
        ValueError
            If span_id is empty.
        """
        if not v or not v.strip():
            raise ValueError("span_id cannot be empty")
        return v.strip()


class SpanRelation(BeadBaseModel):
    """A typed, directed relation between two spans.

    Used for semantic role labeling, relation extraction, entity linking,
    coreference, and similar tasks.

    Attributes
    ----------
    relation_id : str
        Unique identifier within the item.
    source_span_id : str
        ``span_id`` of the source span.
    target_span_id : str
        ``span_id`` of the target span.
    label : SpanLabel | None
        Relation label (reuses SpanLabel for consistency).
    directed : bool
        Whether the relation is directed (A->B) or undirected (A--B).
    relation_metadata : dict[str, MetadataValue]
        Additional relation-specific metadata.
    """

    relation_id: str = Field(..., description="Unique relation ID within item")
    source_span_id: str = Field(..., description="Source span ID")
    target_span_id: str = Field(..., description="Target span ID")
    label: SpanLabel | None = Field(
        default=None, description="Relation label"
    )
    directed: bool = Field(
        default=True, description="Whether relation is directed"
    )
    relation_metadata: dict[str, MetadataValue] = Field(
        default_factory=_empty_relation_metadata,
        description="Relation metadata",
    )

    @field_validator("relation_id")
    @classmethod
    def validate_relation_id(cls, v: str) -> str:
        """Validate relation_id is not empty.

        Parameters
        ----------
        v : str
            Relation ID to validate.

        Returns
        -------
        str
            Validated relation ID.

        Raises
        ------
        ValueError
            If relation_id is empty.
        """
        if not v or not v.strip():
            raise ValueError("relation_id cannot be empty")
        return v.strip()

    @field_validator("source_span_id", "target_span_id")
    @classmethod
    def validate_span_ids(cls, v: str) -> str:
        """Validate span IDs are not empty.

        Parameters
        ----------
        v : str
            Span ID to validate.

        Returns
        -------
        str
            Validated span ID.

        Raises
        ------
        ValueError
            If span ID is empty.
        """
        if not v or not v.strip():
            raise ValueError("span ID cannot be empty")
        return v.strip()


class SpanSpec(BeadBaseModel):
    """Specification for span labeling behavior.

    Configures how spans are displayed, created, and labeled in an
    experiment. Supports both fixed label sets and Wikidata entity search
    for both span labels and relation labels.

    Attributes
    ----------
    index_mode : SpanIndexMode
        Whether spans index by token or character position.
    interaction_mode : SpanInteractionMode
        "static" for read-only highlights, "interactive" for participant
        annotation.
    label_source : LabelSourceType
        Source of span labels ("fixed" or "wikidata").
    labels : list[str] | None
        Fixed span label set (when label_source is "fixed").
    label_colors : dict[str, str] | None
        CSS colors keyed by label name.
    allow_overlapping : bool
        Whether overlapping spans are permitted.
    min_spans : int | None
        Minimum number of spans required (interactive mode).
    max_spans : int | None
        Maximum number of spans allowed (interactive mode).
    enable_relations : bool
        Whether relation annotation is enabled.
    relation_label_source : LabelSourceType
        Source of relation labels.
    relation_labels : list[str] | None
        Fixed relation label set.
    relation_label_colors : dict[str, str] | None
        CSS colors keyed by relation label name.
    relation_directed : bool
        Default directionality for new relations.
    min_relations : int | None
        Minimum number of relations required (interactive mode).
    max_relations : int | None
        Maximum number of relations allowed (interactive mode).
    wikidata_language : str
        Language for Wikidata entity search.
    wikidata_entity_types : list[str] | None
        Restrict Wikidata search to these entity types.
    wikidata_result_limit : int
        Maximum number of Wikidata search results.
    """

    index_mode: SpanIndexMode = Field(
        default="token", description="Span indexing mode"
    )
    interaction_mode: SpanInteractionMode = Field(
        default="static", description="Span interaction mode"
    )
    # Span label config
    label_source: LabelSourceType = Field(
        default="fixed", description="Span label source"
    )
    labels: list[str] | None = Field(
        default=None, description="Fixed span label set"
    )
    label_colors: dict[str, str] | None = Field(
        default=None, description="CSS colors per span label"
    )
    allow_overlapping: bool = Field(
        default=True, description="Whether overlapping spans are allowed"
    )
    min_spans: int | None = Field(
        default=None, description="Minimum required spans (interactive)"
    )
    max_spans: int | None = Field(
        default=None, description="Maximum allowed spans (interactive)"
    )
    # Relation config
    enable_relations: bool = Field(
        default=False, description="Whether relation annotation is enabled"
    )
    relation_label_source: LabelSourceType = Field(
        default="fixed", description="Relation label source"
    )
    relation_labels: list[str] | None = Field(
        default=None, description="Fixed relation label set"
    )
    relation_label_colors: dict[str, str] | None = Field(
        default=None, description="CSS colors per relation label"
    )
    relation_directed: bool = Field(
        default=True, description="Default directionality for relations"
    )
    min_relations: int | None = Field(
        default=None, description="Minimum required relations (interactive)"
    )
    max_relations: int | None = Field(
        default=None, description="Maximum allowed relations (interactive)"
    )
    # Wikidata config (shared by span labels and relation labels)
    wikidata_language: str = Field(
        default="en", description="Language for Wikidata entity search"
    )
    wikidata_entity_types: list[str] | None = Field(
        default=None, description="Restrict Wikidata entity types"
    )
    wikidata_result_limit: int = Field(
        default=10, description="Max Wikidata search results"
    )
