"""Participant data models.

This module provides Participant and ParticipantIDMapping models for
storing participant information with privacy-preserving external ID mapping.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from pydantic import Field, field_validator

from bead.data.base import BeadBaseModel, JsonValue
from bead.data.timestamps import now_iso8601

if TYPE_CHECKING:
    from bead.participants.metadata_spec import ParticipantMetadataSpec


def _empty_metadata_dict() -> dict[str, JsonValue]:
    """Return empty metadata dict."""
    return {}


def _empty_session_list() -> list[str]:
    """Return empty session list."""
    return []


class Participant(BeadBaseModel):
    """A study participant with demographic and session metadata.

    Inherits UUID, timestamps, version, and metadata from BeadBaseModel.
    The internal `id` (UUID) is used for all analysis; external IDs
    (e.g., Prolific IDs) are stored separately for privacy.

    Attributes
    ----------
    id : UUID
        Internal unique identifier (UUIDv7, inherited from BeadBaseModel).
    created_at : datetime
        When participant record was created (inherited).
    modified_at : datetime
        When participant record was last modified (inherited).
    participant_metadata : dict[str, JsonValue]
        Demographic and other participant attributes (e.g., age, education).
        Keys should match a ParticipantMetadataSpec for validation.
    study_id : str | None
        Optional study identifier this participant belongs to.
    session_ids : list[str]
        Session identifiers for this participant (for longitudinal studies).
    consent_timestamp : datetime | None
        When participant provided consent.
    notes : str | None
        Free-text notes about this participant.

    Examples
    --------
    >>> participant = Participant(
    ...     participant_metadata={
    ...         "age": 25,
    ...         "education": "bachelors",
    ...         "native_speaker": True,
    ...     },
    ...     study_id="study_001",
    ... )
    >>> participant.participant_metadata["age"]
    25
    >>> str(participant.id)  # doctest: +SKIP
    '019...'  # UUIDv7
    """

    participant_metadata: dict[str, JsonValue] = Field(
        default_factory=_empty_metadata_dict,
        description="Participant attributes (demographics, etc.)",
    )
    study_id: str | None = Field(default=None, description="Study identifier")
    session_ids: list[str] = Field(
        default_factory=_empty_session_list, description="Session identifiers"
    )
    consent_timestamp: datetime | None = Field(
        default=None, description="Consent timestamp"
    )
    notes: str | None = Field(default=None, description="Free-text notes")

    def validate_against_spec(
        self, spec: ParticipantMetadataSpec
    ) -> tuple[bool, list[str]]:
        """Validate participant_metadata against a specification.

        Parameters
        ----------
        spec : ParticipantMetadataSpec
            Specification to validate against.

        Returns
        -------
        tuple[bool, list[str]]
            (is_valid, list of error messages)

        Examples
        --------
        >>> from bead.participants.metadata_spec import (
        ...     FieldSpec, ParticipantMetadataSpec
        ... )
        >>> spec = ParticipantMetadataSpec(
        ...     name="test",
        ...     fields=[FieldSpec(name="age", field_type="int", required=True)]
        ... )
        >>> p = Participant(participant_metadata={"age": 25})
        >>> p.validate_against_spec(spec)
        (True, [])
        """
        # Convert JsonValue dict to the expected type for validation
        metadata: dict[str, str | int | float | bool | None] = {}
        for key, value in self.participant_metadata.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                metadata[key] = value
            # Skip complex values (lists, dicts) - they won't match FieldSpec types
        return spec.validate_metadata(metadata)

    def get_attribute(self, key: str, default: JsonValue = None) -> JsonValue:
        """Get a metadata attribute with optional default.

        Parameters
        ----------
        key : str
            Attribute name.
        default : JsonValue
            Default value if attribute not found.

        Returns
        -------
        JsonValue
            Attribute value or default.

        Examples
        --------
        >>> p = Participant(participant_metadata={"age": 25})
        >>> p.get_attribute("age")
        25
        >>> p.get_attribute("unknown", default="N/A")
        'N/A'
        """
        return self.participant_metadata.get(key, default)

    def set_attribute(self, key: str, value: JsonValue) -> None:
        """Set a metadata attribute.

        Parameters
        ----------
        key : str
            Attribute name.
        value : JsonValue
            Attribute value.

        Examples
        --------
        >>> p = Participant()
        >>> p.set_attribute("age", 25)
        >>> p.participant_metadata["age"]
        25
        """
        self.participant_metadata[key] = value
        self.update_modified_time()

    def add_session(self, session_id: str) -> None:
        """Add a session ID to this participant.

        Parameters
        ----------
        session_id : str
            Session identifier to add.

        Examples
        --------
        >>> p = Participant()
        >>> p.add_session("session_001")
        >>> p.session_ids
        ['session_001']
        """
        self.session_ids.append(session_id)
        self.update_modified_time()


class ParticipantIDMapping(BeadBaseModel):
    """Mapping between external participant IDs and internal UUIDs.

    This model is stored SEPARATELY from participant data for IRB/privacy
    compliance. The external ID (e.g., Prolific PID) can be deleted while
    retaining the internal UUID for analysis.

    Attributes
    ----------
    id : UUID
        Unique identifier for this mapping record (inherited).
    external_id : str
        External participant identifier (e.g., Prolific PID).
    external_source : str
        Source of the external ID (e.g., "prolific", "mturk", "sona").
    participant_id : UUID
        Internal participant UUID (references Participant.id).
    mapping_timestamp : datetime
        When this mapping was created.
    is_active : bool
        Whether this mapping is active (for soft deletion).

    Examples
    --------
    >>> from uuid import UUID
    >>> mapping = ParticipantIDMapping(
    ...     external_id="PROLIFIC_ABC123",
    ...     external_source="prolific",
    ...     participant_id=UUID("01234567-89ab-cdef-0123-456789abcdef"),
    ... )
    >>> mapping.external_source
    'prolific'
    """

    external_id: str = Field(..., description="External participant ID")
    external_source: str = Field(..., description="Source of external ID")
    participant_id: UUID = Field(..., description="Internal participant UUID")
    mapping_timestamp: datetime = Field(
        default_factory=now_iso8601, description="When mapping was created"
    )
    is_active: bool = Field(default=True, description="Whether mapping is active")

    @field_validator("external_id", "external_source")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        """Validate string fields are non-empty.

        Parameters
        ----------
        v : str
            String value to validate.

        Returns
        -------
        str
            Validated string.

        Raises
        ------
        ValueError
            If string is empty or whitespace only.
        """
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()

    def deactivate(self) -> None:
        """Soft-delete this mapping (for privacy compliance).

        Sets is_active to False without deleting the record. This allows
        the mapping to be retained for audit purposes while marking it
        as no longer valid.

        Examples
        --------
        >>> from uuid import uuid4
        >>> mapping = ParticipantIDMapping(
        ...     external_id="ABC123",
        ...     external_source="prolific",
        ...     participant_id=uuid4(),
        ... )
        >>> mapping.is_active
        True
        >>> mapping.deactivate()
        >>> mapping.is_active
        False
        """
        self.is_active = False
        self.update_modified_time()
