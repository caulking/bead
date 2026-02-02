"""Participant metadata system for bead.

This module provides data models and utilities for managing participant
metadata with privacy-preserving external ID mapping. It supports:

- Configurable metadata fields with validation (FieldSpec, ParticipantMetadataSpec)
- Participant data models with UUID-based identification (Participant)
- Privacy-compliant external ID mapping (ParticipantIDMapping)
- Collection classes with JSONL I/O (ParticipantCollection, IDMappingCollection)
- DataFrame merge utilities for analysis (merge_participant_metadata, etc.)

All DataFrame operations support both pandas and polars backends.

Examples
--------
>>> from bead.participants import (
...     Participant,
...     ParticipantCollection,
...     FieldSpec,
...     ParticipantMetadataSpec,
... )

>>> # Define metadata schema
>>> spec = ParticipantMetadataSpec(
...     name="study_demographics",
...     fields=[
...         FieldSpec(name="age", field_type="int", required=True),
...         FieldSpec(
...             name="education",
...             field_type="str",
...             allowed_values=["high_school", "bachelors", "masters", "phd"],
...         ),
...     ],
... )

>>> # Create participant with metadata
>>> p = Participant(
...     participant_metadata={"age": 25, "education": "bachelors"},
...     study_id="study_001",
... )

>>> # Validate against spec
>>> is_valid, errors = p.validate_against_spec(spec)
>>> is_valid
True
"""

from bead.participants.collection import IDMappingCollection, ParticipantCollection
from bead.participants.merging import (
    build_participant_lookup,
    create_analysis_dataframe,
    merge_participant_metadata,
    resolve_external_ids,
)
from bead.participants.metadata_spec import FieldSpec, ParticipantMetadataSpec
from bead.participants.models import Participant, ParticipantIDMapping

__all__ = [
    # Models
    "Participant",
    "ParticipantIDMapping",
    # Metadata specification
    "FieldSpec",
    "ParticipantMetadataSpec",
    # Collections
    "ParticipantCollection",
    "IDMappingCollection",
    # Merge utilities
    "merge_participant_metadata",
    "resolve_external_ids",
    "create_analysis_dataframe",
    "build_participant_lookup",
]
