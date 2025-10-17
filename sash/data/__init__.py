"""Data infrastructure for sash package.

This module provides core data models, identifiers, timestamps,
serialization utilities, metadata tracking, repository pattern,
and validation utilities.
"""

from __future__ import annotations

from sash.data.base import SashBaseModel
from sash.data.identifiers import extract_timestamp, generate_uuid, is_valid_uuid7
from sash.data.metadata import (
    MetadataTracker,
    ProcessingRecord,
    ProvenanceRecord,
)
from sash.data.repository import Repository
from sash.data.serialization import (
    DeserializationError,
    SerializationError,
    append_jsonlines,
    read_jsonlines,
    stream_jsonlines,
    write_jsonlines,
)
from sash.data.timestamps import format_iso8601, now_iso8601, parse_iso8601
from sash.data.validation import (
    ValidationReport,
    validate_jsonlines_file,
    validate_provenance_chain,
    validate_uuid_references,
)

__all__ = [
    # Base model
    "SashBaseModel",
    # Identifiers
    "generate_uuid",
    "extract_timestamp",
    "is_valid_uuid7",
    # Timestamps
    "now_iso8601",
    "parse_iso8601",
    "format_iso8601",
    # Serialization
    "write_jsonlines",
    "read_jsonlines",
    "stream_jsonlines",
    "append_jsonlines",
    "SerializationError",
    "DeserializationError",
    # Metadata
    "MetadataTracker",
    "ProvenanceRecord",
    "ProcessingRecord",
    # Repository
    "Repository",
    # Validation
    "ValidationReport",
    "validate_jsonlines_file",
    "validate_uuid_references",
    "validate_provenance_chain",
]
