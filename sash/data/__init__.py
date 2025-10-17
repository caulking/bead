"""Data infrastructure for sash package.

This module provides core data models, identifiers, timestamps,
and serialization utilities.
"""

from __future__ import annotations

from sash.data.base import SashBaseModel
from sash.data.identifiers import extract_timestamp, generate_uuid, is_valid_uuid7
from sash.data.serialization import (
    DeserializationError,
    SerializationError,
    append_jsonlines,
    read_jsonlines,
    stream_jsonlines,
    write_jsonlines,
)
from sash.data.timestamps import format_iso8601, now_iso8601, parse_iso8601

__all__ = [
    "SashBaseModel",
    "generate_uuid",
    "extract_timestamp",
    "is_valid_uuid7",
    "now_iso8601",
    "parse_iso8601",
    "format_iso8601",
    "write_jsonlines",
    "read_jsonlines",
    "stream_jsonlines",
    "append_jsonlines",
    "SerializationError",
    "DeserializationError",
]
