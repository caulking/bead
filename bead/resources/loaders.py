"""Lexicon loading utilities for various data formats.

This module provides class methods for loading Lexicon objects from
various data formats (CSV, TSV) with flexible column mapping.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from bead.data.language_codes import LanguageCode
from bead.resources.lexical_item import LexicalItem
from bead.resources.lexicon import Lexicon


def from_csv(
    path: str | Path,
    name: str,
    *,
    column_mapping: dict[str, str] | None = None,
    feature_columns: list[str] | None = None,
    attribute_columns: list[str] | None = None,
    language_code: LanguageCode | None = None,
    description: str | None = None,
    **csv_kwargs: Any,
) -> Lexicon:
    """Load lexicon from CSV file with flexible column mapping.

    Parameters
    ----------
    path : str | Path
        Path to the CSV file.
    name : str
        Name for the lexicon.
    column_mapping : dict[str, str] | None
        Mapping from CSV column names to LexicalItem field names.
        Supported fields: "lemma", "pos", "form", "source".
        If None, assumes CSV columns match field names exactly.
        Example: {"word": "lemma", "part_of_speech": "pos"}
    feature_columns : list[str] | None
        CSV column names to map to LexicalItem.features.
        Values are stored as feature key-value pairs.
        Example: ["number", "tense", "countability"]
    attribute_columns : list[str] | None
        CSV column names to map to LexicalItem.attributes.
        Values are stored as attribute key-value pairs.
        Example: ["semantic_class", "frequency"]
    language_code : LanguageCode | None
        ISO 639-1 or ISO 639-3 language code for the lexicon.
    description : str | None
        Optional description of the lexicon.
    **csv_kwargs : Any
        Additional keyword arguments passed to pandas.read_csv().

    Returns
    -------
    Lexicon
        New lexicon loaded from CSV.

    Raises
    ------
    ValueError
        If required "lemma" column/mapping is missing.
    FileNotFoundError
        If CSV file does not exist.

    Examples
    --------
    Basic usage with column mapping:
    >>> lexicon = from_csv(
    ...     "bleached_nouns.csv",
    ...     "nouns",
    ...     column_mapping={"word": "lemma"},
    ...     feature_columns=["number", "countability"],
    ...     attribute_columns=["semantic_class"],
    ...     language_code="eng"
    ... )  # doctest: +SKIP

    With default column names:
    >>> lexicon = from_csv(
    ...     "verbs.csv",
    ...     "verbs",
    ...     feature_columns=["tense"],
    ...     language_code="eng"
    ... )  # doctest: +SKIP
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    # Read CSV
    df = pd.read_csv(file_path, **csv_kwargs)

    # Set up column mapping
    mapping = column_mapping or {}
    reverse_mapping = {v: k for k, v in mapping.items()}

    # Check for required lemma column
    lemma_col = reverse_mapping.get("lemma", "lemma")
    if lemma_col not in df.columns:
        raise ValueError(
            f"CSV must have a 'lemma' column or provide column_mapping. "
            f"Available columns: {list(df.columns)}"
        )

    # Create lexicon
    lexicon = Lexicon(
        name=name,
        description=description,
        language_code=language_code,
    )

    # Process each row
    for _, row in df.iterrows():
        # Extract base fields
        item_data: dict[str, Any] = {}

        # Map standard fields
        for target_field, source_col in [
            ("lemma", reverse_mapping.get("lemma", "lemma")),
            ("pos", reverse_mapping.get("pos", "pos")),
            ("form", reverse_mapping.get("form", "form")),
            ("source", reverse_mapping.get("source", "source")),
        ]:
            if source_col in df.columns and pd.notna(row[source_col]):
                item_data[target_field] = row[source_col]

        # Extract features
        if feature_columns:
            features = {}
            for col in feature_columns:
                if col in df.columns and pd.notna(row[col]):
                    features[col] = row[col]
            if features:
                item_data["features"] = features

        # Extract attributes
        if attribute_columns:
            attributes = {}
            for col in attribute_columns:
                if col in df.columns and pd.notna(row[col]):
                    attributes[col] = row[col]
            if attributes:
                item_data["attributes"] = attributes

        # Add language code if provided
        if language_code:
            item_data["language_code"] = language_code

        # Create and add item
        item = LexicalItem(**item_data)
        lexicon.add(item)

    return lexicon


def from_tsv(
    path: str | Path,
    name: str,
    *,
    column_mapping: dict[str, str] | None = None,
    feature_columns: list[str] | None = None,
    attribute_columns: list[str] | None = None,
    language_code: LanguageCode | None = None,
    description: str | None = None,
    **tsv_kwargs: Any,
) -> Lexicon:
    r"""Load lexicon from TSV file with flexible column mapping.

    This is a convenience wrapper around from_csv() that sets sep="\t".

    Parameters
    ----------
    path : str | Path
        Path to the TSV file.
    name : str
        Name for the lexicon.
    column_mapping : dict[str, str] | None
        Mapping from TSV column names to LexicalItem field names.
    feature_columns : list[str] | None
        TSV column names to map to LexicalItem.features.
    attribute_columns : list[str] | None
        TSV column names to map to LexicalItem.attributes.
    language_code : LanguageCode | None
        ISO 639-1 or ISO 639-3 language code for the lexicon.
    description : str | None
        Optional description of the lexicon.
    **tsv_kwargs : Any
        Additional keyword arguments passed to pandas.read_csv().

    Returns
    -------
    Lexicon
        New lexicon loaded from TSV.

    Examples
    --------
    >>> lexicon = from_tsv(
    ...     "verbs.tsv",
    ...     "verbs",
    ...     feature_columns=["tense", "aspect"],
    ...     language_code="eng"
    ... )  # doctest: +SKIP
    """
    return from_csv(
        path=path,
        name=name,
        column_mapping=column_mapping,
        feature_columns=feature_columns,
        attribute_columns=attribute_columns,
        language_code=language_code,
        description=description,
        sep="\t",
        **tsv_kwargs,
    )
