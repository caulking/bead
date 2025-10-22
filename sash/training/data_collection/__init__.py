"""Data collection from experimental platforms.

This module provides tools for downloading and merging data from:
- JATOS: Experimental data and results
- Prolific: Participant metadata and demographics

The collected data is used for model training and active learning.
"""

from __future__ import annotations

from sash.training.data_collection.jatos import JATOSDataCollector
from sash.training.data_collection.merger import DataMerger
from sash.training.data_collection.prolific import ProlificDataCollector

__all__ = [
    "JATOSDataCollector",
    "ProlificDataCollector",
    "DataMerger",
]
