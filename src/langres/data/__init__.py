"""Data utilities for langres.

This module provides utilities for loading, splitting, and managing
entity resolution datasets.
"""

from langres.data.loaders import (
    load_iteration_05_data,
    load_labeled_dedup_data,
    load_labeled_dedup_data_legacy,
)
from langres.data.schemas import LabeledDeduplicationDataset, LabeledGroup
from langres.data.splitting import stratified_dedup_split

__all__ = [
    # Schemas
    "LabeledDeduplicationDataset",
    "LabeledGroup",
    # Loaders (new generic API)
    "load_labeled_dedup_data",
    "load_labeled_dedup_data_legacy",
    # Loaders (legacy)
    "load_iteration_05_data",
    # Splitting
    "stratified_dedup_split",
]
