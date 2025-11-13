"""Data utilities for langres.

This module provides utilities for loading, splitting, and managing
entity resolution datasets.
"""

from langres.data.loaders import load_labeled_dedup_data
from langres.data.schemas import LabeledDeduplicationDataset, LabeledGroup
from langres.data.splitting import stratified_dedup_split

__all__ = [
    # Schemas
    "LabeledDeduplicationDataset",
    "LabeledGroup",
    # Loaders
    "load_labeled_dedup_data",
    # Splitting
    "stratified_dedup_split",
]
