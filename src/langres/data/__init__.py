"""Data utilities for langres.

This module provides utilities for loading, splitting, and managing
entity resolution datasets.
"""

from langres.training.finetune import LabeledCandidate
from langres.data._benchmark_utils import BenchmarkDataNotFoundError
from langres.data.loaders import load_labeled_dedup_data
from langres.data.mining import (
    augment_by_attribute,
    denoise_pairs,
    flip_pairs,
    mine_misclassified_pairs,
    sample_negative_pairs,
)
from langres.data.registry import (
    BenchmarkEntry,
    get_benchmark,
    list_benchmarks,
    list_methods,
)
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
    # Training-pair mining (sklearn stays lazy inside the featurizing miners)
    "LabeledCandidate",
    "mine_misclassified_pairs",
    "sample_negative_pairs",
    "augment_by_attribute",
    "flip_pairs",
    "denoise_pairs",
    # Benchmark data availability (corpora are git-checkout-only, not in the wheel)
    "BenchmarkDataNotFoundError",
    # Benchmark registry (import-light manifest)
    "BenchmarkEntry",
    "get_benchmark",
    "list_benchmarks",
    "list_methods",
]
