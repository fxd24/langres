"""
Blocker base class for candidate generation and schema normalization.

This module provides the abstract base class for all blocking (candidate
generation) implementations in the langres framework.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from langres.core.models import ERCandidate
from langres.core.reports import CandidateInspectionReport

# Generic type variable for schema types (must be a Pydantic model)
SchemaT = TypeVar("SchemaT", bound=BaseModel)


class Blocker(ABC, Generic[SchemaT]):
    """Abstract base class for candidate generation and schema normalization.

    The Blocker is the Data Loader & Transformer of the pipeline. It has two
    critical responsibilities:

    1. **Generate Candidate Pairs**: Efficiently find candidate pairs to avoid
       N² comparisons (e.g., using blocking keys, ANN search, or other techniques)
    2. **Normalize Schema**: Act as the ETL layer, transforming raw data from
       one or more sources into the clean, internal Pydantic schema that the
       Module expects

    Design principles:
    - Separation of concerns: Blocker loads and normalizes; Module compares
    - Streaming first: stream() is a generator for memory efficiency
    - Schema normalization: All data cleaning happens here, before Module
    - High recall: Blocking should have ≥95% recall (don't miss true matches)

    The Blocker is responsible for:
    - Reading raw data from various sources (dicts, DataFrames, databases, etc.)
    - Normalizing field names and types to match the internal schema
    - Generating candidate pairs using efficient blocking techniques
    - Yielding ERCandidate[SchemaT] objects ready for Module comparison

    The Blocker is NOT responsible for:
    - Comparing entities (that's the Module's job)
    - Making match decisions
    - Clustering entities

    Example:
        class SimpleBlocker(Blocker[CompanySchema]):
            '''Blocker that generates all pairs (no blocking).'''

            def stream(self, data):
                # 1. Normalize schema
                companies = [
                    CompanySchema(
                        id=row["company_id"],
                        name=row["company_name"],
                        address=row.get("addr")
                    )
                    for row in data
                ]

                # 2. Generate pairs (naive: all pairs)
                for i, left in enumerate(companies):
                    for right in companies[i + 1:]:
                        yield ERCandidate(
                            left=left,
                            right=right,
                            blocker_name="simple_blocker"
                        )

    Example:
        class VectorBlocker(Blocker[CompanySchema]):
            '''Blocker using ANN for efficient candidate generation.'''

            def __init__(self, embedding_model, k=10):
                self.model = embedding_model
                self.k = k  # Top-k nearest neighbors

            def stream(self, data):
                # 1. Normalize schema
                companies = [
                    CompanySchema(id=row["id"], name=row["name"])
                    for row in data
                ]

                # 2. Build embedding index
                embeddings = [self.model.encode(c.name) for c in companies]
                index = build_ann_index(embeddings)

                # 3. Generate pairs via ANN search
                for i, company in enumerate(companies):
                    neighbor_indices = index.search(embeddings[i], self.k)
                    for j in neighbor_indices:
                        if i < j:  # Avoid duplicates
                            yield ERCandidate(
                                left=companies[i],
                                right=companies[j],
                                blocker_name="vector_blocker"
                            )
    """

    @abstractmethod
    def stream(self, data: list[Any]) -> Iterator[ERCandidate[SchemaT]]:
        """Generate candidate pairs from input data.

        This is the core method that all Blocker implementations must define.
        It takes raw data (typically dicts, but could be DataFrames, database
        rows, etc.) and yields normalized ERCandidate pairs.

        The method should:
        1. Normalize the raw data to the internal SchemaT
        2. Generate candidate pairs using blocking logic
        3. Yield ERCandidate objects with normalized entities

        Args:
            data: List of raw data items (typically dicts). The blocker is
                responsible for understanding the structure and extracting
                the necessary fields to create SchemaT objects.

        Yields:
            ERCandidate[SchemaT] objects containing:
            - left: Normalized entity (SchemaT)
            - right: Normalized entity (SchemaT)
            - blocker_name: Name identifying this blocker

        Note:
            Implementations should be generators (use yield) to support
            streaming/lazy evaluation for large datasets. This allows
            processing millions of records without loading everything into memory.

        Note:
            For deduplication tasks, generate pairs within the same dataset.
            For entity linking tasks, you may need stream_against() to generate
            pairs between two different datasets (source vs. target).

        Note:
            The blocking strategy is crucial for performance. Naive all-pairs
            blocking is O(N²) and doesn't scale. Real implementations should
            use techniques like:
            - Blocking keys (group by last name, postal code, etc.)
            - ANN search (embedding similarity)
            - Sorted neighborhood
            - Q-grams
            - LSH (Locality-Sensitive Hashing)
        """
        pass  # pragma: no cover

    @abstractmethod
    def inspect_candidates(
        self,
        candidates: list[ERCandidate[SchemaT]],
        entities: list[SchemaT],
        sample_size: int = 10,
    ) -> CandidateInspectionReport:
        """Explore candidates without ground truth labels.

        Use this method to understand blocker output before labeling:
        - How many candidates were generated?
        - Distribution of candidates per entity
        - Sample examples with readable entity text

        For quality evaluation with ground truth, use
        PipelineDebugger.analyze_candidates() instead.

        Args:
            candidates: List of candidate pairs generated by blocker
            entities: List of normalized entities (for readable text extraction)
            sample_size: Number of examples to include in report (default: 10)

        Returns:
            CandidateInspectionReport with statistics, distribution, examples,
            and actionable recommendations for parameter tuning.

        Note:
            This method is for exploratory analysis. It does NOT require
            ground truth labels. Use it to progressively build and tune
            your pipeline before expensive labeling work.

        Note:
            Subclasses should implement this to provide blocker-specific
            recommendations (e.g., "increase k_neighbors" for VectorBlocker).
        """
        pass  # pragma: no cover
