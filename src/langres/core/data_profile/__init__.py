"""Composable data-profile report: the ``ProfileSection`` bag + its container.

The public surface of the data-layer profiler (plan §2). The report is a *bag of
sections* -- one self-contained :class:`ProfileSection` per metric family (label
structure, corpus fields, separability, embeddings, the KPI hero), composed into a
:class:`DataProfileReport` that renders exactly the sections it is given. Build one
explicitly from the profiler functions, or via the convenience constructors
:func:`from_benchmark` / :func:`from_records` (and the ``[semantic]``-gated
:func:`from_embedder` on-ramp).

A leaf, import-light package: numpy + stdlib + ``core.metrics`` + the shared
render scaffold (:mod:`langres.core._report_html`). It *consumes* precomputed
embeddings (an :class:`EmbeddingSource`); it never generates them, so it carries
no ``[semantic]`` dependency (``from_embedder`` imports sentence-transformers
lazily inside its body). ``tests/test_import_budget.py`` locks the budget: a bare
``import langres`` -- and importing this package -- pulls no torch / faiss /
sentence-transformers / litellm.
"""

from langres.core.data_profile.base import DataProfileReport, ProfileSection
from langres.core.data_profile.builders import (
    from_benchmark,
    from_embedder,
    from_records,
)
from langres.core.data_profile.corpus_field import (
    CorpusFieldSection,
    FieldStat,
    profile_corpus_fields,
)
from langres.core.data_profile.embedding_section import (
    EmbeddingComparisonSection,
    EmbeddingSection,
    profile_embedding,
    profile_embedding_comparison,
)
from langres.core.data_profile.embedding_source import (
    ArraySource,
    EmbeddingSource,
    NpySource,
    cosine_signal,
)
from langres.core.data_profile.hero import HeroSection, build_hero
from langres.core.data_profile.label_structure import (
    LabelStructureSection,
    profile_label_structure,
)
from langres.core.data_profile.separability import (
    SeparabilitySection,
    SimilaritySignal,
    profile_separability,
    string_signal,
)

__all__ = [
    # Seam
    "ProfileSection",
    "DataProfileReport",
    # Sections
    "HeroSection",
    "LabelStructureSection",
    "CorpusFieldSection",
    "FieldStat",
    "SeparabilitySection",
    "EmbeddingSection",
    "EmbeddingComparisonSection",
    # Profiler functions
    "build_hero",
    "profile_label_structure",
    "profile_corpus_fields",
    "profile_separability",
    "profile_embedding",
    "profile_embedding_comparison",
    # Signals + embedding sources
    "SimilaritySignal",
    "string_signal",
    "cosine_signal",
    "EmbeddingSource",
    "ArraySource",
    "NpySource",
    # Convenience constructors
    "from_benchmark",
    "from_records",
    "from_embedder",
]
