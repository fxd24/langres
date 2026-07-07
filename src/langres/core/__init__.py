"""
langres.core: Low-level API for entity resolution.

This module provides the foundational primitives for building custom
entity resolution pipelines.

Import weight (W0.4): most names below are cheap (pydantic/rapidfuzz/networkx,
the core dependencies) and stay eager. A handful pull an optional, heavy
dependency -- the embedding/vector stack (torch/sentence-transformers/faiss/
qdrant-client, the ``[semantic]`` extra), the LLM stack (litellm, the ``[llm]``
extra), the trained-judge stack (scikit-learn, the ``[trained]`` extra --
:class:`~langres.core.modules.rf_judge.RFJudge`), or dev/eval tooling
(ranx/optuna/wandb) -- and are resolved lazily via PEP 562 ``__getattr__``
(see :data:`_LAZY_SUBMODULES` / :data:`_LAZY_SYMBOLS` below): ``from
langres.core import VectorBlocker`` still works, but the actual ``import`` of
``faiss``/``torch``/etc. only happens the first time ``VectorBlocker`` (or
another lazy name) is actually accessed -- so plain ``import langres`` stays
fast and never touches ``sys.modules`` for a dependency the caller hasn't
asked for. Accessing a ``[semantic]``/``[llm]``/``[trained]`` symbol without
that extra installed raises a clear ``ImportError`` naming the extra to
install.
"""

import importlib
from typing import TYPE_CHECKING, Any

from langres.core.adapters.glinker import GLinkerAdapter
from langres.core.anchor_store import AnchorStore, ClusterDelta
from langres.core.blocker import Blocker
from langres.core.blockers.all_pairs import AllPairsBlocker
from langres.core.blockers.composite import CompositeBlocker
from langres.core.blockers.key import KeyBlocker
from langres.core.canonicalizer import Canonicalizer
from langres.core.clusterer import Clusterer
from langres.core.clusterers.correlation import CorrelationClusterer
from langres.core.comparator import Comparator, StringComparator
from langres.core.debugging import (
    CandidateStats,
    ClusterStats,
    ErrorExample,
    PipelineDebugger,
    ScoreStats,
)
from langres.core.feature import ComparisonLevel, ComparisonVector, FeatureSpec
from langres.core.fit import SupervisedFitMixin, UnsupervisedFitMixin
from langres.core.groups import ERCandidateGroup, derive_groups_from_pairs
from langres.core.harvest import (
    Correction,
    CorrectionLog,
    LabeledPair,
    derive_threshold_from_pairs,
    harvest_labeled_pairs,
)
from langres.core.judgement_log import JudgementLog, LoggingModule
from langres.core.judges.embedding_score import EmbeddingScoreJudge
from langres.core.judges.fellegi_sunter import FellegiSunterJudge
from langres.core.judges.weighted_average import WeightedAverageJudge
from langres.core.models import (
    CompanySchema,
    EntityProtocol,
    ERCandidate,
    PairwiseJudgement,
)
from langres.core.module import GroupwiseModule, Module, stamp_group_cost
from langres.core.modules.cascade_judge import CascadeJudge
from langres.core.registry import (
    SchemaNotRegistered,
    UnknownComponentType,
    get_component,
    get_schema,
    register,
    register_schema,
)
from langres.core.resolver import Resolver
from langres.core.review import ReviewItem, ReviewQueue, select_for_review
from langres.core.serialization import (
    ARTIFACT_VERSION,
    ArtifactManifest,
    ComponentSpec,
    SerializableState,
)

if TYPE_CHECKING:
    # Only reached by mypy (never at runtime) -- keeps every lazy name visible
    # to `mypy --strict` without executing the heavy imports below.
    from langres.core import benchmark, metrics, optimizers
    from langres.core.blockers.vector import VectorBlocker
    from langres.core.embeddings import (
        EmbeddingProvider,
        FakeEmbedder,
        FakeSparseEmbedder,
        FastEmbedSparseEmbedder,
        SentenceTransformerEmbedder,
        SparseEmbeddingProvider,
    )
    from langres.core.indexes import (
        FAISSIndex,
        FakeHybridVectorIndex,
        FakeVectorIndex,
        QdrantHybridIndex,
        VectorIndex,
    )
    from langres.core.modules.llm_judge import LLMJudge
    from langres.core.modules.rf_judge import RFJudge
    from langres.core.modules.select_judge import SelectJudge

__all__ = [
    "ARTIFACT_VERSION",
    "AllPairsBlocker",
    "AnchorStore",
    "ArtifactManifest",
    "benchmark",
    "Blocker",
    "CandidateStats",
    "Canonicalizer",
    "CascadeJudge",
    "ClusterDelta",
    "ClusterStats",
    "Clusterer",
    "CompanySchema",
    "Comparator",
    "ComparisonLevel",
    "ComparisonVector",
    "ComponentSpec",
    "CompositeBlocker",
    "Correction",
    "CorrectionLog",
    "CorrelationClusterer",
    "derive_groups_from_pairs",
    "derive_threshold_from_pairs",
    "EmbeddingProvider",
    "EmbeddingScoreJudge",
    "EntityProtocol",
    "ERCandidate",
    "ERCandidateGroup",
    "ErrorExample",
    "FAISSIndex",
    "FakeEmbedder",
    "FakeHybridVectorIndex",
    "FakeSparseEmbedder",
    "FakeVectorIndex",
    "FastEmbedSparseEmbedder",
    "FeatureSpec",
    "FellegiSunterJudge",
    "get_component",
    "get_schema",
    "GLinkerAdapter",
    "GroupwiseModule",
    "harvest_labeled_pairs",
    "JudgementLog",
    "KeyBlocker",
    "LabeledPair",
    "LLMJudge",
    "LoggingModule",
    "metrics",
    "Module",
    "optimizers",
    "PairwiseJudgement",
    "PipelineDebugger",
    "QdrantHybridIndex",
    "register",
    "register_schema",
    "Resolver",
    "ReviewItem",
    "ReviewQueue",
    "RFJudge",
    "ScoreStats",
    "SchemaNotRegistered",
    "select_for_review",
    "SelectJudge",
    "SentenceTransformerEmbedder",
    "SerializableState",
    "SparseEmbeddingProvider",
    "stamp_group_cost",
    "StringComparator",
    "SupervisedFitMixin",
    "UnknownComponentType",
    "UnsupervisedFitMixin",
    "VectorBlocker",
    "VectorIndex",
    "WeightedAverageJudge",
]

#: Names resolved to a *submodule of this package* on first access -- unlike
#: :data:`_LAZY_SYMBOLS`, the value ``__getattr__`` binds is the imported
#: module itself (``langres.core.benchmark``, not an attribute of it). All
#: three eventually need ranx (``metrics``, and ``benchmark`` which imports
#: it) or optuna/wandb (``optimizers``) -- dev/eval tooling, not part of the
#: link()/dedupe() runtime path.
_LAZY_SUBMODULES: frozenset[str] = frozenset({"benchmark", "metrics", "optimizers"})

#: ``name -> owning module`` for symbols resolved on first access. Each entry
#: needs an optional extra installed; see :data:`_EXTRA_BY_SYMBOL` for the
#: ``pip install langres[<extra>]`` hint a missing dependency should surface.
_LAZY_SYMBOLS: dict[str, str] = {
    "VectorBlocker": "langres.core.blockers.vector",
    "EmbeddingProvider": "langres.core.embeddings",
    "FakeEmbedder": "langres.core.embeddings",
    "FakeSparseEmbedder": "langres.core.embeddings",
    "FastEmbedSparseEmbedder": "langres.core.embeddings",
    "SentenceTransformerEmbedder": "langres.core.embeddings",
    "SparseEmbeddingProvider": "langres.core.embeddings",
    "FAISSIndex": "langres.core.indexes",
    "FakeHybridVectorIndex": "langres.core.indexes",
    "FakeVectorIndex": "langres.core.indexes",
    "QdrantHybridIndex": "langres.core.indexes",
    "VectorIndex": "langres.core.indexes",
    "LLMJudge": "langres.core.modules.llm_judge",
    "RFJudge": "langres.core.modules.rf_judge",
    "SelectJudge": "langres.core.modules.select_judge",
}

#: ``name -> extra`` for the lazy symbols a ``pip install langres[<extra>]``
#: actually fixes -- everything in :data:`_LAZY_SYMBOLS` except the three
#: submodules (dev/eval tooling, not distributed as a pip extra; see
#: :data:`_LAZY_SUBMODULES`'s docstring). ``RFJudge`` needs scikit-learn (the
#: ``[trained]`` extra, W1.2's trained-family judge); ``LLMJudge``/
#: ``SelectJudge`` need ``[llm]`` (litellm/dspy-ai); everything else needs
#: ``[semantic]`` (embeddings/vector index/VectorBlocker).
_EXTRA_BY_SYMBOL: dict[str, str] = {
    name: {"LLMJudge": "llm", "SelectJudge": "llm", "RFJudge": "trained"}.get(name, "semantic")
    for name in _LAZY_SYMBOLS
}


def __getattr__(name: str) -> Any:
    """PEP 562: resolve a heavy/optional name the first time it's accessed.

    Raises:
        AttributeError: ``name`` isn't a known attribute of this module.
        ImportError: The owning module's dependency isn't installed --
            re-raised with a ``pip install langres[<extra>]`` hint instead of
            the raw ``ModuleNotFoundError`` (:data:`_EXTRA_BY_SYMBOL`).
    """
    if name in _LAZY_SUBMODULES:
        value: Any = importlib.import_module(f"{__name__}.{name}")
    elif name in _LAZY_SYMBOLS:
        try:
            value = getattr(importlib.import_module(_LAZY_SYMBOLS[name]), name)
        except ImportError as exc:
            extra = _EXTRA_BY_SYMBOL[name]
            raise ImportError(
                f"langres.core.{name} requires the {extra!r} extra: "
                f"pip install 'langres[{extra}]' (or uv add 'langres[{extra}]')"
            ) from exc
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value  # cache: subsequent access skips __getattr__
    return value
