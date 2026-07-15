"""
langres.core: Low-level API for entity resolution.

This module provides the foundational primitives for building custom
entity resolution pipelines.

Import weight (W0.4): most names below are cheap (pydantic/rapidfuzz/networkx,
the core dependencies) and stay eager. A handful pull an optional, heavy
dependency -- the embedding/vector stack (torch/sentence-transformers/faiss/
qdrant-client, the ``[semantic]`` extra), the LLM stack (litellm, the ``[llm]``
extra), the trained-judge stack (scikit-learn, the ``[trained]`` extra --
:class:`~langres.core.matchers.random_forest_judge.RandomForestMatcher`), or dev/eval tooling
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
from langres.core.fit_report import FitReport
from langres.core.harvest import (
    Correction,
    CorrectionLog,
    LabeledPair,
    align_pairs,
    derive_threshold_from_pairs,
    harvest_labeled_pairs,
)
from langres.core.judgement_log import JudgementLog, LoggingMatcher
from langres.core.matchers.embedding_score import EmbeddingScoreMatcher
from langres.core.matchers.fellegi_sunter import FellegiSunterMatcher
from langres.core.matchers.weighted_average import WeightedAverageMatcher
from langres.core.method_registry import (
    DEFAULT_EMBEDDING_MODEL,
    MethodSpec,
    UnknownMethodError,
    get_method,
    list_methods,
    register_method,
)
from langres.core.methods_api import Method
from langres.core.methods_calibrate import Isotonic, Platt
from langres.core.methods_prompt import Bootstrap, GEPA, MIPRO
from langres.core.models import (
    CompanySchema,
    EntityProtocol,
    ERCandidate,
    MatcherAbstainedError,
    PairwiseJudgement,
    predicted_match,
)
from langres.core.matcher import GroupwiseMatcher, Matcher, stamp_group_cost
from langres.core.matchers.cascade_judge import CascadeMatcher
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
from langres.core.runs import (
    RunContext,
    RunRecord,
    RunStore,
    capture_run,
    compute_recipe_id,
    resolve_store,
)
from langres.core.serialization import (
    ARTIFACT_VERSION,
    ArtifactManifest,
    ComponentSpec,
    SerializableState,
)
from langres.core.trackers import (
    ExperimentTracker,
    MultiTracker,
    NoOpTracker,
    resolve_tracker,
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
    from langres.core.calibration import Calibrator
    from langres.core.matchers.llm_judge import LLMMatcher
    from langres.core.matchers.random_forest_judge import RandomForestMatcher
    from langres.core.matchers.select_judge import SelectMatcher
    from langres.core.trackers import MlflowTracker, WandbTracker

__all__ = [
    "ARTIFACT_VERSION",
    "AllPairsBlocker",
    "AnchorStore",
    "ArtifactManifest",
    "benchmark",
    "Blocker",
    "Calibrator",
    "CandidateStats",
    "Canonicalizer",
    "CascadeMatcher",
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
    "DEFAULT_EMBEDDING_MODEL",
    "derive_groups_from_pairs",
    "derive_threshold_from_pairs",
    "EmbeddingProvider",
    "EmbeddingScoreMatcher",
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
    "FellegiSunterMatcher",
    "get_component",
    "get_method",
    "get_schema",
    "GLinkerAdapter",
    "GroupwiseMatcher",
    "harvest_labeled_pairs",
    "MatcherAbstainedError",
    "JudgementLog",
    "KeyBlocker",
    "LabeledPair",
    "LLMMatcher",
    "list_methods",
    "LoggingMatcher",
    "MethodSpec",
    "metrics",
    "Matcher",
    "optimizers",
    "PairwiseJudgement",
    "PipelineDebugger",
    "predicted_match",
    "QdrantHybridIndex",
    "register",
    "register_method",
    "register_schema",
    "Resolver",
    "ReviewItem",
    "ReviewQueue",
    "RandomForestMatcher",
    "ScoreStats",
    "SchemaNotRegistered",
    "select_for_review",
    "SelectMatcher",
    "SentenceTransformerEmbedder",
    "SerializableState",
    "SparseEmbeddingProvider",
    "stamp_group_cost",
    "StringComparator",
    "SupervisedFitMixin",
    "UnknownComponentType",
    "UnknownMethodError",
    "UnsupervisedFitMixin",
    "VectorBlocker",
    "VectorIndex",
    "WeightedAverageMatcher",
    # Experiment tracking (S1): run identity + persistence + pluggable trackers.
    "capture_run",
    "compute_recipe_id",
    "ExperimentTracker",
    "MlflowTracker",
    "MultiTracker",
    "NoOpTracker",
    "resolve_store",
    "resolve_tracker",
    "RunContext",
    "RunRecord",
    "RunStore",
    "WandbTracker",
    # Training surface: the pairs->candidates bridge, the fit digest, and the
    # method objects passed to Resolver.fit(method=...) (import-light config;
    # dspy/sklearn stay lazy in dspy_judge/calibration).
    "align_pairs",
    "Bootstrap",
    "FitReport",
    "GEPA",
    "Isotonic",
    "Method",
    "MIPRO",
    "Platt",
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
    "Calibrator": "langres.core.calibration",
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
    "LLMMatcher": "langres.core.matchers.llm_judge",
    "RandomForestMatcher": "langres.core.matchers.random_forest_judge",
    "SelectMatcher": "langres.core.matchers.select_judge",
    # Backend tracker adapters (S3/S4): the package's own __getattr__ pulls the
    # concrete adapter -- and its mlflow/wandb dependency -- only on access.
    "MlflowTracker": "langres.core.trackers",
    "WandbTracker": "langres.core.trackers",
}

#: ``name -> extra`` for the lazy symbols a ``pip install langres[<extra>]``
#: actually fixes -- everything in :data:`_LAZY_SYMBOLS` except the three
#: submodules (dev/eval tooling, not distributed as a pip extra; see
#: :data:`_LAZY_SUBMODULES`'s docstring). ``RandomForestMatcher`` needs scikit-learn (the
#: ``[trained]`` extra, W1.2's trained-family judge); ``LLMMatcher``/
#: ``SelectMatcher`` need ``[llm]`` (litellm/dspy-ai); everything else needs
#: ``[semantic]`` (embeddings/vector index/VectorBlocker).
_EXTRA_BY_SYMBOL: dict[str, str] = {
    name: {
        "LLMMatcher": "llm",
        "SelectMatcher": "llm",
        "RandomForestMatcher": "trained",
        "Calibrator": "trained",
        "MlflowTracker": "mlflow",
        "WandbTracker": "wandb",
    }.get(name, "semantic")
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
