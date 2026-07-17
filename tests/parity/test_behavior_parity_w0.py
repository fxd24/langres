"""W0 behavior-parity net for epic #193 (the core-algebra breaking rewrite).

**The safety net a multi-wave breaking refactor is measured against.** It runs
the CURRENT pipeline end-to-end on a frozen, offline, ``$0`` record set
(:mod:`tests.parity._fixture_records`) and asserts byte-identical golden output.
Every later wave re-runs this file: if a wave claims to preserve behavior, these
goldens prove it did -- and if a wave means to change behavior, the golden diff
is the deliberate, reviewable record of exactly what moved.

What is captured (all offline, deterministic, ``$0`` -- no network, no paid call):

- **``FuzzyString().dedupe``** -- the ``$0`` offline architecture and the PRIMARY
  parity target. Its clusters + self-describing result metadata + BCubed/pairwise
  metrics against the hand-labeled gold.
- **``FuzzyString().compare``** -- the pair front door: a known duplicate and a
  known non-duplicate, snapshotting every ``LinkVerdict`` field.
- **``Resolver.from_schema(matcher="string")`` pair-level judgements** -- the
  tightest net for #193's W1, which is a *carrier* refactor: the per-pair
  ``PairwiseJudgement`` (ids, score, score_type, decision, decision_step) sorted
  canonically, so a change to how a pair is scored cannot hide behind clustering.

What is deferred:

- **``VectorLLMCascade``** parity -- see :func:`test_vector_llm_cascade_parity_is_deferred`.
  It needs the ``[semantic]`` extra (faiss + a sentence-transformers embedder) and
  a deterministic, download-free embedder plus a ``DummyLM`` harness; not available
  offline in this environment, so a stable snapshot is a later-wave task.

Regenerate the goldens on purpose (a deliberate re-baseline) with::

    LANGRES_PARITY_UPDATE=1 uv run pytest tests/parity --no-cov
"""

from __future__ import annotations

import importlib.util

import pytest

from langres.architectures import FuzzyString
from langres.core.resolver import Resolver
from langres.metrics.metrics import (
    calculate_bcubed_metrics,
    calculate_pairwise_metrics,
)

from tests.parity._fixture_records import GOLD_CLUSTERS, RECORDS, ParityBusinessW0
from tests.parity._golden import canonical_clusters, check_golden

_STRING_THRESHOLD = 0.7


def _record(record_id: str) -> dict[str, object]:
    """The one fixture record with ``id == record_id`` (fixture ids are unique)."""
    return next(record for record in RECORDS if record["id"] == record_id)


def test_fuzzy_string_default_dedupe() -> None:
    """FuzzyString() at its default threshold -- the primary $0 offline snapshot."""
    result = FuzzyString().dedupe(RECORDS)
    payload = {
        "architecture": result.architecture,
        "backbone": result.backbone,
        "score_type": result.score_type,
        "threshold": result.threshold,
        "clusters": canonical_clusters(result),
        "bcubed": calculate_bcubed_metrics(list(result), GOLD_CLUSTERS),
        "pairwise": calculate_pairwise_metrics(list(result), GOLD_CLUSTERS),
    }
    check_golden("fuzzy_string_default_dedupe", payload)


def test_fuzzy_string_compare_verdicts() -> None:
    """FuzzyString().compare on a known duplicate and a known non-duplicate.

    Binds an explicit schema so the LinkVerdict is over a stable, typed pipeline
    (the same one the legacy artifact persists), and snapshots every field.
    """
    model = FuzzyString(schema=ParityBusinessW0)
    verdicts = []
    for left_id, right_id in (("b01", "b02"), ("b01", "b09")):
        verdict = model.compare(_record(left_id), _record(right_id))
        verdicts.append(
            {
                "left_id": left_id,
                "right_id": right_id,
                "match": verdict.match,
                "score": verdict.score,
                "score_type": verdict.score_type,
                "threshold": verdict.threshold,
                "architecture": verdict.architecture,
                "backbone": verdict.backbone,
            }
        )
    check_golden("fuzzy_string_compare_verdicts", {"verdicts": verdicts})


def test_resolver_string_pair_level() -> None:
    """Per-pair PairwiseJudgement parity -- the tightest net for the W1 carrier refactor.

    Snapshots every blocked pair's judgement (canonically sorted), plus the
    clusters and metrics they produce, so a refactor that changes what a pair
    scores is caught at the pair level rather than only if it survives to alter
    the final clusters.
    """
    resolver = Resolver.from_schema(ParityBusinessW0, matcher="string", threshold=_STRING_THRESHOLD)
    judgements = sorted(resolver.predict(RECORDS), key=lambda j: (j.left_id, j.right_id))
    pairs = [
        {
            "left_id": judgement.left_id,
            "right_id": judgement.right_id,
            "score": judgement.score,
            "score_type": judgement.score_type,
            "decision": judgement.decision,
            "decision_step": judgement.decision_step,
        }
        for judgement in judgements
    ]
    clusters = resolver.resolve(RECORDS)
    payload = {
        "threshold": resolver.clusterer.threshold,
        "n_pairs": len(pairs),
        "pairs": pairs,
        "clusters": canonical_clusters(clusters),
        "bcubed": calculate_bcubed_metrics(clusters, GOLD_CLUSTERS),
        "pairwise": calculate_pairwise_metrics(clusters, GOLD_CLUSTERS),
    }
    check_golden("resolver_string_pairs", payload)


def test_vector_llm_cascade_parity_is_deferred() -> None:
    """Record -- visibly, not silently -- that cascade parity is a later wave's job.

    A ``VectorLLMCascade`` snapshot needs the ``[semantic]`` extra (faiss + a
    sentence-transformers embedder) and the ``[llm]`` extra, plus a deterministic,
    download-free embedder and a ``DummyLM``/mock harness. None of that is
    available offline here, and an embedder model download would make the
    snapshot non-``$0`` and platform-dependent. So this deferral is explicit.
    """
    semantic = all(
        importlib.util.find_spec(module) is not None
        for module in ("faiss", "sentence_transformers")
    )
    llm = importlib.util.find_spec("litellm") is not None
    pytest.skip(
        "VectorLLMCascade parity deferred to a later wave "
        f"([semantic] installed={semantic}, [llm] installed={llm}): a stable "
        "snapshot needs a deterministic, download-free embedder + a DummyLM "
        "harness, which is not available offline in this environment."
    )
