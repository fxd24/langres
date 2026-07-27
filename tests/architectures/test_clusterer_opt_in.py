"""The ``clusterer=`` opt-in on the two architectures the docs sell.

``CorrelationClusterer`` measures better than the shipped transitive-closure
default (``docs/research/20260727_closure_diagnostic.md``), but until now
*opting into it was impossible* on :class:`~langres.architectures.FuzzyString`
and :class:`~langres.architectures.vector_llm_cascade.VectorLLMCascade`: neither
constructor took a ``clusterer=``, and the slot was hard-built inside
``_topology``. The only way to get one was to abandon the named architecture and
hand-wire a raw ``ERModel``.

The default is deliberately NOT flipped. So the load-bearing test here is the
*negative* one -- that a caller who passes nothing gets byte-identical behaviour
to before -- with the opt-in proved on top of it.

``VectorLLMCascade``'s ``_topology`` pulls faiss/sentence-transformers/litellm,
so this file only exercises the parts of it that are free: the constructor and
``_build_clusterer``. Wiring the slot is identical code in both classes and is
proved end-to-end on the $0 ``FuzzyString``.
"""

from __future__ import annotations

import pytest

from langres.architectures import FuzzyString, VectorLLMCascade
from langres.core.clusterer import Clusterer
from langres.core.clusterers.correlation import CorrelationClusterer

# The classic over-merge chain. MEASURED scores from FuzzyString's own
# comparator, not guessed: 1-2 = 0.8889, 2-3 = 0.5714, 1-3 = 0.5000. So at
# _CHAIN_THRESHOLD the two chain edges clear and the direct 1-3 edge does not --
# transitive closure merges all three, pivot merges only {1, 2}.
_CHAIN = [
    {"id": "1", "name": "Acme Corporation"},
    {"id": "2", "name": "Acme Corporation Ltd"},
    {"id": "3", "name": "Acme Ltd"},
]
_CHAIN_THRESHOLD = 0.55


# ---------------------------------------------------------------------------
# The default is untouched
# ---------------------------------------------------------------------------


def test_default_clusterer_is_still_the_base_transitive_closure_clusterer() -> None:
    """No ``clusterer=`` -> exactly the base ``Clusterer``, not a subclass."""
    model = FuzzyString(schema=None)
    model.dedupe(_CHAIN)

    assert type(model.clusterer) is Clusterer


def test_default_clusterer_gets_the_models_threshold() -> None:
    """The pre-existing wiring, pinned: ``threshold=`` reaches the clusterer."""
    model = FuzzyString(threshold=0.42)
    model.dedupe(_CHAIN)

    assert model.clusterer.threshold == 0.42


def test_passing_nothing_is_identical_to_never_having_the_parameter() -> None:
    """The zero-behaviour-change guarantee, stated as an output comparison."""
    explicit_none = FuzzyString(threshold=0.6, clusterer=None).dedupe(_CHAIN)
    omitted = FuzzyString(threshold=0.6).dedupe(_CHAIN)

    assert sorted(map(sorted, explicit_none)) == sorted(map(sorted, omitted))
    assert explicit_none.threshold == omitted.threshold
    assert explicit_none.architecture == omitted.architecture


def test_vector_llm_cascade_default_clusterer_is_unchanged() -> None:
    """Same for the paid architecture -- without importing its heavy topology."""
    model = VectorLLMCascade(llm="openrouter/openai/gpt-4o-mini", threshold=0.7)

    built = model._build_clusterer()

    assert type(built) is Clusterer
    assert built.threshold == 0.7


# ---------------------------------------------------------------------------
# The opt-in actually works
# ---------------------------------------------------------------------------


def test_fuzzy_string_accepts_a_correlation_clusterer() -> None:
    """The whole point of the track: this call was impossible before."""
    model = FuzzyString(clusterer=CorrelationClusterer())
    model.dedupe(_CHAIN)

    assert type(model.clusterer) is CorrelationClusterer


def test_vector_llm_cascade_accepts_a_correlation_clusterer() -> None:
    model = VectorLLMCascade(llm="openrouter/openai/gpt-4o-mini", clusterer=CorrelationClusterer())

    assert type(model._build_clusterer()) is CorrelationClusterer


def test_opting_in_changes_the_clustering_not_the_architecture() -> None:
    """Pivot refuses the chain merge that closure makes -- and stays a FuzzyString.

    The measured difference this opt-in exists to deliver, at the front door.
    """
    closure = FuzzyString(threshold=_CHAIN_THRESHOLD).dedupe(_CHAIN)
    pivot = FuzzyString(threshold=_CHAIN_THRESHOLD, clusterer=CorrelationClusterer()).dedupe(_CHAIN)

    assert sorted(map(sorted, closure)) == [["1", "2", "3"]]  # chained through 2
    assert sorted(map(sorted, pivot)) == [["1", "2"]]  # no direct 1-3 edge
    # Swapping the clusterer must NOT mint a new architecture.
    assert pivot.architecture == closure.architecture == "FuzzyString"


def test_opting_in_does_not_introduce_singleton_clusters() -> None:
    """Record 3 is unmerged, so it is ABSENT -- the same shape closure returns.

    A user opting into a better clusterer must not silently get a different
    output *shape*; ``dedupe()`` is documented to return the multi-record
    clusters and leave singletons out.
    """
    pivot = FuzzyString(threshold=_CHAIN_THRESHOLD, clusterer=CorrelationClusterer()).dedupe(_CHAIN)

    assert "3" not in {rid for cluster in pivot for rid in cluster}
    assert all(len(cluster) > 1 for cluster in pivot)


# ---------------------------------------------------------------------------
# ``threshold=`` stays the single match cut
# ---------------------------------------------------------------------------


def test_model_threshold_overrides_the_passed_clusterers_own_threshold() -> None:
    """One argument, one meaning: ``clusterer=`` picks the algorithm, not the cut.

    Otherwise ``FuzzyString(threshold=0.6, clusterer=CorrelationClusterer())``
    would silently cut at the clusterer's default 0.5 while
    ``DedupeResult.threshold`` reported 0.6.
    """
    model = FuzzyString(threshold=0.6, clusterer=CorrelationClusterer(threshold=0.99))
    result = model.dedupe(_CHAIN)

    assert model.clusterer.threshold == 0.6
    assert result.threshold == 0.6


def test_the_passed_clusterer_instance_is_not_mutated() -> None:
    """The caller's object is cloned, never rewritten under them."""
    passed = CorrelationClusterer(threshold=0.99)
    model = FuzzyString(threshold=0.6, clusterer=passed)
    model.dedupe(_CHAIN)

    assert passed.threshold == 0.99
    assert model.clusterer is not passed


def test_an_out_of_range_threshold_still_raises() -> None:
    """Cloning goes through ``__init__``, so validation is not skipped.

    A plain ``copy(clusterer); clusterer.threshold = t`` would have silently
    accepted this.
    """
    model = FuzzyString(threshold=5.0, clusterer=CorrelationClusterer())

    with pytest.raises(ValueError, match="threshold"):
        model.dedupe(_CHAIN)
