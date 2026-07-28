"""Tests for the FEBRL3 deduplication benchmark — the registry's first ``task="dedup"``.

Beyond the shared loader contract, these pin the properties that make this a
*dedup* benchmark rather than a relabelled linkage one: a single source (no
``source`` field to filter candidates by), gold clusters larger than 2, and gold
labels that are entity membership rather than the transitive closure of a
pairwise link file.
"""

import collections

import pytest

from langres.data import _benchmark_utils as _bu
from langres.data.benchmark import gold_pairs_from_clusters
from langres.data.febrl_dedup import (
    ACHIEVED_PC_AT_DEFAULT_K,
    DEDUP_RECALL_GATE,
    DEFAULT_DEDUP_BLOCKING_K,
    GATE_MET,
    PC_REGRESSION_FLOOR,
    FebrlDedupBenchmark,
    FebrlDedupSchema,
    build_dedup_blocker,
    load_febrl_dedup,
    pick_blocking_k,
    sweep_blocking_k,
)
from langres.data.registry import get_benchmark, list_benchmarks
from tests.data._loader_contract import assert_loader_contract

#: Pinned vendored counts (see datasets/febrl_dedup/SOURCE.md).
EXPECTED_RECORDS = 5000
EXPECTED_CLUSTERS = 2000
EXPECTED_GOLD_PAIRS = 6538
#: entities per gold-cluster size — the distribution that makes this a dedup task.
EXPECTED_SIZE_HISTOGRAM = {1: 835, 2: 368, 3: 256, 4: 212, 5: 161, 6: 168}


# --- the shared loader contract -------------------------------------------------


def test_satisfies_the_shared_loader_contract() -> None:
    """Protocol conformance, id scheme, closed-world partition, leakage-free split."""
    assert_loader_contract(
        FebrlDedupBenchmark(),
        expected_corpus_size=EXPECTED_RECORDS,
        expected_gold_pairs=EXPECTED_GOLD_PAIRS,
    )


# --- what makes it a DEDUP benchmark --------------------------------------------


def test_registry_entry_is_the_dedup_task() -> None:
    """``febrl_dedup`` is registered as ``dedup`` — and is the only such entry.

    The registry declared ``BenchmarkTask = Literal["linkage", "dedup"]`` while
    every entry was ``linkage``, so ``dedupe()`` — the primary shipped verb — had
    no benchmark. If a second dedup dataset lands, relax the count; do not delete
    the assertion, or the portfolio can silently drift back to linkage-only.
    """
    by_name = {e.name: e for e in list_benchmarks()}
    assert by_name["febrl_dedup"].task == "dedup"
    assert by_name["febrl_dedup"].domain == "person"
    assert by_name["febrl_dedup"].loadable is True
    assert [e.name for e in list_benchmarks() if e.task == "dedup"] == ["febrl_dedup"]


def test_records_carry_no_source_field() -> None:
    """A single-source corpus: there is no side to filter candidates by.

    This is the structural difference from ``febrl_person`` (FEBRL4), whose
    records carry ``source`` and whose adapter drops every intra-source candidate.
    Here an intra-source pair is exactly what a match *is*.
    """
    assert "source" not in FebrlDedupSchema.model_fields
    corpus, _clusters, _pairs = load_febrl_dedup()
    assert not any(hasattr(record, "source") for record in corpus)


def test_cross_source_filter_cannot_be_applied_to_this_corpus() -> None:
    """The linkage filter fails loudly here rather than silently emptying the set.

    Guards the reason ``_bu.sweep_blocking_k`` grew ``cross_source_only=``: if a
    future edit routed this dataset through the linkage path, this is the failure
    it would hit — not a quiet 0.0 recall.
    """
    corpus, _clusters, _pairs = load_febrl_dedup()
    from langres.core.models import ERCandidate

    candidate = ERCandidate(left=corpus[0], right=corpus[1], score=1.0, blocker_name="test")
    with pytest.raises(AttributeError):
        _bu.cross_source([candidate])


def test_gold_clusters_exceed_pair_size() -> None:
    """Clusters run 1–6, so BCubed and the clusterer's closure are exercised.

    Not a uniqueness claim — ``dblp_scholar`` (37) and ``amazon_google`` (6) also
    exceed two. The difference is provenance: theirs are connected components of
    a cross-source link file, these are the generator's own entities.
    """
    _corpus, gold_clusters, _pairs = load_febrl_dedup()
    histogram = dict(collections.Counter(len(c) for c in gold_clusters))
    assert histogram == EXPECTED_SIZE_HISTOGRAM
    assert len(gold_clusters) == EXPECTED_CLUSTERS
    assert max(histogram) == 6


def test_gold_pairs_are_derived_from_membership_not_a_link_file() -> None:
    """Gold pairs come from the entity partition, so the two can never disagree.

    The vendored ground truth is ``record_id,cluster_id`` membership, *not* a
    pairwise link file whose transitive closure would have to be taken — that
    construction inherits the link file's errors and fuses unrelated records into
    giant components (this repo's DBLP-Scholar 37-record component), inflating
    every cluster-based metric. See ``datasets/febrl_dedup/SOURCE.md``, whose
    generation step asserts the upstream pair index equals these 6538 pairs.
    """
    _corpus, gold_clusters, gold_pairs = load_febrl_dedup()
    assert gold_pairs == gold_pairs_from_clusters(gold_clusters)
    assert len(gold_pairs) == EXPECTED_GOLD_PAIRS


def test_ids_are_opaque_and_do_not_leak_the_entity() -> None:
    """Ids must not encode the entity number the way upstream FEBRL ids do.

    Upstream, ``rec-1496-org`` and ``rec-1496-dup-1`` share the entity number, so
    keeping those ids would hand a schema-less ``dedupe()`` the label. Two records
    in the same gold cluster must therefore have unrelated ids.
    """
    corpus, gold_clusters, _pairs = load_febrl_dedup()
    assert all(record.id.startswith("r") and record.id[1:].isdigit() for record in corpus)
    assert not any("org" in record.id or "dup" in record.id for record in corpus)
    multi = [sorted(c) for c in gold_clusters if len(c) > 1]
    # No cluster is a run of consecutive ids (which would make position a label).
    assert not any(
        all(int(b[1:]) - int(a[1:]) == 1 for a, b in zip(ids, ids[1:], strict=True))
        for ids in multi
    )


# --- schema ---------------------------------------------------------------------


def test_embed_text_composition_order() -> None:
    record = FebrlDedupSchema(id="r0", given_name="ada", surname="lovelace", suburb="richmond")
    assert record.embed_text == "ada lovelace richmond"


def test_embed_text_omits_missing_fields() -> None:
    record = FebrlDedupSchema(id="r0", given_name="ada", suburb=None)
    assert record.embed_text == "ada"


def test_embed_text_serializes_as_computed_field() -> None:
    record = FebrlDedupSchema(id="r0", given_name="ada", surname="lovelace")
    assert record.model_dump()["embed_text"] == "ada lovelace"


def test_blank_cells_load_as_none() -> None:
    """FEBRL blanks fields as a corruption; empty cells must be ``None``, not ``""``."""
    corpus, _clusters, _pairs = load_febrl_dedup()
    blanked = [r for r in corpus if r.given_name is None]
    assert blanked, "expected some records with a corrupted-away given_name"
    assert all(r.given_name is None for r in blanked)


# --- benchmark conformer --------------------------------------------------------


def test_benchmark_exposes_pinned_config() -> None:
    """Pins the blocking ``k`` to the LITERAL 50, not to its own constant.

    Asserting ``blocking_k == DEFAULT_DEDUP_BLOCKING_K`` would be circular: both
    sides move together, so editing the constant to 30 would keep this green
    *and* keep the slow Pair-Completeness test green (it sweeps at whatever the
    constant says, and k=30's 0.9440 still clears the 0.93 floor). The literal is
    what makes changing the pin require a deliberate re-measurement — it is the
    documented 0.9552 value's only anchor.
    """
    benchmark = FebrlDedupBenchmark()
    assert benchmark.name == "febrl_dedup"
    assert benchmark.schema is FebrlDedupSchema
    assert DEFAULT_DEDUP_BLOCKING_K == 50, (
        "the pinned k changed; re-run sweep_blocking_k and update "
        "ACHIEVED_PC_AT_DEFAULT_K + the sweep table in the module docstring"
    )
    assert benchmark.blocking_k == DEFAULT_DEDUP_BLOCKING_K
    assert benchmark.threshold_grid == (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


def test_registry_resolves_the_benchmark() -> None:
    benchmark = get_benchmark("febrl_dedup")
    assert isinstance(benchmark, FebrlDedupBenchmark)
    assert benchmark.name == "febrl_dedup"


def test_split_is_deterministic_and_stratifies_every_cluster_size() -> None:
    """Both sides see clusters of every size, so neither is a pairs-only split."""
    benchmark = FebrlDedupBenchmark()
    corpus, gold_clusters, _pairs = benchmark.load()
    train, test, train_clusters, test_clusters = benchmark.split(corpus, gold_clusters, seed=0)
    again = benchmark.split(corpus, gold_clusters, seed=0)
    assert [r.id for r in train] == [r.id for r in again[0]]
    assert [r.id for r in test] == [r.id for r in again[1]]
    assert {len(c) for c in train_clusters} == set(EXPECTED_SIZE_HISTOGRAM)
    assert {len(c) for c in test_clusters} == set(EXPECTED_SIZE_HISTOGRAM)


def test_pinned_blocking_constants_are_self_consistent() -> None:
    """The recorded gate outcome must follow from the recorded measurement.

    ``GATE_MET`` is derived, so this checks the *floor* the slow test asserts is
    genuinely below the measured value (a floor above it could never fail) and
    genuinely above chance (a floor at 0 would never fail either).
    """
    assert GATE_MET is (ACHIEVED_PC_AT_DEFAULT_K >= DEDUP_RECALL_GATE)
    assert GATE_MET is True
    assert PC_REGRESSION_FLOOR < ACHIEVED_PC_AT_DEFAULT_K
    assert PC_REGRESSION_FLOOR > 0.5


def test_pick_blocking_k_defaults_to_the_dedup_gate() -> None:
    assert pick_blocking_k({10: 0.90, 50: 0.96, 80: 0.97}) == 50
    assert pick_blocking_k({10: 0.90, 50: 0.94}) == 50  # honest fallback: best available
    assert pick_blocking_k({10: 0.90, 50: 0.94}, threshold=0.85) == 10


def test_pick_blocking_k_raises_on_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        pick_blocking_k({})


# --- slow: real embeddings ------------------------------------------------------


@pytest.mark.slow
def test_build_dedup_blocker_returns_fresh_unbuilt_vector_blocker() -> None:
    first = build_dedup_blocker()
    second = build_dedup_blocker()
    assert type(first).__name__ == "VectorBlocker"
    assert first.k_neighbors == DEFAULT_DEDUP_BLOCKING_K
    assert first.vector_index is not second.vector_index


@pytest.mark.slow
def test_blocking_pair_completeness_holds_its_regression_floor() -> None:
    """Pair-Completeness at the pinned ``k`` must not regress.

    Asserts :data:`PC_REGRESSION_FLOOR` rather than :data:`DEDUP_RECALL_GATE`
    deliberately: the measured 0.9552 clears the 0.95 gate by less than the known
    cross-platform MiniLM spread (``febrl_person`` measured over 1.6pp), so
    asserting the gate would test the CI runner, not the dataset. The floor sits
    between the k=10 (0.9137) and k=30 (0.9440) measurements, so what it catches
    is a *gross* failure — a broken ``embed_text``, a corrupted fixture, or a
    cross-source filter reintroduced here (≈0.0, since no two records share a
    source). Small ``k`` drift is pinned separately, by
    ``test_benchmark_exposes_pinned_config``.
    """
    corpus, gold_clusters, _pairs = load_febrl_dedup()
    recalls = sweep_blocking_k(corpus, gold_clusters, ks=(DEFAULT_DEDUP_BLOCKING_K,))
    assert recalls[DEFAULT_DEDUP_BLOCKING_K] >= PC_REGRESSION_FLOOR


@pytest.mark.slow
def test_dedupe_is_evaluable_end_to_end_with_cluster_metrics() -> None:
    """The whole point: race a method through the harness and get BCubed out.

    Uses the $0 offline ``rapidfuzz`` matcher, so this costs nothing and needs no
    key. The assertion is against ``sanity_floor_f1`` — the BCubed F1 of
    predicting all singletons — not an absolute number: all-singletons precision
    is 1.0 and its recall is ``n_clusters / n_records``, so a do-nothing resolver
    already scores ``2r/(1+r)`` = 0.5721 on this split (597/1490). An absolute
    threshold would hide that. Beating the floor is the honest claim.
    """
    from langres.benchmarks.runner import run_methods

    table = run_methods(get_benchmark("febrl_dedup"), ["rapidfuzz"], seed=0)
    (result,) = table.results
    assert result.dataset == "febrl_dedup"
    assert result.pipeline.bcubed_f1 > result.pipeline.sanity_floor_f1
    assert result.pipeline.delta_above_floor > 0.3
    # Pair-level track is reported too, and must not be mistaken for the cluster
    # score: on multi-record clusters they diverge (that is the dedup signal).
    assert result.pair is not None
    assert 0.0 <= result.pair.f1 <= 1.0


@pytest.mark.slow
def test_the_dedupe_front_door_itself_scores_on_this_benchmark() -> None:
    """``ERModel.dedupe`` — the literal verb — not just the harness's ``resolve``.

    ``run_methods`` above goes through ``resolver.resolve``. This closes the gap
    to the documented front door: ``dedupe()`` on the benchmark's own held-out
    split, scored with BCubed against the gold partition. It is the claim the
    benchmark exists to support, so it is asserted rather than assumed.
    """
    from langres.core.resolver import Resolver
    from langres.data.benchmark import complete_partition
    from langres.metrics.metrics import calculate_bcubed_metrics

    benchmark = FebrlDedupBenchmark()
    corpus, gold_clusters, _pairs = benchmark.load()
    _train, test, _train_clusters, test_clusters = benchmark.split(corpus, gold_clusters, seed=0)

    model = Resolver.from_schema(FebrlDedupSchema, matcher="string", threshold=0.6)
    model.blocker = build_dedup_blocker(benchmark.blocking_k)
    result = model.dedupe([record.model_dump() for record in test])

    # A DedupeResult is a list[set[str]] that also reports what produced it.
    assert result.backbone is None, "the $0 string path must report no weighted backbone"
    assert result.threshold == 0.6

    predicted = complete_partition(list(result), [record.id for record in test])
    scores = calculate_bcubed_metrics(predicted, test_clusters)
    floor = calculate_bcubed_metrics([{r.id} for r in test], test_clusters)
    assert scores["f1"] > floor["f1"] + 0.3, (
        f"dedupe() BCubed F1 {scores['f1']:.4f} must clear the all-singletons "
        f"floor {floor['f1']:.4f} by a wide margin"
    )
