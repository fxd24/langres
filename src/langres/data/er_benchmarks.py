"""Entity-resolution benchmark adapters (Fodors-Zagat restaurant matching).

Loads the vendored Fodors-Zagat benchmark into a single corpus of
:class:`RestaurantSchema` records plus cross-source ground-truth pairs, and
provides a blocking k-sweep that pins the Pair-Completeness >= 0.95 gate.

The benchmark is a *linkage* task: two sources (Fodor's, Zagat) each list
restaurants, and the matches we care about are cross-source. The k-sweep
therefore filters candidate pairs to cross-source ones before measuring recall
(intra-source pairs are noise for this task; see DESIGN-REVIEW B2).
"""

import logging
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, Field, computed_field

from langres.core.benchmark import (
    Benchmark,
    complete_partition,
    gold_pairs_from_clusters,
)
from langres.core.benchmark import (
    evaluate_resolver_bcubed as _generic_evaluate_resolver_bcubed,
)
from langres.core.blocker import Blocker
from langres.core.blockers.all_pairs import register_schema_idempotent
from langres.core.blockers.vector import VectorBlocker
from langres.core.clusterer import Clusterer
from langres.core.comparator import Comparator
from langres.core.embeddings import SentenceTransformerEmbedder
from langres.core.indexes.vector_index import FAISSIndex
from langres.core.matchers.weighted_average import WeightedAverageMatcher
from langres.core.metrics import evaluate_blocking
from langres.core.models import ERCandidate
from langres.core.resolver import Resolver
from langres.data import _benchmark_utils as _bu

logger = logging.getLogger(__name__)

# ``complete_partition`` is re-exported above (it moved to the dataset-agnostic
# ``langres.core.benchmark`` harness); importers that still do
# ``from langres.data.er_benchmarks import complete_partition`` keep working.
__all__ = [
    "DEFAULT_BLOCKING_K",
    "DEFAULT_THRESHOLD_GRID",
    "RECALL_GATE",
    "BCubedEvalResult",
    "FodorsZagatBenchmark",
    "RestaurantSchema",
    "build_restaurant_blocker",
    "build_restaurant_resolver",
    "complete_partition",
    "evaluate_resolver_bcubed",
    "load_fodors_zagat",
    "pick_blocking_k",
    "split_restaurant_corpus",
    "sweep_blocking_k",
    "tune_threshold_on_train",
]

_DATASET_PACKAGE = "langres.data.datasets.fodors_zagat"
_FODORS_FILE = "fodors.csv"
_ZAGATS_FILE = "zagats.csv"
_MAPPING_FILE = "fodors-zagats_perfectMapping.csv"

RestaurantSource = Literal["fodors", "zagat"]

# Pinned blocking k that meets Pair-Completeness >= 0.95 on the cross-source
# Fodors-Zagat matches, measured with VectorBlocker over
# SentenceTransformer("all-MiniLM-L6-v2") cosine similarity on ``embed_text``.
#
# Measured sweep (cross-source Pair-Completeness, 112 gold pairs):
#   k= 5 -> 0.9911
#   k=10 -> 0.9911
#   k=20 -> 0.9911
#   k=30 -> 0.9911
#   k=50 -> 0.9911
# Recall is flat: 111/112 pairs are captured within each record's top-5
# cross-source neighbours, so larger k adds no new true matches. The single
# missed pair (f640/z325, "masa's") has *identical* embed_text in both sources;
# VectorBlocker's position-based self-skip drops one exact-duplicate neighbour,
# so this pair is structurally missed at every k (a pre-existing blocker edge
# case, out of scope here). The 0.95 gate is still cleared at the minimum k.
# DEFAULT_BLOCKING_K is the MIN k clearing 0.95 (see ``pick_blocking_k``).
DEFAULT_BLOCKING_K = 5

#: Pair-Completeness gate the blocking k-sweep must clear (DESIGN-REVIEW W7).
RECALL_GATE = 0.95

#: Candidate Clusterer thresholds swept by :func:`tune_threshold_on_train`. The
#: WeightedAverageMatcher scores in ``[0, 1]``; this small grid brackets the
#: useful range without an expensive fine sweep (tuned on TRAIN only).
DEFAULT_THRESHOLD_GRID: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


class RestaurantSchema(BaseModel):
    """A single restaurant record from the Fodors-Zagat benchmark.

    ``id`` is globally unique across both sources (the loader prefixes the raw
    source id with ``f``/``z``). ``embed_text`` is the serializable blocking
    text used by the :class:`VectorBlocker` (referenced as ``text_field``).

    Attributes:
        id: Globally-unique record id (e.g. ``"f534"`` / ``"z219"``).
        name: Restaurant name (always present).
        addr: Street address, if present.
        city: City, if present.
        phone: Phone number, if present.
        type: Cuisine / category, if present.
        source: Originating guide (``"fodors"`` or ``"zagat"``).
    """

    id: str
    name: str
    addr: str | None = None
    city: str | None = None
    phone: str | None = None
    type: str | None = None
    source: RestaurantSource

    @computed_field  # type: ignore[prop-decorator]
    @property
    def embed_text(self) -> str:
        """Blocking text: name, city and address joined by spaces.

        Used as the :class:`VectorBlocker` ``text_field`` and as the text fed to
        the vector index. Omits missing components so absent fields don't inject
        empty tokens.
        """
        return " ".join(p for p in [self.name, self.city, self.addr] if p)


# Register RestaurantSchema at import time so a fresh process that only does
# ``import langres.data.er_benchmarks`` (e.g. to ``Resolver.load`` a saved
# artifact and ``resolve``) finds the schema in the registry without first
# constructing a blocker. The declarative VectorBlocker re-registers idempotently
# in :func:`build_restaurant_blocker`, so this is the single source of truth for
# the registry key and round-trips the saved artifact's ``schema_type_name``.
register_schema_idempotent(RestaurantSchema)


def _unquote(value: str) -> str:
    """Strip the dataset's wrapping single quotes and unescape inner quotes.

    The source wraps text fields in single quotes and escapes inner quotes as
    ``\\'`` (e.g. ``'art\\'s delicatessen'``). Numeric / unquoted fields pass
    through unchanged.

    Args:
        value: A raw CSV cell.

    Returns:
        The cleaned cell value.
    """
    value = value.strip()
    if len(value) >= 2 and value.startswith("'") and value.endswith("'"):
        value = value[1:-1]
    return value.replace("\\'", "'")


def _read_csv_rows(filename: str) -> list[dict[str, str]]:
    """Read a packaged Fodors-Zagat CSV into a list of header-keyed row dicts."""
    return _bu.read_csv_rows(_DATASET_PACKAGE, filename)


def _record_from_row(
    row: dict[str, str], source: RestaurantSource, prefix: str
) -> RestaurantSchema:
    """Build a :class:`RestaurantSchema` from a raw CSV row dict."""

    def field(name: str) -> str | None:
        cleaned = _unquote(row.get(name, ""))
        return cleaned or None

    return RestaurantSchema(
        id=f"{prefix}{_unquote(row['id'])}",
        name=field("name") or "",
        addr=field("addr"),
        city=field("city"),
        phone=field("phone"),
        type=field("type"),
        source=source,
    )


def load_fodors_zagat() -> tuple[list[RestaurantSchema], list[set[str]]]:
    """Load the Fodors-Zagat benchmark as one corpus plus its complete partition.

    Both sources are combined into a single corpus of 864 records with globally
    unique, source-prefixed ids (``f<id>`` for Fodor's, ``z<id>`` for Zagat).
    Ground truth comes from the explicit ``perfectMapping`` file: each row yields
    a 2-element match set ``{f<fodors_id>, z<zagats_id>}``.

    The returned ``gold_clusters`` is the **complete closed-world partition** of
    the corpus: the 2-element match clusters PLUS a singleton ``{id}`` for every
    record that is not in any match. This matters because Fodors-Zagat is fully
    labeled -- any cross-source pair absent from the mapping is a *known*
    non-match, not an unknown. Consumers that score teacher pairs only when both
    ids appear in some cluster (e.g.
    :meth:`~langres.bootstrap.report.BootstrapReport.build`) would otherwise drop
    every pair touching an unmatched record, hiding the teacher's false positives
    and silently shrinking the evaluation set. Singletons add no positive pairs,
    so blocking pair-completeness is unaffected.

    Returns:
        ``(corpus, gold_clusters)`` where ``corpus`` is the combined record list
        and ``gold_clusters`` is the complete partition (match sets + singletons).
    """
    corpus: list[RestaurantSchema] = [
        _record_from_row(row, "fodors", "f") for row in _read_csv_rows(_FODORS_FILE)
    ]
    corpus += [_record_from_row(row, "zagat", "z") for row in _read_csv_rows(_ZAGATS_FILE)]

    match_clusters: list[set[str]] = [
        {f"f{_unquote(row['fodors_id'])}", f"z{_unquote(row['zagats_id'])}"}
        for row in _read_csv_rows(_MAPPING_FILE)
    ]
    matched_ids = {rid for cluster in match_clusters for rid in cluster}
    # Closed-world completion: every unmatched record is its own singleton, so
    # downstream scoring treats it as a known non-match rather than "no truth".
    singletons: list[set[str]] = [{r.id} for r in corpus if r.id not in matched_ids]
    gold_clusters = match_clusters + singletons

    logger.info(
        "Loaded Fodors-Zagat: %d records (%d fodors + %d zagat), "
        "%d gold pairs, %d clusters (%d singletons)",
        len(corpus),
        sum(1 for r in corpus if r.source == "fodors"),
        sum(1 for r in corpus if r.source == "zagat"),
        len(match_clusters),
        len(gold_clusters),
        len(singletons),
    )
    return corpus, gold_clusters


def build_restaurant_blocker(
    k_neighbors: int = DEFAULT_BLOCKING_K,
) -> VectorBlocker[RestaurantSchema]:
    """Build the shared restaurant VectorBlocker (MiniLM + FAISS-cosine).

    This is the one blocking config used across M1 and M2 — extracted here so
    the gold-set run, the k-sweep, and the M2 Resolver all wire the *same*
    candidate generator rather than each re-spelling it. Declarative
    (``schema=`` + ``text_field=``) so the resulting blocker is
    config-serializable: it round-trips through a saved Resolver artifact.

    Each call constructs a *fresh* (unbuilt) :class:`FAISSIndex`; embedding only
    happens when the index is later populated from a corpus (e.g. by
    ``Resolver.resolve``). :func:`sweep_blocking_k` intentionally does **not**
    use this factory: it shares one pre-built index across every ``k`` to embed
    the corpus once, whereas this factory owns its own index per call.

    Args:
        k_neighbors: Nearest neighbours per record. Defaults to the pinned
            :data:`DEFAULT_BLOCKING_K` (clears Pair-Completeness >= 0.95).

    Returns:
        A :class:`VectorBlocker` over ``RestaurantSchema.embed_text``.
    """
    return VectorBlocker(
        vector_index=FAISSIndex(
            embedder=SentenceTransformerEmbedder("all-MiniLM-L6-v2"),
            metric="cosine",
        ),
        schema=RestaurantSchema,
        text_field="embed_text",
        k_neighbors=k_neighbors,
    )


def build_restaurant_resolver(threshold: float, k_neighbors: int = DEFAULT_BLOCKING_K) -> Resolver:
    """Wire the M2 baseline restaurant Resolver (vector-blocked, zero-spend).

    Composes the shared :func:`build_restaurant_blocker` with the missing-aware
    :class:`~langres.core.comparator.StringComparator` (auto-derived from
    ``RestaurantSchema``) and the registered, zero-spend
    :class:`~langres.core.matchers.weighted_average.WeightedAverageMatcher` scoring
    on the same FeatureSpecs, then a connected-components
    :class:`~langres.core.clusterer.Clusterer` at ``threshold``. Mirrors
    ``Resolver.from_schema``'s comparator+judge+clusterer wiring but swaps the
    default all-pairs blocker for the vector blocker.

    ``Comparator.from_schema`` derives one feature per ``str | None`` field and
    already excludes ``id``, the computed ``embed_text``, and ``source`` (a
    ``Literal``, not a string). Excluding ``source`` is essential: Fodors-Zagat's
    true matches are all cross-source, so comparing ``source`` would penalise
    every positive. The resulting Resolver is fully serializable (every slot is a
    registered component), so ``save()`` does not raise.

    Args:
        threshold: Clusterer match threshold (tune on train, score on test).
        k_neighbors: Blocking neighbours; defaults to :data:`DEFAULT_BLOCKING_K`.

    Returns:
        A ready-to-run, serializable :class:`Resolver`.
    """
    comparator: Comparator[RestaurantSchema] = Comparator.from_schema(RestaurantSchema)
    return Resolver(
        blocker=build_restaurant_blocker(k_neighbors),
        comparator=comparator,
        # The judge scores on the same FeatureSpecs the comparator compares on,
        # so the comparison vector and the weights line up.
        matcher=WeightedAverageMatcher(feature_specs=comparator.feature_specs),
        clusterer=Clusterer(threshold=threshold),
    )


def split_restaurant_corpus(
    corpus: list[RestaurantSchema],
    gold_clusters: list[set[str]],
    *,
    test_size: float = 0.3,
    seed: int = 0,
) -> tuple[list[RestaurantSchema], list[RestaurantSchema], list[set[str]], list[set[str]]]:
    """Stratified, leakage-free train/test split over full restaurant records.

    Thin restaurant-typed wrapper over the shared
    :func:`langres.data._benchmark_utils.stratified_corpus_split`: cluster-size
    stratification (singletons distributed separately; matched groups split within
    each size band, whole clusters kept together) over full
    :class:`RestaurantSchema` records — preserving ``source`` and the ``f``/``z``
    ids. Because whole gold clusters are assigned to one side, no match pair ever
    straddles the split (no test-set leakage).

    Args:
        corpus: Records from :func:`load_fodors_zagat` (the complete corpus).
        gold_clusters: The complete closed-world partition (match sets +
            singletons) from :func:`load_fodors_zagat`.
        test_size: Fraction of each stratum assigned to test (in ``(0, 1)``).
        seed: Seed for the deterministic shuffle.

    Returns:
        ``(train_records, test_records, train_clusters, test_clusters)``.

    Raises:
        ValueError: If ``test_size`` is not in the open interval ``(0, 1)``.
    """
    return _bu.stratified_corpus_split(corpus, gold_clusters, test_size=test_size, seed=seed)


def sweep_blocking_k(
    corpus: list[RestaurantSchema],
    gold_clusters: list[set[str]],
    ks: tuple[int, ...] = (5, 10, 20, 30, 50),
) -> dict[int, float]:
    """Measure cross-source Pair-Completeness of vector blocking across ``ks``.

    Restaurant-typed wrapper over the shared
    :func:`langres.data._benchmark_utils.sweep_blocking_k`, binding
    ``RestaurantSchema`` + ``embed_text``: builds the FAISS index once and reuses
    it across all ``k``, filtering to cross-source pairs (DESIGN-REVIEW B2) before
    measuring recall.

    Args:
        corpus: Combined record list from :func:`load_fodors_zagat`.
        gold_clusters: Cross-source match sets from :func:`load_fodors_zagat`.
        ks: Neighbor counts to sweep.

    Returns:
        Mapping of ``k`` to cross-source Pair-Completeness (``candidate_recall``).
    """
    return _bu.sweep_blocking_k(
        corpus, gold_clusters, RestaurantSchema, text_field="embed_text", ks=ks
    )


def pick_blocking_k(recalls: dict[int, float], threshold: float = RECALL_GATE) -> int:
    """Pick the smallest ``k`` whose recall clears ``threshold`` (default the FZ gate).

    Restaurant-typed wrapper over
    :func:`langres.data._benchmark_utils.pick_blocking_k` that defaults
    ``threshold`` to the Fodors-Zagat :data:`RECALL_GATE` (0.95). If no ``k``
    reaches it, returns the ``k`` with the highest recall (honest fallback).

    Args:
        recalls: Mapping of ``k`` to recall, e.g. from :func:`sweep_blocking_k`.
        threshold: Minimum acceptable recall (defaults to :data:`RECALL_GATE`).

    Returns:
        The chosen ``k``.

    Raises:
        ValueError: If ``recalls`` is empty.
    """
    return _bu.pick_blocking_k(recalls, threshold)


# ---------------------------------------------------------------------------
# Held-out BCubed evaluation (M2 Wave 2)
# ---------------------------------------------------------------------------


class BCubedEvalResult(BaseModel):
    """BCubed scores of a resolver on one record split, plus two diagnostics.

    Attributes:
        precision: BCubed precision against the (closed-world) truth partition.
        recall: BCubed recall against the truth partition.
        f1: BCubed F1 (harmonic mean of precision and recall).
        pair_completeness: Cross-source blocking recall on this split — the
            fraction of true match pairs the resolver's blocker even surfaced as
            candidates. It caps achievable recall, so it is reported every run.
        sanity_floor_f1: BCubed F1 of the all-singletons prediction (the resolver
            merging *nothing*). A meaningful run must clear this floor; a resolver
            scoring at it has added no value over "every record is unique". On a
            sparse dataset with many true singletons this floor can be high (e.g.
            ~0.93 on Fodors-Zagat, which is mostly singletons) — that's expected,
            not a weak floor; what matters is that ``f1`` clears it.
    """

    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    f1: float = Field(ge=0.0, le=1.0)
    pair_completeness: float = Field(ge=0.0, le=1.0)
    sanity_floor_f1: float = Field(ge=0.0, le=1.0)


def _cross_source_pair_completeness(
    blocker: Blocker[RestaurantSchema],
    record_dicts: list[dict[str, object]],
    truth_clusters: list[set[str]],
) -> float:
    """Cross-source blocking Pair-Completeness for one split.

    Streams candidates from an already-built ``blocker``, filters to cross-source
    pairs (Fodors-Zagat true matches are all cross-source; intra-source pairs are
    noise — same rule as :func:`sweep_blocking_k`), and measures candidate recall
    against ``truth_clusters``.
    """
    candidates: list[ERCandidate[RestaurantSchema]] = list(blocker.stream(record_dicts))
    return evaluate_blocking(_bu.cross_source(candidates), truth_clusters).candidate_recall


def evaluate_resolver_bcubed(
    resolver: Resolver,
    test_records: list[RestaurantSchema],
    test_truth_clusters: list[set[str]],
) -> BCubedEvalResult:
    """Score a built resolver on a held-out split with BCubed + diagnostics.

    Pure function (no global state): it runs ``resolver.resolve`` on the records,
    completes the predicted partition with singletons, and scores BCubed against
    the closed-world ``test_truth_clusters``. It also reports the all-singletons
    sanity floor and the resolver's own cross-source blocking Pair-Completeness —
    the latter by reusing ``resolver.blocker`` (its index is already built by the
    ``resolve`` call above, so no re-embedding), keeping the measured blocking
    identical to the one used in resolution.

    Args:
        resolver: A built (serializable) Resolver, e.g. from
            :func:`build_restaurant_resolver`.
        test_records: The held-out records (full ``RestaurantSchema`` objects).
        test_truth_clusters: The closed-world truth partition restricted to the
            test split (match sets + singletons), from
            :func:`split_restaurant_corpus`.

    Returns:
        A :class:`BCubedEvalResult` with precision/recall/F1, Pair-Completeness,
        and the sanity floor.
    """
    # Delegate the dataset-agnostic BCubed + sanity-floor scoring to the core
    # harness (which runs ``resolver.resolve`` and builds the index), then add the
    # restaurant-specific cross-source Pair-Completeness by reusing the now-built
    # ``resolver.blocker`` — keeping the measured blocking identical to resolution.
    track = _generic_evaluate_resolver_bcubed(resolver, test_records, test_truth_clusters)
    record_dicts = [r.model_dump() for r in test_records]
    pair_completeness = _cross_source_pair_completeness(
        resolver.blocker, record_dicts, test_truth_clusters
    )

    return BCubedEvalResult(
        precision=track.bcubed_p,
        recall=track.bcubed_r,
        f1=track.bcubed_f1,
        pair_completeness=pair_completeness,
        sanity_floor_f1=track.sanity_floor_f1,
    )


def tune_threshold_on_train(
    train_records: list[RestaurantSchema],
    train_clusters: list[set[str]],
    *,
    thresholds: Sequence[float] = DEFAULT_THRESHOLD_GRID,
    k_neighbors: int = DEFAULT_BLOCKING_K,
) -> float:
    """Select the Clusterer threshold maximizing BCubed F1 on the TRAIN split.

    Builds a fresh :func:`build_restaurant_resolver` per candidate threshold,
    scores it on ``train_records`` via :func:`evaluate_resolver_bcubed`, and
    returns the threshold with the highest train F1. The **test split is never
    touched here** — this is the no-leakage tuning step. Ties keep the first
    (lowest) threshold in ``thresholds`` order.

    Note: each candidate builds a fresh resolver and re-embeds the train corpus,
    even though only ``Clusterer.threshold`` varies (the embeddings and candidate
    pairs are identical across sweeps). At POC / Fodors-Zagat scale this is fine;
    for larger corpora, share one built blocker across thresholds and vary only
    the Clusterer.

    Args:
        train_records: Training records (full ``RestaurantSchema`` objects).
        train_clusters: Closed-world truth partition for the train split.
        thresholds: Candidate Clusterer thresholds to sweep.
        k_neighbors: Blocking neighbours for the built resolvers.

    Returns:
        The best-performing threshold by train BCubed F1.

    Raises:
        ValueError: If ``thresholds`` is empty.
    """
    # Restaurant-specific loop (kept thin and self-contained rather than
    # delegating to the core generic tuner) so it scores via *this* module's
    # ``evaluate_resolver_bcubed`` — the cross-source-PC-aware wrapper. The
    # generic resolver-factory tuner lives in ``langres.core.benchmark`` and is
    # what ``run_method`` uses; the FZ ``run_method`` parity test confirms the two
    # paths agree on the held-out numbers.
    if not thresholds:
        raise ValueError("thresholds is empty; nothing to tune over")

    best_threshold = thresholds[0]
    best_f1 = -1.0
    for threshold in thresholds:
        resolver = build_restaurant_resolver(threshold, k_neighbors=k_neighbors)
        result = evaluate_resolver_bcubed(resolver, train_records, train_clusters)
        logger.info("threshold=%.2f -> train BCubed F1=%.4f", threshold, result.f1)
        if result.f1 > best_f1:
            best_f1 = result.f1
            best_threshold = threshold
    return best_threshold


class FodorsZagatBenchmark(Benchmark[RestaurantSchema]):
    """Fodors-Zagat as a :class:`~langres.core.benchmark.Benchmark` conformer.

    Adapts the restaurant loaders/splitters to the dataset-agnostic harness so
    :func:`~langres.core.benchmark.run_method` can run any resolver factory
    against it: ``load`` returns the corpus, the closed-world gold partition, and
    the gold match pairs; ``split`` delegates to :func:`split_restaurant_corpus`.
    The canonical resolver factory is :func:`build_restaurant_resolver`, passed
    separately to ``run_method``.

    It also conforms to ``langres.methods.BlockingBenchmark`` by exposing its
    record ``schema`` and pinned blocking config (``blocking_k`` +
    :meth:`build_blocker`), so the method registry can race *any* of the five
    methods against it on the identical candidate set.
    """

    name = "fodors_zagat"
    threshold_grid = DEFAULT_THRESHOLD_GRID
    #: Record type, exposed for the method registry's Comparator/rapidfuzz fields.
    schema = RestaurantSchema
    #: Pinned blocking neighbour count (blocking held constant across methods).
    blocking_k = DEFAULT_BLOCKING_K

    def build_blocker(self, k_neighbors: int) -> VectorBlocker[RestaurantSchema]:
        """Return a fresh restaurant VectorBlocker (MiniLM + FAISS-cosine).

        Delegates to :func:`build_restaurant_blocker` so the race shares the exact
        blocking config used elsewhere; each call yields a fresh, unbuilt index.
        """
        return build_restaurant_blocker(k_neighbors)

    def load(self) -> tuple[list[RestaurantSchema], list[set[str]], set[frozenset[str]]]:
        """Return ``(corpus, gold_clusters, gold_pairs)`` for Fodors-Zagat."""
        corpus, gold_clusters = load_fodors_zagat()
        return corpus, gold_clusters, gold_pairs_from_clusters(gold_clusters)

    def split(
        self,
        corpus: list[RestaurantSchema],
        gold_clusters: list[set[str]],
        *,
        seed: int,
    ) -> tuple[list[RestaurantSchema], list[RestaurantSchema], list[set[str]], list[set[str]]]:
        """Leakage-free stratified split, delegating to :func:`split_restaurant_corpus`."""
        return split_restaurant_corpus(corpus, gold_clusters, seed=seed)
