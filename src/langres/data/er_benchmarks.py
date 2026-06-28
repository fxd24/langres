"""Entity-resolution benchmark adapters (Fodors-Zagat restaurant matching).

Loads the vendored Fodors-Zagat benchmark into a single corpus of
:class:`RestaurantSchema` records plus cross-source ground-truth pairs, and
provides a blocking k-sweep that pins the Pair-Completeness >= 0.95 gate.

The benchmark is a *linkage* task: two sources (Fodor's, Zagat) each list
restaurants, and the matches we care about are cross-source. The k-sweep
therefore filters candidate pairs to cross-source ones before measuring recall
(intra-source pairs are noise for this task; see DESIGN-REVIEW B2).
"""

import csv
import logging
import random
from collections import defaultdict
from importlib import resources
from typing import Literal

from pydantic import BaseModel, computed_field

from langres.core.blockers.vector import VectorBlocker
from langres.core.clusterer import Clusterer
from langres.core.comparator import Comparator
from langres.core.embeddings import SentenceTransformerEmbedder
from langres.core.indexes.vector_index import FAISSIndex
from langres.core.judges.weighted_average import WeightedAverageJudge
from langres.core.metrics import evaluate_blocking
from langres.core.models import ERCandidate
from langres.core.resolver import Resolver

logger = logging.getLogger(__name__)

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
    """Read a packaged benchmark CSV into a list of header-keyed row dicts."""
    text = resources.files(_DATASET_PACKAGE).joinpath(filename).read_text(encoding="utf-8")
    reader = csv.DictReader(text.splitlines())
    return [dict(row) for row in reader]


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
    :class:`~langres.core.judges.weighted_average.WeightedAverageJudge` scoring
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
        module=WeightedAverageJudge(feature_specs=comparator.feature_specs),
        clusterer=Clusterer(threshold=threshold),
    )


def _split_stratum(
    clusters: list[set[str]], test_size: float, rng: random.Random
) -> tuple[list[set[str]], list[set[str]]]:
    """Split one same-size cluster stratum into (train, test) whole clusters.

    Clusters are sorted into a canonical order (by their sorted id tuple) before
    shuffling so the split is deterministic given ``rng`` regardless of input
    ordering. At least one cluster always goes to test (mirrors
    :func:`stratified_dedup_split`).
    """
    ordered = sorted(clusters, key=lambda c: tuple(sorted(c)))
    rng.shuffle(ordered)
    n_test = max(1, int(len(ordered) * test_size))
    return ordered[:-n_test], ordered[-n_test:]


def split_restaurant_corpus(
    corpus: list[RestaurantSchema],
    gold_clusters: list[set[str]],
    *,
    test_size: float = 0.3,
    seed: int = 0,
) -> tuple[list[RestaurantSchema], list[RestaurantSchema], list[set[str]], list[set[str]]]:
    """Stratified, leakage-free train/test split over full restaurant records.

    Mirrors :func:`~langres.data.splitting.stratified_dedup_split`'s
    cluster-size stratification (singletons distributed separately; matched
    groups split within each size band, whole clusters kept together) but
    operates on full :class:`RestaurantSchema` records — preserving ``source``
    and the ``f``/``z`` ids — instead of the ``{id, name}``-only int-cast dicts
    that function produces (which can't reconstruct ``RestaurantSchema``).

    Because whole gold clusters are assigned to one side, no match pair ever
    straddles the split (no test-set leakage), and each returned cluster list
    partitions exactly its split's ids.

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
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be in (0, 1); got {test_size}")

    by_id = {r.id: r for r in corpus}
    rng = random.Random(seed)

    # Stratify: singletons in their own band, matched groups by cluster size.
    singletons = [c for c in gold_clusters if len(c) == 1]
    groups_by_size: dict[int, list[set[str]]] = defaultdict(list)
    for cluster in gold_clusters:
        if len(cluster) >= 2:
            groups_by_size[len(cluster)].append(cluster)

    train_clusters: list[set[str]] = []
    test_clusters: list[set[str]] = []
    # Process strata in a fixed order (singletons, then ascending size) so rng
    # consumption — and thus the split — is reproducible.
    strata = [singletons] + [groups_by_size[size] for size in sorted(groups_by_size)]
    for stratum in strata:
        train_part, test_part = _split_stratum(stratum, test_size, rng)
        train_clusters.extend(train_part)
        test_clusters.extend(test_part)

    train_ids = {rid for cluster in train_clusters for rid in cluster}
    test_ids = {rid for cluster in test_clusters for rid in cluster}
    train_records = [by_id[rid] for rid in sorted(train_ids)]
    test_records = [by_id[rid] for rid in sorted(test_ids)]
    logger.info(
        "split_restaurant_corpus: %d train records (%d clusters), %d test records (%d clusters)",
        len(train_records),
        len(train_clusters),
        len(test_records),
        len(test_clusters),
    )
    return train_records, test_records, train_clusters, test_clusters


def _cross_source(
    candidates: list[ERCandidate[RestaurantSchema]],
) -> list[ERCandidate[RestaurantSchema]]:
    """Keep only candidate pairs whose two records come from different sources."""
    return [c for c in candidates if c.left.source != c.right.source]


def sweep_blocking_k(
    corpus: list[RestaurantSchema],
    gold_clusters: list[set[str]],
    ks: tuple[int, ...] = (5, 10, 20, 30, 50),
) -> dict[int, float]:
    """Measure cross-source Pair-Completeness of vector blocking across ``ks``.

    Builds the FAISS index once over ``embed_text`` and reuses it across all
    ``k`` (only ``k_neighbors`` changes). For each ``k`` the candidates are
    filtered to cross-source pairs (DESIGN-REVIEW B2) before recall is measured
    via :func:`evaluate_blocking`.

    Args:
        corpus: Combined record list from :func:`load_fodors_zagat`.
        gold_clusters: Cross-source match sets from :func:`load_fodors_zagat`.
        ks: Neighbor counts to sweep.

    Returns:
        Mapping of ``k`` to cross-source Pair-Completeness (``candidate_recall``).
    """
    embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
    index = FAISSIndex(embedder=embedder, metric="cosine")
    index.create_index([r.embed_text for r in corpus])
    records = [r.model_dump() for r in corpus]

    recalls: dict[int, float] = {}
    for k in ks:
        # Construct a fresh blocker per k (the pre-built FAISS index is reused,
        # so this is cheap) rather than mutating k_neighbors in place.
        blocker: VectorBlocker[RestaurantSchema] = VectorBlocker(
            vector_index=index,
            schema=RestaurantSchema,
            text_field="embed_text",
            k_neighbors=k,
        )
        candidates = _cross_source(list(blocker.stream(records)))
        recall = evaluate_blocking(candidates, gold_clusters).candidate_recall
        recalls[k] = recall
        logger.info("blocking k=%d -> cross-source recall=%.4f", k, recall)
    return recalls


def pick_blocking_k(recalls: dict[int, float], threshold: float = RECALL_GATE) -> int:
    """Pick the smallest ``k`` whose recall clears ``threshold``.

    If no ``k`` reaches ``threshold``, returns the ``k`` with the highest recall
    (the honest best-effort fallback; callers should document the shortfall
    rather than fake the gate).

    Args:
        recalls: Mapping of ``k`` to recall, e.g. from :func:`sweep_blocking_k`.
        threshold: Minimum acceptable recall.

    Returns:
        The chosen ``k``.

    Raises:
        ValueError: If ``recalls`` is empty.
    """
    if not recalls:
        raise ValueError("recalls is empty; nothing to pick from")
    passing = [k for k in sorted(recalls) if recalls[k] >= threshold]
    if passing:
        return passing[0]
    return max(recalls, key=lambda k: recalls[k])
