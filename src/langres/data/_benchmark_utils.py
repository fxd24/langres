"""Shared, dataset-agnostic helpers for the ER benchmark adapters (M3 W3.5).

Both :mod:`langres.data.er_benchmarks` (Fodors-Zagat) and
:mod:`langres.data.amazon_google` (Amazon-Google) need the *same* primitives: a
packaged-CSV reader, a cross-source candidate filter, a vector-blocking k-sweep
and picker, and a leakage-free stratified cluster-level split. This module owns
that genuinely-shared logic once; each adapter keeps its own public,
schema-typed wrappers (preserving its distinct defaults — e.g. the recall gate —
and concrete record type) that delegate here.

Keeping the shared core generic (over the dataset's Pydantic record type) and the
public surface in the adapters means: no public API change, no duplicated
algorithm, and the adapters stay the single source of truth for their pinned
constants. The helpers are intentionally free of any per-dataset schema import,
so this module never imports ``er_benchmarks`` or ``amazon_google``.
"""

import csv
import logging
import random
from collections import defaultdict
from collections.abc import Iterable, Sequence
from importlib import resources
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from langres.core.metrics import evaluate_blocking
from langres.core.models import ERCandidate

# NOTE: the heavy ``[semantic]`` stack (VectorBlocker / SentenceTransformerEmbedder /
# FAISSIndex) is imported LAZILY inside ``sweep_blocking_k`` (its sole consumer), not
# at module scope. This keeps ``import langres.data._benchmark_utils`` — and hence the
# generic loader factory that reuses these helpers — faiss-free, so a dataset can be
# loaded/split offline without pulling faiss + sentence-transformers.

logger = logging.getLogger(__name__)


class _HasId(Protocol):
    """Minimal record contract the stratified split needs: a globally-unique id."""

    id: str


SchemaT = TypeVar("SchemaT", bound=BaseModel)
"""The dataset's Pydantic record type (exposes ``model_dump`` + the text field)."""

RecordT = TypeVar("RecordT", bound=_HasId)
"""A record exposing ``id`` (the only attribute the split logic reads)."""

CandidateT = TypeVar("CandidateT", bound=ERCandidate[Any])
"""A normalized candidate pair (its ``left``/``right`` expose ``source``)."""

_EMBED_MODEL = "all-MiniLM-L6-v2"


class BenchmarkDataNotFoundError(FileNotFoundError):
    """A packaged benchmark CSV is missing from this install.

    Raised by :func:`read_csv_rows` when the requested file is not present in
    the installed ``langres`` package. This is the expected state of every
    pip/uv install from PyPI: the large third-party benchmark corpora
    (DeepMatcher/Magellan — Abt-Buy, Amazon-Google, DBLP-ACM, DBLP-Scholar,
    Walmart-Amazon, WDC Computers) are excluded from the wheel/sdist (see
    ``[tool.hatch.build].exclude`` in ``pyproject.toml``) and ship in the git
    repository only. Subclasses :class:`FileNotFoundError` so callers that
    already catch the generic error keep working.
    """


def read_csv_rows(package: str, filename: str) -> list[dict[str, str]]:
    """Read a packaged benchmark CSV into a list of header-keyed row dicts.

    Args:
        package: The importable package holding the CSV (e.g.
            ``"langres.data.datasets.fodors_zagat"``).
        filename: The CSV filename within ``package``.

    Returns:
        One dict per data row, keyed by the CSV header.

    Raises:
        BenchmarkDataNotFoundError: If the file is not present in this install.
            The large benchmark corpora are not bundled in the PyPI package;
            they are available from a git checkout only.
    """
    try:
        text = resources.files(package).joinpath(filename).read_text(encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError, NotADirectoryError) as exc:
        # ModuleNotFoundError: the whole dataset directory is absent (its files
        # were all excluded from the wheel, so the namespace package is gone).
        # FileNotFoundError/NotADirectoryError: the directory survived (e.g.
        # peeters_sampled_test.csv is still bundled) but this file did not.
        raise BenchmarkDataNotFoundError(
            f"Benchmark file {filename!r} in {package!r} is not available in this "
            "install. The third-party benchmark corpora are not bundled in the PyPI "
            "package (they are ~14 MB and carry no explicit redistribution license); "
            "they ship in the git repository only. To use this benchmark, install "
            "langres from a git checkout:\n"
            "    git clone https://github.com/fxd24/langres\n"
            "    pip install -e ./langres  # or, inside the checkout: uv sync\n"
            "Datasets whose data IS bundled: tiny_fixture (loads on a core install), "
            "fodors_zagat and febrl_person (their loaders need the [semantic] extra)."
        ) from exc
    reader = csv.DictReader(text.splitlines())
    return [dict(row) for row in reader]


def cross_source(candidates: list[CandidateT]) -> list[CandidateT]:
    """Keep only candidate pairs whose two records come from different sources.

    Both benchmarks are *linkage* tasks whose true matches are all cross-source;
    intra-source pairs are noise (see each adapter's module docstring). Generic
    over the candidate type so each adapter keeps its concrete
    ``ERCandidate[Schema]`` typing through this filter.
    """
    return [c for c in candidates if c.left.source != c.right.source]


def sweep_blocking_k(
    corpus: Sequence[SchemaT],
    gold_clusters: list[set[str]],
    schema: type[SchemaT],
    *,
    text_field: str,
    ks: tuple[int, ...],
) -> dict[int, float]:
    """Measure cross-source Pair-Completeness of vector blocking across ``ks``.

    Builds the FAISS index once over ``text_field`` and reuses it across every
    ``k`` (only ``k_neighbors`` changes), so the corpus is embedded a single
    time. For each ``k`` the candidates are filtered to cross-source pairs before
    recall is measured via :func:`~langres.core.metrics.evaluate_blocking`.
    ``candidate_recall`` *is* Pair-Completeness (the fraction of gold match pairs
    surfaced as candidates), which is what the blocking gate is defined on.

    Args:
        corpus: Combined record list for the dataset.
        gold_clusters: The complete closed-world partition (match sets +
            singletons) for the dataset.
        schema: The Pydantic record type, passed declaratively to each
            :class:`VectorBlocker` so the blocker stays config-serializable.
        text_field: Attribute name holding each record's blocking text (e.g.
            ``"embed_text"``).
        ks: Neighbor counts to sweep.

    Returns:
        Mapping of ``k`` to cross-source Pair-Completeness.
    """
    # Lazy [semantic] import (kept out of module scope): only this k-sweep needs
    # the vector stack, so importing it here keeps the module faiss-free for the
    # loaders/factory that only call the light helpers above.
    from langres.core.blockers.vector import VectorBlocker
    from langres.core.embeddings import SentenceTransformerEmbedder
    from langres.core.indexes.vector_index import FAISSIndex

    embedder = SentenceTransformerEmbedder(_EMBED_MODEL)
    index = FAISSIndex(embedder=embedder, metric="cosine")
    index.create_index([getattr(r, text_field) for r in corpus])
    records = [r.model_dump() for r in corpus]

    recalls: dict[int, float] = {}
    for k in ks:
        # Fresh blocker per k (the pre-built FAISS index is reused, so this is
        # cheap); only k_neighbors varies. ``k`` lives on the blocker, not the
        # index, so sharing one index across ks is safe.
        blocker: VectorBlocker[SchemaT] = VectorBlocker(
            vector_index=index,
            schema=schema,
            text_field=text_field,
            k_neighbors=k,
        )
        candidates = cross_source(list(blocker.stream(records)))
        recall = evaluate_blocking(candidates, gold_clusters).candidate_recall
        recalls[k] = recall
        logger.info("blocking k=%d -> cross-source recall=%.4f", k, recall)
    return recalls


def pick_blocking_k(recalls: dict[int, float], threshold: float) -> int:
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


def clusters_from_pairs(gold_pairs: set[frozenset[str]], all_ids: Iterable[str]) -> list[set[str]]:
    """Connected components of the match graph, singleton-completed over ``all_ids``.

    Shared by every many-to-many linkage benchmark (Amazon-Google, Abt-Buy): the
    gold clusters are the connected components of the undirected graph whose
    edges are the positive pairs (a record reachable from another via a chain of
    matches shares its entity). A tiny union-find computes the components; every
    id in ``all_ids`` not touched by any match becomes its own singleton,
    yielding the **complete closed-world partition** (match components +
    singletons) — exactly like Fodors-Zagat's ``perfectMapping`` completion.
    Singletons add no positive pairs, so blocking Pair-Completeness is
    unaffected.

    Args:
        gold_pairs: Positive match pairs as 2-element frozensets of corpus ids.
        all_ids: Every corpus id (e.g. ``[r.id for r in corpus]``); order fixes
            the singleton order for determinism.

    Returns:
        The complete partition: match components followed by one singleton per
        unmatched id (in ``all_ids`` order).
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:  # path compression
            parent[x], x = root, parent[x]
        return root

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for pair in gold_pairs:
        a, b = tuple(pair)
        union(a, b)

    components: dict[str, set[str]] = defaultdict(set)
    for node in parent:
        components[find(node)].add(node)
    match_clusters = list(components.values())

    matched_ids = set(parent)
    singletons = [{rid} for rid in all_ids if rid not in matched_ids]
    return match_clusters + singletons


def _split_stratum(
    clusters: list[set[str]], test_size: float, rng: random.Random
) -> tuple[list[set[str]], list[set[str]]]:
    """Split one same-size cluster stratum into (train, test) whole clusters.

    Clusters are sorted into a canonical order (by their sorted id tuple) before
    shuffling so the split is deterministic given ``rng`` regardless of input
    ordering. At least one cluster always goes to test (mirrors
    ``stratified_dedup_split``).
    """
    ordered = sorted(clusters, key=lambda c: tuple(sorted(c)))
    rng.shuffle(ordered)
    n_test = max(1, int(len(ordered) * test_size))
    return ordered[:-n_test], ordered[-n_test:]


def stratified_corpus_split(
    corpus: Sequence[RecordT],
    gold_clusters: list[set[str]],
    *,
    test_size: float = 0.3,
    seed: int = 0,
) -> tuple[list[RecordT], list[RecordT], list[set[str]], list[set[str]]]:
    """Stratified, leakage-free train/test split over full records.

    Stratifies by gold-cluster size (singletons in their own band, matched groups
    split within each size band, whole clusters kept together). Because whole gold
    clusters are assigned to one side, no match pair ever straddles the split (no
    test-set leakage), and each returned cluster list partitions exactly its
    split's ids. Generic over the record type: any schema exposing ``id`` works,
    so both benchmarks (Fodors-Zagat ``f``/``z`` ids, Amazon-Google ``a``/``g``
    ids) share one implementation.

    Args:
        corpus: The full record list (each record exposes ``id``).
        gold_clusters: The complete closed-world partition (match sets +
            singletons).
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
    groups_by_size: dict[int, list[set[str]]] = {}
    for cluster in gold_clusters:
        if len(cluster) >= 2:
            groups_by_size.setdefault(len(cluster), []).append(cluster)

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

    # Natural sort (a1, a2, a10 — not a1, a10, a2) so the returned record order is
    # intuitive for callers. Assumes ids are a single-letter source prefix + a pure
    # integer (e.g. "f1"/"z42", "a1"/"g123") — the format both benchmarks use; a
    # different id scheme (e.g. "amazon_1") would raise ValueError on int(rid[1:]).
    def _natural(rid: str) -> tuple[str, int]:
        return (rid[0], int(rid[1:]))

    train_records = [by_id[rid] for rid in sorted(train_ids, key=_natural)]
    test_records = [by_id[rid] for rid in sorted(test_ids, key=_natural)]
    logger.info(
        "stratified_corpus_split: %d train records (%d clusters), %d test records (%d clusters)",
        len(train_records),
        len(train_clusters),
        len(test_records),
        len(test_clusters),
    )
    return train_records, test_records, train_clusters, test_clusters
