"""Measure which embedding model to default to at each real parameter count.

The question: at a given number of parameters, which sentence-embedding model
gives the best blocking, and does an **instruction prompt on the query side**
change the answer? Both halves need care, and each has already produced one
wrong number on this repo:

- **Parameter count is measured, never typed.** Every row reports
  ``sum(p.numel() for p in model.parameters())`` from the model that actually
  produced its vectors (``SentenceTransformerEmbedder.parameter_count``). There
  is no name→size table here and no "small/base/large" label anywhere in the
  output: a table drifts silently when a checkpoint is revised, and a size label
  is a marketing word.
- **The metric is candidate recall + separability AUC, not F1 at a threshold.**
  Blocking sets a *ceiling*: a pair the blocker never emits cannot be recovered
  downstream. F1 at some threshold conflates that ceiling with a matcher's
  operating point.
- **Recall has a ceiling below 1.0 that belongs to the benchmark, not the
  model.** These are two-source linkage tasks, so the harness keeps only
  cross-source candidates — and a gold cluster spanning three or more records
  emits intra-source gold pairs no cross-source candidate set can contain.
  ``recall_of_reachable`` is the model comparison; raw ``candidate_recall``
  mixes it with a property of the gold set. See ``_reachable_ceiling``.
- **Deltas measured at ``CI_K`` carry a cluster-resampled interval.** Prompt-arm
  and model-vs-``REFERENCE_MODEL`` differences go through
  ``langres.experiments.statistics.paired_entity_bootstrap``, resampled by gold
  cluster — never by pair row, which would report intervals far too tight. A
  headline delta without an interval is how a +0.9pp single-seed result becomes
  a recommendation. Where an interval is genuinely unavailable (the reference has
  not been measured on that benchmark; fewer than two gold clusters) the report
  prints ``—`` and says so, rather than dropping the row.
- **The prompt axis only became measurable once ``query_prompt`` was fixed.**
  ``FAISSIndex.search_all`` used to hand the cached corpus vectors to
  ``search()``, which never applies a prompt, so prompted and unprompted runs
  returned byte-identical results. Sweeping the axis before that fix would have
  produced a clean, confident "instructions do not help retrieval".

**A model that fails to load is a reported row, never a silent skip.** A sweep
that quietly drops the models it could not load reads as "these are the models
that exist" — see ``status`` / ``error`` on every row.

Run it (from a git checkout — see the reproducibility note in the report):

    uv run --env-file .env python examples/research/embedder_ladder.py \\
        --models all-MiniLM-L6-v2 --benchmarks fodors_zagat

Rows are appended to a tracked JSONL and the markdown report is regenerated
from it after every model, so a long sweep is durable at each step rather than
only at the end. Re-running a model **replaces** its rows, so the harness
reproduces its own committed table instead of appending duplicates. A third
tracked file holds ``REFERENCE_MODEL``'s per-record recall, which is what lets a
model measured next week still get a paired interval against the model that
ships today.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROWS_PATH = REPO_ROOT / "docs" / "research" / "20260727_embedder_ladder_rows.jsonl"
DEFAULT_REPORT_PATH = REPO_ROOT / "docs" / "research" / "20260727_embedder_ladder.md"
#: Per-record recall of ``REFERENCE_MODEL``, so a model measured in a later run
#: can still be compared against it. Tracked, because a CI computed in one
#: process against numbers that died with another process is not reproducible.
DEFAULT_REFERENCE_PATH = (
    REPO_ROOT / "docs" / "research" / "20260727_embedder_ladder_reference_recall.json"
)
DEFAULT_CACHE_DIR = REPO_ROOT / "tmp" / "embedder_ladder_cache"

#: The one instruction used for every model's query side. Deliberately ONE
#: string, not a per-model recipe: this axis answers "does adding a task
#: instruction to the query help?", not "is each model at its documented best?".
#: Each model's own registered prefixes are reported per row
#: (``registered_prompts``) so a reader can see which models had a documented
#: recipe this sweep did not use.
INSTRUCTION = "Find the duplicate product or business record for: "

PROMPT_ARMS: dict[str, str | None] = {"none": None, "instruct": INSTRUCTION}


@dataclass(frozen=True)
class ModelSpec:
    """One candidate embedder. Size is **not** declared here — it is measured."""

    #: Hugging Face / sentence-transformers reference.
    name: str
    #: Whether the checkpoint ships its own modelling code that must be executed.
    #: Off unless the model genuinely cannot load otherwise; it runs arbitrary
    #: Python from the model repo.
    trust_remote_code: bool = False
    #: Load dtype. Default (``None``) is whatever the checkpoint ships, which is
    #: float32 for most of this list. Set explicitly only where full precision
    #: does not fit: a multi-billion-parameter embedder in float32 needs ~4 bytes
    #: per parameter of unified memory before any activations. The value is
    #: recorded on every row, because half precision is a **different
    #: measurement**, not a free speedup, and a table that hid it would compare
    #: models across precisions silently.
    dtype: str | None = None
    #: Smaller batches for models whose activations dominate memory.
    batch_size: int | None = None


#: Listed in roughly ascending expected size so a truncated sweep still covers
#: the cheap tiers completely. Ordering is a *scheduling* hint only — every
#: published number comes from the measured ``parameter_count``.
MODELS: tuple[ModelSpec, ...] = (
    ModelSpec("all-MiniLM-L6-v2"),
    ModelSpec("all-MiniLM-L12-v2"),
    ModelSpec("BAAI/bge-small-en-v1.5"),
    ModelSpec("all-mpnet-base-v2"),
    ModelSpec("BAAI/bge-base-en-v1.5"),
    ModelSpec("intfloat/e5-base-v2"),
    ModelSpec("Alibaba-NLP/gte-base-en-v1.5", trust_remote_code=True),
    ModelSpec("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True),
    ModelSpec("google/embeddinggemma-300m"),
    ModelSpec("BAAI/bge-large-en-v1.5"),
    ModelSpec("mixedbread-ai/mxbai-embed-large-v1"),
    ModelSpec("Qwen/Qwen3-Embedding-0.6B"),
    ModelSpec("Qwen/Qwen3-Embedding-4B", dtype="float16", batch_size=8),
    ModelSpec("Qwen/Qwen3-Embedding-8B", dtype="float16", batch_size=4),
)

MODELS_BY_NAME: dict[str, ModelSpec] = {spec.name: spec for spec in MODELS}

#: ``fodors_zagat`` is the never-regress floor (small, long-solved); the other
#: four are the working set. This portfolio is a **documented prior, not a
#: finding of this sweep** — see the report's saturation caveat.
BENCHMARKS: tuple[str, ...] = (
    "fodors_zagat",
    "abt_buy",
    "amazon_google",
    "wdc_computers",
    "walmart_amazon",
)

#: Saturation verdict per benchmark. **Imported, not measured here** — a separate
#: portfolio stream measured it, and this harness has no way to observe it (a
#: single-family ladder cannot tell "all embedders agree" from "the benchmark is
#: solved"). Carried per row so no reader has to remember which of these numbers
#: is a finding of this sweep and which is inherited context.
SATURATION: dict[str, str] = {
    "fodors_zagat": "saturated (methods span 0.047)",
    "abt_buy": "not saturated",
    "amazon_google": "not saturated",
    "wdc_computers": "not saturated",
    "walmart_amazon": "not saturated",
}

K_VALUES: tuple[int, ...] = (5, 10, 20, 50)

#: Negatives sampled per benchmark for the separability AUC (seeded).
NEGATIVE_SAMPLE = 20_000
SEED = 0

#: Bumped whenever a metric here CHANGES MEANING, so a row measured under an
#: older definition can be recognised instead of quietly sharing a column with a
#: newer one. This is not versioning for its own sake: rows survive in a tracked
#: JSONL across code changes, and the one-directional separability score that
#: `8806cef` replaced systematically depressed the prompted arm — mixing the two
#: in one `Δ AUC` column reads as a finding about base-size models rather than as
#: the artefact it is.
#:
#: 1 = the first honest revision: reachable-recall ceiling reported, separability
#: scored in both directions, index-build seconds paired with an encode count,
#: prompt delta reported as the statistic its interval bounds.
METRIC_REVISION = 1

#: The `k` at which paired confidence intervals are computed and the reference
#: per-record scores are persisted. One `k` on purpose: the sidecar is a tracked
#: file, and a CI at every `k` would quadruple it to answer the same question.
CI_K = 20

#: The baseline every model's delta is measured against: langres's current
#: `DEFAULT_EMBEDDING_MODEL` (`core/model_ref.py`). A ladder's decision-relevant
#: question is "is this better than what ships today?", so the default IS the
#: baseline. This sweep does not change that default.
REFERENCE_MODEL = "all-MiniLM-L6-v2"


@dataclass
class LadderRow:
    """One measured cell: (model, benchmark, prompt arm, k)."""

    model: str
    benchmark: str
    prompt_arm: str
    k: int
    status: str
    parameter_count: int | None = None
    dtype: str | None = None
    embedding_dim: int | None = None
    n_records: int | None = None
    n_gold_pairs: int | None = None
    candidate_recall: float | None = None
    #: Highest recall this measurement *can* report — see ``_reachable_ceiling``.
    #: Not a property of the model: it is what the cross-source candidate filter
    #: makes unreachable in this benchmark's gold set.
    reachable_recall_ceiling: float | None = None
    #: ``candidate_recall / reachable_recall_ceiling`` — how much of what is
    #: reachable the model actually retrieved. This is the model comparison;
    #: ``candidate_recall`` alone mixes it with a benchmark artefact.
    recall_of_reachable: float | None = None
    candidate_precision: float | None = None
    reduction_ratio: float | None = None
    total_candidates: int | None = None
    candidates_per_unit_recall: float | None = None
    separability_auc: float | None = None
    index_build_seconds: float | None = None
    #: Texts actually encoded during the index build (cache misses). ``0`` means
    #: the build read a warm embedding cache and its seconds are a cache-read
    #: time, NOT an encode time — the two differ by orders of magnitude.
    index_build_encoded: int | None = None
    search_seconds: float | None = None
    registered_prompts: list[str] = field(default_factory=list)
    #: Which revision of the metric definitions produced this row. Bumped whenever
    #: a metric CHANGES MEANING, so the report can refuse to print two definitions
    #: in one column. Rows below ``METRIC_REVISION`` are re-run, not re-rendered.
    metric_revision: int = 0
    #: Mean **per-record** recall difference (instruct minus none) for this model
    #: on this benchmark, with its paired-bootstrap CI (by gold cluster). Only on
    #: ``instruct`` rows at ``CI_K``. The point estimate is the same statistic the
    #: interval bounds — deliberately NOT the difference of the two aggregate
    #: recalls, which is a different number (they disagreed by 34% on
    #: walmart_amazon). An interval spanning 0 means the delta is not
    #: distinguishable from noise on this benchmark.
    prompt_delta: float | None = None
    prompt_delta_ci_low: float | None = None
    prompt_delta_ci_high: float | None = None
    #: Mean per-record recall difference against ``REFERENCE_MODEL`` in the same
    #: arm, with its paired-bootstrap CI. Only at ``CI_K``, and never on the
    #: reference model's own rows. Note this is the mean of per-record deltas, not
    #: the difference of the two aggregate recalls — the bootstrap resamples the
    #: per-record units, so the point estimate has to be the same statistic.
    vs_reference_delta: float | None = None
    vs_reference_ci_low: float | None = None
    vs_reference_ci_high: float | None = None
    #: Number of independent gold clusters the interval above was resampled over.
    #: The honest denominator: it is NOT the record count, because pair rows
    #: inside one entity are dependent.
    ci_clusters: int | None = None
    #: Saturation is measured by a SEPARATE stream; never restate it as measured here.
    saturation: str = "not measured here"
    error: str | None = None


def _load_benchmark(name: str) -> tuple[list[Any], list[set[str]], set[frozenset[str]]]:
    from langres.data.registry import get_benchmark

    return get_benchmark(name).load()


def _blocking_texts(corpus: Sequence[Any]) -> list[str]:
    """Blocking text per record: every comparable string field, space-joined.

    Uses the registered ``concat_comparable_fields`` extractor rather than a
    per-dataset field name, so the same code covers all five schemas without a
    hand-maintained mapping that would silently go stale.
    """
    from langres.core.blockers.vector import concat_comparable_fields

    return [concat_comparable_fields(record) for record in corpus]


def _source_sizes(corpus: Sequence[Any]) -> tuple[int, int] | None:
    """``(n_left, n_right)`` for a two-source linkage corpus, else ``None``.

    Mirrors ``langres.optimize._source_sizes``: exactly two distinct ``source``
    values is the cross-source case, where the reduction ratio must use
    ``|A| * |B|`` rather than the dedup ``n(n-1)/2``.
    """
    from collections import Counter

    counts: Counter[Any] = Counter(getattr(record, "source", None) for record in corpus)
    counts.pop(None, None)
    if len(counts) != 2:
        return None
    left, right = (counts[key] for key in sorted(counts, key=str))
    return (left, right)


def _gold_pair_set(gold_clusters: Sequence[set[str]]) -> set[frozenset[str]]:
    from itertools import combinations

    return {
        frozenset(pair) for cluster in gold_clusters for pair in combinations(sorted(cluster), 2)
    }


def _reachable_ceiling(corpus: Sequence[Any], gold_pairs: set[frozenset[str]]) -> float:
    """Fraction of gold pairs a cross-source candidate set can possibly contain.

    On a two-source linkage benchmark this harness drops intra-source candidate
    pairs (mirroring ``langres.optimize._score_loaded``, so the numbers here and
    the ones ``optimize()`` reports are the same measurement). **That filter does
    not leave recall unchanged.** A gold *cluster* with three or more records can
    put two of them on the same side, and the transitive closure that turns
    clusters into pairs then emits an intra-source gold pair which no
    cross-source candidate set can ever contain.

    It is not hypothetical: on ``amazon_google`` this ceiling measures ~0.8396,
    which is why no model in this sweep reports recall above ~0.84 there — a fact
    about the gold set, not about embeddings. An earlier version of this comment
    asserted the opposite ("all gold matches are inter-source"), and the same
    false claim is still written at ``src/langres/optimize.py:135-139``.

    Returns:
        ``1.0`` when there is no cross-source filter or no gold pairs.
    """
    source = {str(record.id): getattr(record, "source", None) for record in corpus}
    if not gold_pairs:
        return 1.0
    reachable = sum(1 for pair in gold_pairs if len({source.get(rid) for rid in pair}) == 2)
    return reachable / len(gold_pairs)


def separability_auc(
    doc_vectors: np.ndarray,
    query_vectors: np.ndarray,
    ids: Sequence[str],
    gold_pairs: set[frozenset[str]],
    *,
    seed: int = SEED,
) -> float | None:
    """ROC-AUC of the raw embedding similarity separating gold from non-gold pairs.

    Blocker-independent on purpose: candidate recall depends on ``k``, so it
    measures a *chosen operating point*. This measures the embedding's ability
    to rank a true pair above a false one at all, which is what actually differs
    between models.

    **Both directions are scored and the maximum taken**, which is what the
    blocker effectively does — every record is a query once, so a pair is
    retrieved if *either* direction ranks it. That symmetry is load-bearing under
    a prompt, not cosmetic. Gold pairs come out of a ``sorted()`` id tuple, and
    every cross-source loader here prefixes ids by source (``a``/``b``,
    ``a``/``g``), so sorting makes **every positive** an A-query→B-document pair;
    the uniformly sampled negatives are ~25% each of A→B, B→A, A→A, B→B. Scoring
    one direction only would therefore compare the two classes under different
    direction distributions the moment ``query_vectors is not doc_vectors`` —
    i.e. in the prompted arm exactly, which is the arm whose delta gets
    published as "does an instruction help".

    Negatives are a seeded uniform sample of non-gold pairs (the full set is
    quadratic and hugely imbalanced), so this is an AUC over a *sample*, not the
    population — and a uniformly random pair is trivially dissimilar, which is
    why every usable model scores near the ceiling here.

    Returns:
        The AUC, or ``None`` when either class is empty.
    """
    from langres.metrics.metrics import roc_auc_score

    position = {record_id: index for index, record_id in enumerate(ids)}
    positives = [
        (position[a], position[b])
        for a, b in (tuple(sorted(pair)) for pair in gold_pairs)
        if a in position and b in position
    ]
    if not positives:
        return None

    rng = random.Random(seed)
    n = len(ids)
    negatives: list[tuple[int, int]] = []
    seen = {frozenset(pair) for pair in positives}
    attempts = 0
    while len(negatives) < NEGATIVE_SAMPLE and attempts < NEGATIVE_SAMPLE * 10:
        attempts += 1
        i, j = rng.randrange(n), rng.randrange(n)
        if i == j:
            continue
        key = frozenset((i, j))
        if key in seen or frozenset({ids[i], ids[j]}) in gold_pairs:
            continue
        seen.add(key)
        negatives.append((i, j))
    if not negatives:
        return None

    def scores(pairs: list[tuple[int, int]]) -> np.ndarray:
        first = [i for i, _ in pairs]
        second = [j for _, j in pairs]
        forward = np.einsum("ij,ij->i", query_vectors[first], doc_vectors[second])
        backward = np.einsum("ij,ij->i", query_vectors[second], doc_vectors[first])
        return np.asarray(np.maximum(forward, backward))

    labels = [True] * len(positives) + [False] * len(negatives)
    values = np.concatenate([scores(positives), scores(negatives)])
    return float(roc_auc_score(labels, values.tolist()))


def per_record_recall(
    candidate_pairs: set[frozenset[str]],
    gold_clusters: Sequence[set[str]],
) -> dict[str, tuple[float, str]]:
    """Per-record blocking recall plus its gold cluster id, for paired bootstrap.

    The unit of a model-vs-model comparison must not be the pair row: pair rows
    inside one entity are dependent, and bootstrapping them produces intervals
    that are far too tight. Scoring per record and resampling by *cluster* is
    what ``langres.experiments.statistics.paired_entity_bootstrap`` expects.

    Returns:
        ``{record_id: (fraction of its gold partners captured, cluster_id)}``,
        containing only records that have at least one gold partner.
    """
    scores: dict[str, tuple[float, str]] = {}
    for index, cluster in enumerate(gold_clusters):
        if len(cluster) < 2:
            continue
        cluster_id = f"c{index}"
        for record_id in sorted(cluster):
            partners = [other for other in cluster if other != record_id]
            captured = sum(
                1 for other in partners if frozenset({record_id, other}) in candidate_pairs
            )
            scores[record_id] = (captured / len(partners), cluster_id)
    return scores


#: ``{record_id: (per-record recall, gold cluster id)}`` for one measured cell.
RecallByRecord = dict[str, tuple[float, str]]


def paired_interval(baseline: RecallByRecord, candidate: RecallByRecord) -> Any | None:
    """Cluster-resampled CI for ``candidate - baseline`` per-record recall.

    Delegates to ``langres.experiments.statistics.paired_entity_bootstrap``
    rather than rolling a bootstrap here — that function exists precisely because
    resampling *pair rows* (which are dependent inside one entity) yields
    intervals that are far too tight, and a second implementation is a second
    chance to reintroduce the bug it was written to prevent.

    Returns:
        A ``BootstrapInterval``, or ``None`` when the two cells share fewer than
        two records (nothing to resample).
    """
    from langres.experiments.statistics import PairedScore, paired_entity_bootstrap

    shared = sorted(set(baseline) & set(candidate))
    if len(shared) < 2:
        return None
    observations = tuple(
        PairedScore(
            entity_id=record_id,
            baseline=baseline[record_id][0],
            candidate=candidate[record_id][0],
            cluster_id=candidate[record_id][1],
        )
        for record_id in shared
    )
    return paired_entity_bootstrap(observations, seed=SEED)


def _reference_key(benchmark: str, arm: str) -> str:
    """Sidecar key for one reference cell.

    ``REFERENCE_MODEL`` and ``CI_K`` are in the key, not just in the filename:
    both are one-line constants, and if either changed while the file stayed put,
    ``read_reference`` would return the old contents as the baseline and every
    ``vs_reference_*`` would silently become a comparison against a different
    model — or a different ``k`` — with no error and no visible change in the
    report.
    """
    return f"{REFERENCE_MODEL}|k{CI_K}|{benchmark}|{arm}"


def read_reference(path: Path) -> dict[str, RecallByRecord]:
    """Load the reference model's per-record recall, keyed by :func:`_reference_key`."""
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    return {
        key: {record_id: (float(value[0]), str(value[1])) for record_id, value in cell.items()}
        for key, cell in raw.items()
    }


def write_reference(path: Path, store: dict[str, RecallByRecord]) -> None:
    """Persist the reference per-record recall, sorted so the file diffs readably."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        key: {
            record_id: [round(score, 6), cluster]
            for record_id, (score, cluster) in sorted(cell.items())
        }
        for key, cell in sorted(store.items())
    }
    path.write_text(json.dumps(payload, indent=0, sort_keys=True) + "\n")


def _build_embedder(spec: ModelSpec, cache_dir: Path, device: str | None, batch_size: int) -> Any:
    """A disk-cached embedder for ``spec``.

    The cache is keyed on (text, prompt) and namespaced per model, which makes a
    long sweep resumable and keeps the prompted arm from re-encoding the corpus
    once per ``k`` (a prompted ``search_all`` re-encodes queries by design).
    """
    from langres.core.embeddings import DiskCachedEmbedder, SentenceTransformerEmbedder

    base = SentenceTransformerEmbedder(
        spec.name,
        batch_size=spec.batch_size or batch_size,
        device=device,
        trust_remote_code=spec.trust_remote_code,
        dtype=spec.dtype,  # type: ignore[arg-type]
    )
    # The cache namespace carries the dtype: half-precision vectors are DIFFERENT
    # vectors, and reusing a float32 cache entry for a float16 run would silently
    # publish a number the run never computed.
    namespace = f"{spec.name.replace('/', '__')}__{spec.dtype or 'default'}"
    return base, DiskCachedEmbedder(base, cache_dir=cache_dir, namespace=namespace)


def _registered_prompts(base_embedder: Any) -> list[str]:
    """Prompt names the checkpoint registers with a NON-EMPTY prefix.

    Measured off the loaded model rather than tabulated by hand. Empty prefixes
    are excluded because sentence-transformers registers placeholder
    ``query``/``document`` entries with empty strings for models that have no
    real instruction recipe (measured on ``all-MiniLM-L6-v2``:
    ``{'query': '', 'document': ''}``) — reporting those names would claim a
    documented recipe the checkpoint does not actually have.
    """
    model = getattr(base_embedder, "_model", None)
    if model is None:
        return []
    return sorted(name for name, prefix in (getattr(model, "prompts", {}) or {}).items() if prefix)


def evaluate_model_on_benchmark(
    spec: ModelSpec,
    benchmark: str,
    *,
    k_values: Sequence[int],
    prompt_arms: dict[str, str | None],
    cache_dir: Path,
    device: str | None,
    batch_size: int,
    reference: dict[str, RecallByRecord] | None = None,
) -> tuple[list[LadderRow], dict[str, RecallByRecord]]:
    """Measure one model on one benchmark.

    Args:
        reference: ``REFERENCE_MODEL``'s per-record recall keyed
            ``"{benchmark}|{arm}"``, used to attach a paired CI at ``CI_K``.
            Missing keys simply leave the CI unset — a delta without an interval
            is honest, a fabricated interval is not.

    Returns:
        ``(rows, reference_updates)``. The second element is non-empty only when
        ``spec`` **is** the reference model, in which case it carries the cells a
        later run needs to compare against.
    """
    from langres.core.blockers.vector import VectorBlocker
    from langres.core.indexes.vector_index import FAISSIndex
    from langres.metrics.metrics import evaluate_blocking

    rows: list[LadderRow] = []
    reference_updates: dict[str, RecallByRecord] = {}
    reference = reference or {}
    #: Per-record recall at CI_K, per arm — the input to both paired intervals.
    per_arm_recall: dict[str, RecallByRecord] = {}

    try:
        corpus, gold_clusters, _ = _load_benchmark(benchmark)
        texts = _blocking_texts(corpus)
        ids = [str(record.id) for record in corpus]
        gold_pairs = _gold_pair_set(gold_clusters)
        sizes = _source_sizes(corpus)
        ceiling = _reachable_ceiling(corpus, gold_pairs) if sizes is not None else 1.0
        schema = type(corpus[0])
        records = [record.model_dump() for record in corpus]

        base, embedder = _build_embedder(spec, cache_dir, device, batch_size)

        started = time.perf_counter()
        index = FAISSIndex(embedder=embedder, metric="cosine")
        index.create_index(texts)
        index_build_seconds = time.perf_counter() - started
        # How many texts this build actually encoded. The embedder is disk-cached
        # and freshly constructed per cell, so `misses` counts exactly the texts
        # that went through the model. Zero means the seconds above are a SQLite
        # read, not an encode, and must not be quoted as an encoding cost.
        index_build_encoded = int(embedder.cache_info()["misses"])

        parameter_count = base.parameter_count
        embedding_dim = base.embedding_dim
        prompts = _registered_prompts(base)
        doc_vectors = embedder.encode(texts)
    except Exception as exc:  # noqa: BLE001 - a failure IS a result, never a skip
        logger.exception("model %s failed on %s", spec.name, benchmark)
        return (
            [
                LadderRow(
                    model=spec.name,
                    benchmark=benchmark,
                    prompt_arm="-",
                    k=0,
                    status="failed",
                    metric_revision=METRIC_REVISION,
                    error=f"{type(exc).__name__}: {exc}"[:400],
                )
            ],
            {},
        )

    for arm, prompt in prompt_arms.items():
        query_vectors = embedder.encode(texts, prompt=prompt) if prompt else doc_vectors
        auc = separability_auc(doc_vectors, query_vectors, ids, gold_pairs)

        for k in k_values:
            try:
                blocker = VectorBlocker(
                    vector_index=index,
                    schema=schema,
                    text_field_extractor="concat_comparable_fields",
                    k_neighbors=k,
                    query_prompt=prompt,
                )
                search_started = time.perf_counter()
                candidates = list(blocker.stream(records))
                search_seconds = time.perf_counter() - search_started

                if sizes is not None:
                    # Cross-source linkage: keep only inter-source candidates so
                    # the reduction ratio uses |A|*|B|. Mirrors
                    # langres.optimize._score_loaded, deliberately — the point is
                    # that these numbers ARE what optimize() would report.
                    # This filter is NOT recall-neutral: gold clusters spanning
                    # three or more records emit intra-source gold pairs that no
                    # cross-source candidate set can contain. See
                    # `_reachable_ceiling`, reported on every row.
                    candidates = [c for c in candidates if c.left.source != c.right.source]
                    stats = evaluate_blocking(
                        candidates, gold_clusters, n_left=sizes[0], n_right=sizes[1]
                    )
                else:
                    stats = evaluate_blocking(candidates, gold_clusters, num_records=len(corpus))

                recall = stats.candidate_recall
                if k == CI_K:
                    candidate_pairs = {
                        frozenset({str(c.left.id), str(c.right.id)}) for c in candidates
                    }
                    per_arm_recall[arm] = per_record_recall(candidate_pairs, gold_clusters)
                rows.append(
                    LadderRow(
                        model=spec.name,
                        benchmark=benchmark,
                        prompt_arm=arm,
                        k=k,
                        status="ok",
                        parameter_count=parameter_count,
                        dtype=spec.dtype,
                        embedding_dim=embedding_dim,
                        n_records=len(corpus),
                        n_gold_pairs=len(gold_pairs),
                        candidate_recall=recall,
                        reachable_recall_ceiling=ceiling,
                        recall_of_reachable=(recall / ceiling if ceiling > 0 else None),
                        candidate_precision=stats.candidate_precision,
                        reduction_ratio=stats.reduction_ratio,
                        total_candidates=stats.total_candidates,
                        candidates_per_unit_recall=(
                            stats.total_candidates / recall if recall > 0 else None
                        ),
                        separability_auc=auc,
                        index_build_seconds=index_build_seconds,
                        index_build_encoded=index_build_encoded,
                        search_seconds=search_seconds,
                        registered_prompts=prompts,
                        metric_revision=METRIC_REVISION,
                        saturation=SATURATION.get(benchmark, "not measured here"),
                    )
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("model %s failed on %s k=%s", spec.name, benchmark, k)
                rows.append(
                    LadderRow(
                        model=spec.name,
                        benchmark=benchmark,
                        prompt_arm=arm,
                        k=k,
                        status="failed",
                        parameter_count=parameter_count,
                        dtype=spec.dtype,
                        metric_revision=METRIC_REVISION,
                        error=f"{type(exc).__name__}: {exc}"[:400],
                    )
                )

    _attach_intervals(
        rows,
        spec=spec,
        benchmark=benchmark,
        per_arm_recall=per_arm_recall,
        reference=reference,
    )
    if spec.name == REFERENCE_MODEL:
        reference_updates = {
            _reference_key(benchmark, arm): cell for arm, cell in per_arm_recall.items()
        }
    return rows, reference_updates


def _attach_intervals(
    rows: list[LadderRow],
    *,
    spec: ModelSpec,
    benchmark: str,
    per_arm_recall: dict[str, RecallByRecord],
    reference: dict[str, RecallByRecord],
) -> None:
    """Fill in the paired confidence intervals on the ``CI_K`` rows, in place.

    Two comparisons, both paired per record and resampled by gold cluster:

    - **prompt arm**: instruct minus none, same model, same index. The only thing
      that differs is the query encoding, so this is as clean a paired test as
      the harness can make.
    - **vs. the shipped default**: this model minus ``REFERENCE_MODEL`` in the
      same arm. Left unset when the reference has not been measured on this
      benchmark — an absent interval is a fact, an invented one is not.

    Both paths store ``interval.observed_difference`` as the point estimate, not
    a difference of aggregates computed elsewhere. That is the whole reason this
    function writes the delta at all: the bootstrap resamples per-record units,
    so the number printed beside an interval has to be the number the interval
    bounds. Both paths also require ``status == "available"`` —
    ``paired_entity_bootstrap`` returns a real ``observed_difference`` with
    ``lower=upper=None`` when there are fewer than two clusters, and publishing
    that delta in a table whose premise is "no delta without an interval" would
    be the same bug wearing a different hat.
    """
    baseline_arm = per_arm_recall.get("none")
    for row in rows:
        if row.k != CI_K or row.status != "ok":
            continue
        current = per_arm_recall.get(row.prompt_arm)
        if current is None:
            continue

        if row.prompt_arm != "none" and baseline_arm is not None:
            interval = paired_interval(baseline_arm, current)
            if interval is not None and interval.status == "available":
                row.prompt_delta = interval.observed_difference
                row.prompt_delta_ci_low = interval.lower
                row.prompt_delta_ci_high = interval.upper
                row.ci_clusters = interval.n_clusters

        reference_cell = reference.get(_reference_key(benchmark, row.prompt_arm))
        if spec.name != REFERENCE_MODEL and reference_cell:
            interval = paired_interval(reference_cell, current)
            if interval is not None and interval.status == "available":
                row.vs_reference_delta = interval.observed_difference
                row.vs_reference_ci_low = interval.lower
                row.vs_reference_ci_high = interval.upper
                row.ci_clusters = interval.n_clusters


# ---------------------------------------------------------------------------
# Persistence + reporting
# ---------------------------------------------------------------------------


def read_rows(path: Path) -> list[LadderRow]:
    """Load previously measured rows (missing file = no rows yet)."""
    if not path.exists():
        return []
    return [LadderRow(**json.loads(line)) for line in path.read_text().splitlines() if line.strip()]


def write_rows(path: Path, rows: Sequence[LadderRow]) -> None:
    """Persist rows, model-then-benchmark ordered so the file diffs readably."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(
        rows, key=lambda r: (r.parameter_count or 0, r.model, r.benchmark, r.prompt_arm, r.k)
    )
    path.write_text("".join(json.dumps(asdict(row)) + "\n" for row in ordered))


def merge_rows(existing: Sequence[LadderRow], fresh: Sequence[LadderRow]) -> list[LadderRow]:
    """Replace a model's rows rather than appending duplicates.

    A re-run must reproduce the committed table, not grow it — so any
    ``(model, benchmark)`` present in ``fresh`` is dropped from ``existing``.

    Re-measuring ``REFERENCE_MODEL`` additionally **clears every other model's
    ``vs_reference_*``** on the benchmarks it touched. Those numbers were
    computed against per-record scores that no longer exist: leaving them would
    publish a delta against a baseline the file cannot reproduce, and nothing
    would have looked wrong. Clearing makes the gap visible until those models
    are re-run.
    """
    replaced = {(row.model, row.benchmark) for row in fresh}
    kept = [row for row in existing if (row.model, row.benchmark) not in replaced]

    refreshed_benchmarks = {row.benchmark for row in fresh if row.model == REFERENCE_MODEL}
    for row in kept:
        if row.benchmark in refreshed_benchmarks:
            row.vs_reference_delta = None
            row.vs_reference_ci_low = None
            row.vs_reference_ci_high = None
    return kept + list(fresh)


def _fmt(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:.{digits}f}"


def _millions(count: int | None) -> str:
    return "n/a" if count is None else f"{count / 1e6:,.1f}M"


def _ci(low: float | None, high: float | None) -> str:
    """Render an interval, marking the ones that contain 0 as inconclusive.

    A **degenerate** ``[0, 0]`` is marked ``(exactly 0)``, not ``(spans 0)``:
    ``low <= 0 <= high`` is true for it, but "not distinguishable from noise" is
    the wrong reading of a benchmark where every record scored identically in
    both arms (``fodors_zagat`` — both arms hit recall 1.0 everywhere). That is
    certainty about a zero effect, which is a result, not an absence of one.
    """
    if low is None or high is None:
        return "n/a"
    if low == high == 0.0:
        return "[+0.0000, +0.0000] (exactly 0)"
    spans_zero = " (spans 0)" if low <= 0.0 <= high else ""
    return f"[{low:+.4f}, {high:+.4f}]{spans_zero}"


def render_report(rows: Sequence[LadderRow], headline_k: int = 20) -> str:
    """Render the markdown report from rows measured at the CURRENT metric revision.

    Rows below :data:`METRIC_REVISION` are **excluded from every table** and
    listed separately as needing a re-run. They are not deleted — the JSONL keeps
    them — but they must not share a column with rows measured under a different
    definition of the same metric. Publishing both is how "instructions badly
    hurt separability for the base-size models" gets read off a table where half
    the rows carry a known one-directional scoring artefact.
    """
    stale = [row for row in rows if row.metric_revision < METRIC_REVISION]
    current = [row for row in rows if row.metric_revision >= METRIC_REVISION]
    ok = [row for row in current if row.status == "ok"]
    failures = [row for row in current if row.status == "failed"]
    benchmarks = sorted({row.benchmark for row in ok} | {row.benchmark for row in failures})
    models = sorted(
        {row.model for row in current},
        key=lambda name: (
            min((r.parameter_count or 10**12) for r in current if r.model == name),
            name,
        ),
    )

    out: list[str] = []
    out.append("# Embedder ladder: which embedding model at which parameter count\n")
    out.append(
        "Generated by `examples/research/embedder_ladder.py`. Every number here is "
        "measured by that script at metric revision "
        f"**{METRIC_REVISION}**; re-running it regenerates this file from "
        "`20260727_embedder_ladder_rows.jsonl`.\n"
    )
    if stale:
        pending = sorted({row.model for row in stale})
        out.append(
            f"\n> **{len(pending)} model(s) are NOT in the tables below**, because they "
            "were measured under an older metric definition and would not be the same "
            "statistic as their neighbours: "
            + ", ".join(f"`{name}`" for name in pending)
            + ". Their rows are still in the JSONL. Re-run them to publish them.\n"
        )

    out.append("\n## How to read this (please read before quoting a number)\n")
    out.append(
        "- **Parameter counts are measured**, not looked up: each is "
        "`sum(p.numel() for p in model.parameters())` from the model that produced "
        "that row's vectors. There is no size table and no small/base/large label.\n"
        "- **The metric is candidate recall and separability AUC, not F1.** Blocking "
        "sets a ceiling; a pair never emitted cannot be recovered downstream.\n"
        "- **Recall alone does not rank models.** Recall is trivially bought with a "
        "bigger `k`. Read it against `cands/recall` (candidates per unit recall) — "
        "the cost of that ceiling — and note that `langres.optimize` already "
        "sweeps `k` as a search axis (`SearchSpace.k_neighbors`, enumerated by "
        "`autoresearch.loop.run_loop`'s keep-if-better loop), so an "
        "operating-point win is not the same as a model win: `optimize()` would "
        "have found the better `k` on its own.\n"
        "- **`recall` cannot reach 1.0 on these benchmarks, and that is not the "
        "model's fault.** Every benchmark here is a two-source *linkage* task, so "
        "the harness keeps only cross-source candidate pairs (mirroring "
        "`langres.optimize._score_loaded`). A gold cluster spanning three or more "
        "records emits intra-source gold pairs that no cross-source candidate set "
        "can contain. `ceiling` is that structural limit and `recall/ceil` is the "
        "share of what was reachable — **`recall/ceil` is the model comparison**; "
        "raw `recall` mixes it with a property of the gold set. (An earlier version "
        "of this harness asserted the filter was recall-neutral. It is not: on "
        "`amazon_google` the ceiling is ~0.84. The same false claim is still "
        "written in `src/langres/optimize.py:135-139` — reported, not silently "
        "edited, since that file is outside this change's scope.)\n"
        "- **This says nothing about deduplication.** All five benchmarks are "
        "cross-source linkage; the registry contains no single-source dedup "
        "benchmark. A within-source blocking ladder is unmeasured here.\n"
        "- **`index build (s)` is only an encoding cost when `enc` is non-zero.** "
        "The harness embeds through a disk cache, so a re-run reads SQLite instead "
        "of the model and the same column then measures a cache read — orders of "
        "magnitude faster. `enc` is the number of texts that actually went through "
        "the model during that build.\n"
        "- **The separability AUC sits near its ceiling and that compresses it.** "
        f"Negatives are a seeded *uniform* sample of up to {NEGATIVE_SAMPLE:,} non-gold "
        "pairs, and a uniformly random pair of records is trivially dissimilar — so "
        "every usable model scores ~0.99 and small AUC gaps are not proportional to "
        "the recall gaps beside them. It is a floor check ('does this model separate "
        "at all'), not a fine-grained ranking. A hard-negative-mined variant would "
        "discriminate better and is not measured here.\n"
        "- **Saturation is not measured here — it is imported.** A ladder measured "
        "on a saturated benchmark measures the benchmark, so every row carries a "
        "`saturation` verdict from a *separate* portfolio stream. This harness "
        "cannot produce that verdict: a one-family ladder cannot distinguish 'all "
        "embedders agree' from 'the task is solved'. `fodors_zagat` is saturated "
        "(the floor, kept as a never-regress check, not as evidence); the other "
        "four are not. Do not average across benchmarks, and do not read this "
        "benchmark set as a finding — it is a documented prior this sweep "
        "inherited.\n"
        "- **The claim that started this sweep was not strong enough to act on.** "
        "A pilot reported `BAAI/bge-small-en-v1.5` at 0.8295 candidate recall "
        "versus `all-MiniLM-L6-v2` at 0.8201 on `amazon_google` — **+0.9pp, one "
        "benchmark, one seed, no interval** — alongside '2.5x fewer candidates at "
        "equal recall'. Those are two different claims: the second is an "
        "*operating-point* comparison over a `k` that `langres.optimize` already "
        "searches, so it is a statement about where each model was sampled, not "
        "about which model is better. This sweep re-measures both across "
        "benchmarks and attaches a cluster-resampled interval to the first — see "
        "'Is it better than what ships today?' below for whether the gap survives "
        "it. (The pilot numbers above are quoted as the motivation, not "
        "re-measured here — read this document's own tables for the "
        "measurement.)\n"
        "- **Reproducible from a git checkout only, not from `pip install langres`.** "
        "The wheel deliberately excludes the `amazon_google`, `abt_buy`, "
        "`walmart_amazon` and `wdc_computers` corpora "
        "(`[tool.hatch.build] exclude` in `pyproject.toml`), so only `fodors_zagat` "
        "here is loadable from a PyPI install; the rest raise "
        "`BenchmarkDataNotFoundError`.\n"
    )

    out.append("\n## Models that were measured\n")
    out.append(
        "\n`dtype` is the load precision. It is shown because half precision is a "
        "**different measurement**, not a free speedup: a row measured in float16 is "
        "not directly comparable to a float32 row, and only the models that do not "
        "fit in unified memory at full precision were loaded that way.\n"
    )
    out.append(
        "\n| model | parameters | dtype | dim | own prompt names |\n|---|---:|---|---:|---|\n"
    )
    for model in models:
        sample = next((row for row in ok if row.model == model), None)
        if sample is None:
            continue
        prompts = ", ".join(sample.registered_prompts) or "—"
        out.append(
            f"| `{model}` | {_millions(sample.parameter_count)} | "
            f"{sample.dtype or 'default (fp32)'} | "
            f"{sample.embedding_dim} | {prompts} |\n"
        )

    out.append(f"\n## Candidate recall at k={headline_k}, no instruction\n")
    out.append(
        "\nThe unprompted operating point, one column per benchmark. "
        "`cands/recall` is the candidate count divided by recall — lower is a "
        "cheaper ceiling.\n"
    )
    for benchmark in benchmarks:
        # Prefer a row that carries the ceiling: rows measured before the ceiling
        # existed are still readable, and picking one of those would print "n/a"
        # for a benchmark that has in fact been measured.
        sample = next(
            (r for r in ok if r.benchmark == benchmark and r.reachable_recall_ceiling is not None),
            next((r for r in ok if r.benchmark == benchmark), None),
        )
        out.append(f"\n### {benchmark}\n")
        if sample is not None:
            out.append(
                f"\n{_fmt(sample.n_records)} records, {_fmt(sample.n_gold_pairs)} gold pairs, "
                f"reachable recall ceiling **{_fmt(sample.reachable_recall_ceiling)}** "
                f"— saturation: {sample.saturation} (imported, see above).\n"
            )
        out.append(
            "\n| model | parameters | recall | recall/ceil | sep. AUC | candidates | "
            "cands/recall | index build (s) | enc |\n"
        )
        out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for model in models:
            row = next(
                (
                    r
                    for r in ok
                    if r.model == model
                    and r.benchmark == benchmark
                    and r.k == headline_k
                    and r.prompt_arm == "none"
                ),
                None,
            )
            if row is None:
                failed = next(
                    (r for r in failures if r.model == model and r.benchmark == benchmark), None
                )
                if failed is not None:
                    out.append(
                        f"| `{model}` | — | **failed** | — | — | — | — | — | {failed.error} |\n"
                    )
                continue
            out.append(
                f"| `{model}` | {_millions(row.parameter_count)} | "
                f"{_fmt(row.candidate_recall)} | {_fmt(row.recall_of_reachable)} | "
                f"{_fmt(row.separability_auc)} | "
                f"{_fmt(row.total_candidates)} | {_fmt(row.candidates_per_unit_recall, 0)} | "
                f"{_fmt(row.index_build_seconds, 1)} | {_fmt(row.index_build_encoded)} |\n"
            )

    out.append(f"\n## Does an instruction prompt help? (k={headline_k})\n")
    out.append(
        "\nSame model, same `k`, same index — only the query side is re-encoded with "
        f"a single fixed instruction (`{INSTRUCTION!r}`). This is deliberately **one** "
        "instruction for every model, so it answers 'does a task instruction on the "
        "query help', not 'is each model at its documented best'. The `own prompt "
        "names` column above shows which models ship a documented recipe this sweep "
        "did not use.\n"
    )
    out.append(
        "\n**`Δ per-record` is the statistic the interval bounds** — the mean "
        "per-record recall difference (instruct minus none), resampled by gold "
        "cluster via `langres.experiments.statistics.paired_entity_bootstrap`; "
        "never by pair row, because pair rows inside one entity are dependent and "
        "resampling them produces intervals that are far too tight. It is "
        "deliberately **not** the difference of the two aggregate `recall` columns "
        "beside it, which is a different number (they disagreed by 34% on "
        "`walmart_amazon`); both are shown so the gap is visible rather than "
        "hidden. **An interval spanning 0 means the delta is not distinguishable "
        "from noise**; `(exactly 0)` means every record scored identically in both "
        "arms, which is certainty about a zero effect, not an absence of one.\n"
    )
    out.append(
        "\n| model | benchmark | recall (none) | recall (instruct) | Δ aggregate | "
        "Δ per-record | 95% CI | AUC (none) | AUC (instruct) | Δ AUC |\n"
    )
    out.append("|---|---|---:|---:|---:|---:|---|---:|---:|---:|\n")
    for model in models:
        for benchmark in benchmarks:
            plain = next(
                (
                    r
                    for r in ok
                    if r.model == model
                    and r.benchmark == benchmark
                    and r.k == headline_k
                    and r.prompt_arm == "none"
                ),
                None,
            )
            prompted = next(
                (
                    r
                    for r in ok
                    if r.model == model
                    and r.benchmark == benchmark
                    and r.k == headline_k
                    and r.prompt_arm == "instruct"
                ),
                None,
            )
            if plain is None or prompted is None:
                continue
            d_recall = (prompted.candidate_recall or 0) - (plain.candidate_recall or 0)
            d_auc = (
                None
                if plain.separability_auc is None or prompted.separability_auc is None
                else prompted.separability_auc - plain.separability_auc
            )
            per_record = (
                "n/a" if prompted.prompt_delta is None else f"{prompted.prompt_delta:+.4f}"
            )
            out.append(
                f"| `{model}` | {benchmark} | {_fmt(plain.candidate_recall)} | "
                f"{_fmt(prompted.candidate_recall)} | {d_recall:+.4f} | {per_record} | "
                f"{_ci(prompted.prompt_delta_ci_low, prompted.prompt_delta_ci_high)} | "
                f"{_fmt(plain.separability_auc)} | {_fmt(prompted.separability_auc)} | "
                f"{'n/a' if d_auc is None else f'{d_auc:+.4f}'} |\n"
            )

    out.append(f"\n## Is it better than what ships today? (k={headline_k}, no instruction)\n")
    out.append(
        f"\nEvery model against `{REFERENCE_MODEL}` — langres's current "
        "`DEFAULT_EMBEDDING_MODEL` — on the same records, paired per record and "
        "resampled by gold cluster. **Δ is the mean per-record recall difference**, "
        "not the difference of the two aggregate recalls above: the bootstrap "
        "resamples per-record units, so the point estimate has to be the same "
        "statistic the interval is built from. `clusters` is the number of "
        "independent units resampled — the honest denominator, and much smaller "
        "than the record count.\n"
        "\n**This sweep does not change the default.** A CI spanning 0 is not "
        "evidence of a better default; it is evidence the measurement cannot tell "
        "them apart on that benchmark.\n"
    )
    out.append(
        "\nA model with no interval is shown with `—` rather than dropped: a "
        "measured model that vanishes from the decision table is exactly the "
        "silent skip this harness refuses to do with failures.\n"
    )
    out.append("\n| model | benchmark | Δ per-record recall | 95% CI | clusters |\n")
    out.append("|---|---|---:|---|---:|\n")
    for model in models:
        if model == REFERENCE_MODEL:
            continue
        for benchmark in benchmarks:
            row = next(
                (
                    r
                    for r in ok
                    if r.model == model
                    and r.benchmark == benchmark
                    and r.k == headline_k
                    and r.prompt_arm == "none"
                ),
                None,
            )
            if row is None:
                continue
            if row.vs_reference_delta is None:
                out.append(
                    f"| `{model}` | {benchmark} | — | not measured against the "
                    f"current `{REFERENCE_MODEL}` reference | — |\n"
                )
                continue
            out.append(
                f"| `{model}` | {benchmark} | {row.vs_reference_delta:+.4f} | "
                f"{_ci(row.vs_reference_ci_low, row.vs_reference_ci_high)} | "
                f"{_fmt(row.ci_clusters)} |\n"
            )

    out.append("\n## The recall/cost frontier (every k)\n")
    out.append(
        "\nRecall is bought with `k`. This is the table that makes an "
        "operating-point comparison distinguishable from a model comparison: two "
        "models at equal recall but different candidate counts differ in cost, not "
        "in ceiling.\n"
    )
    out.append("\n| model | benchmark | prompt | k | recall | candidates | cands/recall |\n")
    out.append("|---|---|---|---:|---:|---:|---:|\n")
    for row in sorted(
        ok, key=lambda r: (r.parameter_count or 0, r.model, r.benchmark, r.prompt_arm, r.k)
    ):
        out.append(
            f"| `{row.model}` | {row.benchmark} | {row.prompt_arm} | {row.k} | "
            f"{_fmt(row.candidate_recall)} | {_fmt(row.total_candidates)} | "
            f"{_fmt(row.candidates_per_unit_recall, 0)} |\n"
        )

    out.append("\n## Failures (reported, not skipped)\n")
    if failures:
        out.append("\n| model | benchmark | k | error |\n|---|---|---:|---|\n")
        for row in failures:
            out.append(f"| `{row.model}` | {row.benchmark} | {row.k} | {row.error} |\n")
    else:
        out.append("\nNo model failed to load or run in the rows recorded here.\n")

    return "".join(out)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the sweep for the requested models and regenerate the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", default=[m.name for m in MODELS])
    parser.add_argument("--benchmarks", nargs="*", default=list(BENCHMARKS))
    parser.add_argument("--k", nargs="*", type=int, default=list(K_VALUES))
    parser.add_argument("--prompts", nargs="*", default=list(PROMPT_ARMS))
    parser.add_argument("--rows", type=Path, default=DEFAULT_ROWS_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE_PATH)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--device", default=None, help="torch device, e.g. mps / cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--headline-k", type=int, default=20)
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="Regenerate the report from the recorded rows without measuring anything.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.render_only:
        rows = read_rows(args.rows)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(render_report(rows, headline_k=args.headline_k))
        logger.info("rendered %d rows to %s", len(rows), args.report)
        return

    arms = {name: PROMPT_ARMS[name] for name in args.prompts}

    for model_name in args.models:
        spec = MODELS_BY_NAME.get(model_name) or ModelSpec(model_name)
        # Re-read per model: the reference model may have been measured earlier in
        # THIS sweep, and a stale in-memory copy would silently skip its CIs.
        reference = read_reference(args.reference)
        fresh: list[LadderRow] = []
        reference_updates: dict[str, RecallByRecord] = {}
        for benchmark in args.benchmarks:
            logger.info("=== %s on %s ===", spec.name, benchmark)
            rows, updates = evaluate_model_on_benchmark(
                spec,
                benchmark,
                k_values=args.k,
                prompt_arms=arms,
                cache_dir=args.cache_dir,
                device=args.device,
                batch_size=args.batch_size,
                reference=reference,
            )
            fresh.extend(rows)
            reference_updates.update(updates)
        # Persist after EVERY model: a long sweep must be durable at each step,
        # not only at the end.
        if reference_updates:
            write_reference(args.reference, {**reference, **reference_updates})
        merged = merge_rows(read_rows(args.rows), fresh)
        write_rows(args.rows, merged)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(render_report(merged, headline_k=args.headline_k))
        logger.info("wrote %d rows for %s", len(fresh), spec.name)


if __name__ == "__main__":
    main()
