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

#: Arm name -> ``(document_prompt, query_prompt)``.
#:
#: A pair, not a single string, because the recipe that matters is **asymmetric**.
#: EmbeddingGemma's model card prefixes documents with ``"title: none | text: "``
#: and queries with ``"task: search result | query: "``; prefixing queries against
#: *bare* documents — which is what ``instruct`` does — is a structurally
#: different configuration, not the documented one with different wording.
PROMPT_ARMS: dict[str, tuple[str | None, str | None]] = {
    "none": (None, None),
    "instruct": (None, INSTRUCTION),
}


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
    #: ``(document_prefix, query_prefix)`` from the checkpoint's OWN
    #: ``config_sentence_transformers.json`` — the recipe a user following the
    #: model card would actually run. Set ONLY for checkpoints trained with a
    #: query-side instruction: for every other model the generic ``instruct`` arm
    #: already answers the question, and a second flavour of the same negative is
    #: not worth the queue time.
    documented_arm: tuple[str | None, str] | None = None
    #: The licence identifier **the checkpoint itself declares**, read from its
    #: own model card (the ``license:`` key in the README front matter, which is
    #: what the Hub API's ``cardData.license`` summarises). Not remembered, not
    #: inferred from the publisher: this repo has already been wrong twice in
    #: opposite directions about a dataset licence it did not read.
    #:
    #: It is a first-class field rather than a footnote because langres is
    #: Apache-2.0, so a ladder that ranks checkpoints without stating whether the
    #: winner may be shipped as a default is ranking on the wrong axis.
    license: str = "unknown"


#: Licences that are OSI-approved, so a langres default carrying one adds no
#: use restriction on top of Apache-2.0. Deliberately an ALLOW list: an unknown
#: or new licence must read as "not clearly OSI" and require a human to look,
#: rather than passing by absence from a deny list.
OSI_APPROVED_LICENSES: frozenset[str] = frozenset({"apache-2.0", "mit", "bsd-3-clause"})


def _is_osi(spec: ModelSpec) -> bool:
    """Whether ``spec``'s declared licence is on the OSI allow list."""
    return spec.license in OSI_APPROVED_LICENSES


#: Listed in roughly ascending expected size so a truncated sweep still covers
#: the cheap tiers completely. Ordering is a *scheduling* hint only — every
#: published number comes from the measured ``parameter_count``.
#:
#: **Scheduling caveat, learned the expensive way.** Ascending size is the right
#: default for *recall vs. parameters*, but it is the wrong order for the prompt
#: axis: every model in the cheap half (MiniLM, mpnet, BGE v1.5) was trained
#: WITHOUT a query-side instruction, so a sweep truncated there measures only
#: "what does a prompt do to a model that never saw one" and cannot speak to
#: instruction-following at all. When time is short, run
#: ``google/embeddinggemma-300m`` and a ``Qwen/Qwen3-Embedding-*`` tier before
#: finishing the ladder — they are the ones that make the axis mean anything.
#:
#: Every ``license=`` below was read from that checkpoint's own model card on
#: 2026-07-27 (README front matter / Hub ``cardData.license``). Exactly one is
#: not OSI: ``google/embeddinggemma-300m`` ships under the **Gemma Terms of
#: Use**, which carries a prohibited-use policy that survives redistribution —
#: a use restriction Apache-2.0 does not have.
MODELS: tuple[ModelSpec, ...] = (
    ModelSpec("all-MiniLM-L6-v2", license="apache-2.0"),
    ModelSpec("all-MiniLM-L12-v2", license="apache-2.0"),
    ModelSpec("BAAI/bge-small-en-v1.5", license="mit"),
    ModelSpec("all-mpnet-base-v2", license="apache-2.0"),
    ModelSpec("BAAI/bge-base-en-v1.5", license="mit"),
    ModelSpec("intfloat/e5-base-v2", license="mit"),
    ModelSpec("Alibaba-NLP/gte-base-en-v1.5", trust_remote_code=True, license="apache-2.0"),
    ModelSpec("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True, license="apache-2.0"),
    # Prefixes read from the checkpoint's own config_sentence_transformers.json
    # (Retrieval-document / Retrieval-query), not from memory or a model card.
    ModelSpec(
        "google/embeddinggemma-300m",
        documented_arm=("title: none | text: ", "task: search result | query: "),
        license="gemma",
    ),
    ModelSpec("BAAI/bge-large-en-v1.5", license="mit"),
    ModelSpec("mixedbread-ai/mxbai-embed-large-v1", license="apache-2.0"),
    # Qwen3's own recipe is query-side only (its "document" prompt is ""), and
    # its query instruction is about retrieving WEB SEARCH PASSAGES, not matching
    # entities -- so this arm measures "the documented recipe applied outside its
    # stated domain", which is what a user following the model card would get.
    ModelSpec(
        "Qwen/Qwen3-Embedding-0.6B",
        documented_arm=(
            None,
            "Instruct: Given a web search query, retrieve relevant passages that "
            "answer the query\nQuery:",
        ),
        license="apache-2.0",
    ),
    ModelSpec("Qwen/Qwen3-Embedding-4B", dtype="float16", batch_size=8, license="apache-2.0"),
    ModelSpec("Qwen/Qwen3-Embedding-8B", dtype="float16", batch_size=4, license="apache-2.0"),
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
    #: Mean **per-record** recall difference (this row's arm minus ``none``) for
    #: this model on this benchmark, with its paired-bootstrap CI (by gold
    #: cluster). Set on prompted rows at ``CI_K`` — ``instruct`` (query side only)
    #: and ``documented`` (the checkpoint's own document AND query prefixes, so
    #: the corpus index is rebuilt). The two are not the same experiment; only
    #: ``instruct`` isolates the query side. The point estimate is the statistic the
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


def arms_for(
    spec: ModelSpec, base_arms: dict[str, tuple[str | None, str | None]]
) -> dict[str, tuple[str | None, str | None]]:
    """The prompt arms to measure for ``spec``.

    Adds a ``documented`` arm only for checkpoints that ship a query-side
    instruction recipe. The distinction the third arm buys: ``instruct`` answers
    "does *any* instruction help", while ``documented`` answers "should I follow
    the model card" — and only the second is the configuration a real user ends
    up running, so it is the one a shipping default should be chosen on.
    """
    if spec.documented_arm is None:
        return dict(base_arms)
    return {**base_arms, "documented": spec.documented_arm}


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

    ``REFERENCE_MODEL``, ``CI_K`` and ``METRIC_REVISION`` are in the key, not just
    in the filename: each is a one-line constant, and if any changed while the
    file stayed put, ``read_reference`` would return the old contents as the
    baseline and every ``vs_reference_*`` would silently become a comparison
    against a different model, a different ``k``, or a different *definition of
    recall* — with no error and no visible change in the report. The metric
    revision is the one that bites hardest: the rows measured under it are
    excluded from every table, but the per-record sidecar they produced is not a
    table and would be read straight back in. (Found by cross-model review.)
    """
    return f"{REFERENCE_MODEL}|k{CI_K}|rev{METRIC_REVISION}|{benchmark}|{arm}"


def read_reference(path: Path) -> dict[str, RecallByRecord]:
    """Load the reference model's per-record recall, keyed by :func:`_reference_key`."""
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    return {
        key: {record_id: (float(value[0]), str(value[1])) for record_id, value in cell.items()}
        for key, cell in raw.items()
    }


def refresh_reference(
    existing: dict[str, RecallByRecord],
    updates: dict[str, RecallByRecord],
    touched: set[str],
) -> dict[str, RecallByRecord]:
    """The sidecar after a reference run, with the cells it invalidated removed.

    ``updates`` carries only the cells that **succeeded**, so merging it over
    ``existing`` is not enough: an arm that failed on a re-run writes a failure
    row (and :func:`merge_rows` voids the rows it invalidated), while its old
    per-record recall survives in the sidecar. A later model then reads that
    cell and publishes a ``vs_reference_*`` interval against a reference
    measurement the current rows no longer contain — a confidence interval whose
    baseline exists nowhere in the data.

    So every cell the run *attempted* is voided first, and only the ones that
    produced a measurement come back. The sidecar can then never outlive its
    rows. (Found by cross-model review.)
    """
    kept = {key: cell for key, cell in existing.items() if key not in touched}
    return {**kept, **updates}


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


#: Text encoded once per model per run to prove the warm cache still belongs to
#: the checkpoint that is loaded now. Content is irrelevant — only that it is
#: fixed forever, so the entry written on the first run is the one compared
#: against on every later one.
_CANARY_TEXT = "langres embedder-ladder cache canary"

#: How far a cached canary vector may sit from a freshly encoded one before the
#: cache is declared stale. Generous on purpose: run-to-run float noise (device,
#: batch composition, BLAS version) is ~1e-6 on normalized vectors, while a
#: different checkpoint moves them by order 1e-1. Anything between those is not
#: a case worth guessing about — it is a case worth stopping on.
_CANARY_TOLERANCE = 1e-4


class StaleEmbeddingCacheError(RuntimeError):
    """The cached vectors were produced by a different checkpoint than the loaded one."""


def _cache_entry_count(db_path: Path) -> int:
    """How many vectors ``db_path`` already holds; 0 if it does not exist yet.

    Read directly rather than via ``cache_info()``, which reports hit/miss
    *counters for this process* and is therefore 0 for a cache written last week.
    """
    import sqlite3

    if not db_path.exists():
        return 0
    with sqlite3.connect(db_path) as conn:
        return int(conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0])


def _canary_is_cached(cached: Any, db_path: Path) -> bool:
    """Is the canary already in ``db_path``, *without* putting it there?

    Asking by encoding would answer the question by changing it: a legacy
    namespace would gain a canary row on the very run that refuses it, and the
    NEXT run would then find a canary present and pass — so re-running would
    bypass the refusal. The refusal has to be idempotent, which means reading the
    key rather than writing it.

    Uses ``DiskCachedEmbedder._hash_text``, which is private, on purpose: the
    whole point is to ask for *the exact key the cache would use*, including its
    embedder discriminator. Re-deriving it here would be a second implementation
    of the key, free to drift from the first — the failure mode this file keeps
    arguing against.
    """
    import sqlite3

    if not db_path.exists():
        return False
    key = cached._hash_text(_CANARY_TEXT)
    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT 1 FROM embeddings WHERE text_hash = ?", (key,)).fetchone()
    return row is not None


def _assert_cache_matches_checkpoint(
    base: Any, cached: Any, namespace: str, cache_dir: Path, *, adopt_legacy: bool = False
) -> None:
    """Stop the run if the warm cache no longer agrees with the loaded checkpoint.

    The namespace is keyed on model name + dtype, **not** on a Hub revision. So
    if a checkpoint is re-uploaded under the same name, the namespace still hits:
    the run would read the NEW checkpoint's ``parameter_count``/``embedding_dim``
    while reusing the OLD checkpoint's vectors, and publish a row mixing the two.

    Putting the revision in the namespace would close that — and invalidate every
    cached vector in the ladder to do it. This closes it without re-encoding
    anything: encode one fixed canary through the cache (served from disk on a
    warm run) and through the base embedder (always fresh), and compare. They can
    only disagree if the cached vectors came from something the loaded model no
    longer is.

    That is deliberately a *stronger* check than a revision pin rather than a
    cheaper one. A revision pin answers "is this the same Hub commit"; this
    answers "does this checkpoint still produce these vectors", which is the
    property the measurements actually depend on. It therefore also catches
    drift a revision cannot see — a sentence-transformers upgrade that changes
    pooling, a tokenizer fix, a different dtype path on a new device.

    **A cache that predates the canary cannot be vouched for, and must not pass
    silently.** On such a cache the canary simply *misses*: it is encoded fresh
    from the checkpoint loaded right now, written to the otherwise-unvouched
    database, and then compared against another fresh encoding of itself. It
    always matches — while every corpus vector beside it may still belong to a
    different checkpoint. That is the same defect this function exists to close,
    reintroduced one level up, so the entry count is read *before* the canary is
    written: a non-empty namespace with no canary in it is refused outright.

    Args:
        adopt_legacy: Vouch for an existing unverified namespace instead of
            refusing it (``--trust-existing-cache``). This exists because the
            refusal above is otherwise retroactive: every namespace written
            before this check — 1.8 GB and hours of encoding, for the ladder as
            it stands — has no canary and would demand a full re-measure of a
            cache the operator knows is current. It vouches **once**: the canary
            written during adoption pins the checkpoint, so the next run is
            checked normally. It is a human assertion, and it is logged as one.

    Raises:
        StaleEmbeddingCacheError: Naming the namespace file to delete. Deliberately
            fatal rather than a warning: the failure it guards against is one that
            publishes a plausible number, and a warning scrolls past.
    """
    import numpy as np

    db_path = cache_dir / f"{namespace}.db"
    before = _cache_entry_count(db_path)

    # Asked WITHOUT writing, so a refused run leaves the namespace exactly as it
    # found it. Encoding first would answer the question by changing it: the
    # refused run would deposit a canary, and the next run would find one present
    # and sail through -- a refusal you get past by running it twice.
    if before > 0 and not _canary_is_cached(cached, db_path):
        if not adopt_legacy:
            raise StaleEmbeddingCacheError(
                f"the embedding cache for namespace {namespace!r} holds {before} vectors "
                f"written before this check existed, so nothing in it has ever been "
                f"verified against a checkpoint. It may predate an upstream re-upload, "
                f"and a run using it would publish rows mixing two checkpoints. Delete "
                f"{db_path} and re-measure this model; the cache it writes will be "
                f"checked from its first run onward. If you know this cache was written "
                f"by the checkpoint loaded now, pass --trust-existing-cache to adopt it "
                f"-- that vouches for it once and pins the canary from then on."
            )
        # Adoption is a human assertion, so it is logged as one, and it PINS: the
        # canary written here is what the next run compares against, so the flag
        # vouches for a cache once rather than turning the check off.
        cached.encode([_CANARY_TEXT])
        logger.warning(
            "--trust-existing-cache: adopting %d unverified vectors in namespace %r as "
            "belonging to the checkpoint loaded now. Nothing verified this; you did. "
            "Subsequent runs are checked against the canary written now.",
            before,
            namespace,
        )
        return

    from_cache = np.asarray(cached.encode([_CANARY_TEXT])[0], dtype=np.float64)
    fresh = np.asarray(base.encode([_CANARY_TEXT])[0], dtype=np.float64)

    if from_cache.shape != fresh.shape:
        drift: float = float("inf")
    elif not (np.isfinite(from_cache).all() and np.isfinite(fresh).all()):
        # A NaN anywhere makes `drift` NaN, and `NaN > tolerance` is **False** --
        # so the guard would ACCEPT a cache it cannot compare and the ladder would
        # publish rows computed from non-finite vectors. Unstable half precision
        # on some devices and a truncated cached blob both produce this. Treat it
        # as maximal drift: unusable is not the same as equal. (Cross-model review.)
        drift = float("inf")
    else:
        drift = float(np.abs(from_cache - fresh).max())

    if drift > _CANARY_TOLERANCE:
        raise StaleEmbeddingCacheError(
            f"the embedding cache for namespace {namespace!r} was written by a "
            f"different checkpoint than the one loaded now (canary vectors differ "
            f"by {drift:.3g} > {_CANARY_TOLERANCE:g}). Every cached vector in it is "
            f"suspect, so continuing would publish rows mixing two checkpoints. "
            f"Delete {cache_dir / f'{namespace}.db'} and re-measure this model."
        )


def _build_embedder(
    spec: ModelSpec,
    cache_dir: Path,
    device: str | None,
    batch_size: int,
    *,
    adopt_legacy_cache: bool = False,
) -> Any:
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
    #
    # It does NOT carry a Hub revision. Adding one would invalidate every cached
    # vector in the ladder, so the re-upload hazard is caught by measurement
    # instead: `_assert_cache_matches_checkpoint` re-encodes one fixed canary and
    # stops the run if the warm cache disagrees with the loaded model. That needs
    # no revision, costs one short encode, and catches drift a revision cannot
    # see (a pooling change, a tokenizer fix, a different dtype path).
    namespace = f"{spec.name.replace('/', '__')}__{spec.dtype or 'default'}"
    cached = DiskCachedEmbedder(base, cache_dir=cache_dir, namespace=namespace)
    _assert_cache_matches_checkpoint(
        base, cached, namespace, cache_dir, adopt_legacy=adopt_legacy_cache
    )
    return base, cached


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


def build_prompted_index(
    embedder: Any, texts: Sequence[str], document_prompt: str | None
) -> tuple[Any, np.ndarray, float, int]:
    """Index the corpus under one document-side prefix.

    Returns ``(index, corpus vectors, build seconds, texts encoded)``.

    The prefix is applied by plain string concatenation rather than by binding
    ``prompt_name=`` on the embedder. **Not because the latter does not work** —
    it does, and ``tests/core/blockers/test_asymmetric_prompt_recipe.py`` proves
    the two produce identical results — but because these rows must keep
    reproducing the table measured before that recipe was documented. An earlier
    version of this docstring called it a product gap; that claim is corrected in
    the report. The two are equivalent for the checkpoints measured here,
    verified rather than assumed:
    sentence-transformers applies a prompt as ``prompt + text``
    (``sentence_transformers/base/model.py:560``) and only excludes its tokens
    from pooling when ``include_prompt=False``
    (``sentence_transformers/sentence_transformer/modules/pooling.py:125``),
    which both documented checkpoints ship as ``true``. A checkpoint shipping
    ``include_prompt=false`` would NOT be equivalent and must not be added to
    ``documented_arm`` without changing this.

    **The query side is rebound to the bare text**, and that line is the whole
    reason this is a function worth testing. ``create_index`` snapshots the texts
    it was handed, and ``search_all(query_prompt=...)`` re-encodes *those* to
    build the query vectors — so on a documented arm the query would come out as
    ``query_prompt + document_prompt + text``, a double-prefixed recipe no model
    card describes, and the arm's recall would measure something nobody asked
    for. Reaching behind ``create_index`` is not elegance; it is the cost of
    hand-prefixing the corpus instead of using ``prompt_name=``, which does not
    have this problem because the prefix lives on the embedder, not in the
    snapshotted text.
    """
    from langres.core.indexes.vector_index import FAISSIndex

    doc_texts = [document_prompt + text for text in texts] if document_prompt else list(texts)
    # `misses` is cumulative over the embedder, so the delta — not the total — is
    # what this build encoded. Zero means the seconds below are a SQLite read,
    # not an encode, and must not be quoted as an encoding cost.
    before = int(embedder.cache_info()["misses"])
    started = time.perf_counter()
    index = FAISSIndex(embedder=embedder, metric="cosine")
    index.create_index(doc_texts)
    seconds = time.perf_counter() - started
    encoded = int(embedder.cache_info()["misses"]) - before
    if document_prompt:
        index._corpus_texts = list(texts)
    return index, embedder.encode(doc_texts), seconds, encoded


def evaluate_model_on_benchmark(
    spec: ModelSpec,
    benchmark: str,
    *,
    k_values: Sequence[int],
    prompt_arms: dict[str, tuple[str | None, str | None]],
    cache_dir: Path,
    device: str | None,
    batch_size: int,
    reference: dict[str, RecallByRecord] | None = None,
    adopt_legacy_cache: bool = False,
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

        base, embedder = _build_embedder(
            spec, cache_dir, device, batch_size, adopt_legacy_cache=adopt_legacy_cache
        )

        # Force the load HERE, before any arm runs, so a checkpoint that cannot
        # load at all is one failure row rather than one per arm.
        parameter_count = base.parameter_count
        embedding_dim = base.embedding_dim
        prompts = _registered_prompts(base)
    except StaleEmbeddingCacheError:
        # NOT a result. Every other exception here is a fact about the model --
        # it did not load, it ran out of memory -- and recording it as a failure
        # row is the honest thing. A cache-integrity refusal is a fact about the
        # HARNESS, and turning it into a row is actively destructive: `main()`
        # persists that row and `merge_rows()` voids every previously recorded
        # cell for this (model, benchmark), so a refusal would DELETE good
        # measurements from the tracked jsonl. `run_ladder.sh` would then see
        # exit 0 and commit the deletion. Let it out. (Cross-model review.)
        raise
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

    #: ``document_prompt -> (index, corpus vectors, build seconds, texts encoded)``.
    #: Keyed by the document-side prefix because arms that share one share the
    #: index: only the arms that actually change the corpus pay to rebuild it.
    built: dict[str | None, tuple[Any, np.ndarray, float, int]] = {}

    for arm, (document_prompt, query_prompt) in prompt_arms.items():
        try:
            if document_prompt not in built:
                built[document_prompt] = build_prompted_index(embedder, texts, document_prompt)
            index, doc_vectors, index_build_seconds, index_build_encoded = built[document_prompt]
            # `prompt=None` is a cache hit on the corpus encode whenever the arm
            # leaves the document side bare, so this is one line, not a branch.
            query_vectors = embedder.encode(texts, prompt=query_prompt)
            auc = separability_auc(doc_vectors, query_vectors, ids, gold_pairs)
        except Exception as exc:  # noqa: BLE001
            logger.exception("model %s failed on %s arm=%s", spec.name, benchmark, arm)
            rows.append(
                LadderRow(
                    model=spec.name,
                    benchmark=benchmark,
                    prompt_arm=arm,
                    k=0,
                    status="failed",
                    parameter_count=parameter_count,
                    dtype=spec.dtype,
                    metric_revision=METRIC_REVISION,
                    error=f"{type(exc).__name__}: {exc}"[:400],
                )
            )
            continue

        for k in k_values:
            try:
                blocker = VectorBlocker(
                    vector_index=index,
                    schema=schema,
                    text_field_extractor="concat_comparable_fields",
                    k_neighbors=k,
                    query_prompt=query_prompt,
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

    - **prompt arm**: this arm minus ``none``, same model. What differs depends on
      the arm and they are NOT the same experiment: ``instruct`` changes the query
      encoding only, against the same bare-document index; ``documented`` also
      applies the checkpoint's document prefix, so the corpus vectors are rebuilt
      and the delta is document-plus-query vs. neither. Both are paired per record
      the same way; only ``instruct`` is a query-only test.
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
    """Replace the re-measured cells rather than appending duplicates.

    A re-run must reproduce the committed table, not grow it — so every
    ``(model, benchmark, arm, k)`` cell present in ``fresh`` replaces its
    predecessor, and a **failure row** (``k == 0``) additionally voids what it
    stands for: ``prompt_arm == "-"`` means the model never loaded, so the whole
    ``(model, benchmark)`` cell is gone; a named arm voids that arm alone.
    Without the second rule a failed re-run would leave the previous run's
    successful rows sitting beside the new failure, and the report would print
    both.

    Re-measuring ``REFERENCE_MODEL`` additionally **clears every other model's
    ``vs_reference_*``** on the benchmarks it touched. Those numbers were
    computed against per-record scores that no longer exist: leaving them would
    publish a delta against a baseline the file cannot reproduce, and nothing
    would have looked wrong. Clearing makes the gap visible until those models
    are re-run.
    """
    # Replace the exact CELLS that were re-measured, not every row of the
    # ``(model, benchmark)`` pair. `--k` and `--prompts` make a partial re-run a
    # normal thing to do, and coarser keying silently deleted the arms and
    # operating points that run did not touch: the file shrank, the report
    # rendered, and only the "what did not run" section would ever have said so.
    # (Found by cross-model review, which re-ran one arm and watched the others
    # disappear.)
    replaced = {(row.model, row.benchmark, row.prompt_arm, row.k) for row in fresh}
    void_cells = {(r.model, r.benchmark) for r in fresh if r.k == 0 and r.prompt_arm == "-"}
    void_arms = {
        (r.model, r.benchmark, r.prompt_arm) for r in fresh if r.k == 0 and r.prompt_arm != "-"
    }
    kept = [
        row
        for row in existing
        if (row.model, row.benchmark) not in void_cells
        and (row.model, row.benchmark, row.prompt_arm) not in void_arms
        and (row.model, row.benchmark, row.prompt_arm, row.k) not in replaced
    ]

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


def _excludes_zero(low: float | None, high: float | None) -> bool:
    """Whether an interval is entirely on one side of 0. Absent bounds are not."""
    return low is not None and high is not None and not (low <= 0.0 <= high)


def _render_recommendation(
    ok: Sequence[LadderRow],
    models: Sequence[str],
    benchmarks: Sequence[str],
    headline_k: int,
) -> list[str]:
    """The recommendation, split by licence and derived entirely from the rows.

    Two separate questions, deliberately not merged into one ranking:

    1. **What is the best model measured here?** Answered by the table above,
       which does not care about licences.
    2. **What may langres ship as a default?** langres is Apache-2.0, so a
       checkpoint under a use-restricted licence can be a *documented opt-in*
       with its terms stated, but not a silent default. That is a licence fact,
       not a measurement, so the two are reported side by side rather than
       collapsed into a single "winner".

    Nothing here is hand-written except the licence identifiers on
    :data:`MODELS`, which were read from the model cards. The winners, the
    counts, and the coverage denominator all come from ``ok``.
    """

    def wins(model: str) -> list[LadderRow]:
        """Benchmarks where ``model`` beats the reference with the CI clear of 0."""
        found = []
        for benchmark in benchmarks:
            row = next(
                (
                    r
                    for r in ok
                    if r.model == model
                    and r.benchmark == benchmark
                    and r.k == headline_k
                    and r.prompt_arm == "none"
                    and r.vs_reference_delta is not None
                    and r.vs_reference_delta > 0
                    and _excludes_zero(r.vs_reference_ci_low, r.vs_reference_ci_high)
                ),
                None,
            )
            if row is not None:
                found.append(row)
        return found

    measured = sorted({row.model for row in ok})
    candidates = [name for name in models if name != REFERENCE_MODEL and name in measured]
    osi = [name for name in candidates if _is_osi(MODELS_BY_NAME.get(name) or ModelSpec(name))]
    restricted = [name for name in candidates if name not in osi]

    out: list[str] = []
    out.append(f"\n## Recommendation (k={headline_k}, no instruction)\n")
    # The denominator is the LADDER plus every model that RAN, not ``len(MODELS)``
    # and not the successes. ``--models`` accepts a name outside the fixed tuple
    # (main() falls back to a bare ``ModelSpec``), so such a model never appears in
    # ``MODELS`` — and a fixed denominator would print "15 of the 14 models" and
    # point at a 'What did not run' section that only iterates ``MODELS`` and so
    # cannot account for the difference. ``models`` and not ``measured``: measured
    # is derived from ``ok`` rows only, so a custom model that FAILED would drop
    # out of the denominator entirely while still being listed in the Failures
    # table — the ladder would appear to shrink because a run went badly.
    ladder = {spec.name for spec in MODELS} | set(models)
    out.append(
        f"\n**{len(measured)} of the {len(ladder)} models in the ladder have a row at "
        f"metric revision {METRIC_REVISION}.** Everything below is a statement about "
        "those and only those; the rest are named under 'What did not run'. A "
        "recommendation drawn from a partial field is still a recommendation, but it "
        "is not a survey — do not read the absence of a model here as evidence "
        "against it.\n"
        "\n**This document does not change `DEFAULT_EMBEDDING_MODEL`.** It states what "
        "was measured and what the licences are; the default is a human decision.\n"
    )

    out.append("\n### The OSI-licensed field — the only candidates for a default\n")
    out.append(
        f"\nlangres ships under Apache-2.0. A default that carries a use-restricted "
        f"licence pushes that restriction onto every user who never chose it, so the "
        f"candidates for `DEFAULT_EMBEDDING_MODEL` are exactly the OSI-licensed "
        f"models — including the incumbent, `{REFERENCE_MODEL}` "
        f"({(MODELS_BY_NAME.get(REFERENCE_MODEL) or ModelSpec(REFERENCE_MODEL)).license}).\n"
    )
    if not osi:
        out.append(
            "\n**No OSI-licensed challenger has a row at this metric revision**, so "
            "this sweep has nothing to say about replacing the default. That is a gap "
            "in the measurement, not a verdict on the incumbent.\n"
        )
    else:
        out.append("\n| model | licence | benchmarks beaten (CI clear of 0) | best Δ | on |\n")
        out.append("|---|---|---:|---:|---|\n")
        for name in osi:
            spec = MODELS_BY_NAME.get(name) or ModelSpec(name)
            won = wins(name)
            if won:
                best = max(won, key=lambda r: r.vs_reference_delta or 0.0)
                out.append(
                    f"| `{name}` | {spec.license} | {len(won)} of {len(benchmarks)} | "
                    f"{best.vs_reference_delta:+.4f} | {best.benchmark} |\n"
                )
            else:
                out.append(f"| `{name}` | {spec.license} | 0 of {len(benchmarks)} | — | — |\n")
        # Rank on wins first, then on the largest single win, then on the name so
        # the file is byte-stable across re-renders. A tie broken by name alone
        # would silently promote a model for being alphabetically early.
        ranked = [
            (
                len(wins(name)),
                max((r.vs_reference_delta or 0.0) for r in wins(name)) if wins(name) else 0.0,
                name,
            )
            for name in osi
        ]
        best_count, _best_delta, best_name = max(ranked)
        if best_count == 0:
            out.append(
                f"\n**No OSI-licensed model beats `{REFERENCE_MODEL}` on any benchmark "
                "with an interval clear of zero.** The measured recommendation is "
                "therefore to **keep the current default** — not because the "
                "challengers are bad, but because on this evidence the measurement "
                "cannot tell them apart, and 'indistinguishable' is not a reason to "
                "move.\n"
            )
        else:
            out.append(
                f"\n**Best OSI-licensed candidate: `{best_name}`**, ahead of "
                f"`{REFERENCE_MODEL}` on {best_count} of {len(benchmarks)} benchmarks "
                "with the interval clear of zero. Read it against the same model's "
                "row in the table above before adopting it: a win on some benchmarks "
                "and a loss on others is the normal shape here, and this column counts "
                "only the wins.\n"
            )

    out.append("\n### Use-restricted checkpoints — documented opt-in, never a silent default\n")
    if not restricted:
        out.append(
            "\nNo measured model carries a non-OSI licence, so this section has no "
            "entries at this metric revision.\n"
        )
    else:
        for name in restricted:
            spec = MODELS_BY_NAME.get(name) or ModelSpec(name)
            won = wins(name)
            summary = (
                ", ".join(f"{r.benchmark} {r.vs_reference_delta:+.4f}" for r in won)
                if won
                else "no benchmark, with an interval clear of zero"
            )
            out.append(
                f"\n- **`{name}` — licence `{spec.license}`, which is NOT OSI-approved.** "
                f"Measured ahead of `{REFERENCE_MODEL}` on: {summary}. Recommended as a "
                "**documented opt-in**: a user who names it accepts its terms; a user "
                "who names nothing must not be given them. Anyone shipping it must read "
                "the checkpoint's own licence — in Gemma's case a prohibited-use policy "
                "that survives redistribution, which Apache-2.0 does not impose.\n"
                f"\n  ```python\n"
                f"  # opt in explicitly, having read the licence\n"
                f'  SentenceTransformerEmbedder("{name}")\n'
                f"  ```\n"
            )

    return out


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

    # Provenance, derived from the rows rather than typed. A number whose
    # producing command is not written down cannot be sourced later, and an
    # unsourceable number is one that eventually has to be retracted.
    measured_ks = sorted({row.k for row in ok})
    out.append("\n## How to reproduce these numbers\n")
    out.append(
        "\n```bash\n"
        "# every model, committing and pushing after each one\n"
        "bash examples/research/run_ladder.sh\n"
        "\n# or one model / a subset (space-separated)\n"
        'LADDER_MODELS="intfloat/e5-base-v2" bash examples/research/run_ladder.sh\n'
        "\n# render this file from the existing rows, measuring nothing\n"
        "uv run python examples/research/embedder_ladder.py --render-only\n"
        "```\n"
    )
    out.append(
        f"\n**$0 — no paid API and no key.** Not offline, though: the first run of a "
        "checkpoint downloads it from the Hugging Face Hub "
        "(`SentenceTransformerEmbedder` leaves `local_files_only` false), and `uv run` "
        "may resolve dependencies. With networking disabled and a cold cache the "
        "script records failure rows rather than reproducing this table. Once the "
        "checkpoints and the embedding cache are warm, a re-render is genuinely "
        "offline. Metric revision "
        f"**{METRIC_REVISION}**; `k` values `{measured_ks}`; benchmarks "
        + ", ".join(f"`{b}`" for b in benchmarks)
        + ". Every row records its own `model`, `benchmark`, `k`, `prompt_arm`, "
        "`metric_revision`, `parameter_count` and `embedding_dim`, so a table cell "
        "can always be traced to the row that produced it.\n"
    )
    out.append(
        "\n> **Checkpoints are pinned by name only, and the cache is checked rather "
        "than trusted.** The rows record `parameter_count` and `embedding_dim`, "
        "**not** a Hub revision, and the embedding cache is namespaced on model name "
        "+ dtype (`embedder_ladder.py::_build_embedder`) — also not a revision. Left "
        "there, an upstream re-upload under the same name would let a warm re-run "
        "read the **new** checkpoint's metadata while reusing the **old** "
        "checkpoint's vectors and publish a row mixing the two, with nothing in the "
        "row revealing it. So every run re-encodes one fixed canary string and "
        "compares it to the cached entry "
        "(`embedder_ladder.py::_assert_cache_matches_checkpoint`); a mismatch aborts "
        "the run and names the namespace file to delete. A namespace written *before* "
        "that check existed carries no canary and has therefore never been verified "
        "against anything, so it is refused rather than adopted silently — every "
        "namespace behind this table is in that state. Pass `--trust-existing-cache` "
        "to vouch for one: it pins the canary once, and every later run is checked "
        "normally. Putting the revision in the "
        "namespace instead would invalidate every cached vector to close the same "
        "hazard, and would still only answer *is this the same Hub commit* — the "
        "canary answers *does this checkpoint still produce these vectors*, which is "
        "what the measurements depend on, and so also catches a pooling change, a "
        "tokenizer fix, or a different dtype path.\n"
    )
    out.append(
        "\nRequires `OMP_NUM_THREADS=1` and `KMP_DUPLICATE_LIB_OK=1` on macOS "
        "(`run_ladder.sh` defaults both when unset): torch, faiss and scikit-learn "
        "each bundle a `libomp.dylib`, and with two runtimes loaded the sweep "
        "deadlocks at 0% CPU rather than failing.\n"
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

    # The headline is COMPUTED from the rows, not written beside them: a
    # hand-typed lead paragraph is the first thing to go stale when the table
    # under it is re-measured.
    spreads: list[tuple[float, str, tuple[float, LadderRow], tuple[float, LadderRow]]] = []
    for model in models:
        scored: list[tuple[float, LadderRow]] = []
        for row in ok:
            delta = row.vs_reference_delta
            if (
                row.model == model
                and row.k == headline_k
                and row.prompt_arm == "none"
                and delta is not None
            ):
                scored.append((delta, row))
        if len(scored) < 2:
            continue
        best = max(scored, key=lambda item: item[0])
        worst = min(scored, key=lambda item: item[0])
        spreads.append((best[0] - worst[0], model, best, worst))
    if spreads:
        _, model, (high, high_row), (low, low_row) = max(spreads, key=lambda item: item[0])
        # Only make the strong claim when the measurement actually supports it: a
        # sign flip whose intervals straddle 0 is noise wearing a headline.
        flips = (
            high > 0 > low
            and _excludes_zero(high_row.vs_reference_ci_low, high_row.vs_reference_ci_high)
            and _excludes_zero(low_row.vs_reference_ci_low, low_row.vs_reference_ci_high)
        )
        out.append("\n## Headline: there is no single winner — the answer is per benchmark\n")
        out.append(
            f"\nThe widest disagreement measured here is `{model}` against "
            f"`{REFERENCE_MODEL}` (langres's current default) at k={headline_k}: "
            f"**{high:+.4f}** per-record recall on `{high_row.benchmark}` "
            f"{_ci(high_row.vs_reference_ci_low, high_row.vs_reference_ci_high)} and "
            f"**{low:+.4f}** on `{low_row.benchmark}` "
            f"{_ci(low_row.vs_reference_ci_low, low_row.vs_reference_ci_high)}"
            + (
                " — the same model, better on one benchmark and **worse** on the "
                "other, with both intervals clear of zero.\n"
                if flips
                else " — the same model, spread across benchmarks.\n"
            )
        )
        out.append(
            "\nAveraging those into one number would report a middling win or loss "
            "and hide both. **This document deliberately publishes no cross-benchmark "
            "mean.** Pick the model against the data you actually have; the "
            "per-benchmark tables below are the unit of decision.\n"
        )
        # The arm these headline numbers came from, and the half-driven arm they
        # were once wrongly attributed to. Both derived from the rows: a
        # correction that hand-types the number it corrects is the same defect.
        instruct_row = next(
            (
                r
                for r in ok
                if r.model == model
                and r.benchmark == high_row.benchmark
                and r.k == headline_k
                and r.prompt_arm == "instruct"
            ),
            None,
        )
        out.append(
            "\n**Both of those numbers come from the `none` arm — neither the query "
            "side nor the document side carries a prompt.** `render_report` filters "
            "`prompt_arm == 'none'` for this comparison and the section heading below "
            "says `no instruction`. That is langres's **default** configuration: a "
            "`VectorBlocker` with no `query_prompt` over an embedder with no "
            "`prompt_name`. The headline is therefore a statement about the two models "
            "as they ship, not about a half-driven one.\n"
        )
        if instruct_row is not None and instruct_row.prompt_delta is not None:
            out.append(
                "\n> **Correction — supersedes the merged #239 PR body.** That body "
                "said: *'Every number below, including the "
                f"**{high:+.4f}** on `{high_row.benchmark}`, was measured with only the "
                "query side driven.'* **That is false**, by the arm filter cited above. "
                "The genuinely half-driven arm is `instruct`, and on "
                f"`{high_row.benchmark}` it moves `{model}` by "
                f"**{instruct_row.prompt_delta:+.4f}** "
                f"{_ci(instruct_row.prompt_delta_ci_low, instruct_row.prompt_delta_ci_high)} "
                "— a different number, from a different table. The correction runs in "
                "the direction that makes the headline **stronger**, not weaker: it was "
                "measured in the configuration langres actually ships.\n"
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
        "`amazon_google` the ceiling is ~0.84. `langres.optimize._score_loaded` "
        "carried the same false claim in its docstring and this change corrects "
        "it there too, since `optimize()` reports the same capped recall.)\n"
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

    # Computed from the rows this report actually publishes. It used to name
    # `all-mpnet-base-v2`, measured under an earlier metric revision — a claim the
    # reader could not check against any table on the page, because those rows are
    # excluded. A claim whose evidence is not in the document is not measured here.
    ref_params = next(
        (r.parameter_count for r in ok if r.model == REFERENCE_MODEL and r.parameter_count), None
    )
    inversion: tuple[float, LadderRow] | None = None
    for row in ok if ref_params else []:
        assert ref_params is not None
        if (
            row.k == headline_k
            and row.prompt_arm == "none"
            and row.model != REFERENCE_MODEL
            and (row.parameter_count or 0) > ref_params
            and (row.vs_reference_delta or 0.0) < 0.0
            and _excludes_zero(row.vs_reference_ci_low, row.vs_reference_ci_high)
        ):
            ratio = (row.parameter_count or 0) / ref_params
            if inversion is None or ratio > inversion[0]:
                inversion = (ratio, row)
    if inversion is not None:
        ratio, row = inversion
        out.append(
            "- **Parameter count is not the axis.** Measured, not asserted: "
            f"`{row.model}` carries {ratio:.0f}x the parameters of "
            f"`{REFERENCE_MODEL}` ({_millions(row.parameter_count)} vs "
            f"{_millions(ref_params)}) and is **{row.vs_reference_delta:+.4f}** "
            f"per-record recall on `{row.benchmark}` "
            f"{_ci(row.vs_reference_ci_low, row.vs_reference_ci_high)} — worse, with "
            "the interval clear of 0. A ladder that assumed bigger-is-better, or "
            "that read sizes off a table instead of off the loaded model, would "
            "have hidden that.\n"
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
        "\n**This paragraph describes the `instruct` arm only.** For it: same model, "
        "same `k`, same index — only the query side is re-encoded with "
        f"a single fixed instruction (`{INSTRUCTION!r}`). This is deliberately **one** "
        "instruction for every model, so it answers 'does a task instruction on the "
        "query help', not 'is each model at its documented best'. The `own prompt "
        "names` column above shows which models ship a documented recipe this sweep "
        "did not use.\n"
        "\n**The `instruct` arm prefixes queries against BARE documents**, which is a "
        "structurally different configuration from the asymmetric recipe an "
        "instruction-trained checkpoint documents — not the same recipe with "
        "different wording. `google/embeddinggemma-300m` prefixes documents with "
        "`'title: none | text: '` and queries with `'task: search result | query: '`; "
        "running only `instruct` measures half of that. Checkpoints that ship a "
        "query-side instruction therefore carry a third `documented` arm, read from "
        "their own `config_sentence_transformers.json` rather than from a model card "
        "or from memory. Where that arm is missing from the tables it is listed under "
        "'What did not run'.\n"
        "\n`Qwen/Qwen3-Embedding-*`'s documented instruction is about retrieving **web "
        "search passages**, and its document side is empty. Applying it to entity "
        "matching is out of its stated domain — which is precisely what a user "
        "following the model card would do, so it is worth measuring, but a poor "
        "result for that arm is a statement about a transplanted instruction, not "
        "about the model's instruction-following.\n"
    )
    out.append(
        "\n> ### Do not read this table as 'instructions do not help retrieval'\n"
        ">\n"
        "> **A negative result here is mostly a statement about the models in it.**\n"
        "> `all-MiniLM-*`, `all-mpnet-base-v2` and the BGE v1.5 family were not\n"
        "> trained with a task instruction on the query side. Prepending one to a\n"
        "> model that never saw one in training does not give it an instruction to\n"
        "> follow — it moves the query vector away from the document vectors it is\n"
        "> supposed to match. Measuring that it hurts is a real, useful, and\n"
        "> actionable finding: **do not switch these models on by default with a\n"
        "> prompt.** It is not evidence about instruction-following embedders.\n"
        ">\n"
        "> The hypothesis this axis exists to test is about the models trained for\n"
        "> it — `google/embeddinggemma-300m`, `Qwen/Qwen3-Embedding-*`, and E5's\n"
        "> native `query:`/`passage:` scheme (which this sweep's single generic\n"
        "> instruction is **not**; see `own prompt names`). Until those are in the\n"
        "> table, the instruction-following question is **open**, not answered.\n"
        ">\n"
        "> This distinction is the whole reason the axis was worth fixing. Before\n"
        "> the `query_prompt` no-op was repaired every cell here read exactly\n"
        "> `0.0000`, and the conclusion would have been a confident 'instructions\n"
        "> do not help'. Publishing an unprompted-model-only sweep as if it settled\n"
        "> the question would reach the same false negative by a different route.\n"
    )
    out.append(
        "\n**`Δ per-record` is the statistic the interval bounds** — the mean "
        "per-record recall difference (**this arm** minus `none`), resampled by gold "
        "cluster via `langres.experiments.statistics.paired_entity_bootstrap`. "
        "**Which sides moved depends on the arm, and they are not the same "
        "experiment**: `instruct` re-encodes the *query* side only, against the "
        "same bare-document index as `none`; `documented` applies the checkpoint's "
        "own document prefix too, so `build_prompted_index` rebuilds the corpus "
        "vectors and its Δ is **document-plus-query vs. neither**. Reading a "
        "`documented` Δ as a query-only effect overstates what the query "
        "instruction did. It is also the more useful number, because it is the "
        "configuration the model card actually describes. Resampling is identical "
        "for both — "
        "never by pair row, because pair rows inside one entity are dependent and "
        "resampling them produces intervals that are far too tight. It is "
        "deliberately **not** the difference of the two aggregate `recall` columns "
        "beside it, which is a different number (they disagreed by 34% on "
        "`walmart_amazon`); both are shown so the gap is visible rather than "
        "hidden. **An interval spanning 0 means the delta is not distinguishable "
        "from noise**; `(exactly 0)` means every record scored identically in both "
        "arms, which is certainty about a zero effect, not an absence of one.\n"
    )
    # Every non-baseline arm present in the rows gets a line, rather than only
    # `instruct`: an arm that was measured and then left out of the one table that
    # compares arms is the silent skip this harness refuses to do elsewhere.
    prompted_arms = sorted({row.prompt_arm for row in ok} - {"none"})
    out.append(
        "\n| model | benchmark | arm | recall (none) | recall (arm) | Δ aggregate | "
        "Δ per-record | 95% CI | AUC (none) | AUC (arm) | Δ AUC |\n"
    )
    out.append("|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|\n")
    for model in models:
        for benchmark in benchmarks:
            arm_rows = {
                r.prompt_arm: r
                for r in ok
                if r.model == model and r.benchmark == benchmark and r.k == headline_k
            }
            plain = arm_rows.get("none")
            if plain is None:
                continue
            for arm in prompted_arms:
                prompted = arm_rows.get(arm)
                if prompted is None:
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
                    f"| `{model}` | {benchmark} | {arm} | {_fmt(plain.candidate_recall)} | "
                    f"{_fmt(prompted.candidate_recall)} | {d_recall:+.4f} | {per_record} | "
                    f"{_ci(prompted.prompt_delta_ci_low, prompted.prompt_delta_ci_high)} | "
                    f"{_fmt(plain.separability_auc)} | {_fmt(prompted.separability_auc)} | "
                    f"{'n/a' if d_auc is None else f'{d_auc:+.4f}'} |\n"
                )

    out.append("\n### The asymmetric recipe, and how to drive it (corrected)\n")
    out.append(
        "\n> **Correction.** An earlier version of this section was headed *'langres's "
        "blocking path has no document-side prompt'* and said the asymmetric recipe was "
        "**'not expressible through the blocking API'**. That is **false**. It is "
        "expressible today, with no API change — the defect was that nothing documented "
        "it and nothing checked it.\n"
        "\n`create_index(texts)` indeed takes no prompt argument, but the document-side "
        "prompt does not travel through that argument: it is bound to the **embedder** "
        "the index owns. `SentenceTransformerEmbedder` forwards `prompts=` and "
        "`prompt_name=` into the `SentenceTransformer` constructor as "
        "`default_prompt_name` (`src/langres/core/embeddings.py`), and "
        "sentence-transformers resolves an `encode(prompt=None)` call back to that "
        "default (`base/model.py`, `_resolve_prompt`). So every text `create_index` "
        "encodes already carries the document prefix. `search_all(query_prompt=...)` "
        "then passes an **explicit** prompt, which takes precedence over the default — "
        "queries get the query prefix, documents keep the document prefix.\n"
        "\nThat is the whole asymmetric recipe, and it is verified rather than argued: "
        "`tests/core/blockers/test_asymmetric_prompt_recipe.py` builds it through the "
        "shipped API and asserts it is **byte-identical** to this harness's "
        "prefix-the-corpus-by-hand workaround, with two controls — dropping the "
        "document prompt changes the result, and dropping the query prompt changes the "
        "result. (A `query_prompt` that silently did nothing already shipped here once "
        "and made every prompt cell read `0.0000`.)\n"
        "\n```python\n"
        "embedder = SentenceTransformerEmbedder(\n"
        '    "google/embeddinggemma-300m",\n'
        "    prompts={\n"
        '        "document": "title: none | text: ",\n'
        '        "query": "task: search result | query: ",\n'
        "    },\n"
        '    prompt_name="document",   # <- the DOCUMENT side, applied by create_index\n'
        ")\n"
        "blocker = VectorBlocker(\n"
        "    vector_index=FAISSIndex(embedder),\n"
        '    query_prompt="task: search result | query: ",  # <- the QUERY side\n'
        "    schema=MySchema,\n"
        '    text_field="name",\n'
        ")\n"
        "```\n"
        "\n**What was actually broken, and is fixed in the same change as this "
        "correction:**\n"
        "\n- Nothing checked that the two sides agree. Setting the embedder's "
        "`prompt_name` and forgetting the blocker's `query_prompt` is silently *worse* "
        "than setting neither: `search_all(query_prompt=None)` reuses the cached corpus "
        "vectors as queries, so the queries are encoded with the **document** prefix. "
        "`VectorBlocker` now warns on exactly that combination.\n"
        "- `QdrantHybridIndex.search_all` accepted `query_prompt` and **discarded** it, "
        "so sweeping the axis over that index returned a flat, meaningless result "
        "instead of an error. It now raises.\n"
        "\nThis harness still prefixes the corpus text itself, because it must reproduce "
        "rows measured before the recipe was documented. That is exactly equivalent "
        "**for these checkpoints**, verified rather than assumed: sentence-transformers "
        "applies a prompt as `prompt + text`, and excludes its tokens from pooling only "
        "when the checkpoint sets `include_prompt=false`, which neither of these does. "
        "A checkpoint that did set it would need the real API above.\n"
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

    out.extend(_render_recommendation(ok, models, benchmarks, headline_k))

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

    out.append("\n## What did not run (and why)\n")
    out.append(
        "\nDerived from the recorded rows, not written by hand — a hand-kept list of "
        "gaps is exactly the thing that goes stale and turns a partial sweep into a "
        "table that reads as complete.\n"
    )
    measured = {row.model for row in ok}
    stale_models = {row.model for row in stale}
    failed_models = {row.model for row in failures}
    never = [spec for spec in MODELS if spec.name not in measured]
    # Same denominator as the recommendation section, and computed the same way:
    # every model that RAN, not just the ones that succeeded. Counting only
    # `measured` here would make a custom model's failure shrink the ladder.
    ladder_size = len({spec.name for spec in MODELS} | measured | stale_models | failed_models)
    if never:
        out.append(
            f"\n**{len(never)} of the {ladder_size} models in the ladder have no usable "
            f"row at metric revision {METRIC_REVISION}.** The `state` column says why "
            "for each — this table cannot speak about any of them.\n"
        )
        out.append("\n| model | state |\n|---|---|\n")
        for spec in never:
            if spec.name in failed_models:
                # A model whose only rows are failures did NOT go unmeasured. It
                # was tried and it broke; saying "not run" would file a reported
                # failure under "never reached".
                state = "**failed** — see Failures below"
            elif spec.name in stale_models:
                state = "measured under an older metric revision — re-run to publish"
            else:
                state = "not run"
            out.append(f"| `{spec.name}` | {state} |\n")

    # The grid, cell by cell — NOT "does each benchmark appear somewhere" and
    # "does each arm appear somewhere" separately. Those marginals are both
    # satisfied by a grid with a hole in it, and cross-model review demonstrated
    # exactly that: one missing (benchmark x arm) cell while the report declared
    # every model, benchmark and arm measured.
    gaps: list[str] = []
    for spec in MODELS:
        if spec.name not in measured:
            continue
        cells = {(row.benchmark, row.prompt_arm, row.k) for row in ok if row.model == spec.name}
        missing = [
            (benchmark, arm, k)
            for benchmark in BENCHMARKS
            for arm in arms_for(spec, PROMPT_ARMS)
            for k in K_VALUES
            if (benchmark, arm, k) not in cells
        ]
        if missing:
            # Collapse to the coarsest true statement so the table stays readable:
            # a whole missing benchmark is one line, not len(arms) * len(k) lines.
            whole_benchmarks = sorted(
                {b for b, _, _ in missing} - {row.benchmark for row in ok if row.model == spec.name}
            )
            rest = [cell for cell in missing if cell[0] not in whole_benchmarks]
            parts = []
            if whole_benchmarks:
                parts.append("benchmarks " + ", ".join(f"`{b}`" for b in whole_benchmarks))
            if rest:
                parts.append(
                    "cells "
                    + ", ".join(f"`{b}`/`{arm}`/k={k}" for b, arm, k in sorted(rest)[:12])
                    + (f" (+{len(rest) - 12} more)" if len(rest) > 12 else "")
                )
            gaps.append(f"| `{spec.name}` | {'; '.join(parts)} |\n")
    if gaps:
        out.append(
            "\nMeasured models with an incomplete grid — a blank cell in the tables "
            "above is one of these, never a silent failure:\n"
        )
        out.append("\n| model | missing |\n|---|---|\n")
        out.extend(gaps)

    # A checkpoint that ships its own query-side prefix but was measured without a
    # `documented` arm. Derived from `registered_prompts`, measured off the loaded
    # model — so it cannot go stale the way a hand-kept list of instruction-trained
    # families does. `ModelSpec.documented_arm` is only ever set from a checkpoint
    # config that was actually read, so the ones nobody has downloaded yet show up
    # here the moment they are measured.
    undocumented = sorted(
        {
            row.model
            for row in ok
            if any(name in ("query", "Retrieval-query") for name in row.registered_prompts)
            and "documented" not in {r.prompt_arm for r in ok if r.model == row.model}
        }
    )
    if undocumented:
        out.append(
            "\nMeasured checkpoints that register their own query-side prefix but were "
            "measured without a `documented` arm — the generic `instruct` arm is not "
            "their documented recipe, so this sweep does not show them at their "
            "documented best: " + ", ".join(f"`{name}`" for name in undocumented) + ".\n"
        )

    if not never and not gaps and not undocumented:
        out.append("\nEvery model, benchmark and prompt arm in the ladder was measured.\n")

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
    # The base arms only: the third `documented` arm is a property of the
    # checkpoint (`ModelSpec.documented_arm`), not a thing to ask for by name.
    parser.add_argument(
        "--prompts", nargs="*", choices=list(PROMPT_ARMS), default=list(PROMPT_ARMS)
    )
    parser.add_argument("--rows", type=Path, default=DEFAULT_ROWS_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE_PATH)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--trust-existing-cache",
        action="store_true",
        help=(
            "Adopt an existing cache namespace that has never been verified against "
            "a checkpoint, instead of refusing it. Use only when you know the cache "
            "was written by the checkpoint that is loaded now: it vouches once, then "
            "the canary it pins is checked normally on every later run."
        ),
    )
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

    # Vouching is per cache, so it has to be asked for per cache. `--models`
    # defaults to the WHOLE ladder, so a bare `--trust-existing-cache` would
    # silently bless every unverified namespace it met -- six of them, as the
    # cache stands -- from a flag documented as vouching for one. Requiring an
    # explicit single `--models` makes the operator name what they are vouching
    # for, which is the whole content of the assertion. (Cross-model review.)
    if args.trust_existing_cache and len(args.models) != 1:
        parser.error(
            "--trust-existing-cache vouches for ONE cache, so it requires exactly one "
            f"--models NAME (got {len(args.models)}). Adopting a namespace asserts that "
            "its vectors came from the checkpoint loaded now; that is a claim about a "
            "specific model, and it is not one you can make for the whole ladder at once."
        )

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
                prompt_arms=arms_for(spec, arms),
                cache_dir=args.cache_dir,
                device=args.device,
                batch_size=args.batch_size,
                reference=reference,
                adopt_legacy_cache=args.trust_existing_cache,
            )
            fresh.extend(rows)
            reference_updates.update(updates)
        # Persist after EVERY model: a long sweep must be durable at each step,
        # not only at the end.
        if spec.name == REFERENCE_MODEL:
            # Unconditional, NOT `if reference_updates:` — a reference run where
            # every arm failed produces no updates at all, and that is exactly
            # the run whose stale cells must go.
            touched = {
                _reference_key(benchmark, arm)
                for benchmark in args.benchmarks
                for arm in arms_for(spec, arms)
            }
            write_reference(
                args.reference, refresh_reference(reference, reference_updates, touched)
            )
        merged = merge_rows(read_rows(args.rows), fresh)
        write_rows(args.rows, merged)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(render_report(merged, headline_k=args.headline_k))
        logger.info("wrote %d rows for %s", len(fresh), spec.name)


if __name__ == "__main__":
    main()
