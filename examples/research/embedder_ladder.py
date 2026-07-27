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
reproduces its own committed table instead of appending duplicates.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROWS_PATH = REPO_ROOT / "docs" / "research" / "20260727_embedder_ladder_rows.jsonl"
DEFAULT_REPORT_PATH = REPO_ROOT / "docs" / "research" / "20260727_embedder_ladder.md"
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
    ModelSpec("Qwen/Qwen3-Embedding-4B"),
    ModelSpec("Qwen/Qwen3-Embedding-8B"),
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

K_VALUES: tuple[int, ...] = (5, 10, 20, 50)

#: Negatives sampled per benchmark for the separability AUC (seeded).
NEGATIVE_SAMPLE = 20_000
SEED = 0


@dataclass
class LadderRow:
    """One measured cell: (model, benchmark, prompt arm, k)."""

    model: str
    benchmark: str
    prompt_arm: str
    k: int
    status: str
    parameter_count: int | None = None
    embedding_dim: int | None = None
    n_records: int | None = None
    n_gold_pairs: int | None = None
    candidate_recall: float | None = None
    candidate_precision: float | None = None
    reduction_ratio: float | None = None
    total_candidates: int | None = None
    candidates_per_unit_recall: float | None = None
    separability_auc: float | None = None
    index_build_seconds: float | None = None
    search_seconds: float | None = None
    registered_prompts: list[str] = field(default_factory=list)
    #: Saturation is measured by a SEPARATE stream; never restate it as measured here.
    saturation: str = "unknown (stream B)"
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
    between models. Scored asymmetrically the same way search is —
    ``dot(query_vector[a], doc_vector[b])`` — so the prompt arm is reflected.

    Negatives are a seeded uniform sample of non-gold pairs (the full set is
    quadratic and hugely imbalanced), so this is an AUC over a *sample*, not the
    population.

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
        left = query_vectors[[i for i, _ in pairs]]
        right = doc_vectors[[j for _, j in pairs]]
        return np.einsum("ij,ij->i", left, right)

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


def _build_embedder(spec: ModelSpec, cache_dir: Path, device: str | None, batch_size: int) -> Any:
    """A disk-cached embedder for ``spec``.

    The cache is keyed on (text, prompt) and namespaced per model, which makes a
    long sweep resumable and keeps the prompted arm from re-encoding the corpus
    once per ``k`` (a prompted ``search_all`` re-encodes queries by design).
    """
    from langres.core.embeddings import DiskCachedEmbedder, SentenceTransformerEmbedder

    base = SentenceTransformerEmbedder(
        spec.name,
        batch_size=batch_size,
        device=device,
        trust_remote_code=spec.trust_remote_code,
    )
    namespace = spec.name.replace("/", "__")
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
) -> Iterator[LadderRow]:
    """Yield one row per (prompt arm, k), or a single failure row."""
    from langres.core.blockers.vector import VectorBlocker
    from langres.core.indexes.vector_index import FAISSIndex
    from langres.metrics.metrics import evaluate_blocking

    try:
        corpus, gold_clusters, _ = _load_benchmark(benchmark)
        texts = _blocking_texts(corpus)
        ids = [str(record.id) for record in corpus]
        gold_pairs = _gold_pair_set(gold_clusters)
        sizes = _source_sizes(corpus)
        schema = type(corpus[0])
        records = [record.model_dump() for record in corpus]

        base, embedder = _build_embedder(spec, cache_dir, device, batch_size)

        started = time.perf_counter()
        index = FAISSIndex(embedder=embedder, metric="cosine")
        index.create_index(texts)
        index_build_seconds = time.perf_counter() - started

        parameter_count = base.parameter_count
        embedding_dim = base.embedding_dim
        prompts = _registered_prompts(base)
        doc_vectors = embedder.encode(texts)
    except Exception as exc:  # noqa: BLE001 - a failure IS a result, never a skip
        logger.exception("model %s failed on %s", spec.name, benchmark)
        yield LadderRow(
            model=spec.name,
            benchmark=benchmark,
            prompt_arm="-",
            k=0,
            status="failed",
            error=f"{type(exc).__name__}: {exc}"[:400],
        )
        return

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
                    # Cross-source linkage: all gold matches are inter-source, so
                    # dropping intra-source pairs leaves recall unchanged while
                    # making the reduction ratio use |A|*|B|. Mirrors
                    # langres.optimize._score_loaded.
                    candidates = [c for c in candidates if c.left.source != c.right.source]
                    stats = evaluate_blocking(
                        candidates, gold_clusters, n_left=sizes[0], n_right=sizes[1]
                    )
                else:
                    stats = evaluate_blocking(candidates, gold_clusters, num_records=len(corpus))

                recall = stats.candidate_recall
                yield LadderRow(
                    model=spec.name,
                    benchmark=benchmark,
                    prompt_arm=arm,
                    k=k,
                    status="ok",
                    parameter_count=parameter_count,
                    embedding_dim=embedding_dim,
                    n_records=len(corpus),
                    n_gold_pairs=len(gold_pairs),
                    candidate_recall=recall,
                    candidate_precision=stats.candidate_precision,
                    reduction_ratio=stats.reduction_ratio,
                    total_candidates=stats.total_candidates,
                    candidates_per_unit_recall=(
                        stats.total_candidates / recall if recall > 0 else None
                    ),
                    separability_auc=auc,
                    index_build_seconds=index_build_seconds,
                    search_seconds=search_seconds,
                    registered_prompts=prompts,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("model %s failed on %s k=%s", spec.name, benchmark, k)
                yield LadderRow(
                    model=spec.name,
                    benchmark=benchmark,
                    prompt_arm=arm,
                    k=k,
                    status="failed",
                    parameter_count=parameter_count,
                    error=f"{type(exc).__name__}: {exc}"[:400],
                )


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
    """
    replaced = {(row.model, row.benchmark) for row in fresh}
    return [row for row in existing if (row.model, row.benchmark) not in replaced] + list(fresh)


def _fmt(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:.{digits}f}"


def _millions(count: int | None) -> str:
    return "n/a" if count is None else f"{count / 1e6:,.1f}M"


def render_report(rows: Sequence[LadderRow], headline_k: int = 20) -> str:
    """Render the markdown report from measured rows only."""
    ok = [row for row in rows if row.status == "ok"]
    failures = [row for row in rows if row.status == "failed"]
    benchmarks = sorted({row.benchmark for row in ok} | {row.benchmark for row in failures})
    models = sorted(
        {row.model for row in rows},
        key=lambda name: (
            min((r.parameter_count or 10**12) for r in rows if r.model == name),
            name,
        ),
    )

    out: list[str] = []
    out.append("# Embedder ladder: which embedding model at which parameter count\n")
    out.append(
        "Generated by `examples/research/embedder_ladder.py`. Every number here is "
        "measured by that script; re-running it regenerates this file from "
        "`20260727_embedder_ladder_rows.jsonl`.\n"
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
        "hill-climbs `k`, so an operating-point win is not the same as a model win.\n"
        "- **Saturation is not measured here.** The `saturation` column is carried as "
        "`unknown (stream B)` on every row: a parallel stream measures which of these "
        "benchmarks are saturated, and a ladder measured on a saturated benchmark "
        "measures the benchmark. Do not average across benchmarks whose saturation "
        "status is unknown, and do not read this benchmark set as a finding — it is a "
        "documented prior this sweep inherited.\n"
        "- **Reproducible from a git checkout only, not from `pip install langres`.** "
        "The wheel deliberately excludes the `amazon_google`, `abt_buy`, "
        "`walmart_amazon` and `wdc_computers` corpora "
        "(`[tool.hatch.build] exclude` in `pyproject.toml`), so only `fodors_zagat` "
        "here is loadable from a PyPI install; the rest raise "
        "`BenchmarkDataNotFoundError`.\n"
    )

    out.append("\n## Models that were measured\n")
    out.append("\n| model | parameters | dim | own prompt names |\n|---|---:|---:|---|\n")
    for model in models:
        sample = next((row for row in ok if row.model == model), None)
        if sample is None:
            continue
        prompts = ", ".join(sample.registered_prompts) or "—"
        out.append(
            f"| `{model}` | {_millions(sample.parameter_count)} | "
            f"{sample.embedding_dim} | {prompts} |\n"
        )

    out.append(f"\n## Candidate recall at k={headline_k}, no instruction\n")
    out.append(
        "\nThe unprompted operating point, one column per benchmark. "
        "`cands/recall` is the candidate count divided by recall — lower is a "
        "cheaper ceiling.\n"
    )
    for benchmark in benchmarks:
        out.append(f"\n### {benchmark}\n")
        out.append(
            "\n| model | parameters | recall | sep. AUC | candidates | cands/recall | index build (s) | saturation |\n"
        )
        out.append("|---|---:|---:|---:|---:|---:|---:|---|\n")
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
                    out.append(f"| `{model}` | — | **failed** | — | — | — | — | {failed.error} |\n")
                continue
            out.append(
                f"| `{model}` | {_millions(row.parameter_count)} | "
                f"{_fmt(row.candidate_recall)} | {_fmt(row.separability_auc)} | "
                f"{_fmt(row.total_candidates)} | {_fmt(row.candidates_per_unit_recall, 0)} | "
                f"{_fmt(row.index_build_seconds, 1)} | {row.saturation} |\n"
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
        "\n| model | benchmark | recall (none) | recall (instruct) | Δ recall | AUC (none) | AUC (instruct) | Δ AUC |\n"
    )
    out.append("|---|---|---:|---:|---:|---:|---:|---:|\n")
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
            out.append(
                f"| `{model}` | {benchmark} | {_fmt(plain.candidate_recall)} | "
                f"{_fmt(prompted.candidate_recall)} | {d_recall:+.4f} | "
                f"{_fmt(plain.separability_auc)} | {_fmt(prompted.separability_auc)} | "
                f"{'n/a' if d_auc is None else f'{d_auc:+.4f}'} |\n"
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
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--device", default=None, help="torch device, e.g. mps / cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--headline-k", type=int, default=20)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    arms = {name: PROMPT_ARMS[name] for name in args.prompts}

    for model_name in args.models:
        spec = MODELS_BY_NAME.get(model_name) or ModelSpec(model_name)
        fresh: list[LadderRow] = []
        for benchmark in args.benchmarks:
            logger.info("=== %s on %s ===", spec.name, benchmark)
            fresh.extend(
                evaluate_model_on_benchmark(
                    spec,
                    benchmark,
                    k_values=args.k,
                    prompt_arms=arms,
                    cache_dir=args.cache_dir,
                    device=args.device,
                    batch_size=args.batch_size,
                )
            )
        # Persist after EVERY model: a long sweep must be durable at each step,
        # not only at the end.
        merged = merge_rows(read_rows(args.rows), fresh)
        write_rows(args.rows, merged)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(render_report(merged, headline_k=args.headline_k))
        logger.info("wrote %d rows for %s", len(fresh), spec.name)


if __name__ == "__main__":
    main()
