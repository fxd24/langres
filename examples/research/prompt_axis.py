"""Do instruction prompts make embedders better at ER blocking? Per model, per benchmark.

Modern embedders are instruction-following: e5, bge, EmbeddingGemma and Qwen3 all
ship a prescribed prompt format and their published numbers assume it. This
harness answers two questions the ladder (``embedder_ladder.py``) left open:

1. Does *the model's own documented recipe* beat no prompt at all on blocking
   recall?
2. Can an ER-specific instruction — written for "find records describing the
   same real-world entity" rather than "retrieve passages answering a query" —
   beat the documented one?

**Every prompt string here was read from a primary source** (the checkpoint's own
``config_sentence_transformers.json`` or its model card in the local Hugging Face
snapshot), never inferred from another model's convention. Each recipe carries
its ``provenance`` into the output rows, and ``PROMPT_SOURCES`` records the exact
file and line. These formats genuinely differ: e5 wants ``query: ``/``passage: ``,
bge wants a query-side sentence and *explicitly no document prefix*, Gemma wants
``task: … | query: `` / ``title: … | text: ``, Qwen3 wants ``Instruct: …\\nQuery:``.
Guessing one from another produces a silently wrong measurement.

**Both halves of every recipe are driven deliberately.** A prompt on the document
side with a bare query side is not "half a treatment", it is a *different and
worse* treatment: the query inherits no prefix the checkpoint was trained to see
while the corpus carries one. The ``er_query_only`` arm exists to *measure* that
trap rather than to fall into it, and it is labelled as such. Where a card
prescribes a query-only recipe (bge, Qwen3) that asymmetry is the documented
recipe, and the arm says so.

**A prompt that never reaches the encoder produces identical numbers, which reads
as "instructions do not help".** That bug shipped here once (``search_all`` served
queries from cached corpus vectors). So every arm records three independent
proofs that its prompt reached the encoder, and the run *fails* rather than
reports a flat result if they say it did not — see :func:`_prompt_reached_encoder`.

Recall is measured as **candidate recall** at fixed ``k`` — never F1 at a
threshold, which would confound the retrieval effect with the cut.

Usage::

    uv run --env-file .env python examples/research/prompt_axis.py \\
        --models intfloat/e5-base-v2 --benchmarks abt_buy

Zero paid API calls: every model runs locally.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("prompt_axis")

DEFAULT_ROWS_PATH = Path("docs/research/20260728_prompt_axis_rows.jsonl")
DEFAULT_REPORT_PATH = Path("docs/research/20260728_prompt_axis.md")
DEFAULT_CACHE_DIR = Path("tmp/prompt_axis_cache")

K_VALUES: tuple[int, ...] = (5, 10, 20, 50)
HEADLINE_K = 20
SEED = 0

BENCHMARKS: tuple[str, ...] = (
    "fodors_zagat",
    "abt_buy",
    "amazon_google",
    "wdc_computers",
    "walmart_amazon",
)

# --------------------------------------------------------------------------
# The prompts, and where each one came from.
# --------------------------------------------------------------------------

#: Primary source for every documented prompt string below. Paths are inside the
#: local Hugging Face snapshot cache, which is the checkpoint as its author
#: published it -- not a blog post, not another model's convention.
PROMPT_SOURCES: dict[str, str] = {
    "intfloat/e5-base-v2": (
        "model card README.md (snapshot f52bf8ec8c7124536f0efb74aca902b2995e5bcd), "
        'L2631 \'Each input text should start with "query: " or "passage: "\' and '
        "FAQ 1 L2690-2694: 'query:'/'passage:' for ASYMMETRIC tasks, 'query:' on "
        "both sides for SYMMETRIC tasks (semantic similarity, paraphrase "
        "retrieval) and for clustering. The checkpoint ships NO "
        "config_sentence_transformers.json, so the card is the only source."
    ),
    "BAAI/bge-base-en-v1.5": (
        "model card README.md (snapshot a5beb1e3e68b9ab74eb54cfd186867f64f240e1a), "
        "Model List table L2679 gives the query instruction verbatim; note [1] "
        "L2692: 'In all cases, no instruction needs to be added to passages'; "
        "L2738-2744: v1.5 was tuned to work WITHOUT the instruction and 'the best "
        "method to decide whether to add instructions for queries is choosing the "
        "setting that achieves better performance on your task'. Its "
        "config_sentence_transformers.json registers no prompts at all."
    ),
    "google/embeddinggemma-300m": (
        "config_sentence_transformers.json (snapshot "
        "57c266a740f537b4dc058e1b0cda161fd15afa75) 'prompts' mapping, cross-checked "
        "against the model card's 'Prompt Instructions' section L344 + task table "
        "L366-416. Query form is 'task: {task description} | query: ', document "
        "form is 'title: {title | \"none\"} | text: '. The card lists a Semantic "
        "Similarity template ('task: sentence similarity | query: ', flagged 'not "
        "intended for retrieval use cases') and a Clustering template."
    ),
    "Qwen/Qwen3-Embedding-0.6B": (
        "config_sentence_transformers.json (snapshot "
        "c54f2e6e80b2d7b7de06f51cec4959f6b3e03418): prompts = {'query': 'Instruct: "
        "Given a web search query, retrieve relevant passages that answer the "
        "query\\nQuery:', 'document': ''} -- the document prompt is literally empty. "
        "Model card L129 gives the template f'Instruct: {task_description}\\nQuery:"
        "{query}', L138 'No need to add instruction for retrieval documents', and "
        "L54 'using instructions typically yields an improvement of 1% to 5% ... we "
        "recommend that developers create tailored instructions specific to their "
        "tasks'."
    ),
    "sentence-transformers/all-MiniLM-L6-v2": (
        "config_sentence_transformers.json (snapshot "
        "c9745ed1d9f207416be6d2e6f8de32d1f16199bf) registers no usable prompts. "
        "Not instruction-trained -- included as the CONTROL: whatever a generic "
        "instruction does here is what it does to a model that was never taught to "
        "read one."
    ),
}

#: Our ER-specific instruction. Frames the task as record-to-record identity
#: rather than query-to-passage relevance, which is what blocking actually is.
ER_INSTRUCTION = "Find records that describe the same real-world entity as: "

BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
QWEN_OFFICIAL_QUERY = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
)
QWEN_ER_QUERY = (
    "Instruct: Given a product or business record, retrieve other records that describe "
    "the same real-world entity\nQuery:"
)


@dataclass(frozen=True)
class Recipe:
    """One prompt setting, with both halves stated and its origin recorded.

    ``document_prompt`` is applied by the index build (via the embedder's
    ``prompt_name``); ``query_prompt`` is applied at search time (via
    ``VectorBlocker(query_prompt=...)``). ``None`` on a side means that side is
    encoded bare.
    """

    arm: str
    document_prompt: str | None
    query_prompt: str | None
    #: "documented" (straight from the card/config), "ours" (an ER instruction we
    #: wrote), or "trap" (a deliberately incoherent recipe, measured on purpose).
    kind: str
    note: str


def _shared_arms() -> tuple[Recipe, ...]:
    """Arms every model gets, so the ER instruction is comparable across models."""
    return (
        Recipe("none", None, None, "baseline", "No prompt on either side."),
        Recipe(
            "er_symmetric",
            ER_INSTRUCTION,
            ER_INSTRUCTION,
            "ours",
            "Our ER instruction on BOTH sides -- coherent, since ER blocking is a "
            "symmetric record-to-record task.",
        ),
        Recipe(
            "er_query_only",
            None,
            ER_INSTRUCTION,
            "trap",
            "Our ER instruction on the QUERY side only, documents bare. The "
            "half-driven recipe #242 warns about; measured here rather than assumed.",
        ),
    )


@dataclass(frozen=True)
class ModelSpec:
    name: str
    license: str
    osi_approved: bool
    extra_arms: tuple[Recipe, ...] = ()
    dtype: str | None = None
    batch_size: int = 64
    note: str = ""

    @property
    def recipes(self) -> tuple[Recipe, ...]:
        return (*_shared_arms(), *self.extra_arms)


MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        name="sentence-transformers/all-MiniLM-L6-v2",
        license="apache-2.0",
        osi_approved=True,
        note="Control: not instruction-trained, ships no prompts.",
    ),
    ModelSpec(
        name="BAAI/bge-base-en-v1.5",
        license="mit",
        osi_approved=True,
        extra_arms=(
            Recipe(
                "official_query_instruction",
                None,
                BGE_QUERY_INSTRUCTION,
                "documented",
                "The card's own recipe. Query-side only is DOCUMENTED here: "
                "'In all cases, no instruction needs to be added to passages.'",
            ),
            Recipe(
                "official_symmetric",
                BGE_QUERY_INSTRUCTION,
                BGE_QUERY_INSTRUCTION,
                "ours",
                "The official instruction made symmetric -- NOT documented. Tests "
                "whether a symmetric task wants a symmetric prefix.",
            ),
        ),
    ),
    ModelSpec(
        name="intfloat/e5-base-v2",
        license="mit",
        osi_approved=True,
        extra_arms=(
            Recipe(
                "official_asymmetric",
                "passage: ",
                "query: ",
                "documented",
                "The card's retrieval recipe: 'query:'/'passage:' for asymmetric tasks.",
            ),
            Recipe(
                "official_symmetric",
                "query: ",
                "query: ",
                "documented",
                "The card's SYMMETRIC recipe -- 'use query: prefix for symmetric "
                "tasks'. ER blocking is symmetric, so this is the documented recipe "
                "for this use, not the retrieval one.",
            ),
        ),
    ),
    ModelSpec(
        name="google/embeddinggemma-300m",
        license="gemma (NOT OSI-approved; use-restricted)",
        osi_approved=False,
        extra_arms=(
            Recipe(
                "official_retrieval",
                "title: none | text: ",
                "task: search result | query: ",
                "documented",
                "The config's Retrieval-document / Retrieval-query pair.",
            ),
            Recipe(
                "official_sts",
                "task: sentence similarity | query: ",
                "task: sentence similarity | query: ",
                "documented",
                "The config's STS / PairClassification template, applied to both "
                "sides as a symmetric template must be.",
            ),
            Recipe(
                "official_clustering",
                "task: clustering | query: ",
                "task: clustering | query: ",
                "documented",
                "The config's Clustering template, both sides.",
            ),
            Recipe(
                "er_in_official_template",
                "task: entity resolution | query: ",
                "task: entity resolution | query: ",
                "ours",
                "An ER task description dropped into the card's OWN template shape. "
                "Tests whether the template or the task word is what matters.",
            ),
        ),
        note="Gated + use-restricted licence. Never the default recommendation.",
    ),
    ModelSpec(
        name="Qwen/Qwen3-Embedding-0.6B",
        license="apache-2.0",
        osi_approved=True,
        dtype="float16",
        batch_size=16,
        extra_arms=(
            Recipe(
                "official_query_instruct",
                None,
                QWEN_OFFICIAL_QUERY,
                "documented",
                "The config's registered 'query' prompt verbatim; its 'document' "
                "prompt is the empty string, so a bare document side IS the "
                "documented recipe. Note the task text is about WEB SEARCH, so this "
                "arm measures the card's default applied outside its stated domain.",
            ),
            Recipe(
                "er_in_official_template",
                None,
                QWEN_ER_QUERY,
                "ours",
                "The card's own template with an ER task description substituted -- "
                "exactly what the card tells developers to do ('create tailored "
                "instructions specific to your tasks'). Document side stays bare, "
                "per the card.",
            ),
        ),
    ),
)

MODELS_BY_NAME = {spec.name: spec for spec in MODELS}


@dataclass
class Row:
    """One measured (model, benchmark, arm, k) cell."""

    model: str
    benchmark: str
    arm: str
    kind: str
    k: int
    document_prompt: str | None
    query_prompt: str | None
    note: str
    candidate_recall: float
    candidate_precision: float
    reduction_ratio: float
    total_candidates: int
    reachable_ceiling: float
    recall_of_reachable: float
    # Evidence the prompt actually reached the encoder (constant across k).
    doc_shift_vs_none: float
    query_shift_vs_none: float
    doc_query_cosine: float
    pair_jaccard_vs_none: float | None
    # Effect vs the no-prompt arm, bootstrapped by gold cluster (headline k only).
    delta_recall_vs_none: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    n_clusters: int | None = None
    parameter_count: int | None = None
    embedding_dim: int | None = None
    license: str = "unknown"
    osi_approved: bool = False
    seconds: float = 0.0


# --------------------------------------------------------------------------
# Proof that the prompt reached the encoder
# --------------------------------------------------------------------------


def _mean_cosine(left: np.ndarray, right: np.ndarray) -> float:
    """Mean row-wise cosine. Vectors arrive L2-normalized, so this is a dot."""
    if left.shape != right.shape:
        return float("nan")
    return float(np.mean(np.einsum("ij,ij->i", left, right)))


def _shift(vectors: np.ndarray, baseline: np.ndarray) -> float:
    """``1 - mean cosine`` against the no-prompt vectors. Exactly 0.0 = untouched."""
    return 1.0 - _mean_cosine(vectors, baseline)


def _prompt_reached_encoder(
    recipe: Recipe,
    doc_shift: float,
    query_shift: float,
    doc_query_cosine: float,
) -> None:
    """Fail loudly when a non-empty prompt left the vectors untouched.

    Three independent signals, because a single one can be satisfied by accident:

    1. ``doc_shift`` -- the corpus vectors moved. Proves ``prompt_name`` reached
       ``SentenceTransformer.encode`` through the *index build* path.
    2. ``query_shift`` -- the query vectors moved. Proves ``query_prompt`` reached
       the encoder through the *search* path, which is the exact seam that used to
       discard it.
    3. ``doc_query_cosine`` -- for a symmetric recipe the two sides must agree
       (cosine ~1), for an asymmetric one they must not. This catches a prompt
       that reached one path but was silently rewritten on the other, which
       neither shift alone can see.

    A byte-identical result is a harness bug until proven otherwise, so this
    raises rather than recording a flat row.
    """
    if recipe.document_prompt and doc_shift == 0.0:
        raise RuntimeError(
            f"arm {recipe.arm!r} sets document_prompt={recipe.document_prompt!r} but the "
            "corpus vectors are identical to the no-prompt arm. The prompt never "
            "reached the encoder -- this is a harness bug, not evidence that "
            "instructions do not help."
        )
    if recipe.query_prompt and query_shift == 0.0:
        raise RuntimeError(
            f"arm {recipe.arm!r} sets query_prompt={recipe.query_prompt!r} but the query "
            "vectors are identical to the no-prompt arm. The prompt never reached "
            "the encoder (this is exactly the search_all() bug #242 hardened)."
        )
    symmetric = recipe.document_prompt == recipe.query_prompt
    if symmetric and not math.isclose(doc_query_cosine, 1.0, abs_tol=1e-4):
        raise RuntimeError(
            f"arm {recipe.arm!r} is symmetric ({recipe.document_prompt!r} on both sides) "
            f"but its document and query vectors disagree (cosine={doc_query_cosine:.6f}). "
            "One of the two encode paths is not applying the prompt it was given."
        )
    if not symmetric and math.isclose(doc_query_cosine, 1.0, abs_tol=1e-6):
        raise RuntimeError(
            f"arm {recipe.arm!r} is asymmetric (document={recipe.document_prompt!r}, "
            f"query={recipe.query_prompt!r}) but both sides produced identical vectors. "
            "At least one side ignored its prompt."
        )


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------


def _load_benchmark(name: str) -> tuple[list[Any], list[set[str]], set[frozenset[str]]]:
    from langres.data.registry import get_benchmark

    corpus, gold_clusters, gold_pairs = get_benchmark(name).load()
    return list(corpus), list(gold_clusters), set(gold_pairs)


def _reachable_ceiling(corpus: Sequence[Any], gold_pairs: set[frozenset[str]]) -> float:
    """Fraction of gold pairs that cross sources, i.e. that blocking may find at all."""
    source = {str(record.id): getattr(record, "source", None) for record in corpus}
    if len({value for value in source.values() if value is not None}) < 2:
        return 1.0
    reachable = sum(1 for pair in gold_pairs if len({source.get(rid) for rid in pair}) == 2)
    return reachable / len(gold_pairs) if gold_pairs else 1.0


def _source_sizes(corpus: Sequence[Any]) -> tuple[int, int] | None:
    from collections import Counter

    counts = Counter(getattr(record, "source", None) for record in corpus)
    if len(counts) != 2 or None in counts:
        return None
    left, right = (counts[key] for key in sorted(counts))
    return left, right


def _per_record_recall(
    candidate_pairs: set[frozenset[str]], gold_clusters: Sequence[set[str]]
) -> dict[str, tuple[float, str]]:
    """Per record: what fraction of its gold partners blocking captured."""
    scores: dict[str, tuple[float, str]] = {}
    for index, cluster in enumerate(gold_clusters):
        if len(cluster) < 2:
            continue
        cluster_id = f"c{index}"
        for record_id in cluster:
            partners = cluster - {record_id}
            captured = sum(
                1 for partner in partners if frozenset({record_id, partner}) in candidate_pairs
            )
            scores[record_id] = (captured / len(partners), cluster_id)
    return scores


def _paired_interval(
    baseline: dict[str, tuple[float, str]], candidate: dict[str, tuple[float, str]]
) -> Any:
    from langres.experiments.statistics import PairedScore, paired_entity_bootstrap

    shared = sorted(set(baseline) & set(candidate))
    if len(shared) < 2:
        return None
    observations = tuple(
        PairedScore(
            entity_id=record_id,
            baseline=baseline[record_id][0],
            candidate=candidate[record_id][0],
            cluster_id=baseline[record_id][1],
        )
        for record_id in shared
    )
    return paired_entity_bootstrap(observations, seed=SEED)


def _build_embedder(spec: ModelSpec, document_prompt: str | None, cache_dir: Path) -> Any:
    """An embedder whose *document* side carries ``document_prompt``.

    Uses the shipped ``prompts=``/``prompt_name=`` API rather than hand-prefixing
    the corpus text: ``create_index`` takes no prompt argument, so
    ``default_prompt_name`` is how the document half of an asymmetric recipe is
    actually driven in library code. Measuring the path users would take is the
    point.
    """
    import hashlib

    from langres.core.embeddings import DiskCachedEmbedder, SentenceTransformerEmbedder

    kwargs: dict[str, Any] = {}
    if document_prompt:
        kwargs = {"prompts": {"document": document_prompt}, "prompt_name": "document"}
    base = SentenceTransformerEmbedder(
        spec.name,
        batch_size=spec.batch_size,
        normalize_embeddings=True,
        dtype=spec.dtype,  # type: ignore[arg-type]
        **kwargs,
    )
    digest = hashlib.blake2b((document_prompt or "").encode(), digest_size=8).hexdigest()
    namespace = f"{spec.name.replace('/', '__')}__{spec.dtype or 'default'}__doc{digest}"
    return DiskCachedEmbedder(base, cache_dir=cache_dir, namespace=namespace)


def evaluate_model_on_benchmark(
    spec: ModelSpec,
    benchmark: str,
    *,
    recipes: Sequence[Recipe],
    k_values: Sequence[int],
    cache_dir: Path,
) -> list[Row]:
    from langres.core.blockers import VectorBlocker
    from langres.core.blockers.vector import concat_comparable_fields
    from langres.core.indexes import FAISSIndex
    from langres.metrics.metrics import evaluate_blocking

    corpus, gold_clusters, gold_pairs = _load_benchmark(benchmark)
    schema = type(corpus[0])
    texts = [concat_comparable_fields(record) for record in corpus]
    # VectorBlocker(schema=...) builds a dict-consuming factory, so stream() wants
    # plain dicts even though the benchmark hands back schema instances.
    records = [record.model_dump() for record in corpus]
    ceiling = _reachable_ceiling(corpus, gold_pairs)
    sizes = _source_sizes(corpus)

    # One index per distinct document prompt; arms sharing a document side share it.
    indexes: dict[str | None, tuple[Any, np.ndarray, Any]] = {}
    baseline_vectors: np.ndarray | None = None
    baseline_pairs: set[frozenset[str]] | None = None
    baseline_recall: dict[str, tuple[float, str]] | None = None
    rows: list[Row] = []
    facts: dict[str, int | None] = {"parameter_count": None, "embedding_dim": None}

    for recipe in recipes:
        started = time.perf_counter()
        if recipe.document_prompt not in indexes:
            embedder = _build_embedder(spec, recipe.document_prompt, cache_dir)
            index = FAISSIndex(embedder=embedder, metric="cosine")
            index.create_index(texts)
            indexes[recipe.document_prompt] = (index, embedder.encode(texts), embedder)
        index, doc_vectors, embedder = indexes[recipe.document_prompt]
        if facts["parameter_count"] is None:
            facts["parameter_count"] = getattr(embedder.embedder, "parameter_count", None)
            facts["embedding_dim"] = int(doc_vectors.shape[1])

        query_vectors = embedder.encode(texts, prompt=recipe.query_prompt)
        if recipe.arm == "none":
            baseline_vectors = doc_vectors
        assert baseline_vectors is not None, "the 'none' arm must run first"

        doc_shift = _shift(doc_vectors, baseline_vectors)
        query_shift = _shift(query_vectors, baseline_vectors)
        doc_query_cosine = _mean_cosine(doc_vectors, query_vectors)
        _prompt_reached_encoder(recipe, doc_shift, query_shift, doc_query_cosine)

        for k in k_values:
            blocker = VectorBlocker(
                vector_index=index,
                schema=schema,
                text_field_extractor="concat_comparable_fields",
                k_neighbors=k,
                query_prompt=recipe.query_prompt,
            )
            candidates = list(blocker.stream(records))
            if sizes is not None:
                candidates = [c for c in candidates if c.left.source != c.right.source]
                stats = evaluate_blocking(
                    candidates, gold_clusters, n_left=sizes[0], n_right=sizes[1]
                )
            else:
                stats = evaluate_blocking(candidates, gold_clusters, num_records=len(corpus))

            pairs = {frozenset({str(c.left.id), str(c.right.id)}) for c in candidates}
            jaccard: float | None = None
            if k == HEADLINE_K:
                if recipe.arm == "none":
                    baseline_pairs = pairs
                    baseline_recall = _per_record_recall(pairs, gold_clusters)
                if baseline_pairs is not None:
                    union = len(pairs | baseline_pairs)
                    jaccard = len(pairs & baseline_pairs) / union if union else 1.0

            row = Row(
                model=spec.name,
                benchmark=benchmark,
                arm=recipe.arm,
                kind=recipe.kind,
                k=k,
                document_prompt=recipe.document_prompt,
                query_prompt=recipe.query_prompt,
                note=recipe.note,
                candidate_recall=stats.candidate_recall,
                candidate_precision=stats.candidate_precision,
                reduction_ratio=stats.reduction_ratio,
                total_candidates=stats.total_candidates,
                reachable_ceiling=ceiling,
                recall_of_reachable=(stats.candidate_recall / ceiling) if ceiling else 0.0,
                doc_shift_vs_none=doc_shift,
                query_shift_vs_none=query_shift,
                doc_query_cosine=doc_query_cosine,
                pair_jaccard_vs_none=jaccard,
                parameter_count=facts["parameter_count"],
                embedding_dim=facts["embedding_dim"],
                license=spec.license,
                osi_approved=spec.osi_approved,
                seconds=time.perf_counter() - started,
            )
            if k == HEADLINE_K and baseline_recall is not None:
                arm_recall = _per_record_recall(pairs, gold_clusters)
                interval = _paired_interval(baseline_recall, arm_recall)
                if interval is not None:
                    row.delta_recall_vs_none = interval.observed_difference
                    row.ci_low = interval.lower
                    row.ci_high = interval.upper
                    row.n_clusters = interval.n_clusters
            rows.append(row)
        recalls = {r.k: r.candidate_recall for r in rows if r.arm == recipe.arm}
        logger.info(
            "%s | %s | %s: recall@%s=%.4f doc_shift=%.4g query_shift=%.4g cos=%.4f",
            spec.name,
            benchmark,
            recipe.arm,
            max(recalls),
            recalls[max(recalls)],
            doc_shift,
            query_shift,
            doc_query_cosine,
        )
    return rows


# --------------------------------------------------------------------------
# Persistence + report
# --------------------------------------------------------------------------


def read_rows(path: Path) -> list[Row]:
    if not path.exists():
        return []
    return [Row(**json.loads(line)) for line in path.read_text().splitlines() if line.strip()]


def merge_rows(existing: Iterable[Row], fresh: Iterable[Row]) -> list[Row]:
    """Fresh cells replace old ones with the same (model, benchmark, arm, k)."""
    merged = {(row.model, row.benchmark, row.arm, row.k): row for row in existing}
    merged.update({(row.model, row.benchmark, row.arm, row.k): row for row in fresh})
    return sorted(merged.values(), key=lambda r: (r.model, r.benchmark, r.arm, r.k))


def write_rows(rows: Sequence[Row], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(asdict(row)) for row in rows) + "\n")


def _fmt(value: float | None, digits: int = 4) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _interval(row: Row) -> str:
    if row.ci_low is None or row.ci_high is None:
        return "-"
    spans_zero = row.ci_low <= 0.0 <= row.ci_high
    marker = " (spans 0)" if spans_zero else " **"
    return f"[{row.ci_low:+.4f}, {row.ci_high:+.4f}]{marker}"


def render_report(rows: Sequence[Row]) -> str:
    lines: list[str] = []
    add = lines.append
    add("# Do instruction prompts help embedding blockers? Per model, per benchmark")
    add("")
    add(
        "Generated by `examples/research/prompt_axis.py`. Every number is **candidate "
        "recall** at a fixed `k` -- no threshold anywhere, so nothing here confounds "
        "the retrieval effect with a cut. Confidence intervals are paired bootstraps "
        "resampled **by gold cluster** (`langres.experiments.statistics."
        "paired_entity_bootstrap`), never by pair row."
    )
    add("")
    add("**Results are reported per model. They are deliberately not averaged across ")
    add("models -- the whole question is whether the effect is model-specific.**")
    add("")

    add("## Where each prompt came from")
    add("")
    add("| model | licence | source of its documented prompt |")
    add("|---|---|---|")
    for spec in MODELS:
        source = PROMPT_SOURCES.get(spec.name, "-").replace("\n", " ")
        add(f"| `{spec.name}` | {spec.license} | {source} |")
    add("")

    add("## The arms")
    add("")
    add("| model | arm | kind | document side | query side | note |")
    add("|---|---|---|---|---|---|")
    for spec in MODELS:
        for recipe in spec.recipes:
            doc = "-" if recipe.document_prompt is None else f"`{recipe.document_prompt!r}`"
            query = "-" if recipe.query_prompt is None else f"`{recipe.query_prompt!r}`"
            add(
                f"| `{spec.name}` | `{recipe.arm}` | {recipe.kind} | {doc} | {query} "
                f"| {recipe.note} |"
            )
    add("")

    headline = [row for row in rows if row.k == HEADLINE_K]
    add(f"## Effect on candidate recall at k={HEADLINE_K}")
    add("")
    add(
        "`doc shift` / `query shift` are `1 - mean cosine` against the no-prompt arm's "
        "vectors: **any non-zero value is proof the prompt reached the encoder**, and a "
        "prompted arm measuring exactly 0 aborts the run instead of reporting a flat "
        "result. `pair J` is the Jaccard overlap of the candidate-pair set with the "
        "no-prompt arm -- proof the changed vectors changed the *retrieved neighbours*, "
        "not just the geometry. `**` on an interval means it excludes zero."
    )
    add("")
    for spec in MODELS:
        model_rows = [row for row in headline if row.model == spec.name]
        if not model_rows:
            continue
        params = next((r.parameter_count for r in model_rows if r.parameter_count), None)
        dim = next((r.embedding_dim for r in model_rows if r.embedding_dim), None)
        osi = "OSI-approved" if spec.osi_approved else "**NOT OSI-approved**"
        add(f"### `{spec.name}`")
        add("")
        add(f"{params or '?'} params, dim {dim or '?'}, licence `{spec.license}` ({osi}).")
        if spec.note:
            add("")
            add(spec.note)
        add("")
        add(
            "| benchmark | arm | kind | recall | Δ vs none | 95% CI | doc shift | "
            "query shift | doc·query | pair J | candidates |"
        )
        add("|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|")
        for benchmark in BENCHMARKS:
            bench_rows = [row for row in model_rows if row.benchmark == benchmark]
            for row in sorted(bench_rows, key=lambda r: (r.kind != "baseline", r.arm)):
                add(
                    f"| {benchmark} | `{row.arm}` | {row.kind} | {row.candidate_recall:.4f} "
                    f"| {_fmt(row.delta_recall_vs_none)} | {_interval(row)} "
                    f"| {row.doc_shift_vs_none:.4g} | {row.query_shift_vs_none:.4g} "
                    f"| {row.doc_query_cosine:.4f} | {_fmt(row.pair_jaccard_vs_none, 3)} "
                    f"| {row.total_candidates} |"
                )
        add("")

    add("## Every k")
    add("")
    add("| model | benchmark | arm | k | recall | recall/reachable | candidates |")
    add("|---|---|---|---:|---:|---:|---:|")
    for row in rows:
        add(
            f"| `{row.model}` | {row.benchmark} | `{row.arm}` | {row.k} "
            f"| {row.candidate_recall:.4f} | {row.recall_of_reachable:.4f} "
            f"| {row.total_candidates} |"
        )
    add("")
    return "\n".join(lines)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", default=[spec.name for spec in MODELS])
    parser.add_argument("--benchmarks", nargs="*", default=list(BENCHMARKS))
    parser.add_argument("--k", nargs="*", type=int, default=list(K_VALUES))
    parser.add_argument("--arms", nargs="*", default=None)
    parser.add_argument("--rows", type=Path, default=DEFAULT_ROWS_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()

    if args.render_only:
        args.report.write_text(render_report(read_rows(args.rows)))
        return 0

    for name in args.models:
        spec = MODELS_BY_NAME[name]
        recipes = [r for r in spec.recipes if args.arms is None or r.arm in args.arms]
        if recipes and recipes[0].arm != "none":
            recipes = [next(r for r in spec.recipes if r.arm == "none"), *recipes]
        for benchmark in args.benchmarks:
            fresh = evaluate_model_on_benchmark(
                spec,
                benchmark,
                recipes=recipes,
                k_values=args.k,
                cache_dir=args.cache_dir,
            )
            rows = merge_rows(read_rows(args.rows), fresh)
            write_rows(rows, args.rows)
            args.report.write_text(render_report(rows))
            logger.info("wrote %d rows -> %s", len(rows), args.rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
