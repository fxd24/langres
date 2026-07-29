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

**``--env-file .env`` is load-bearing, and ``OMP_NUM_THREADS=1`` is the reason.**
Without it this harness DEADLOCKS -- silently, at 0% CPU, with no error and no
traceback -- when it loads ``embeddinggemma-300m`` after faiss is already
imported. Measured: with faiss imported first, the Gemma load never returns
(killed at 180s); with ``OMP_NUM_THREADS=1`` it returns in 1.6s; without faiss
imported at all it returns in 1.3s either way. It is the faiss/torch duplicate
OpenMP runtime conflict on macOS, and note that ``KMP_DUPLICATE_LIB_OK`` alone
is **not** sufficient -- that suppresses the abort, not the deadlock. The
earlier models (MiniLM, bge, e5) complete fine, so the hang looks like "stuck on
Gemma" rather than an environment problem, which is what makes it expensive to
diagnose. If ``.env`` is absent (e.g. in a fresh worktree), set the three
variables explicitly instead::

    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \\
        uv run python examples/research/prompt_axis.py

Zero paid API calls: every model runs locally.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

logger = logging.getLogger("prompt_axis")

DEFAULT_ROWS_PATH = Path("docs/research/20260728_prompt_axis_rows.jsonl")
DEFAULT_REPORT_PATH = Path("docs/research/20260728_prompt_axis.md")
DEFAULT_CACHE_DIR = Path("tmp/prompt_axis_cache")

K_VALUES: tuple[int, ...] = (5, 10, 20, 50)
HEADLINE_K = 20
SEED = 0

#: Bootstrap replicates per comparison. Well above the library default of 1000,
#: and the reason is the multiplicity correction rather than the intervals.
#: Holm tests at `alpha/m` -- as low as 0.0125 here -- and a p-value estimated
#: from B replicates carries Monte-Carlo error `sqrt(p(2-p)/B)` (a *doubled* tail
#: proportion, so `2-p`, not the `1-p` of an ordinary proportion): at p≈0.02 that
#: is **0.00629** with B=1000, which is **half** the threshold being tested, so
#: the verdict would be decided by the draw rather than by the data. At B=20000
#: it is **0.00141**, 11% of it. Nothing below `2/(B+1)` is observable either,
#: and that floor must sit under the smallest threshold or strong effects cannot
#: be separated from merely-significant ones. Costs ~2 min for the largest
#: comparison; the embedding cache means a re-run pays this and almost nothing
#: else.
BOOTSTRAP_SAMPLES = 20000

#: Family-wise error rate for the Holm correction.
ALPHA = 0.05

#: The four benchmarks the committed study measured. `walmart_amazon` was in this
#: default but was never run, so the documented command did not reproduce the
#: committed artifact -- it silently added a fifth, unmeasured benchmark and
#: rewrote the report. A default sweep must reproduce the study it ships with.
#: (Found by automated review on PR #252.)
BENCHMARKS: tuple[str, ...] = (
    "fodors_zagat",
    "abt_buy",
    "amazon_google",
    "wdc_computers",
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
        "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3): prompts = {'query': 'Instruct: "
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
        "1110a243fdf4706b3f48f1d95db1a4f5529b4d41) registers no usable prompts. "
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
    #: The immutable commit these numbers were measured on. Enforced at load time
    #: by `_resolve_checkpoint_revision`, and folded into the cache namespace so a
    #: partition can never outlive the weights that wrote it.
    revision: str
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
        revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        license="apache-2.0",
        osi_approved=True,
        note="Control: not instruction-trained, ships no prompts.",
    ),
    ModelSpec(
        name="BAAI/bge-base-en-v1.5",
        revision="a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
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
        revision="f52bf8ec8c7124536f0efb74aca902b2995e5bcd",
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
        revision="57c266a740f537b4dc058e1b0cda161fd15afa75",
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
        revision="97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
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
    #
    # NOTE THE ESTIMATOR DIFFERS from `candidate_recall` above, and the two are
    # not interchangeable. `candidate_recall` is MICRO -- the fraction of all gold
    # pairs captured. This delta is MACRO over records -- the mean per-record
    # fraction of that record's gold partners captured -- because a paired
    # bootstrap needs a per-entity score to resample by cluster, and a single
    # corpus-wide micro ratio has no per-entity decomposition to resample.
    # They diverge whenever clusters differ in size: for bge/wdc_computers micro
    # recall moves +0.1098 while this macro delta reads +0.1224. Naming it
    # `delta_recall_vs_none` invited reading it as the difference of the recall
    # column beside it, which it is not. (Found by automated review on PR #252.)
    delta_per_record_recall: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    #: Achieved significance level from the SAME replicates as the interval, so
    #: `p_value <= t` means "the 1-t percentile interval excludes zero" at every
    #: `t` -- which is what Holm reads and what an interval alone cannot answer.
    #: An earlier draft recovered this by turning an interval endpoint into a
    #: standard error and assuming normality; that is calibrated at 0.95 and
    #: nowhere else, i.e. not at any level Holm actually tests at, so the
    #: retraction it produced rested on numbers the method could not supply.
    #: (Found by automated review on PR #252.)
    p_value: float | None = None
    #: Monte-Carlo error of `p_value`. A verdict within a few of these of its
    #: Holm threshold is resolution-limited, not settled, and the report says so
    #: rather than presenting the draw's noise as a finding.
    p_value_standard_error: float | None = None
    bootstrap_samples: int | None = None
    n_clusters: int | None = None
    parameter_count: int | None = None
    embedding_dim: int | None = None
    # Provenance: WHICH weights and WHICH prompt strings produced this row.
    #
    # Without these, `--resume` matched on (arm, k) alone, so changing a pinned
    # revision or editing a prompt's text left every old cell looking complete
    # and the report silently kept measurements of the previous study under the
    # new definition. `None` means "recorded before this field existed", which
    # `_cell_complete` treats as NOT complete -- unknown provenance must not
    # count as matching provenance. (Found by automated review on PR #252.)
    revision: str | None = None
    recipe_fingerprint: str | None = None
    license: str = "unknown"
    osi_approved: bool = False
    seconds: float = 0.0


# --------------------------------------------------------------------------
# Proof that the prompt reached the encoder
# --------------------------------------------------------------------------


#: How far the two encode paths may disagree on a *symmetric* recipe before we
#: call it a bug. Not a free parameter -- it is bounded on both sides by measured
#: values:
#:
#: * **Above**: encoder round-off. A ``float16`` checkpoint
#:   (``Qwen/Qwen3-Embedding-0.6B``) produced ``cosine=1.000126`` for the
#:   no-prompt arm, where both sides are literally the same call. A cosine
#:   *greater than 1* is arithmetically impossible for two distinct unit vectors,
#:   so that value is accumulation error and nothing else. ``float32`` models
#:   land within ``1e-7``.
#: * **Below**: the smallest real failure this check exists to catch. When one
#:   side silently drops its prompt the cosine falls to the asymmetric range --
#:   the *closest* such value measured anywhere in this sweep is ``0.9446``
#:   (``e5-base-v2``, ``official_asymmetric``), i.e. a deviation of ``0.055``.
#:
#: ``5e-3`` therefore sits ~5x above the worst observed round-off and ~11x below
#: the smallest genuine divergence. Widening it to swallow a *real* failure would
#: take a 10x further increase.
_SYMMETRIC_TOLERANCE = 5e-3


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
    doc_vectors: np.ndarray,
    query_vectors: np.ndarray,
    baseline_vectors: np.ndarray,
    doc_query_cosine: float,
) -> None:
    """Fail loudly when a non-empty prompt left the vectors untouched.

    Three independent signals, because a single one can be satisfied by accident:

    1. the corpus vectors moved -- proves ``prompt_name`` reached
       ``SentenceTransformer.encode`` through the *index build* path.
    2. the query vectors moved -- proves ``query_prompt`` reached the encoder
       through the *search* path, the exact seam that used to discard it.
    3. ``doc_query_cosine`` -- for a symmetric recipe the two sides must agree
       (cosine ~1), for an asymmetric one they must not. This catches a prompt
       that reached one path but was silently rewritten on the other, which
       neither of the first two can see alone.

    **(1) and (2) compare the arrays directly rather than thresholding the
    cosine shift, and that distinction is load-bearing.** An earlier version
    tested ``shift == 0.0``. On a ``float16`` checkpoint an *ignored* prompt does
    not produce exactly zero -- the committed Qwen3 baseline rows carry residuals
    like ``-0.0002207`` because the self-cosine lands just above 1 -- so a dropped
    prompt would have slipped past the exact comparison, and an ignored symmetric
    recipe would also have satisfied the cosine tolerance. The harness would then
    publish a flat result as evidence the prompt was applied: precisely the
    failure this guard exists to prevent. Array equality has no tolerance to tune
    and cannot drift: if the encoder ignored the prompt, the two calls saw
    identical input and returned identical bytes. (Found by automated review on
    PR #252.)

    Note the asymmetry in what each check compares. (1) and (2) compare *the same
    code path with and without a prompt*, so bit-identity is the correct
    predicate. (3) compares *two different code paths*, which legitimately differ
    in last-bit round-off, so it keeps a tolerance.
    """
    if recipe.document_prompt and np.array_equal(doc_vectors, baseline_vectors):
        raise RuntimeError(
            f"arm {recipe.arm!r} sets document_prompt={recipe.document_prompt!r} but the "
            "corpus vectors are bit-identical to the no-prompt arm. The prompt never "
            "reached the encoder -- this is a harness bug, not evidence that "
            "instructions do not help."
        )
    if recipe.query_prompt and np.array_equal(query_vectors, baseline_vectors):
        raise RuntimeError(
            f"arm {recipe.arm!r} sets query_prompt={recipe.query_prompt!r} but the query "
            "vectors are bit-identical to the no-prompt arm. The prompt never reached "
            "the encoder (this is exactly the search_all() bug #242 hardened)."
        )
    symmetric = recipe.document_prompt == recipe.query_prompt
    if symmetric and not math.isclose(doc_query_cosine, 1.0, abs_tol=_SYMMETRIC_TOLERANCE):
        raise RuntimeError(
            f"arm {recipe.arm!r} is symmetric ({recipe.document_prompt!r} on both sides) "
            f"but its document and query vectors disagree (cosine={doc_query_cosine:.6f}). "
            "One of the two encode paths is not applying the prompt it was given."
        )
    if not symmetric and np.array_equal(doc_vectors, query_vectors):
        raise RuntimeError(
            f"arm {recipe.arm!r} is asymmetric (document={recipe.document_prompt!r}, "
            f"query={recipe.query_prompt!r}) but both sides produced bit-identical "
            "vectors. At least one side ignored its prompt."
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
    """Per record: what fraction of its gold partners blocking captured.

    Clusters are labelled from a **deterministically ordered** copy, not from the
    loader's incoming order. The gold clusters arrive as ``set`` objects, so their
    enumeration order varies between processes under Python's per-process string
    hash randomisation. That order reaches the paired bootstrap as the cluster-id
    list ``random.Random(seed).choice`` samples from, which made the "seeded,
    reproducible" interval **not reproducible**: re-running an identical cell
    moved the CI bounds by ~0.002 while leaving the point estimate exactly
    unchanged (the observed difference is an order-independent mean; only the
    resampling saw the shuffle). Sorting by each cluster's smallest member is
    stable across processes and makes the seed mean what it claims.
    """
    scores: dict[str, tuple[float, str]] = {}
    for index, cluster in enumerate(sorted(gold_clusters, key=lambda c: min(c))):
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
    return paired_entity_bootstrap(observations, seed=SEED, samples=BOOTSTRAP_SAMPLES)


def _build_embedder(
    spec: ModelSpec,
    document_prompt: str | None,
    cache_dir: Path,
    *,
    texts: Sequence[str] = (),
    query_prompts: Sequence[str | None] = (),
) -> Any:
    """An embedder whose *document* side carries ``document_prompt``.

    Uses the shipped ``prompts=``/``prompt_name=`` API rather than hand-prefixing
    the corpus text: ``create_index`` takes no prompt argument, so
    ``default_prompt_name`` is how the document half of an asymmetric recipe is
    actually driven in library code. Measuring the path users would take is the
    point.
    """
    import hashlib

    from langres.core.embeddings import DiskCachedEmbedder, SentenceTransformerEmbedder

    revision, checkpoint_path = _resolve_checkpoint_revision(spec)
    kwargs: dict[str, Any] = {}
    if document_prompt:
        kwargs = {"prompts": {"document": document_prompt}, "prompt_name": "document"}
    base = SentenceTransformerEmbedder(
        # Load the resolved snapshot DIRECTORY, not the bare repo name. Comparing
        # `spec.revision` against whatever `main` points at only detects drift --
        # it cannot survive it, so once upstream advanced, the documented "fetch
        # the pinned revision" remedy was impossible and the study became
        # unreproducible. Loading the path pins the weights for real.
        # (Found by automated review on PR #252.)
        checkpoint_path,
        batch_size=spec.batch_size,
        normalize_embeddings=True,
        dtype=spec.dtype,  # type: ignore[arg-type]
        **kwargs,
    )
    digest = hashlib.blake2b((document_prompt or "").encode(), digest_size=8).hexdigest()
    # The revision belongs in the namespace, not just in a log line: it is what
    # makes the partition key IMMUTABLE. Keyed only on the mutable repo name, a
    # cache that outlived a checkpoint update kept hitting, and no prompt-reach
    # guard could see it -- those guards compare arms against each other, so
    # uniformly stale vectors look perfectly consistent. (Found by automated
    # review on PR #252.)
    namespace = (
        f"{spec.name.replace('/', '__')}__{revision[:12]}__{spec.dtype or 'default'}__doc{digest}"
    )
    cached = DiskCachedEmbedder(base, cache_dir=cache_dir, namespace=namespace)
    _assert_cache_matches_checkpoint(base, cached, texts, query_prompts)
    return cached


def _resolve_checkpoint_revision(spec: ModelSpec) -> tuple[str, str]:
    """Fetch the study's pinned commit and return ``(revision, local snapshot path)``.

    The harness used to load every model by its *mutable* repo name, so "which
    weights ran" was pinned by nothing: a re-run after an upstream update would
    silently measure different weights under the same ``model`` value, and
    ``--resume`` would merge the two into one table.

    That is not hypothetical here. Two ``Qwen3-Embedding-0.6B`` snapshots and
    three ``all-MiniLM-L6-v2`` snapshots are present in this machine's HF cache,
    and for both models ``refs/main`` resolves to a *different* commit than the
    one this study originally cited. (Both Qwen3 snapshots ship byte-identical
    ``config_sentence_transformers.json`` prompts and the resolved MiniLM
    revision still registers no prompts, so the measurements and the documented
    formats stand -- but nothing in the harness had established that.)

    Resolution asks for ``spec.revision`` explicitly rather than comparing it
    against ``main`` afterwards. Comparing only *detects* drift; it cannot
    survive it, so the moment upstream moved, the pinned revision could still be
    sitting in the cache while the run aborted and the study became impossible to
    reproduce. Asking for the commit works whether or not ``main`` has moved.

    (Found by automated review on PR #252, across two rounds.)
    """
    from huggingface_hub import snapshot_download

    try:
        path = snapshot_download(spec.name, revision=spec.revision, local_files_only=True)
    except Exception as exc:  # noqa: BLE001 -- any resolution failure is the same story
        raise RuntimeError(
            f"{spec.name} revision {spec.revision} is not in the local HF cache "
            f"({type(exc).__name__}). This study's numbers describe that commit. Fetch it "
            f"(`huggingface_hub.snapshot_download('{spec.name}', revision='{spec.revision}')`) "
            f"rather than measuring whatever `main` currently points at."
        ) from exc
    return spec.revision, path


def _assert_cache_matches_checkpoint(
    base: Any,
    cached: Any,
    texts: Sequence[str],
    query_prompts: Sequence[str | None] = (),
) -> None:
    """Refuse any cache partition whose vectors the live checkpoint disagrees with.

    The namespace now pins the checkpoint revision, so a partition cannot outlive
    the *weights* that wrote it. It can still outlive the **runtime**: a
    sentence-transformers change to how ``prompt=`` is applied would leave the
    weights identical and the prompted vectors wrong. So this re-encodes one real
    corpus text through the raw checkpoint and compares it to what the cache
    serves.

    Two properties this needs, both learned the hard way:

    1. **Every partition, not just the document one.** ``DiskCachedEmbedder`` keys
       each explicit query prompt separately, so checking only the default/document
       partition left every bge / Qwen3 / ER query-prompt partition unvalidated --
       and stale prompted vectors still *differ* from the baseline, so the
       prompt-reach guards pass on them too.
    2. **Never a vacuous pass.** On a cache miss the wrapper delegates to ``base``
       and stores the result, so comparing a miss to a fresh encode is comparing a
       value to itself -- a check that cannot fail. Only *populated* entries are
       probed, looked up directly rather than through ``encode``.

    (Found by automated review on PR #252, across two rounds.)
    """
    if not texts:
        return
    text = texts[0]
    for prompt in dict.fromkeys((None, *query_prompts)):
        if cached._get_from_db(cached._hash_text(text, prompt)) is None:
            continue  # nothing stored for this partition: nothing to validate
        fresh = base.encode([text]) if prompt is None else base.encode([text], prompt=prompt)
        served = cached.encode([text]) if prompt is None else cached.encode([text], prompt=prompt)
        if np.allclose(fresh, served, atol=_CANARY_TOLERANCE):
            continue
        raise RuntimeError(
            f"embedding cache {cached.db_path} disagrees with the live checkpoint on the "
            f"{'document/default' if prompt is None else repr(prompt)} partition "
            f"(max |delta| = {float(np.max(np.abs(fresh - served))):.3g}). It was written "
            "by a different embedding runtime. Delete the cache directory and re-run rather "
            "than publishing a measurement of a model you are no longer loading."
        )


#: Canary agreement bound. Well above float16 encode jitter, far below the
#: disagreement a genuinely different checkpoint produces.
_CANARY_TOLERANCE = 1e-3


def _document_prompt_order(recipes: Sequence[Recipe]) -> list[str | None]:
    """Distinct document prompts, bare-document group first.

    The ``none`` arm lives in the bare group and establishes the baseline vectors
    and per-record recall every other arm is measured against, so its group must
    run first.
    """
    seen: list[str | None] = []
    for recipe in sorted(recipes, key=lambda r: r.document_prompt is not None):
        if recipe.document_prompt not in seen:
            seen.append(recipe.document_prompt)
    return seen


def evaluate_model_on_benchmark(
    spec: ModelSpec,
    benchmark: str,
    *,
    recipes: Sequence[Recipe],
    k_values: Sequence[int],
    cache_dir: Path,
) -> Iterator[list[Row]]:
    """Yield one arm's rows at a time so the caller can persist as they land.

    A sweep cell costs minutes of encoding on the larger checkpoints, so the
    caller flushes after every arm rather than at the end: a crash at 90% should
    cost the last arm, not the run.
    """
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

    baseline_vectors: np.ndarray | None = None
    baseline_pairs: set[frozenset[str]] | None = None
    baseline_recall: dict[str, tuple[float, str]] | None = None
    rows: list[Row] = []
    facts: dict[str, int | None] = {"parameter_count": None, "embedding_dim": None}

    # Arms are walked GROUPED BY DOCUMENT PROMPT, with the bare-document group
    # first (it holds the `none` baseline every other arm is measured against).
    # Grouping is what keeps exactly one checkpoint resident: a per-arm loop would
    # hold one loaded model per distinct document prompt, and EmbeddingGemma has
    # six of them -- roughly 7.2 GB of fp32 weights before inference or FAISS
    # allocations, enough to OOM an 8 GB GPU even though the arms run
    # sequentially. (Found by automated review on PR #252.)
    for document_prompt in _document_prompt_order(recipes):
        group = [r for r in recipes if r.document_prompt == document_prompt]
        embedder = _build_embedder(
            spec,
            document_prompt,
            cache_dir,
            texts=texts,
            # Every query prompt this group will actually use, so the canary
            # validates the partitions the run is about to read.
            query_prompts=list(dict.fromkeys(r.query_prompt for r in group)),
        )
        index = FAISSIndex(embedder=embedder, metric="cosine")
        index.create_index(texts)
        doc_vectors = embedder.encode(texts)
        if facts["parameter_count"] is None:
            facts["parameter_count"] = getattr(embedder.embedder, "parameter_count", None)
            facts["embedding_dim"] = int(doc_vectors.shape[1])

        for recipe in group:
            started = time.perf_counter()
            query_vectors = embedder.encode(texts, prompt=recipe.query_prompt)
            if recipe.arm == "none":
                baseline_vectors = doc_vectors
            assert baseline_vectors is not None, "the 'none' arm must run first"

            doc_query_cosine = _mean_cosine(doc_vectors, query_vectors)
            _prompt_reached_encoder(
                recipe, doc_vectors, query_vectors, baseline_vectors, doc_query_cosine
            )
            doc_shift = _shift(doc_vectors, baseline_vectors)
            query_shift = _shift(query_vectors, baseline_vectors)

            arm_rows: list[Row] = []
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
                    revision=spec.revision,
                    recipe_fingerprint=_recipe_fingerprint(recipe),
                )
                if k == HEADLINE_K and baseline_recall is not None:
                    arm_recall = _per_record_recall(pairs, gold_clusters)
                    interval = _paired_interval(baseline_recall, arm_recall)
                    if interval is not None:
                        row.delta_per_record_recall = interval.observed_difference
                        row.ci_low = interval.lower
                        row.ci_high = interval.upper
                        row.p_value = interval.p_value
                        row.p_value_standard_error = interval.p_value_standard_error
                        row.bootstrap_samples = interval.samples
                        row.n_clusters = interval.n_clusters
                arm_rows.append(row)
            rows.extend(arm_rows)
            recalls = {r.k: r.candidate_recall for r in arm_rows}
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
            yield arm_rows

        # Drop this document prompt's checkpoint before loading the next one.
        # `blocker` MUST be in this list: it is bound in the innermost loop and
        # leaks out of it (Python loop variables outlive the loop), and it holds
        # `index` -> the cached embedder -> the loaded model. Deleting only the
        # three names below left the checkpoint reachable through `blocker`, so
        # the next group loaded a second model before this one was collected --
        # the exact OOM the grouping exists to prevent. A fix that does not
        # release the last reference is not a fix. (Found by automated review on
        # PR #252, on the first round's own fix.)
        del embedder, index, doc_vectors, blocker


# --------------------------------------------------------------------------
# Persistence + report
# --------------------------------------------------------------------------


def read_rows(path: Path) -> list[Row]:
    if not path.exists():
        return []
    return [Row(**json.loads(line)) for line in path.read_text().splitlines() if line.strip()]


def merge_rows(existing: Iterable[Row], fresh: Iterable[Row]) -> list[Row]:
    """Fresh cells replace old ones with the same (model, benchmark, arm, k).

    Deliberately does **not** enforce single-provenance. A sweep is mixed for
    most of its life: it rewrites one ``(model, benchmark)`` at a time, so from
    the first flush until the last the file legitimately holds new rows beside
    not-yet-recomputed old ones. An earlier version raised here, which meant a
    revision bump made every flush fail -- and because the check ran *before*
    ``write_rows``, each failure discarded an arm that had just been computed.
    A guard that destroys the results it was written to protect is worse than no
    guard. (Found by automated review on PR #252.)

    Enforcement lives at the two points where a mixed file is actually wrong:
    :func:`find_mixed_provenance` at the end of a run, and
    :func:`_assert_rows_match_declaration` before anything is published.
    """
    merged = {(row.model, row.benchmark, row.arm, row.k): row for row in existing}
    merged.update({(row.model, row.benchmark, row.arm, row.k): row for row in fresh})
    return sorted(merged.values(), key=lambda r: (r.model, r.benchmark, r.arm, r.k))


def find_mixed_provenance(rows: Iterable[Row]) -> dict[str, list[str]]:
    """Models carrying more than one checkpoint revision, if any.

    ``_cell_complete`` refuses to *resume* a cell whose revision moved, so a
    re-run recomputes it -- but it decides one cell at a time, and a run narrowed
    with ``--benchmarks``/``--models`` only recomputes what it was pointed at.
    The rest keep the old checkpoint's numbers, and the file then holds two sets
    of weights under one model name with nothing on the surface to show it. Only
    a pass over the whole set can see that.

    Reports rather than resolves. Which revision was meant is a question about
    intent that this cannot answer, and an earlier draft of this harness resolved
    a similar ambiguity by deleting rows -- destroying measured data three review
    rounds running. Callers name the conflict and stop; nothing here removes a
    measurement.
    """
    by_model: dict[str, set[str | None]] = {}
    for row in rows:
        by_model.setdefault(row.model, set()).add(row.revision)
    return {
        model: sorted(str(revision) for revision in revisions)
        for model, revisions in sorted(by_model.items())
        if len(revisions) > 1
    }


def write_rows(rows: Sequence[Row], path: Path) -> None:
    """Replace the rows file atomically.

    This file is the *sole* artifact of a sweep that costs hours, and it is
    rewritten in full after every arm. A plain ``write_text`` truncates first, so
    a kill or a full disk between truncate and write leaves it empty or
    half-parsed -- destroying every previously completed cell, which is exactly
    the accident `.claude/rules/data-safety.md` exists for and which this repo has
    paid for once already. Writing a sibling temp file and ``os.replace``-ing it
    means the destination is always either the old complete file or the new one.
    (Found by automated review on PR #252.)
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(asdict(row)) for row in rows) + "\n"
    temporary = path.with_name(f"{path.name}.tmp{os.getpid()}")
    temporary.write_text(payload)
    os.replace(temporary, path)


def _missing_p_value(row: Row) -> bool:
    """A headline comparison carrying an interval but no p-value.

    That combination is only producible by an *older* harness: the bootstrap
    reports a p-value from the same draw as the bounds, so a row with
    ``ci_low`` set and ``p_value`` absent was measured before the p-value
    existed. Such a row is silently dropped by :func:`_row_p_value`, and a
    dropped row shrinks its Holm family -- the exact data-dependent denominator
    the correction is written to avoid -- while the report still renders and
    still prints a family size, because nothing checked.

    One predicate, read by both the resume check and the publish guard, so a
    stale row can neither survive ``--resume`` nor reach the report.
    """
    return (
        row.k == HEADLINE_K and row.arm != "none" and row.ci_low is not None and row.p_value is None
    )


def _cell_complete(
    existing: Sequence[Row],
    spec: ModelSpec,
    benchmark: str,
    recipes: Sequence[Recipe],
    k_values: Sequence[int],
) -> bool:
    """True when every (arm, k) of this cell is already recorded.

    Resume is deliberately at *cell* granularity, not arm granularity: every
    arm's delta and interval are computed against the ``none`` arm's vectors and
    per-record recall, so a partially-resumed cell would have no baseline to
    compare against. Re-running an incomplete cell is cheap anyway -- the
    document and query vectors come back from the on-disk embedding cache.

    Completeness also includes **carrying a p-value** (see
    :func:`_missing_p_value`) -- a cell measured by a harness that did not
    record one is re-run rather than resumed, because the correction cannot
    read it.

    Completeness includes **provenance**, not just coverage. Matching on
    ``(arm, k)`` alone meant a changed checkpoint revision or an edited prompt
    string left the old cell looking complete, so the report kept measurements
    of the previous study under the new definition. A row whose ``revision`` or
    ``recipe_fingerprint`` is absent or different is not a hit -- unknown
    provenance is treated as *not* matching, never as matching. (Found by
    automated review on PR #252.)
    """
    want = {(recipe.arm, k): _recipe_fingerprint(recipe) for recipe in recipes for k in k_values}
    have = {
        (row.arm, row.k)
        for row in existing
        if row.model == spec.name
        and row.benchmark == benchmark
        and row.revision == spec.revision
        and want.get((row.arm, row.k)) == row.recipe_fingerprint
        and not _missing_p_value(row)
    }
    return set(want) <= have


def _recipe_fingerprint(recipe: Recipe) -> str:
    """A stable hash of the two prompt strings -- what actually defines the treatment."""
    import hashlib

    payload = f"{recipe.document_prompt!r}\x00{recipe.query_prompt!r}"
    return hashlib.blake2b(payload.encode(), digest_size=8).hexdigest()


def _benchmark_order(rows: Sequence[Row]) -> list[str]:
    """Benchmarks present in ``rows``: the known ones in declared order, then any extras."""
    present = {row.benchmark for row in rows}
    return [b for b in BENCHMARKS if b in present] + sorted(present - set(BENCHMARKS))


def _cell(text: str) -> str:
    """Text made safe for one GitHub-Flavoured Markdown table cell.

    Backticks do NOT protect a pipe from GFM's table parser -- it splits columns
    before it looks at inline code -- and EmbeddingGemma's own prompts contain
    literal pipes (``title: none | text: ``). Rendering them raw silently split
    those rows into extra, misaligned cells. (Found by automated review on PR
    #252.)
    """
    return text.replace("\n", " ").replace("|", "\\|")


def _fmt(value: float | None, digits: int = 4) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _spans_zero(row: Row) -> bool:
    return row.ci_low is None or row.ci_high is None or row.ci_low <= 0.0 <= row.ci_high


def _on_boundary(row: Row) -> bool:
    """An interval that *closes* on zero -- one bound resting exactly there.

    Detected from the stored bound rather than from the p-value: an equality
    test against a computed p is at the mercy of float round-off, and the first
    version of this classifier used one and reported zero boundary cells over
    four real ones.

    A *degenerate* comparison is excluded. Its bounds are both exactly zero, so
    it satisfies the letter of the test, but it is already reported as "no effect
    to measure" and listing it here again as a near-miss says the opposite of
    what it is.
    """
    return not _is_degenerate(row) and (row.ci_low == 0.0 or row.ci_high == 0.0)


def _holm_cell(row: Row, verdict: tuple[float | None, bool] | None) -> str:
    """This comparison's label in the results tables.

    ``withdrawn`` is reserved for its literal meaning: a comparison that *was*
    significant on its own and lost that to the multiplicity correction. It used
    to be printed for **every** non-rejected comparison whose interval excluded
    zero, which silently swept in a second, different case -- one whose p-value
    never cleared the uncorrected ``alpha`` in the first place, so the correction
    took nothing away from it. Those get ``n.s.`` instead. (The two can differ:
    ``p <= t`` implies the ``1-t`` interval excludes zero, but the converse can
    fail inside a band of ``4/(B+1)``, where an interval clears zero and the
    counting rule does not.)
    """
    if verdict is None:
        return "-"
    p_value, rejected = verdict
    if p_value is None:
        return "n/a"
    if rejected:
        return "**holds**"
    if _spans_zero(row):
        return "-"
    return "n.s." if p_value > ALPHA else "withdrawn"


def _interval(row: Row) -> str:
    if row.ci_low is None or row.ci_high is None:
        return "-"
    spans_zero = row.ci_low <= 0.0 <= row.ci_high
    marker = " (spans 0)" if spans_zero else " **"
    return f"[{row.ci_low:+.4f}, {row.ci_high:+.4f}]{marker}"


def _row_p_value(row: Row) -> float | None:
    """The p-value Holm should use for this comparison.

    Read from the row, where it was written by the same bootstrap draw that
    produced the interval beside it -- so ``p <= t`` *implies* "the ``1-t``
    percentile interval excludes zero" at every ``t``, exactly, including the
    ``alpha/m`` levels the correction actually tests at. (The converse holds only
    up to the draw's order-statistic granularity; see
    :func:`langres.experiments.statistics._achieved_significance_level`.)

    A *degenerate* comparison -- the arm changed no record's recall, so the
    estimate and both interval bounds are exactly zero -- is **not** dropped. It
    is a genuine non-rejection and returns ``p = 1.0``, keeping it in its family.
    Omitting it instead made the Holm denominator depend on the observed result:
    a family was size 3 or size 4 according to whether an arm happened to move
    the saturated benchmark, which is a data-dependent family size and invalid
    however well the p-values are calibrated. (Found by automated review on
    PR #252.) ``None`` is reserved for a comparison that was never measured.
    """
    if row.delta_per_record_recall is None or row.ci_low is None or row.ci_high is None:
        return None
    if row.delta_per_record_recall == 0.0 and row.ci_low == 0.0 and row.ci_high == 0.0:
        return 1.0
    return row.p_value


def _is_degenerate(row: Row) -> bool:
    """No record's recall moved, so there is nothing for the bootstrap to resample."""
    return row.delta_per_record_recall == 0.0 and row.ci_low == 0.0 and row.ci_high == 0.0


class HolmStep(NamedTuple):
    """One comparison's place in Holm's step-down order."""

    key: str
    p_value: float
    #: ``m - i + 1`` -- how many hypotheses are still live at this step. Carried
    #: as an exact integer rather than recovered as ``alpha / threshold``, which
    #: is not exact for a multiplier that is not a power of two and would put
    #: float noise on a boundary comparison.
    multiplier: int
    #: ``alpha / multiplier`` -- the level this step is tested at.
    threshold: float


def _holm_steps(p_values: dict[str, float], alpha: float = ALPHA) -> list[HolmStep]:
    """The family in Holm's step-down order, with each step's level.

    Holm sorts a family ascending and tests ``p(i)`` against
    ``alpha / (m - i + 1)``. Both the verdicts and the resolution check need to
    agree on *which comparison was tested against what*, and deriving that order
    twice would let the two drift -- so the order is produced once, here. (They
    then read different fields of the same step: :func:`_holm` compares
    ``multiplier * p`` against ``alpha`` and :func:`_resolution_limited` measures
    distance to ``threshold``. Equivalent in exact arithmetic; the integer
    multiplier is preferred where a boundary comparison could turn on float
    round-off.)
    """
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    total = len(ordered)
    return [
        HolmStep(key, p_value, total - index, alpha / (total - index))
        for index, (key, p_value) in enumerate(ordered)
    ]


def _holm(p_values: dict[str, float], alpha: float = ALPHA) -> set[str]:
    """Keys Holm's step-down rejects at family-wise ``alpha``.

    Implemented through Holm's **adjusted** p-values,
    ``p~(i) = max(p~(i-1), min(1, (m - i + 1) * p(i)))``, rejecting where
    ``p~ <= alpha``.

    **This is exactly equivalent to "walk the sorted family and stop at the first
    p above its threshold" -- for every input, not merely for distinct
    p-values.** Since ``alpha < 1``, ``min(1, mult * p) > alpha`` iff
    ``mult * p > alpha``, so the running maximum first exceeds ``alpha`` at
    precisely the index where the threshold form first fails. Ties were never at
    risk under either form: the thresholds ``alpha / (m - i + 1)`` increase with
    ``i``, so if the earlier (harder) member of a tie passes, the later one does
    too, and if it fails the walk stops before reaching the later one. The
    adjusted-p shape is kept because it makes that behaviour *visible* -- tied
    comparisons carry equal adjusted p by construction -- rather than leaving it
    resting on the monotonicity of a threshold sequence a reader has to notice.
    Ties are common here: ``p = 1`` for a comparison that moved no record's
    recall, and the floor ``2/(B+1)`` for every sufficiently strong effect, so
    both ends of the range are atoms.

    Holm is uniformly at least as powerful as Bonferroni and needs no
    independence assumption. On where the dependence actually lives, see the
    report section this feeds: *within* a family the comparisons sit on disjoint
    benchmarks, and the strong dependence is *across* families.
    """
    rejected: set[str] = set()
    adjusted = 0.0
    for step in _holm_steps(p_values, alpha):
        adjusted = max(adjusted, min(1.0, step.multiplier * step.p_value))
        if adjusted > alpha:
            break
        rejected.add(step.key)
    return rejected


def _resolution_limited(
    headline: Sequence[Row],
    verdicts: dict[tuple[str, str, str], tuple[float | None, bool]],
    margin: float = 3.0,
) -> list[tuple[str, str, str]]:
    """Verdicts close enough to their Holm threshold that the draw size decided them.

    Every p-value here is estimated from a finite number of replicates and carries
    Monte-Carlo error. A comparison whose p sits within a few of those of the
    threshold it is being tested against has not been settled by the data -- it
    has been settled by how many times the bootstrap happened to resample. Naming
    those is the difference between a correction that reports its own resolution
    and one that launders sampling noise into a finding.
    """
    by_family: dict[tuple[str, str], dict[str, float]] = {}
    for (model, arm, benchmark), (p_value, _) in verdicts.items():
        if p_value is not None:
            by_family.setdefault((model, arm), {})[benchmark] = p_value
    errors = {(row.model, row.arm, row.benchmark): row.p_value_standard_error for row in headline}
    limited: list[tuple[str, str, str]] = []
    for (model, arm), family in by_family.items():
        for step in _holm_steps(family):
            error = errors.get((model, arm, step.key))
            if error and abs(step.p_value - step.threshold) < margin * error:
                limited.append((model, arm, step.key))
    return sorted(limited)


def _multiplicity(rows: Sequence[Row]) -> dict[tuple[str, str, str], tuple[float | None, bool]]:
    """Per ``(model, arm, benchmark)``: its p-value and its Holm verdict.

    **The family is one ``(model, arm)`` across benchmarks**, which is the unit
    the claims are stated in -- "this recipe helps this checkpoint" is a claim
    about a row of the results table, and it is the multiplicity a reader of that
    row is exposed to. It is declared here rather than left implicit because the
    choice changes the answer: correcting over the whole sweep instead would be
    stricter, and the report prints that sensitivity check next to this one.

    **Every measured benchmark stays in its family**, including the ones an arm
    left untouched (``p = 1``). The family size is therefore fixed by the design
    -- how many benchmarks were measured -- and not by which of them happened to
    move.
    """
    verdicts: dict[tuple[str, str, str], tuple[float | None, bool]] = {}
    families: dict[tuple[str, str], dict[str, float]] = {}
    for row in rows:
        if row.k != HEADLINE_K or row.arm == "none":
            continue
        p_value = _row_p_value(row)
        if p_value is None:
            continue
        verdicts[(row.model, row.arm, row.benchmark)] = (p_value, False)
        families.setdefault((row.model, row.arm), {})[row.benchmark] = p_value
    for (model, arm), family in families.items():
        for benchmark in _holm(family):
            p_value, _ = verdicts[(model, arm, benchmark)]
            verdicts[(model, arm, benchmark)] = (p_value, True)
    return verdicts


def _assert_rows_match_declaration(rows: Sequence[Row]) -> None:
    """Refuse to render numbers that were not measured under the prompts we print.

    The report's "Where each prompt came from" and "The arms" tables are rendered
    from ``MODELS`` -- today's specs -- while every number below them comes from
    the rows file. Nothing in the layout couples the two, so editing a prompt
    string or bumping a checkpoint and re-rendering without re-measuring produces
    a document that *describes* one experiment and *reports* another, with no
    visible seam. That is the failure this study was warned about: a caption and
    a measurement drifting apart silently.

    So before rendering, every row must carry provenance, and that provenance
    must equal the spec and recipe the prose is about. Absent provenance fails --
    a row that cannot say what produced it cannot be shown to agree.

    It also refuses a headline comparison that has an interval but no p-value
    (:func:`_missing_p_value`). That row is not merely incomplete: the
    correction drops it, the family it belonged to silently shrinks, and the
    report goes on to print a family size derived from the survivors -- an
    expectation regenerated from the very thing that broke. The guard has to
    read the row, because the rendered number cannot see its own absence.
    """
    problems: list[str] = []
    for row in rows:
        spec = MODELS_BY_NAME.get(row.model)
        if spec is None:
            problems.append(f"{row.model}/{row.benchmark}/{row.arm}: model is not declared")
            continue
        recipe = next((r for r in spec.recipes if r.arm == row.arm), None)
        if recipe is None:
            problems.append(f"{row.model}/{row.benchmark}/{row.arm}: arm is not declared")
            continue
        if row.revision != spec.revision:
            problems.append(
                f"{row.model}/{row.benchmark}/{row.arm}: measured on revision "
                f"{row.revision!r}, report describes {spec.revision!r}"
            )
        want = _recipe_fingerprint(recipe)
        if row.recipe_fingerprint != want:
            problems.append(
                f"{row.model}/{row.benchmark}/{row.arm}: measured under prompt "
                f"fingerprint {row.recipe_fingerprint!r}, report describes {want!r}"
            )
        if _missing_p_value(row):
            problems.append(
                f"{row.model}/{row.benchmark}/{row.arm}: has a k={HEADLINE_K} interval but "
                "no p-value, so the correction would drop it and shrink its family"
            )
    if problems:
        shown = sorted(set(problems))
        detail = "\n  ".join(shown[:10])
        more = f"\n  ... and {len(shown) - 10} more" if len(shown) > 10 else ""
        raise ValueError(
            "refusing to render: these rows do not match the prompts and checkpoints "
            f"the report describes.\n  {detail}{more}\n"
            "Re-measure the affected cells, or render from the rows file that matches "
            "this harness. Nothing was written."
        )


def render_report(rows: Sequence[Row]) -> str:
    _assert_rows_match_declaration(rows)
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
        source = _cell(PROMPT_SOURCES.get(spec.name, "-"))
        add(f"| `{spec.name}` | {spec.license} | {source} |")
    add("")

    add("## The arms")
    add("")
    add("| model | arm | kind | document side | query side | note |")
    add("|---|---|---|---|---|---|")
    for spec in MODELS:
        for recipe in spec.recipes:
            doc = "-" if recipe.document_prompt is None else f"`{_cell(recipe.document_prompt)}`"
            query = "-" if recipe.query_prompt is None else f"`{_cell(recipe.query_prompt)}`"
            add(
                f"| `{spec.name}` | `{recipe.arm}` | {recipe.kind} | {doc} | {query} "
                f"| {recipe.note} |"
            )
    add("")

    headline = [row for row in rows if row.k == HEADLINE_K]
    verdicts = _multiplicity(rows)
    add(f"## Effect on candidate recall at k={HEADLINE_K}")
    add("")
    add(
        "**`recall` and `Δ per-record` are different estimators and the second is NOT "
        "the difference of the first.** `recall` is *micro* candidate recall -- the "
        "fraction of all gold pairs captured. `Δ per-record` is *macro* over records -- "
        "the mean per-record fraction of that record's gold partners captured -- "
        "because a paired bootstrap needs a per-entity score to resample by cluster. "
        "They diverge when clusters differ in size (bge/wdc_computers: micro moves "
        "+0.1098, macro reads +0.1224). The CI belongs to the macro quantity."
    )
    add("")
    add(
        "`doc shift` / `query shift` are `1 - mean cosine` against the no-prompt arm's "
        "vectors, reported as magnitudes. The *guard* does not use them: it compares the "
        "arrays directly, because on float16 an ignored prompt yields a small non-zero "
        "residual rather than exactly 0. `pair J` is the Jaccard overlap of the "
        "candidate-pair set with the no-prompt arm -- proof the changed vectors changed "
        "the *retrieved neighbours*, not just the geometry. `**` on an interval means it "
        "excludes zero."
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
            "| benchmark | arm | kind | recall (micro) | Δ per-record (macro) | 95% CI "
            "| Holm | doc shift | query shift | doc·query | pair J | candidates |"
        )
        add("|---|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|")
        # Order by BENCHMARKS, but render every benchmark actually present in the
        # rows. Iterating BENCHMARKS alone dropped any benchmark reached via
        # `--benchmarks` from every headline table while still writing its rows --
        # a measurement paid for and then hidden. (Found by automated review on
        # PR #252.)
        for benchmark in _benchmark_order(model_rows):
            bench_rows = [row for row in model_rows if row.benchmark == benchmark]
            for row in sorted(bench_rows, key=lambda r: (r.kind != "baseline", r.arm)):
                add(
                    f"| {benchmark} | `{row.arm}` | {row.kind} | {row.candidate_recall:.4f} "
                    f"| {_fmt(row.delta_per_record_recall)} | {_interval(row)} "
                    f"| {_holm_cell(row, verdicts.get((row.model, row.arm, row.benchmark)))} "
                    f"| {row.doc_shift_vs_none:.4g} | {row.query_shift_vs_none:.4g} "
                    f"| {row.doc_query_cosine:.4f} | {_fmt(row.pair_jaccard_vs_none, 3)} "
                    f"| {row.total_candidates} |"
                )
        add("")

    add("## What survives correcting for multiplicity")
    add("")
    add(
        "Every interval above is a separate 95% interval, and this sweep contains "
        f"{sum(1 for row in headline if row.arm != 'none')} of them. Read one at a "
        "time each controls its own error rate; read as a set -- which is how a "
        'claim like *"this recipe helps this checkpoint"* is actually read -- they '
        "do not. So each `(model, arm)` row of the tables above is treated as one "
        "**family** across benchmarks and corrected with **Holm's step-down** at "
        "family-wise α=0.05. Holm is uniformly at least as powerful as Bonferroni "
        "and assumes nothing about dependence."
    )
    add("")
    add(
        "**Where the dependence actually is.** Inside one family the comparisons sit "
        "on four *disjoint* benchmarks -- separate corpora, separate gold clusters -- "
        "so they are close to independent, and Holm's freedom from an independence "
        "assumption is not what carries them. The strong dependence is **across** "
        "families: every arm of a checkpoint is measured against the same `none` "
        "baseline over the same corpus. Holm controls the error rate *within* each "
        "family under any dependence and says nothing about the sweep as a whole -- "
        "which is exactly what the study-wide sensitivity check at the end of this "
        "section is for. (An earlier version of this paragraph cited the shared "
        "baseline as the reason Holm was needed, pointing at dependence that lives "
        "between families rather than in the one being corrected. Found by automated "
        "review on PR #252.)"
    )
    add("")
    samples = next((row.bootstrap_samples for row in headline if row.bootstrap_samples), None)
    add(
        "**The p-values come from the same bootstrap draw as the intervals**, as the "
        "*achieved significance level* -- `2 * min(P(diff* <= 0), P(diff* >= 0))` over "
        f"the {samples or '?'} replicates, with the usual `(count+1)/(B+1)` correction. "
        "That is the percentile interval **inverted**: `p <= t` *implies* "
        '"the `1-t` percentile interval over this draw excludes zero" at every `t`, '
        "exactly, which is what makes it legitimate to read at the `α/m` levels Holm "
        "tests at. (The converse can fail inside a band of `4/(B+1)` -- "
        f"{4 / ((samples or 20000) + 1):.0e} here -- because the interval interpolates "
        "between order statistics near the block of replicates equal to zero. The "
        "p-value is the conservative side of that band, and the band is narrower than "
        "the **3** Monte-Carlo standard errors the resolution check below already "
        "flags at -- so it cannot reach a verdict that check has not named. One "
        "standard error would *not* suffice: at the smallest attainable p-values it "
        "is smaller than the band, which an earlier version of this sentence claimed "
        "otherwise.) "
        "(An earlier version recovered p by turning an interval endpoint into a "
        "standard error and assuming normality. That is pinned to agree at 0.95 and "
        "uncalibrated everywhere else -- including at every level Holm actually uses. "
        "Found by automated review on PR #252.)"
    )
    add("")
    observed_families = {key[:2] for key in verdicts}
    covered = {
        family: frozenset(key[2] for key in verdicts if key[:2] == family)
        for family in observed_families
    }
    sizes = sorted({len(benchmarks) for benchmarks in covered.values()})
    # The paragraph below *claims* the family size is fixed by the design. Nothing made
    # that true: both the sizes AND the set of families are counted from the verdicts,
    # so a lost comparison shrinks its family and a wholly lost family simply vanishes
    # -- and the sentence then reports the survivors' shape as if it were the design.
    # An expectation regenerated from the thing that broke cannot detect it breaking, so
    # both halves are checked against something upstream of the loss: the families
    # against `MODELS`, and each family's benchmark **set** against the other families'.
    #
    # Set identity, not size. Comparing counts let a family made of three declared
    # benchmarks plus a substituted fourth pass as complete, and simultaneously rejected
    # a legitimate `--benchmarks` sweep that adds a fifth to every family -- wrong in
    # both directions at once. Identity is also the property the correction actually
    # needs: what invalidates Holm is a denominator that varies with the outcome, which
    # is exactly two families disagreeing about which benchmarks they contain. A sweep
    # that deliberately measures a different set is coherent, and every table names the
    # benchmarks it used. (Found by automated review on PR #252.)
    #
    # An empty set is a different condition (nothing comparable was measured at all,
    # e.g. a single-arm smoke render) and is not this guard's business.
    if len(set(covered.values())) > 1:
        widest = max(covered.values(), key=len)
        short = {
            f"{model}/{arm}": sorted(widest - benchmarks)
            for (model, arm), benchmarks in sorted(covered.items())
            if benchmarks != widest
        }
        raise ValueError(
            "refusing to render: Holm families do not all span the same benchmarks, so "
            "the correction's denominator depends on what the data happened to do -- "
            "the data-dependent family size this section says it avoids. "
            f"Widest family spans {sorted(widest)}; these are short of it: {short}. "
            "Re-measure the missing cells. Nothing was written."
        )
    declared_families = {
        (spec.name, recipe.arm)
        for spec in MODELS
        for recipe in spec.recipes
        if recipe.arm != "none"
    }
    if observed_families and observed_families != declared_families:
        # Reachable through the documented CLI: `--models <one model>` against a fresh
        # rows file leaves every surviving family the full four benchmarks wide, so the
        # size check reads clean -- while "Where each prompt came from" and "The arms"
        # above still describe all of `MODELS`. That is the caption-and-measurement
        # drift this whole guard family exists to prevent, one level up from where it
        # was being checked. (Found by review of the round-6 fixes.)
        missing = sorted(declared_families - observed_families)
        extra = sorted(observed_families - declared_families)
        raise ValueError(
            "refusing to render: the report describes every arm in MODELS, so the "
            "results must cover every one of them. "
            f"Missing from the rows: {missing or 'none'}. Not declared: {extra or 'none'}. "
            "Render from a complete sweep, or narrow MODELS to match what was measured. "
            "Nothing was written."
        )
    thresholds = (
        " / ".join(f"{ALPHA / (sizes[0] - i):.4g}" for i in range(sizes[0])) if sizes else "-"
    )
    add(
        "**Every measured benchmark stays in its family, including the ones an arm "
        "left untouched** (`p = 1`). Dropping them would make the Holm denominator "
        "depend on the observed result -- a family of 3 or of 4 according to whether "
        "an arm happened to move the saturated benchmark -- and a data-dependent "
        "family size is invalid however well the p-values are calibrated. Family "
        f"sizes here: **m = {', '.join(str(size) for size in sizes)}**"
        + (f", so the thresholds are {thresholds}." if len(sizes) == 1 else ".")
        + " (Also found by automated review on PR #252.)"
    )
    add("")

    testable = {key: value for key, value in verdicts.items() if value[0] is not None}
    degenerate = [row for row in headline if row.arm != "none" and _is_degenerate(row)]
    boundary = sorted(
        (row.model, row.arm, row.benchmark)
        for row in headline
        if row.arm != "none"
        and (row.model, row.arm, row.benchmark) in testable
        and _on_boundary(row)
    )
    excluded_zero = {
        (row.model, row.arm, row.benchmark)
        for row in headline
        if row.arm != "none" and not _spans_zero(row)
    }
    survives = {key for key, (_, rejected) in testable.items() if rejected}
    where = sorted({row.benchmark for row in degenerate})
    baselines = sorted(
        {row.candidate_recall for row in headline if row.arm == "none" and row.benchmark in where}
    )
    span = ""
    if baselines:
        span = (
            f"{baselines[0]:.4f}"
            if len(baselines) == 1
            else f"{baselines[0]:.4f}-{baselines[-1]:.4f}"
        )
    add(
        f"- **{len(testable)}** comparisons entered the correction. For "
        f"**{len(degenerate)}** of them the replicate distribution is a **point mass at "
        "zero** -- the estimate and both bounds land exactly there, which is what "
        "happens when no record's recall moved; they carry `p = 1` and are retained."
        + (
            f" They fall entirely on {', '.join(f'`{b}`' for b in where)}"
            + (f", which every model already reaches at no-prompt recall {span}." if span else ".")
            if degenerate
            else ""
        )
    )
    named = ", ".join(f"`{model.split('/')[-1]}`/`{arm}`/{bench}" for model, arm, bench in boundary)
    add(
        f"- **{len(boundary)}** comparisons rest a bound *exactly* on zero "
        f"({named or 'none'}). An interval closing on zero does not exclude it, so "
        "none of them counted as an effect before the correction either."
    )
    lost = excluded_zero - survives
    withdrawn = {key for key in lost if (testable.get(key, (1.0, False))[0] or 1.0) <= ALPHA}
    add(
        f"- Read uncorrected, **{len(excluded_zero)}** intervals exclude zero. After "
        f"Holm, **{len(survives)}** survive. Of the rest, **{len(withdrawn)}** were "
        "significant on their own and lost it to the correction -- those are the "
        f"withdrawals; the other **{len(lost - withdrawn)}** never cleared the "
        "uncorrected α=0.05 p-value, so the correction took nothing from them."
    )

    limited = _resolution_limited(headline, verdicts)
    add(
        f"- **{len(limited)}** verdicts sit within 3 Monte-Carlo standard errors of "
        "their own Holm threshold"
        + (
            ": "
            + ", ".join(f"`{m.split('/')[-1]}`/`{a}`/{b}" for m, a, b in limited)
            + ". Those are decided by the size of the bootstrap draw as much as by the "
            "data, and more replicates -- not more argument -- is what would settle them."
            if limited
            else ". So no verdict here turns on a p-value within 3 Monte-Carlo standard "
            "errors of the threshold it was tested against. That bounds how much the "
            "replicate count could have moved these verdicts; it is not a claim about "
            "how large the effects are, nor about anything the p-values cannot see."
        )
    )
    add("")
    add("| model | arm | benchmark | Δ per-record | 95% CI | p | ± MC | Holm α=0.05 |")
    add("|---|---|---|---:|---|---:|---:|---|")
    for row in sorted(headline, key=lambda r: (r.model, r.arm, r.benchmark)):
        verdict = verdicts.get((row.model, row.arm, row.benchmark))
        if verdict is None or verdict[0] is None:
            continue
        key = (row.model, row.arm, row.benchmark)
        if key not in excluded_zero and not verdict[1]:
            continue
        mc = "-" if row.p_value_standard_error is None else f"{row.p_value_standard_error:.4f}"
        add(
            f"| `{row.model}` | `{row.arm}` | {row.benchmark} "
            f"| {_fmt(row.delta_per_record_recall)} | {_interval(row)} "
            f"| {verdict[0]:.3g} | {mc} | {_holm_cell(row, verdict)} |"
        )
    add("")

    study_wide = _holm(
        {"|".join(key): p_value for key, (p_value, _) in testable.items() if p_value is not None}
    )
    add(
        "**Sensitivity to the family choice.** Correcting instead over the whole "
        f"sweep as one family of {len(testable)} would leave **{len(study_wide)}** "
        f"comparisons standing rather than {len(survives)}. That is the stricter "
        "reading, and it is reported here so the per-`(model, arm)` choice above is "
        "visible as a choice rather than assumed."
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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip (model, benchmark) cells already fully recorded in --rows.",
    )
    args = parser.parse_args()

    if args.render_only:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        try:
            args.report.write_text(render_report(read_rows(args.rows)))
        except ValueError as error:
            logger.error("refusing to publish: %s", error)
            return 1
        return 0

    # Reject an unknown --arms up front. Nothing downstream deletes rows any more,
    # so a typo is no longer destructive -- but it would still silently produce a
    # narrower sweep than asked for, and a study that quietly measures less than
    # its command says is its own failure. Fail loudly instead.
    if args.arms is not None:
        known = {recipe.arm for spec in MODELS for recipe in spec.recipes}
        unknown = sorted(set(args.arms) - known)
        if unknown:
            parser.error(f"unknown arm(s): {', '.join(unknown)}. Known arms: {sorted(known)}")

    for name in args.models:
        spec = MODELS_BY_NAME[name]
        recipes = [r for r in spec.recipes if args.arms is None or r.arm in args.arms]
        if not recipes:
            logger.info("skipping %s -- no requested arm applies to it", spec.name)
            continue
        if recipes[0].arm != "none":
            recipes = [next(r for r in spec.recipes if r.arm == "none"), *recipes]
        for benchmark in args.benchmarks:
            if args.resume and _cell_complete(
                read_rows(args.rows), spec, benchmark, recipes, args.k
            ):
                logger.info("skipping %s | %s -- already complete", spec.name, benchmark)
                continue
            # NOTE: nothing is deleted up front. An earlier revision cleared the
            # cell here so a crashed rerun would look partial to `--resume`, and
            # that one destructive write produced a bug in every subsequent review
            # round: an unknown `--arms` value wiped cells and exited 0; a *valid*
            # narrowing selector like `--arms none --k 20` silently deleted every
            # unselected measurement; and a failure to load the pinned snapshot or
            # the benchmark destroyed the cell before anything could replace it.
            #
            # The provenance fields subsume what the deletion was for. A row is
            # only reusable if its `revision` and `recipe_fingerprint` match, so a
            # cell rebuilt under changed weights or edited prompts is rejected by
            # `_cell_complete` regardless of what is left lying beside it, and rows
            # that survive are by construction measurements of the *same* treatment
            # on the same weights. `merge_rows` then replaces each
            # (model, benchmark, arm, k) as its replacement actually arrives.
            # Deleting data to keep a bookkeeping invariant was the wrong trade;
            # recording what the data *is* costs nothing and cannot lose a row.
            # (Found by automated review on PR #252, across three rounds.)
            for arm_rows in evaluate_model_on_benchmark(
                spec,
                benchmark,
                recipes=recipes,
                k_values=args.k,
                cache_dir=args.cache_dir,
            ):
                # Flush after every arm, not every benchmark: the big checkpoints
                # spend minutes per arm and a crash must not cost the whole cell.
                rows = merge_rows(read_rows(args.rows), arm_rows)
                write_rows(rows, args.rows)
            logger.info("wrote rows -> %s", args.rows)
            # Best-effort mid-sweep render. Until the last cell lands, the file
            # still holds rows from the previous study, and the publish guard is
            # right to refuse them -- but refusing must not end the run. An
            # earlier version let that exception propagate, so a revision or
            # prompt change made every rerun die on its first completed cell.
            # (Found by automated review on PR #252.)
            args.report.parent.mkdir(parents=True, exist_ok=True)
            try:
                args.report.write_text(render_report(read_rows(args.rows)))
            except ValueError as error:
                logger.info("report not regenerated yet -- sweep still in progress: %s", error)

    # Enforce at completion, where a mixed file really is wrong.
    final_rows = read_rows(args.rows)
    mixed = find_mixed_provenance(final_rows)
    if mixed:
        logger.error(
            "rows hold more than one checkpoint per model: %s. Every row is safely "
            "on disk and nothing was deleted -- re-run the stale cells so each "
            "model comes from one revision.",
            "; ".join(f"{model}: {revisions}" for model, revisions in mixed.items()),
        )
        return 1
    # Publish -- but only *enforce* publication when this invocation was asked to
    # produce the whole declared design. `prompt_axis_sweep.sh` runs one
    # (model, benchmark) per process, so a scoped run legitimately leaves the rows
    # short and the publish guard is right to refuse them; treating that refusal as
    # an error made the documented sweep abort after its first (expensive) cell
    # under `set -e`. The sweep publishes once at the end with `--render-only`,
    # which is where the refusal is a real failure. (Found by automated review on
    # PR #252 -- a guard added one round earlier turned this from "renders a partial
    # report" into "kills the sweep".)
    scope = []
    if set(args.models) != {spec.name for spec in MODELS}:
        scope.append("--models")
    if set(args.benchmarks) != set(BENCHMARKS):
        scope.append("--benchmarks")
    if args.arms is not None:
        scope.append("--arms")
    try:
        args.report.write_text(render_report(final_rows))
    except ValueError as error:
        if not scope:
            logger.error("rows are complete but cannot be published: %s", error)
            return 1
        logger.info(
            "not publishing: this run was narrowed by %s, so the rows are not expected "
            "to cover the declared design yet. Every measured row is on disk; run with "
            "--render-only once the sweep is complete. (%s)",
            "/".join(scope),
            error,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
