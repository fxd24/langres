"""Reproduce *published* embedding-blocking numbers with langres's own components.

langres had never re-derived anyone else's *blocking* number. (It had replicated a
matching result -- ``langres.data.peeters`` -- but that replays their archived model
answers rather than re-deriving a metric from our own models.) This script does one
thing: it re-runs the candidate-set protocol that DeepBlocker (PVLDB 2021) and
UniBlocker (arXiv:2404.14831) describe, on langres's shipped benchmark corpora,
through langres's own :class:`SentenceTransformerEmbedder` and
:class:`FAISSIndex`, and writes the measured numbers next to the published ones.

**The deliverable is confidence, not a good-looking number.** A gap that is
explained is a success; a match obtained by tuning is a failure. Nothing here is
tuned: the serialization is langres's shipped ``concat_comparable_fields``, the
models are named on the command line, and no threshold or ``k`` is fitted.

The published protocol (verified against the papers' own arithmetic -- see
``docs/research/20260728_external_reproduction_reference.json``)::

    C = union over a in A of topK(a -> B)      so |C| = K * |A|
    PC = |C & G| / |G|                          (pair completeness, aka recall)
    PQ = |C & G| / |C|                          (pair quality, aka precision)
    mAP = sum_k (PC_k - PC_{k-1}) * PQ_k

Two things about this differ from ``langres.optimize.score_blocking`` and are
the whole reason this script exists rather than a call to that function:

1. **Direction.** ``score_blocking`` builds a symmetric kNN over the pooled
   corpus and then drops same-source pairs. The papers index table B and query
   with table A only. Those are different candidate sets at the same ``k``.
2. **Gold set.** ``score_blocking`` scores against the *transitive closure* of
   the gold clusters (``Benchmark.load()``'s third return), which on a
   many-to-many benchmark is much larger than the raw positive list -- 13,763 vs
   5,347 pairs on ``dblp_scholar``. The papers score against the raw positive
   list. This script uses the raw list and reports both counts.

Run::

    OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE uv run --extra semantic python \\
        examples/research/external_reproduction.py

    # one benchmark / one model. Name the model EXACTLY as it appears in
    # MODEL_REVISIONS -- the revision lookup is an exact match, so the bare
    # `all-mpnet-base-v2` alias would take the unpinned branch.
    ... external_reproduction.py --benchmarks abt_buy \\
        --models sentence-transformers/all-mpnet-base-v2

    # re-render the markdown from existing rows, measuring nothing.
    # Needs no extras -- stdlib + numpy only, no model and no index.
    uv run python examples/research/external_reproduction.py --render-only

``--extra semantic`` is not optional for a measuring run: ``FAISSIndex`` needs
``faiss-cpu`` and ``SentenceTransformerEmbedder`` needs ``sentence-transformers``,
and both live in that extra, so a bare ``uv sync`` checkout fails on the first
import.

$0: no paid API and no key. Not offline -- the first run of a checkpoint
downloads it from the Hugging Face Hub.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = REPO_ROOT / "docs" / "research"
REFERENCE_PATH = RESEARCH_DIR / "20260728_external_reproduction_reference.json"
ROWS_PATH = RESEARCH_DIR / "20260728_external_reproduction_rows.jsonl"
CROSSCHECK_PATH = RESEARCH_DIR / "20260728_external_reproduction_crosscheck.json"
REPORT_PATH = RESEARCH_DIR / "20260728_external_reproduction.md"
CACHE_DIR = REPO_ROOT / "tmp" / "external_reproduction_cache"

#: Largest ``k`` we retrieve. 150 because DeepBlocker's Table 6 uses K=150 on
#: DBLP-Google; UniBlocker caps its own budget at 100 and we report both.
K_MAX = 150
#: UniBlocker's comparison budget (Section 4.2).
UNIBLOCKER_K_CAP = 100
#: UniBlocker's pair-completeness threshold (Section 4.2).
PC_THRESHOLD = 0.90

#: Metric revision. Bump when the measurement changes meaning; every row records
#: its own value so a stale row can never be silently mixed into a fresh table.
METRIC_REVISION = 1


@dataclass(frozen=True)
class BenchSpec:
    """How to load one benchmark and which side is the papers' table A.

    ``a_source`` is the value of ``record.source`` for the query table. Getting
    this backwards silently changes |C| and the whole comparison, so
    ``expect_a``/``expect_b`` are asserted at load time rather than trusted.
    """

    module: str
    loader: str
    a_source: str
    expect_a: int
    expect_b: int


BENCHMARKS: dict[str, BenchSpec] = {
    # A-side is the table the papers query with; sizes are asserted on load.
    "fodors_zagat": BenchSpec(
        "langres.data.er_benchmarks", "load_fodors_zagat", "fodors", 533, 331
    ),
    "abt_buy": BenchSpec("langres.data.abt_buy", "load_abt_buy", "abt", 1081, 1092),
    "amazon_google": BenchSpec(
        "langres.data.amazon_google", "load_amazon_google", "amazon", 1363, 3226
    ),
    "dblp_acm": BenchSpec("langres.data.dblp_acm", "load_dblp_acm", "a", 2616, 2294),
    "dblp_scholar": BenchSpec("langres.data.dblp_scholar", "load_dblp_scholar", "a", 2616, 64263),
    "walmart_amazon": BenchSpec(
        "langres.data.walmart_amazon", "load_walmart_amazon", "a", 2554, 22074
    ),
    "wdc_computers": BenchSpec("langres.data.wdc_computers", "load_wdc_computers", "a", 2204, 2443),
    "febrl_person": BenchSpec("langres.data.febrl_person", "load_febrl_person", "a", 500, 500),
}

#: Default models. ``all-mpnet-base-v2`` is the checkpoint UniBlocker's
#: "STransformer" baseline is built on (see the reference file's
#: ``stransformer_identity``, which flags that identity as an inference).
#: ``all-MiniLM-L6-v2`` is langres's own default embedder, so it answers the
#: separate question "does what langres ships out of the box reach that bar".
DEFAULT_MODELS = ("sentence-transformers/all-mpnet-base-v2", "all-MiniLM-L6-v2")

#: Hub commit each default checkpoint is pinned to. A bare ``org/name`` resolves
#: to whatever ``main`` points at on the day you run it, so an unpinned rerun can
#: load different weights than the committed rows were measured with -- and the
#: disk cache, keyed on the bare name, would happily serve the old vectors under
#: the new identity. That is silent drift in exactly the artifact this study
#: exists to make trustworthy.
#:
#: Provenance, so this is checkable rather than asserted: each SHA is what this
#: machine's Hugging Face cache has in ``refs/main`` for that repo, read after the
#: measuring run. ``refs/main`` is by definition what an *unpinned* load resolves
#: to, so pinning here changes nothing about what the committed rows used -- it
#: only stops a future run from quietly resolving to something else.
MODEL_REVISIONS: dict[str, str] = {
    "sentence-transformers/all-mpnet-base-v2": "e8c3b32edf5434bc2275fc9bab85f82640a19130",
    "all-MiniLM-L6-v2": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
}


def _load(spec: BenchSpec) -> tuple[list[Any], list[Any], set[frozenset[str]], list[set[str]]]:
    """Load one benchmark, split into (A, B) and return the RAW gold pairs.

    Returns ``(records_a, records_b, raw_gold_pairs, gold_clusters)``.

    ``raw_gold_pairs`` is the pooled ``label == 1`` list, i.e. the papers' ``G``.
    It is deliberately NOT the transitive closure that ``Benchmark.load()``
    returns -- see the module docstring.
    """
    import importlib

    module = importlib.import_module(spec.module)
    loaded = getattr(module, spec.loader)()

    if len(loaded) == 3:
        corpus, clusters, raw_gold = loaded
    else:
        # Fodors-Zagat's loader returns (corpus, clusters) only: its ground truth
        # is the perfectMapping file, which is already exactly the 2-element
        # clusters, so raw == closure and we can derive it.
        corpus, clusters = loaded
        raw_gold = {frozenset(c) for c in clusters if len(c) == 2}

    records_a = [r for r in corpus if str(getattr(r, "source")) == spec.a_source]
    records_b = [r for r in corpus if str(getattr(r, "source")) != spec.a_source]

    if len(records_a) != spec.expect_a or len(records_b) != spec.expect_b:
        raise AssertionError(
            f"table sizes moved: got A={len(records_a)} B={len(records_b)}, "
            f"expected A={spec.expect_a} B={spec.expect_b}. The A/B assignment or "
            f"the shipped data changed; every |C| = k*|A| comparison below would "
            f"be measuring something other than the published protocol."
        )
    return records_a, records_b, raw_gold, clusters


def _closure_size(clusters: Sequence[set[str]]) -> int:
    """Number of pairs in the transitive closure of the gold clusters."""
    return sum(len(c) * (len(c) - 1) // 2 for c in clusters if len(c) >= 2)


def _build_embedder(model_name: str, cache_dir: Path) -> tuple[Any, str | None]:
    """langres's own sentence-transformer embedder behind a disk cache.

    Returns ``(embedder, revision)``. ``revision`` is the pinned Hub commit from
    :data:`MODEL_REVISIONS`, or ``None`` for a model this script does not pin --
    in which case the caller records the absence rather than implying a pin.

    The cache namespace carries the revision as well as the name, so a rerun on a
    different commit cannot be served stale vectors from the old one. It is also
    prefixed so it can never collide with the embedder ladder's namespaces, whose
    canary-pinning protocol this script does not implement.
    """
    from langres.core.embeddings import DiskCachedEmbedder, SentenceTransformerEmbedder
    from langres.core.model_ref import ModelRef

    revision = MODEL_REVISIONS.get(model_name)
    if revision is None:
        logger.warning(
            "%s is not in MODEL_REVISIONS, so this run is NOT revision-pinned and "
            "may not reproduce: whatever the Hub's `main` points at today is what "
            "gets loaded. Caching is DISABLED for it -- a shared namespace keyed on "
            "the name alone would serve the previous checkpoint's vectors after "
            "`main` moves, which is worse than paying to re-encode. Add its commit "
            "to MODEL_REVISIONS to get both the pin and the cache.",
            model_name,
        )
        return SentenceTransformerEmbedder(model_name, normalize_embeddings=True), None

    base = SentenceTransformerEmbedder(
        ModelRef(base=model_name, kind="hf", revision=revision), normalize_embeddings=True
    )
    namespace = "extrepro__" + model_name.replace("/", "__") + "__" + revision[:12]
    cached = DiskCachedEmbedder(embedder=base, cache_dir=cache_dir, namespace=namespace)
    return cached, revision


def measure(
    benchmark: str,
    model_name: str,
    *,
    cache_dir: Path,
    k_max: int = K_MAX,
) -> dict[str, Any]:
    """Run the published protocol for one (benchmark, model) cell.

    Returns a JSON-serializable row. ``pc``/``pq`` are 1-indexed conceptually but
    stored 0-indexed: ``pc[0]`` is PC at k=1.
    """
    from langres.core.blockers.vector import concat_comparable_fields
    from langres.core.comparators import StringComparator
    from langres.core.indexes.vector_index import FAISSIndex

    spec = BENCHMARKS[benchmark]
    records_a, records_b, raw_gold, clusters = _load(spec)

    texts_a = [concat_comparable_fields(r) for r in records_a]
    texts_b = [concat_comparable_fields(r) for r in records_b]

    started = time.perf_counter()
    embedder, model_revision = _build_embedder(model_name, cache_dir)
    index = FAISSIndex(embedder=embedder, metric="cosine")
    index.create_index(texts_b)
    build_seconds = time.perf_counter() - started

    search_started = time.perf_counter()
    k_eff = min(k_max, len(records_b))
    _distances, neighbours = index.search(texts_a, k_eff)
    search_seconds = time.perf_counter() - search_started

    a_pos = {str(r.id): i for i, r in enumerate(records_a)}
    b_pos = {str(r.id): j for j, r in enumerate(records_b)}

    # rank_of[i][j] = position of B-record j in A-record i's neighbour list.
    rank_of: list[dict[int, int]] = [
        {int(col): rank for rank, col in enumerate(row) if col >= 0} for row in neighbours
    ]

    # A gold pair is in C_k iff its B-side is within its A-side's top-k. Gold
    # pairs whose endpoints are not one-per-side are counted as unreachable
    # rather than dropped: the papers' G is cross-source by construction, and
    # silently shrinking the denominator would inflate PC.
    ranks: list[int | None] = []
    non_cross = 0
    for pair in raw_gold:
        left, right = sorted(pair)
        i = a_pos.get(left, a_pos.get(right))
        j = b_pos.get(right, b_pos.get(left))
        if i is None or j is None:
            non_cross += 1
            ranks.append(None)
            continue
        ranks.append(rank_of[i].get(j))

    gold_n = len(raw_gold)
    hits = np.zeros(k_eff, dtype=np.int64)
    for rank in ranks:
        if rank is not None and rank < k_eff:
            hits[rank] += 1
    cumulative = np.cumsum(hits)

    pc = (cumulative / gold_n).tolist()
    pq = [float(cumulative[k - 1]) / float(k * len(records_a)) for k in range(1, k_eff + 1)]

    def _map_at(cap: int) -> float:
        cap = min(cap, k_eff)
        total = 0.0
        previous = 0.0
        for k in range(1, cap + 1):
            total += (pc[k - 1] - previous) * pq[k - 1]
            previous = pc[k - 1]
        return total

    def _k_at(threshold: float, cap: int) -> int | None:
        cap = min(cap, k_eff)
        for k in range(1, cap + 1):
            if pc[k - 1] >= threshold:
                return k
        return None

    k90 = _k_at(PC_THRESHOLD, UNIBLOCKER_K_CAP)

    return {
        "metric_revision": METRIC_REVISION,
        "benchmark": benchmark,
        "model": model_name,
        "model_revision": model_revision,
        "n_a": len(records_a),
        "n_b": len(records_b),
        "gold_raw": gold_n,
        "gold_closure": _closure_size(clusters),
        "gold_non_cross_source": non_cross,
        "k_max": k_eff,
        "pc": pc,
        "pq": pq,
        "map_at_100": _map_at(UNIBLOCKER_K_CAP),
        "k_at_pc90": k90,
        "pc_at_k90": pc[k90 - 1] if k90 else None,
        "pq_at_k90": pq[k90 - 1] if k90 else None,
        "index_build_seconds": round(build_seconds, 1),
        "search_seconds": round(search_seconds, 1),
        "serialization": "concat_comparable_fields",
        "serialization_fields": [
            s.name for s in StringComparator.from_schema(type(records_a[0])).feature_specs
        ],
        "serialization_sample_a": texts_a[0][:200],
        "serialization_sample_b": texts_b[0][:200],
    }


def crosscheck(
    benchmark: str,
    model_name: str,
    k: int,
    *,
    cache_dir: Path,
) -> dict[str, Any]:
    """Attribute the gap between the published protocol and langres's own.

    Three recalls off the *same* embeddings, changing one thing at a time:

    ``paper``
        directional ``A -> B`` top-k, scored against the raw positive list.
        This is what DeepBlocker and UniBlocker report.
    ``paper_direction_closure_gold``
        the same candidate set, scored against the transitive closure. Isolates
        the gold-set definition.
    ``langres_score_blocking``
        symmetric top-k over the pooled corpus with same-source pairs dropped,
        scored against the closure. This is what
        :func:`langres.optimize.score_blocking` reports.

    Any difference between the first and the last is protocol, not model.
    """
    from langres.core.blockers.vector import concat_comparable_fields
    from langres.core.indexes.vector_index import FAISSIndex

    spec = BENCHMARKS[benchmark]
    records_a, records_b, raw_gold, clusters = _load(spec)
    corpus = records_a + records_b
    texts = [concat_comparable_fields(r) for r in corpus]

    embedder, model_revision = _build_embedder(model_name, cache_dir)

    # Directional: index B, query A.
    directional = FAISSIndex(embedder=embedder, metric="cosine")
    directional.create_index(texts[len(records_a) :])
    _d, neigh_a = directional.search(texts[: len(records_a)], min(k, len(records_b)))
    ids_a = [str(r.id) for r in records_a]
    ids_b = [str(r.id) for r in records_b]
    directional_pairs = {
        frozenset({ids_a[i], ids_b[int(j)]}) for i, row in enumerate(neigh_a) for j in row if j >= 0
    }

    # Symmetric: one pooled index, every record queries it (score_blocking's shape).
    pooled = FAISSIndex(embedder=embedder, metric="cosine")
    pooled.create_index(texts)
    _d2, neigh_all = pooled.search_all(min(k + 1, len(corpus)))
    ids_all = [str(r.id) for r in corpus]
    source_of = {str(r.id): str(getattr(r, "source")) for r in corpus}
    symmetric_pairs = {
        frozenset({ids_all[i], ids_all[int(j)]})
        for i, row in enumerate(neigh_all)
        for j in row
        if j >= 0 and int(j) != i
    }
    symmetric_pairs = {p for p in symmetric_pairs if len({source_of[x] for x in p}) == 2}

    closure_gold = {
        frozenset({a, b})
        for cluster in clusters
        if len(cluster) >= 2
        for i, a in enumerate(sorted(cluster))
        for b in sorted(cluster)[i + 1 :]
    }

    def _recall(cands: set[frozenset[str]], gold: set[frozenset[str]]) -> float:
        return len(cands & gold) / len(gold) if gold else 0.0

    return {
        "metric_revision": METRIC_REVISION,
        "benchmark": benchmark,
        "model": model_name,
        "model_revision": model_revision,
        "k": k,
        "paper": _recall(directional_pairs, raw_gold),
        "paper_direction_closure_gold": _recall(directional_pairs, closure_gold),
        "langres_score_blocking": _recall(symmetric_pairs, closure_gold),
        "n_candidates_directional": len(directional_pairs),
        "n_candidates_symmetric": len(symmetric_pairs),
        "gold_raw": len(raw_gold),
        "gold_closure": len(closure_gold),
    }


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

#: Maps langres benchmark names to the names the papers use.
PAPER_NAMES = {
    "fodors_zagat": {"uniblocker": "fodors-zagats_homo"},
    "abt_buy": {"uniblocker": "abt-buy_homo", "deepblocker": "Abt-Buy"},
    "amazon_google": {"uniblocker": "amazon-google", "deepblocker": "Amazon-Google1"},
    "dblp_acm": {"uniblocker": "dblp-acm", "deepblocker": "DBLP-ACM1"},
    "dblp_scholar": {"uniblocker": "dblp-scholar", "deepblocker": "DBLP-Google1"},
    "walmart_amazon": {"uniblocker": "walmart-amazon_homo", "deepblocker": "Walmart-Amazon1"},
    "wdc_computers": {},
    "febrl_person": {},
}


PREAMBLE = """
<!-- Generated by examples/research/external_reproduction.py. Prose and tables both. -->
<!-- Re-render with `--render-only`; that measures nothing. -->

# Reproducing published blocking numbers

Agenda item **E1** (`20260717_research_agenda.md`) asks whether a published ER result
survives contact with our harness, and says the point is to test *our instrument*:
*"When a replication misses, the prior should not be 'the paper is wrong' — it is
usually us."* This document is E1 applied to **blocking**, which the agenda's own
target table does not cover (Ditto, AnyMatch and Jellyfish are all matching targets).

## What was and was not reproduced before this

langres had already replicated a *matching* result — Peeters, Steiner & Bizer
(`src/langres/data/peeters.py`) — but that replication **replays their archived model
answers**. It proves our serialization and parsing agree with theirs; it does not
re-derive a published metric from our own models end to end. No **blocking** number
from any paper had ever been re-measured here. That is what this run does.

By the agenda's own evidence ladder — *"archived predictions >> archived weights >> a
table in a PDF"* — this is the **weakest tier**. There is no artifact to diff against:
only printed numbers. What that buys, and what it does not, is stated in the verdict.

## The protocol, and why it is not `score_blocking`

DeepBlocker and UniBlocker use the same candidate-set construction, and we verified
that against **their own arithmetic** rather than trusting the prose (see
`_arithmetic_check` in the reference JSON):

    C = union over a in A of topK(a -> B)        so |C| = K * |A|
    PC = |C & G| / |G|                            (pair completeness, aka recall)
    PQ = |C & G| / |C|                            (pair quality, aka precision)

`langres.optimize.score_blocking` differs on two axes, and both change the number:

1. **Direction.** It builds a *symmetric* kNN over the pooled corpus and then drops
   same-source pairs. The papers index table B and query with table A only.
2. **Gold set.** It scores against the **transitive closure** of the gold clusters
   (`Benchmark.load()`'s third return). The papers score against the raw positive
   list. On `dblp_scholar` that is 13,763 pairs versus 5,347 — a 2.6x difference in
   the denominator alone. Section F attributes the gap to each axis separately.

So these numbers are **not** `score_blocking` output and must not be compared to the
embedder ladder's. Everything else is langres's: the corpora and loaders from
`langres.data`, the text from `concat_comparable_fields`, the embeddings from
`SentenceTransformerEmbedder`, the exact cosine top-k from `FAISSIndex`
(`IndexFlatIP` over L2-normalized vectors — exact, not ANN, so no approximation loss
is folded into the recall).

## Nothing here is tuned

No threshold, no `k`, and no serialization was chosen after looking at a result. The
serialization is whatever `concat_comparable_fields` already produces for each shipped
schema (Section E prints it). A number that matched because it was tuned until it
matched would be worthless for the question being asked.

## Models

`all-mpnet-base-v2` is the checkpoint UniBlocker's **STransformer** baseline is built
on, so it is the like-for-like comparison. `all-MiniLM-L6-v2` is langres's own default
embedder, which answers the separate question of whether what ships out of the box
clears the same bar.

> **The STransformer identity is an inference, not a quote.** UniBlocker §5.1 says
> *"We chose to use the all-mpnet-base-v2 from STransformer [49] as the initial
> pre-training weight"* and §5.2 says *"The performance of STransformer, which served
> as our initialized model for pre-training, is detailed in Table 3."* Together those
> imply STransformer = `all-mpnet-base-v2` zero-shot. The paper never writes it in one
> sentence. If that inference is wrong, the model-level comparisons below are between
> two different checkpoints and only the protocol-level agreement survives.

## Reading order

Section **C** first. The gold-set sizes decide which comparisons are meaningful: on
three benchmarks our gold set holds *fewer pairs* than the papers report, and **if**
ours nest inside theirs — argued from provenance, verified nowhere, since neither
paper publishes a pair list — then a higher PC on those is *expected* and is not
evidence of anything. Section B2 carries that assumption in full.
"""

VERDICT = """
## G. Verdict

**Two different answers, and they must not be run together.**

1. **Does the harness compute the literature's quantity? Yes, and this part is
   unconditional** — it is arithmetic on candidate sets, which needs no assumption
   about anyone's gold pairs.
2. **Do we reproduce their recall? Mostly, not uniformly, and never
   unconditionally.** Two rows land within 0.7–1.5 pp (`dblp_scholar` +0.72,
   `dblp_acm` +1.48). One is 5.4 pp off in our favour and unexplained
   (`fodors_zagat`). The three product benchmarks are 4–7 pp apart as raw points
   and only *overlap* once the gold-set interval is applied — which is not the same
   as agreeing. **And all of it is conditional on an assumption we cannot verify:**
   that our gold pairs correspond to theirs. Neither paper publishes a pair list.

The second answer is the one a reader wants and the weaker one. Read it as *"our
numbers are where theirs are on two benchmarks, reachable on three more, and
unexplained on one — if the two ground truths are the same ground truth"*, not as a
settled reproduction.

### What is unconditional

None of this depends on whose gold pairs are whose:

- **The candidate-set construction is exactly theirs.** On `dblp_scholar` at
  DeepBlocker's K = 150, |A| = 2,616 and |B| = 64,263 match their Table 4, and our
  candidate set is 150 x 2,616 = **392,400** against their printed **`392.4k`**. That
  is `C = ∪_a topK(a → B)` reproduced to the pair.
- **`PQ = hits / (k * |A|)` is their formula.** Recomputing UniBlocker's own printed
  PC/PQ from it reproduces their PQ to 2 dp on five of six rows, and on
  `fodors-zagats` only if |A| = 533 rather than the 553 their Table 2 prints — i.e.
  the check is sharp enough to find a typo in their table.
- **The `score_blocking` crosscheck** (Section F) is langres-internal on both sides,
  so it needs no paper at all: it reproduces the embedder ladder's published recalls
  *and* candidate counts to every printed digit.

### What is conditional — including the rows that look cleanest

- **`dblp_scholar` vs DeepBlocker Table 6 (DBLP-Google1).** Our raw gold is 5,347
  pairs, *the same count* as their Table 4 `#Matches`, and at their K = 150 we measure
  **98.82%** against their published **98.1%** — 0.72 pp, with an *untrained* embedder
  against their *trained* Autoencoder. **An equal count is not an identical pair set.**
  Recall over our 5,347 pairs bounds nothing about recall over a *different* 5,347
  pairs. What the equal count does buy is narrower: it removes "they scored more pairs
  than we did" as an explanation. It does not establish that we scored the same ones.
- **`dblp_acm` vs UniBlocker Table 3, STransformer column.** Same checkpoint, and at
  their k = 1 we measure **96.76 / PQ 82.11** against their **95.28 / 81.00**. The
  interval here is 4 pairs wide — but that width is itself computed *under* the nesting
  assumption, so a narrow interval is not independent evidence.

**`fodors_zagat` proves the protocol but not the model.** Its gold count is identical
(112 pairs) and our PQ at k=1 reproduces UniBlocker's printed PQ **to the digit**
(20.83 and 21.01). That is arithmetic rather than luck — it pins |A| = 533 and confirms
`PQ = hits / (k * |A|)` — so it belongs in the unconditional list above: it is evidence
about the *harness*. It is not evidence about the model, because our PC there sits
5.4 pp above their STransformer row (see below).

### The like-for-like test: 4 of 6 published values land inside our bound

The sharpest test is Section B2's second table — `all-mpnet-base-v2` against
UniBlocker's STransformer column, same checkpoint, at *their* k. Because our gold set
is smaller than theirs on three benchmarks (Section C), the honest question is
not "do the numbers match" but "is their published value reachable from what we
measured". For the same checkpoint:

| benchmark | k | our bound vs their gold | their PC | inside? |
|---|---:|---|---:|:--:|
| `abt_buy` | 13 | 89.62 – 95.99 | 90.52 | **yes** |
| `amazon_google` | 13 | 84.62 – 94.85 | 90.46 | **yes** |
| `dblp_scholar` | 8 | 90.88 – 92.55 | 91.49 | **yes** |
| `walmart_amazon` | 27 | 80.85 – 97.49 | 90.03 | **yes** |
| `dblp_acm` | 1 | 96.58 – 96.76 | 95.28 | no (+1.3 pp) |
| `fodors_zagat` | 2 | 99.11 (point) | 93.75 | no (+5.4 pp) |

Four are consistent outright. `dblp_acm` misses by 1.3 pp on a bound only 4 pairs wide
— that is a real but small residual, not a category error.

**This whole table is conditional on `G_ours` nesting inside `G_theirs`, which we
argue from provenance and cannot verify** (Section B2's note). Neither paper publishes
its pair list. If that nesting fails, the intervals are not bounds and this count is
not 4 of 6. **No row escapes this, including the ones where the counts match** — an
equal count is not an identical pair set, and a narrow interval is narrow only under
the same assumption that produced it. What survives unconditionally is the
candidate-set arithmetic and the PQ formula check, not any PC comparison.

`fodors_zagat` is the one genuine outlier. Its gold *count* is identical to theirs
(112), so a size difference cannot explain the residual — but the same caveat applies
here as everywhere: an equal count is not an identical pair set, and a different 112
pairs would give a different recall, so "not a denominator effect" is as far as this
goes. It is *their* column that looks odd rather than ours: UniBlocker's own table reports
DeepBlocker at **100.00**,
Sudowoodo at **99.11** and UniBlocker at **100.00**, with STransformer alone at
**93.75**. We measure **100.00** (`all-MiniLM-L6-v2`) and **99.11 / PQ 20.83**
(`all-mpnet-base-v2`) — landing on their *other three* rows rather than their
STransformer one. **We have no verified explanation for why their STransformer
underperforms every other method in their own table here.**

One observation worth recording, explicitly as a **hypothesis we did not test**:
UniBlocker's Table 2 lists `fodors-zagats_homo` as **6 attributes** where our schema
exposes **5** (Section E). The shipped CSV's sixth non-`id` column is `class` — and
`class` is *a cluster id shared by matching records* (`fodors_zagat/SOURCE.md`). langres
excludes it, so nothing here is contaminated. But **any method that serialized that
column would be reading the answer key**, which is one way a table could show three
methods at 99–100% and a fourth, differently-plumbed one at 93.75%. We are not
asserting that happened — we cannot see their preprocessing — only that the attribute
count differs and that the extra column is a label. It is also the reason `fodors_zagat`
should not be used to rank blockers regardless: it is already flagged saturated in
`20260727_portfolio_annotation.md`.

### Do not read the product benchmarks as langres winning

On `abt_buy`, `amazon_google` and `walmart_amazon` the raw PC columns put us well above
the published values, and **that is an artifact, not a result.** The CSVs we ship are
the DeepMatcher labelled splits, and those positives were themselves produced by an
earlier blocking pass — **the pairs that survived are, by construction, pairs some
blocker already found.** The ones it missed are the hard ones, and they are absent from
our denominator. Our PC is therefore biased upward by an amount we can bound but not
measure, which is exactly what B2 does, and against the like-for-like baseline all
three come back *consistent* rather than better.

Against DeepBlocker's *trained* Autoencoder/Hybrid the `abt_buy` bound does not close:
even assuming every one of the 69 pairs we lack was missed, our floor at their K=20 is
**91.25%** against their **87.2%**. That residual is most likely a method difference,
not a harness one — UniBlocker independently measures DeepBlocker's Autoencoder needing
**k=90** to clear 90% PC on `abt-buy_homo`, so their method is weak on this benchmark
by their competitors' measurement too. Attribute sets also differ (ours joins
`name description price`; DeepBlocker's Abt-Buy is 3 attributes, UniBlocker's
`abt-buy_homo` only 2). **We did not run the ablation that would separate those two, so
this stays an unexplained residual rather than a claim.**

### The gap between our published numbers and the literature is a *definition*, not quality

Section F is the practically important table, and it comes with its own validation.
Configured to `score_blocking`'s shape, this script reproduces the **embedder ladder's
own published numbers exactly** for `all-mpnet-base-v2` at k=20 — recall *and*
candidate count, to every printed digit:

| benchmark | ladder recall | ladder candidates | this script, `score_blocking` shape |
|---|---:|---:|---|
| `abt_buy` | 0.9368 | 10,897 | 0.93678 / 10,897 |
| `amazon_google` | 0.8108 | 22,886 | 0.81079 / 22,886 |

(`20260727_embedder_ladder.md`, "Candidate recall at k=20, no instruction".) Two
independent implementations agreeing to four decimals means the delta below is a real
protocol difference and not a bug in either.

And the delta is large. On `amazon_google` at k=20 the *same embeddings* give **95.80%**
under the papers' protocol and **81.08%** under `score_blocking`'s. Section F splits it:
swapping the raw positive list for the transitive closure costs **-15.4 pp**; swapping
directional for symmetric retrieval then *adds* **+0.6 pp**. So essentially the entire
gap is the **gold-set definition**, not the retrieval.

**Be precise about what that ablation does and does not show.** It is entirely
langres-internal: one set of *our* embeddings, scored under two of *our* protocols. It
establishes that **our own** 0.81 and **our own** 95.80 are the same blocker measured
two ways, and that the gold-set definition is what separates them —
`score_blocking`'s is the stricter one, because a transitive closure over gold clusters
manufactures intra-source pairs that a cross-source candidate set structurally cannot
contain. The ladder already documents this as its "reachable ceiling"; what is new is
the measurement of how much of it the closure accounts for.

It does **not** show that the literature's ~0.90 is that same blocker. Reaching their
number would additionally need the checkpoint identity (inferred), the gold-pair
correspondence (unverified) and the serialization to line up — and on
`amazon_google` the serialization demonstrably does not: three fields to
UniBlocker's four (Section E). The honest statement is that the closure explains the
whole gap **between our two protocols**, and is a *sufficient-looking* explanation for
the distance to the literature, not a demonstrated one.

### What could still be wrong

- **This is the weakest reproduction tier.** Printed tables, no archived predictions
  or candidate sets. We can show our number is close to theirs; we cannot show we
  retrieved the *same pairs*. A per-pair agreement check — the thing that localizes a
  harness bug, per E1 — is not possible against a PDF.
- **The STransformer identity is inferred**, not quoted (see the preamble). If it is
  wrong, the `all-mpnet-base-v2` rows compare two different checkpoints.
- **The committed rows were measured with an *unpinned* checkpoint load.** A bare
  `all-mpnet-base-v2` resolves to whatever the Hub's `main` points at on the day you
  run it. Read from this machine's Hub cache afterwards, `refs/main` was
  `e8c3b32edf5434bc2275fc9bab85f82640a19130` for `all-mpnet-base-v2` and
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` for `all-MiniLM-L6-v2`; since `refs/main`
  is by definition what an unpinned load resolves to, those are the weights behind
  these numbers. The harness now pins both explicitly (`MODEL_REVISIONS`) and carries
  the revision in its cache namespace, so a rerun cannot drift onto new weights or be
  served stale vectors under the same name — but that pin is a guarantee for the
  *next* run, not a fact re-derived for this one.
- **We could not reproduce UniBlocker's mAP column** from the formula their §4.2
  prints. Their PC and PQ reconcile exactly with `|C| = K * |A|`; their mAP does not
  follow from `sum_k (PC_k - PC_{k-1}) PQ_k` given their own PC/PQ — on
  `fodors-zagats_homo` that formula yields their PQ (21.01) where they print 60.51. The
  mAP we report is *our reading of their printed formula* and is almost certainly not
  the same quantity. **Compare the PC/PQ/k columns; ignore the mAP column.**
- **One internal inconsistency in their table**, recorded rather than smoothed over:
  UniBlocker's `dblp-scholar` STransformer PQ (23.38) does not follow from their own
  PC and `|C| = 8 x 2616` (which gives 23.77). Ours is self-consistent by construction.
- **No published baseline exists for `wdc_computers` or `febrl_person`** as we ship
  them. SC-Block's WDC-B is a different construction (5,000-2,000,000 records), and our
  FEBRL file is a 500-per-side subset generated by our own script. Their rows are here
  for completeness and have nothing to be compared against. *(Assessment, not a proven
  negative — we searched and found none.)* Worth noting anyway: **`wdc_computers` never
  reaches 90% PC within k=100 on either model** (89.66 MiniLM, 87.02 mpnet; the
  `our k@PC90` column reads `>100` for both because it is only searched to k=100). It *does*
  cross 90% by k=150 — MiniLM ends at 93.00, mpnet at 91.58 — so the committed curves go
  higher than that column alone suggests. Either way it is by a wide margin the hardest
  benchmark in the registry under this protocol, and its records carry a single `title`
  field with nothing else to key on.

### What this does and does not license

It licenses saying: **langres builds the literature's candidate set and computes the
literature's formulas.** That is what the unconditional evidence above shows — the
candidate set is theirs to the pair (392,400 at K=150 on `dblp_scholar`), the PQ
formula reconciles with their printed values to 2 dp, and the internal `score_blocking`
crosscheck is self-consistent digit-for-digit.

**It does not license "so our ground truth must be right".** Be exact about what those
three checks can and cannot see: `|C| = K * |A|` checks candidate-set *cardinality*,
the PQ reconciliation is arithmetic on *their* printed numbers, and the ladder
crosscheck compares two of *our* implementations on *our* data. **A wrong gold split —
a different pair set with the same table sizes and the same match count — passes all
three unchanged.** They exclude a wrong candidate-set convention and a wrong metric.
They cannot exclude a wrong pair set, and nothing in this study can, because no paper
publishes one.

It licenses saying, **with the assumption stated and the spread stated**: on
`dblp_scholar` and `dblp_acm` our recall sits 0.7–1.5 pp from the published value, and
four of six published values for the same checkpoint are reachable from what we
measured. It does not license generalising that 0.7–1.5 pp to the other four —
`fodors_zagat` is 5.4 pp away and the product rows are 4–7 pp away as points, reaching
"consistent" only through the interval. Every one of these statements is conditional on
our gold pairs corresponding to theirs — an assumption argued from provenance and
**verified nowhere**, on any benchmark, including the ones whose counts match.

It does **not** license calling this a settled reproduction of anyone's recall, nor
quoting the `abt_buy` / `amazon_google` / `walmart_amazon` numbers as beating anyone,
nor comparing any number here to the embedder ladder's — different metric, different
candidate set, different gold set (Section F). And it does not settle `fodors_zagat`,
where we cannot explain their STransformer row.

### What would raise the confidence further, cheaply

Per E1, the thing that localizes a harness bug is **per-pair agreement**, and a PDF
cannot give it. The next rung on the evidence ladder is a paper that publishes its
candidate sets or its code with fixed seeds. DeepBlocker's repo (BSD-3-Clause, so
usable) ships code but no data; running their Autoencoder on *our* CSVs would give a
same-data, same-gold, method-to-method comparison — the one thing this study cannot do
from printed tables alone.
"""


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _current_revision_only(rows: Sequence[dict[str, Any]], label: str) -> list[dict[str, Any]]:
    """Drop rows measured under a superseded ``METRIC_REVISION``.

    The replacement key in :func:`_write_row` includes the revision on purpose --
    bumping it must not overwrite the measurements that justified the bump. The
    cost is that the file then holds two generations at once, and rendering both
    into one table would silently mix incompatible measurements. This is the
    other half of that contract: the file keeps everything, the report shows one
    revision, and anything dropped is named in the log rather than vanishing.
    """
    keep = [r for r in rows if r.get("metric_revision") == METRIC_REVISION]
    dropped = [r for r in rows if r.get("metric_revision") != METRIC_REVISION]
    if dropped:
        logger.warning(
            "%s: ignoring %d row(s) from superseded metric revisions %s "
            "(current is %d). Re-run those cells to bring them forward: %s",
            label,
            len(dropped),
            sorted({r.get("metric_revision") for r in dropped}, key=str),
            METRIC_REVISION,
            sorted({(r.get("benchmark"), r.get("model")) for r in dropped}),
        )
    return keep


def _select_generation(rows: Sequence[dict[str, Any]], label: str) -> list[dict[str, Any]]:
    """Pick one checkpoint generation per cell, after filtering the metric revision.

    Rows are additive across checkpoint pins (see :func:`_write_row`), so a cell can
    hold more than one generation and a table must not show both. Preference order
    within a cell: the row measured on the model's current pin, else the single row
    that is there. A cell holding several *non-current* generations is **dropped**
    rather than guessed at -- picking one silently is the failure this exists to
    prevent, and dropping it makes the verdict gate downgrade the render honestly.

    A row with no ``model_revision`` predates the field. It is accepted as the
    fallback rather than discarded: the tracked artifact is exactly that shape.
    """
    keep = _current_revision_only(rows, label)
    by_cell: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in keep:
        by_cell.setdefault((row["benchmark"], row["model"]), []).append(row)

    selected: list[dict[str, Any]] = []
    for (benchmark, model), group in sorted(by_cell.items()):
        if len(group) == 1:
            selected.append(group[0])
            continue
        # Only the CURRENT pin is a positive selection. Falling back to a legacy row
        # here would silently render an arbitrarily old measurement whenever the
        # group holds several non-current generations -- the group already has more
        # than one row at this point, so the "single row present" fallback does not
        # apply and the policy says drop.
        chosen = [r for r in group if r.get("model_revision") == MODEL_REVISIONS.get(model)]
        if len(chosen) != 1:
            logger.warning(
                "%s: %s x %s holds %d generations and none is the current pin -- "
                "dropping the cell rather than picking one: %s",
                label,
                benchmark,
                model,
                len(group),
                sorted(str(r.get("model_revision")) for r in group),
            )
            continue
        logger.info(
            "%s: %s x %s holds %d generations; rendering revision %s",
            label,
            benchmark,
            model,
            len(group),
            chosen[0].get("model_revision"),
        )
        selected.append(chosen[0])
    return selected


def _write_row(path: Path, row: dict[str, Any]) -> None:
    """Append a row, replacing any earlier row for the same cell."""
    path.parent.mkdir(parents=True, exist_ok=True)

    # The checkpoint revision is part of what makes two rows the same experiment --
    # it already is in the cache namespace and the crosscheck admission rule, and a
    # key that omits it would let a re-pin silently destroy the previous
    # checkpoint's measurement. Rows are additive per generation; render picks one.
    def _key(r: dict[str, Any]) -> tuple[Any, ...]:
        return (r["benchmark"], r["model"], r.get("model_revision"), r["metric_revision"])

    key = _key(row)
    kept = [r for r in _load_rows(path) if _key(r) != key]
    kept.append(row)
    kept.sort(key=lambda r: (r["benchmark"], r["model"], str(r.get("model_revision"))))
    path.write_text("\n".join(json.dumps(r, sort_keys=True) for r in kept) + "\n")


def _fmt(value: float | None, digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


#: The crosscheck cells :data:`VERDICT` cites by name. "Some check survived" is not
#: the same as "the checks the prose talks about survived".
VERDICT_CROSSCHECKS = ("abt_buy", "amazon_google")

#: Every figure :data:`VERDICT` states as a measurement of ours, as
#: ``(benchmark, model, k, field, percent)``. Checked against the rendered rows
#: before the prose is emitted, so a re-pin that moves a number downgrades the
#: report instead of letting fixed text assert a value nobody re-read.
_MPNET = "sentence-transformers/all-mpnet-base-v2"
_MINILM = "all-MiniLM-L6-v2"
VERDICT_CLAIMS: tuple[tuple[str, str, int, str, float], ...] = (
    ("dblp_scholar", _MPNET, 150, "pc", 98.82),
    ("dblp_acm", _MPNET, 1, "pc", 96.76),
    ("dblp_acm", _MPNET, 1, "pq", 82.11),
    ("fodors_zagat", _MPNET, 2, "pc", 99.11),
    ("fodors_zagat", _MPNET, 1, "pq", 20.83),
    ("fodors_zagat", _MINILM, 2, "pc", 100.00),
    # The wdc_computers sentence in "What could still be wrong" quotes all four.
    ("wdc_computers", _MINILM, 100, "pc", 89.66),
    ("wdc_computers", _MPNET, 100, "pc", 87.02),
    ("wdc_computers", _MINILM, 150, "pc", 93.00),
    ("wdc_computers", _MPNET, 150, "pc", 91.58),
)

#: Figures :data:`VERDICT` quotes out of the *crosscheck* artifact, as
#: ``(benchmark, k, field, percent)``. The Section F attribution paragraph and the
#: ladder-agreement table both hard-code these, so they need the same gate as the
#: row values -- a re-run of either named crosscheck must not leave fixed prose
#: asserting the old delta.
VERDICT_CROSSCHECK_CLAIMS: tuple[tuple[str, int, str, float], ...] = (
    ("amazon_google", 20, "paper", 95.80),
    ("amazon_google", 20, "paper_direction_closure_gold", 80.43),
    ("amazon_google", 20, "langres_score_blocking", 81.08),
    ("abt_buy", 20, "langres_score_blocking", 93.68),
)


def _claim_holds(
    rows: Sequence[dict[str, Any]], benchmark: str, model: str, k: int, field: str, value: float
) -> bool:
    """Is a figure quoted in :data:`VERDICT` still what the rows say (to 2 dp)?"""
    for row in rows:
        if row["benchmark"] == benchmark and row["model"] == model:
            series = row[field]
            if k > len(series):
                return False
            return bool(abs(series[k - 1] * 100 - value) < 0.005)
    return False


def _searched_to(row: dict[str, Any]) -> str:
    """How far ``k_at_pc90`` actually looked, for a row that never reached 90%.

    ``_k_at`` searches to ``min(UNIBLOCKER_K_CAP, k_eff)``, so a row measured with
    a smaller ``--k-max`` was never asked about k=100. Printing a flat ``>100``
    would claim the threshold was missed over a range that was never searched.
    """
    return f">{min(UNIBLOCKER_K_CAP, int(row['k_max']))}"


def render(rows: Sequence[dict[str, Any]], reference: dict[str, Any]) -> str:
    """Render the comparison tables. Measures nothing."""
    uni = reference["papers"]["uniblocker"]["table_3"]
    uni_pos = reference["papers"]["uniblocker"]["table_2_datasets"]
    deep_rows = {
        r["dataset"]: r for r in reference["papers"]["deepblocker"]["table_6_best_dl"]["rows"]
    }
    deep_sets = reference["papers"]["deepblocker"]["table_4_datasets"]

    out: list[str] = []
    add = out.append

    add(PREAMBLE.strip())
    add("")
    add("## A. UniBlocker protocol: smallest k reaching PC >= 90%")
    add("")
    add(
        "Protocol: `C = union over a in A of topK(a -> B)`, k capped at 100, "
        "`PC = |C and G| / |G|`, `PQ = |C and G| / |C|`, "
        "`mAP = sum_k (PC_k - PC_{k-1}) PQ_k`. "
        "`G` is the raw positive list, not the transitive closure."
    )
    add("")
    add(
        "| benchmark | model | our gold | paper gold | our k@PC90 | their k | "
        "our PC @ their k | their PC | our PQ @ their k | their PQ |"
    )
    add("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        name = PAPER_NAMES.get(row["benchmark"], {}).get("uniblocker")
        ref = uni["STransformer"].get(name) if name else None
        pos = uni_pos.get(name, {}).get("pos") if name else None
        their_k = int(ref["k"]) if ref else None
        ours_pc = row["pc"][their_k - 1] * 100 if their_k and their_k <= row["k_max"] else None
        ours_pq = row["pq"][their_k - 1] * 100 if their_k and their_k <= row["k_max"] else None
        add(
            f"| `{row['benchmark']}` | `{row['model']}` | {row['gold_raw']:,} | "
            f"{pos if pos else '-'} | "
            f"{row['k_at_pc90'] if row['k_at_pc90'] else _searched_to(row)} | "
            f"{their_k if their_k else '-'} | "
            f"{_fmt(ours_pc)} | {_fmt(ref['pc']) if ref else '-'} | "
            f"{_fmt(ours_pq)} | {_fmt(ref['pq']) if ref else '-'} |"
        )
    add("")
    add(
        "`k@PC90` is the *smallest* k at which each side clears 90%, so the two k columns "
        "are not comparable to each other; the `@ their k` columns are the apples-to-apples "
        "comparison. The mAP columns are deliberately omitted -- see the verdict."
    )
    add("")
    add("## B. DeepBlocker Table 6: pair completeness at their published k")
    add("")
    add(
        "Their `DL` column is a per-dataset **trained** Autoencoder/Hybrid over fastText, "
        "not a zero-shot sentence encoder, so a gap here is first of all a method gap."
    )
    add("")
    add("| benchmark | model | their k | their cands | our cands | their recall | our PC |")
    add("|---|---|---:|---:|---:|---:|---:|")
    for row in rows:
        name = PAPER_NAMES.get(row["benchmark"], {}).get("deepblocker")
        ref = deep_rows.get(name) if name else None
        if ref is None:
            continue
        k = int(ref["k"])
        ours = row["pc"][k - 1] * 100 if k <= row["k_max"] else None
        add(
            f"| `{row['benchmark']}` | `{row['model']}` | {k} | {ref['cand']} | "
            f"{k * row['n_a']:,} | {_fmt(ref['recall_pct'], 1)} | {_fmt(ours)} |"
        )
    add("")
    add("### B2. What our PC would be against *their* gold set (a *conditional* interval)")
    add("")
    add(
        "We cannot score the pairs we do not have, but we can bound the number: the "
        "missing `G_theirs - G_ours` pairs are either all missed (lower limit) or all "
        "retrieved (upper limit). Whenever the paper's value falls inside that "
        "interval, the measurement is *consistent* with theirs and the gold set is a "
        "sufficient explanation for the visible difference."
    )
    add("")
    add(
        "> **This is conditional, not a mathematical bound, and the condition is not "
        "verified.** The arithmetic below is only a bound if `G_ours` is literally a "
        "*subset* of `G_theirs`. What we actually have is two aggregate match counts "
        "and a provenance argument -- the shipped CSVs are the DeepMatcher labelled "
        "splits, whose positives were drawn from the same original benchmarks after a "
        "blocking pass. **Equal or smaller counts do not prove subset membership.** "
        "Neither paper publishes its pair list, so nothing here can settle it; if "
        "either used different preprocessing or a different release, `hits / G_theirs` "
        "is not a floor and the missing-pair term is not a ceiling, and the `inside?` "
        'column and the 4-of-6 headline could both move. Read the interval as *"if the '
        "gold sets nest as their provenance says, then...\"*. Obtaining either paper's "
        "candidate or gold pair list would convert it into a real bound; that is the "
        "same missing artifact called out under *What could still be wrong*."
    )
    add("")
    add(
        "> **`inside? = no` is not a failure, and the column is uninformative when the "
        "gold sets nearly match.** The bound's width is exactly "
        "`(G_theirs - G_ours) / G_theirs`, so where the two gold sets are the same size "
        "(`fodors_zagat`, `dblp_scholar` vs DeepBlocker) the bound collapses to a point "
        "and *any* difference at all, however small, reads `no`. On `dblp_acm` it is 4 "
        'pairs wide. Read the column as "can the gold-set difference alone account for '
        'this?" -- and where it says `no`, read the size of the residual, not the word.'
    )
    add("")

    def _bound_table(
        heading: str,
        note: str,
        targets: list[tuple[dict[str, Any], int, int, float]],
    ) -> None:
        """Emit one bound table. `targets` is (row, k, their_gold, their_pc_frac)."""
        add(heading)
        add("")
        add(note)
        add("")
        add(
            "| benchmark | model | k | our PC (our gold) | their gold | lower | upper "
            "| paper | inside? |"
        )
        add("|---|---|---:|---:|---:|---:|---:|---:|:--:|")
        for row, k, their_gold, paper in targets:
            hits = row["pc"][k - 1] * row["gold_raw"]
            lower = hits / their_gold
            upper = (hits + (their_gold - row["gold_raw"])) / their_gold
            inside = "yes" if lower <= paper <= upper else "no"
            add(
                f"| `{row['benchmark']}` | `{row['model']}` | {k} | "
                f"{_fmt(row['pc'][k - 1] * 100)} | {their_gold:,} | {_fmt(lower * 100)} | "
                f"{_fmt(upper * 100)} | {_fmt(paper * 100)} | {inside} |"
            )
        add("")

    deep_targets: list[tuple[dict[str, Any], int, int, float]] = []
    uni_targets: list[tuple[dict[str, Any], int, int, float]] = []
    for row in rows:
        names = PAPER_NAMES.get(row["benchmark"], {})
        dref = deep_rows.get(names.get("deepblocker", ""))
        dgold = deep_sets.get(names.get("deepblocker", ""), {}).get("matches")
        if dref and dgold and int(dref["k"]) <= row["k_max"]:
            deep_targets.append((row, int(dref["k"]), int(dgold), dref["recall_pct"] / 100.0))
        uref = uni["STransformer"].get(names.get("uniblocker", ""))
        ugold = uni_pos.get(names.get("uniblocker", ""), {}).get("pos")
        if uref and ugold and int(uref["k"]) <= row["k_max"]:
            uni_targets.append((row, int(uref["k"]), int(ugold), uref["pc"] / 100.0))

    _bound_table(
        "**vs DeepBlocker's trained Autoencoder/Hybrid, at their K:**",
        "A different *method* as well as a different gold set, so `no` here can mean either.",
        deep_targets,
    )
    _bound_table(
        "**vs UniBlocker's STransformer, at their k — the like-for-like test:**",
        "Same checkpoint we ran (subject to the identity inference in the preamble), so "
        "a `no` here would point at our harness rather than at a method difference. "
        "**This is the row that matters for the question 'is our instrument right'.**",
        uni_targets,
    )
    add("")
    add("## C. Gold-set sizes: ours vs theirs")
    add("")
    add(
        "`raw` is the pooled `label == 1` list that this script scores against; "
        "`closure` is the transitive closure that `Benchmark.load()` returns and "
        "`score_blocking` scores against."
    )
    add("")
    add(
        "| benchmark | table A | table B | our raw gold | our closure gold | "
        "DeepBlocker #Matches | UniBlocker #Pos |"
    )
    add("|---|---:|---:|---:|---:|---:|---:|")
    seen: set[str] = set()
    for row in rows:
        if row["benchmark"] in seen:
            continue
        seen.add(row["benchmark"])
        names = PAPER_NAMES.get(row["benchmark"], {})
        db = deep_sets.get(names.get("deepblocker", ""), {}).get("matches")
        ub = uni_pos.get(names.get("uniblocker", ""), {}).get("pos")
        add(
            f"| `{row['benchmark']}` | {row['n_a']:,} | {row['n_b']:,} | {row['gold_raw']:,} | "
            f"{row['gold_closure']:,} | {db if db else '-'} | {ub if ub else '-'} |"
        )
    add("")
    add("## D. Full PC curves (PC % at each k)")
    add("")
    ks = [1, 2, 3, 5, 8, 10, 13, 20, 27, 50, 77, 90, 100, 150]
    add("| benchmark | model | " + " | ".join(f"k={k}" for k in ks) + " |")
    add("|---|---|" + "---:|" * len(ks))
    for row in rows:
        cells = [_fmt(row["pc"][k - 1] * 100, 1) if k <= row["k_max"] else "-" for k in ks]
        add(f"| `{row['benchmark']}` | `{row['model']}` | " + " | ".join(cells) + " |")
    add("")
    add("## E. Serialization actually used, and whether it matches theirs")
    add("")
    add(
        "Both papers concatenate attribute *values* (DeepBlocker §3.1; UniBlocker §5.1), "
        "which is what `concat_comparable_fields` does — every non-`id` string field of "
        "the schema, space-joined, empties skipped, no case folding. What can still "
        "differ is *which attributes the shipped CSV has at all*, so the field count is "
        "compared against each paper's `#Attr` below. **Where those disagree we are "
        "serializing a different record than they did**, and the comparison for that "
        "benchmark is correspondingly weaker."
    )
    add("")
    add(
        "| benchmark | fields joined | ours | DeepBlocker #Attr | UniBlocker #Attr | "
        "first A record (truncated) |"
    )
    add("|---|---|---:|---:|---:|---|")
    seen.clear()
    for row in rows:
        if row["benchmark"] in seen:
            continue
        seen.add(row["benchmark"])
        names = PAPER_NAMES.get(row["benchmark"], {})
        fields = ", ".join(f"`{f}`" for f in row["serialization_fields"])
        db_attr = deep_sets.get(names.get("deepblocker", ""), {}).get("attrs")
        ub_attr = uni_pos.get(names.get("uniblocker", ""), {}).get("attrs")
        sample = row["serialization_sample_a"].replace("|", "\\|")[:100]
        add(
            f"| `{row['benchmark']}` | {fields} | {len(row['serialization_fields'])} | "
            f"{db_attr if db_attr else '-'} | {ub_attr if ub_attr else '-'} | {sample} |"
        )
    add("")
    add(
        "The two benchmarks the verdict rests on, `dblp_acm` and `dblp_scholar`, are "
        "**4 attributes on all three sides** — so on exactly the benchmarks where the "
        "gold sets also line up, the serialized record is the same shape too. The "
        "product benchmarks are where they diverge: our `amazon_google` has 3 fields "
        "against their 4 (the DeepMatcher release carries no `description`), our "
        "`abt_buy` 3 against UniBlocker's 2, our `walmart_amazon` 5 against "
        "DeepBlocker's 6."
    )
    add("")

    crosscheck_path = CROSSCHECK_PATH
    checks: list[dict[str, Any]] = []
    if crosscheck_path.exists():
        # This artifact is a SEPARATE run from the rows above, so it can describe a
        # different experiment: a stale metric revision, or a (benchmark, model)
        # cell that is not even in the report. Section F would then silently
        # attribute a gap measured elsewhere. Admit only entries whose revision is
        # current AND whose cell is actually rendered here.
        # The checkpoint revision is part of the identity, not decoration: Section F
        # claims "one set of embeddings", which is false if the rows were remeasured
        # on a new pin and the crosscheck was not. Both artifacts predating the field
        # read None on each side and still match, which is correct -- they were one run.
        rendered_cells = {(r["benchmark"], r["model"], r.get("model_revision")) for r in rows}

        def _cell(entry: dict[str, Any]) -> tuple[Any, Any, Any]:
            return (entry["benchmark"], entry["model"], entry.get("model_revision"))

        raw_checks = _current_revision_only(
            json.loads(crosscheck_path.read_text()), str(crosscheck_path)
        )
        checks = [c for c in raw_checks if _cell(c) in rendered_cells]
        orphans = [c for c in raw_checks if _cell(c) not in rendered_cells]
        if orphans:
            logger.warning(
                "crosscheck: ignoring %d entr(ies) whose (benchmark, model, revision) "
                "is not among the rendered rows: %s",
                len(orphans),
                sorted(str(_cell(c)) for c in orphans),
            )
    if checks:
        add("## F. Where the published protocol and `score_blocking` diverge")
        add("")
        add(
            "One set of embeddings, three recalls, one thing changed at a time. "
            "`paper` is directional `A -> B` scored against the raw positive list. "
            "`+closure gold` keeps that candidate set and swaps in the transitive "
            "closure. `score_blocking` additionally swaps directional retrieval for "
            "the symmetric pooled kNN with same-source pairs dropped. The last column "
            "is the whole difference between what a paper would print and what "
            "`score_blocking` prints for the same model at the same k."
        )
        add("")
        if any(c.get("model_revision") is None for c in checks):
            add(
                '> **On "one set of embeddings", for the committed artifacts.** This '
                "crosscheck is a *separate invocation* from the sweep, and both predate "
                "the revision pin, so neither records which checkpoint it loaded. What "
                "makes them the same weights is not the code admitting `None == None` — "
                "it is that the machine that produced both holds exactly **one** cached "
                "snapshot of `all-mpnet-base-v2` "
                "(`e8c3b32edf5434bc2275fc9bab85f82640a19130`), which is also what its "
                "`refs/main` points at, so there was no other weight set available to "
                "either run. That is evidence about this machine, not a guarantee from "
                "the artifact. Every run from here on records its revision and the "
                "admission rule compares it, so this note applies only to the artifacts "
                "committed with this study."
            )
            add("")
        add(
            "| benchmark | model | k | paper protocol | + closure gold | "
            "`score_blocking` shape | total delta |"
        )
        add("|---|---|---:|---:|---:|---:|---:|")
        for c in checks:
            delta = (c["langres_score_blocking"] - c["paper"]) * 100
            add(
                f"| `{c['benchmark']}` | `{c['model']}` | {c['k']} | "
                f"{_fmt(c['paper'] * 100)} | {_fmt(c['paper_direction_closure_gold'] * 100)} | "
                f"{_fmt(c['langres_score_blocking'] * 100)} | {delta:+.2f} |"
            )
        add("")

    # VERDICT is fixed prose written against the FULL canonical sweep -- it names
    # specific benchmarks, specific numbers and "4 of 6". Appending it to a partial
    # render would make the report claim results it does not contain (rendering only
    # the febrl_person row would still assert the DBLP 98.82%). Emit it only when the
    # rows actually CONTAIN those measurements, and say why when they do not.
    #
    # Cell presence alone is not enough: re-measuring one cell with `--k-max 20`
    # leaves every key in place while destroying the k=150 number the prose quotes.
    # So a cell counts only if it was searched to the full canonical depth, the
    # SPECIFIC crosschecks the prose cites survived admission, and -- because a new
    # checkpoint pin can move the measurements without moving any key -- the actual
    # figures VERDICT quotes still hold. Prose that asserts numbers it did not read
    # is the same decoupled-gate shape this file keeps guarding against.
    present = {
        (r["benchmark"], r["model"]) for r in rows if int(r["k_max"]) >= min(K_MAX, int(r["n_b"]))
    }
    missing = {(b, m) for b in BENCHMARKS for m in DEFAULT_MODELS} - present
    missing |= {
        ("<crosscheck>", f"{b} k=20")
        for b in VERDICT_CROSSCHECKS
        if not any(c["benchmark"] == b and c["k"] == 20 for c in checks)
    }
    missing |= {
        ("<quoted value>", f"{b} x {m} @k={k}: {value:.2f}")
        for b, m, k, field, value in VERDICT_CLAIMS
        if not _claim_holds(rows, b, m, k, field, value)
    }
    missing |= {
        ("<quoted crosscheck>", f"{b} k={k} {field}: {value:.2f}")
        for b, k, field, value in VERDICT_CROSSCHECK_CLAIMS
        if not any(
            c["benchmark"] == b and c["k"] == k and abs(c[field] * 100 - value) < 0.005
            for c in checks
        )
    }
    if missing:
        logger.warning(
            "incomplete render (%d canonical input(s) absent or measured at a reduced "
            "k) -- omitting the verdict, which is fixed prose about the full sweep: %s",
            len(missing),
            sorted(missing),
        )
        add("## G. Verdict")
        add("")
        add(
            "*Omitted: this is a **partial** render. The verdict is fixed prose written "
            "against the full canonical sweep — every benchmark x both default models, "
            "each searched to the full k, plus the Section F crosscheck — and would "
            "assert results these tables do not contain. "
            f"{len(missing)} required input(s) are absent or were measured at a reduced "
            "k. Re-render from the full "
            "`20260728_external_reproduction_rows.jsonl` to get it.*"
        )
        add("")
    elif VERDICT.strip():
        add(VERDICT.strip())
        add("")

    add("## References")
    add("")
    add(
        "Every paper below was downloaded as a PDF and converted locally with "
        "`pdftotext -layout`; every number quoted in this document was read off a "
        "table in that text and is tagged with the table it came from in "
        "`20260728_external_reproduction_reference.json`. No number here comes from a "
        "blog post, an abstract, another paper's description of a result, or recall."
    )
    add("")
    for key in ("deepblocker", "uniblocker", "scblock", "sudowoodo"):
        paper = reference["papers"][key]
        add(
            f"- **{paper['title']}** — {paper['authors']}, {paper['venue']}. "
            f"[PDF]({paper['pdf']}) *(retrieved {paper['retrieved']})*"
        )
    add(
        "- **Pre-trained Embeddings for Entity Resolution: An Experimental Analysis** — "
        "Zeakis, Papadakis, Skoutas, Koubarakis, PVLDB 16(9), 2023. "
        "[arXiv:2304.12329](https://arxiv.org/abs/2304.12329) *(retrieved 2026-07-28)*. "
        "This is UniBlocker's citation [63] for STransformer being \"the best blocking "
        'solution with pre-trained embeddings", and it benchmarks sentence-transformer '
        "models for blocking on these same datasets. **We quote no number from it**: "
        "its blocking recall is published only as plots (Figure 3), and reading values "
        "off a figure is not a citation. Its protocol also differs — it queries with "
        "*the smaller* of the two tables, which on `dblp_acm` is ACM (2,294), not the "
        "DBLP side the other two papers query with."
    )
    add("")
    add(
        "**Retrieved but yielding no usable number:** DeepBlocker's technical report "
        "(the `[1]` Dropbox link in the paper's references) downloads and contains the "
        "same Tables 4-6 as the PVLDB version — its per-solution R-C curves, which are "
        "where an SBERT number would live, are figures. So DeepBlocker publishes **no "
        "numeric off-the-shelf sentence-encoder baseline** anywhere we could find."
    )
    add("")
    return "\n".join(out)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmarks", nargs="*", default=list(BENCHMARKS))
    parser.add_argument("--models", nargs="*", default=list(DEFAULT_MODELS))
    parser.add_argument("--k-max", type=int, default=K_MAX)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--rows", type=Path, default=ROWS_PATH)
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="Re-render the tables from existing rows, measuring nothing.",
    )
    parser.add_argument(
        "--crosscheck",
        type=int,
        default=None,
        metavar="K",
        help=(
            "Instead of the sweep, attribute the published-protocol vs "
            "score_blocking gap at this k and write the crosscheck JSON."
        ),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.crosscheck is not None:
        results = [
            crosscheck(b, m, args.crosscheck, cache_dir=args.cache_dir)
            for b in args.benchmarks
            for m in args.models
        ]
        # MERGE, never replace. An exploratory `--crosscheck 5 --benchmarks
        # febrl_person` used to truncate the file to that one cell, destroying the
        # committed k=20 evidence Section F and the verdict are written against --
        # a paid-or-not measurement is still a measurement you cannot get back from
        # git once it is overwritten. Cells are keyed on everything that makes two
        # entries the same experiment, so a rerun replaces its own cell and nothing
        # else.
        path = CROSSCHECK_PATH

        def _key(entry: dict[str, Any]) -> tuple[Any, ...]:
            return (
                entry["benchmark"],
                entry["model"],
                entry.get("model_revision"),
                entry["k"],
                entry.get("metric_revision"),
            )

        existing = json.loads(path.read_text()) if path.exists() else []
        fresh = {_key(r) for r in results}
        kept = [e for e in existing if _key(e) not in fresh]
        if len(kept) != len(existing):
            logger.info("crosscheck: replacing %d existing cell(s)", len(existing) - len(kept))
        merged = sorted(kept + results, key=lambda e: tuple(str(x) for x in _key(e)))
        path.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n")
        logger.info("wrote %s (%d cell(s), %d new)", path, len(merged), len(results))
        return 0

    if not args.render_only:
        for benchmark in args.benchmarks:
            if benchmark not in BENCHMARKS:
                raise SystemExit(f"unknown benchmark: {benchmark}")
            for model in args.models:
                logger.info("measuring %s x %s", benchmark, model)
                row = measure(benchmark, model, cache_dir=args.cache_dir, k_max=args.k_max)
                _write_row(args.rows, row)
                logger.info(
                    "%s x %s: k@PC90=%s PC=%.4f mAP=%.4f",
                    benchmark,
                    model,
                    row["k_at_pc90"],
                    row["pc_at_k90"] or 0.0,
                    row["map_at_100"],
                )

    rows = _select_generation(_load_rows(args.rows), str(args.rows))
    if not rows:
        raise SystemExit(
            f"no rows at metric revision {METRIC_REVISION} in {args.rows}; "
            "nothing to render. Run a measuring pass first."
        )
    reference = json.loads(REFERENCE_PATH.read_text())
    tables = render(rows, reference)
    # The canonical report is the artifact this PR's claims are read from. A trial
    # run against `--rows tmp/...` must not silently replace it with partial tables;
    # an alternate rows file gets an alternate report beside it.
    out = REPORT_PATH if Path(args.rows) == ROWS_PATH else Path(args.rows).with_suffix(".md")
    if out != REPORT_PATH:
        logger.warning(
            "non-canonical --rows %s: writing %s instead of the tracked report %s",
            args.rows,
            out,
            REPORT_PATH,
        )
    out.write_text(tables)
    logger.info("wrote %s (%d rows)", out, len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
