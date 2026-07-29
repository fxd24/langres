"""Is there a FIXED per-score-family match threshold that beats the shipped 0.5?

The follow-up PR #250 asked for and deliberately did not run.
``docs/research/20260728_threshold_default.md`` measured that *deriving* the cut
from labels beats ``0.5`` in 45 of 54 cells -- but §5.2 of that study states the
limit plainly: it measured F1 **only** at ``0.5`` and at each dataset's own
derived cut. It never scored a *shared replacement constant* anywhere. So the
question "would a better fixed constant capture the same gain, for the users who
have no labels at all?" is, as of that study, **unmeasured** -- and the derived
cuts' medians are a documented prior, not a measurement of the constant.

This script measures it. For every registered, loadable benchmark, for each $0
scorer and seed, it scores the **held-out** corpus once and then evaluates
pair-F1 at *every* constant on a fixed grid, at the shipped ``0.5``, at the
label-derived (Youden's J on the disjoint train corpus) cut, and at the
**oracle** cut -- the exact held-out argmax, an upper bound no shipped constant
can beat.

Design decisions, each load-bearing:

* **Held-out means a DISJOINT CORPUS.** ``Benchmark.split(seed=...)`` assigns
  whole gold clusters to one side, so no entity and no match pair straddles the
  boundary. The grid is *never* evaluated on the corpus the derived cut was
  fitted on.
* **The grid sweep is free.** Thresholding does not change the scores, so one
  scoring pass yields every constant's F1 exactly. The expensive part (blocking +
  scoring) happens once per cell, not once per candidate constant.
* **The oracle is exact, not the grid argmax.** It sweeps every distinct score,
  so "how much of the achievable gain does a constant capture?" is measured
  against the true ceiling rather than against the best point of an arbitrary
  grid.
* **Leave-one-benchmark-out (LOBO) selection.** A constant chosen on the same
  benchmarks it is then reported on is an in-sample number and would overstate
  what shipping it does for a *new* dataset. So for each held-out benchmark the
  constant is re-selected from the *others* only, and reported on the held-out
  one. The in-sample argmax is printed too, labeled as optimistic.
* **Confidence intervals from a paired cluster bootstrap.** Candidate pair rows
  are *not* independent -- all pairs touching one entity rise and fall together
  -- so resampling pair rows would give intervals that are far too narrow. The
  resampling unit is the **gold cluster** of the held-out corpus; each judged
  pair is assigned to ``min`` of its two endpoints' cluster units, so every pair
  is counted exactly once. The same resample grades both the candidate constant
  and ``0.5`` (paired), and the interval is on their *difference*.

  **Exactly what that ``min`` does and does not buy** (raised in review, and the
  earlier wording here overclaimed it). A **gold** pair has both endpoints in the
  *same* cluster by definition, so ``min`` is a no-op for it: every gold pair --
  the true-positive numerator and the recall denominator -- is attributed to its
  own cluster, and an entity's gold pairs really do move as a block. A **non-gold
  candidate** pair spans two clusters, and attributing it to one endpoint means
  resampling the *other* endpoint's cluster does not move it. So the dependency
  model is exact on the recall side and approximate on the precision side, where
  it under-counts co-movement of false positives and therefore biases intervals
  slightly **too narrow**.

  This is disclosed rather than fixed because no verdict in the study turns on
  it, and the alternative changes the estimand: requiring both clusters to be
  drawn makes a pair's inclusion probability quadratic, which shrinks the
  effective sample and measures a different quantity. Neither verdict is
  marginal. ``sim_cos`` ships on point estimates that are positive on every
  eligible cell (``+0.05`` to ``+0.87``), and a narrower interval cannot turn a
  positive point estimate negative. ``heuristic``'s veto is the direction that
  *could* be sensitive -- but its ``abt_buy`` deltas sit several interval
  half-widths below zero on all three seeds independently, and the veto's effect
  is to keep the incumbent, so an error here fails safe.
* **Never an average across benchmarks in a reported number.** Aggregation
  appears in exactly one place -- the LOBO *selection* criterion, which must
  reduce the other benchmarks to one ordering -- and it is a median over
  benchmarks of a mean over seeds, labeled as a selection statistic, never
  reported as a performance figure.
* **The ship/no-ship rule is pre-registered** (:data:`SHIP_RULE`), evaluated by
  :func:`to_verdict_markdown` from the artifact. A family that fails it does not
  get a constant, and "no constant works for family X" is a result, not a
  failure.

**What this cannot measure.** Only the two $0 score families are swept:
``heuristic`` (rapidfuzz) and ``sim_cos`` (embedding cosine). ``prob_llm`` /
``prob_group_llm`` need paid calls; ``prob_fs`` / ``prob_rf`` are *fitted*
matchers that a user without labels cannot run at all, so an out-of-the-box
default for them is not the same question. Those families are left alone.

**A ``sim_cos`` constant is a cut on a cosine scale, and the scale belongs to
the checkpoint.** Every benchmark loader in this repo pins ``all-MiniLM-L6-v2``
for its blocker, but ``DEFAULT_EMBEDDING_MODEL`` is ``intfloat/e5-base-v2`` --
so a constant selected from the portfolio as-is is a constant *for MiniLM
cosine*, and whether it is also right for the shipped default is a separate
question that must be measured rather than assumed. ``--embedder`` re-runs the
identical protocol on another checkpoint precisely so that question has an
answer::

    ... threshold_constant_sweep.py --methods embedding_cosine \\
        --embedder intfloat/e5-base-v2 --out tmp/e5.json

Everything here is **$0 in spend**. It is *not* dependency-free on a cold cache:
blocking is the benchmark's own ``VectorBlocker``, so a run needs the
``[semantic]`` extra (and ``[trained]``, for ``derive_threshold``).

Run (offline, $0)::

    OMP_NUM_THREADS=1 uv run --env-file .env python \\
        examples/research/threshold_constant_sweep.py

``OMP_NUM_THREADS=1`` is not optional on macOS. faiss and a torch model in one
process **deadlock silently** without it -- no error, no output, no CPU, forever;
``KMP_DUPLICATE_LIB_OK`` suppresses the OpenMP *abort* but not the *deadlock*.
The module sets it via ``os.environ.setdefault`` above before any import that
could pull either in, which is the layer that protects a run however it was
launched; the command line and ``.env`` are belt and braces.

Each benchmark runs in its **own subprocess** (see
:func:`run_benchmark_isolated`) to bound torch's non-releasing MPS allocator, and
every cell is checkpointed, so an interrupted sweep continues with::

    ... threshold_constant_sweep.py --resume

A narrowed run must name its own ``--out``; the write replaces the file
wholesale, so pointing a subset at the canonical artifact would shrink it::

    uv run --env-file .env python examples/research/threshold_constant_sweep.py \\
        --fast --out tmp/fast.json

``--render`` reprints every table from an existing artifact without measuring
anything, so the write-up's tables are generated rather than transcribed::

    uv run --env-file .env python examples/research/threshold_constant_sweep.py \\
        --render examples/research/results/threshold_constant_sweep.json

``print`` is allowed in examples (this is an operator tool).
"""

import os

# Pin OpenMP / FAISS threading BEFORE importing anything that pulls torch/faiss
# (macOS libomp duplicate-load guard -- mirrors threshold_default_study.py).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import logging  # noqa: E402
import statistics  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections.abc import Iterator, Sequence  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Protocol, cast  # noqa: E402

import numpy as np  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from langres.core.method_registry import get_method  # noqa: E402
from langres.core.models import PairwiseJudgement  # noqa: E402
from langres.data.benchmark import Benchmark, gold_pairs_from_clusters  # noqa: E402
from langres.eval import get_benchmark, list_benchmarks  # noqa: E402
from langres.methods import (  # noqa: E402
    ZERO_SPEND_METHODS,
    BlockingBenchmark,
    make_resolver_factory,
)
from langres.metrics.metrics import classify_pairs  # noqa: E402
from langres.training.calibration import derive_threshold  # noqa: E402

logger = logging.getLogger("threshold_constant_sweep")


def progress(message: str) -> None:
    """Emit a timestamped liveness line.

    Not decoration. Every stage below (load, split, block+score, bootstrap) can
    take minutes on the large benchmarks, and the known failure mode on macOS --
    faiss and a torch model in one process without ``OMP_NUM_THREADS=1`` -- is a
    **silent OpenMP deadlock**: no error, no output, no CPU, forever.
    ``KMP_DUPLICATE_LIB_OK`` suppresses the abort but not the deadlock. Without a
    per-stage heartbeat "still working" and "hung" are indistinguishable, and that
    ambiguity has already cost this project multi-hour runs. The timestamps are
    what make a hang diagnosable: a stage line with no successor is the hang, and
    it names the stage.
    """
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


class CheckpointError(RuntimeError):
    """The run cannot persist its checkpoint (distinct from a benchmark failing)."""


#: The candidate constants. 0.01..0.99 at 0.01, so the shipped 0.5 is exactly
#: ``GRID[49]`` and no interpolation is ever needed to read it off.
GRID: tuple[float, ...] = tuple(round(0.01 * i, 2) for i in range(1, 100))

#: The constant the six architectures hard-code today -- the incumbent every
#: delta in this study is measured against.
SHIPPED_THRESHOLD = 0.5
SHIPPED_INDEX = GRID.index(SHIPPED_THRESHOLD)

#: The $0 scorers. ``rapidfuzz`` stands in for the ``heuristic`` family
#: (``FuzzyString``), ``embedding_cosine`` for ``sim_cos``
#: (``architectures/retrieval.py``) -- the two families the hard-coded 0.5s
#: sit on. Both are rankers, so the cut is the whole decision.
DEFAULT_METHODS: tuple[str, ...] = ("rapidfuzz", "embedding_cosine")

#: Split seeds. A conclusion that flips between seeds is not a conclusion.
DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2)

#: Small, in-repo datasets for a quick pass (``--fast``).
FAST_SUBSET: frozenset[str] = frozenset({"tiny_fixture", "fodors_zagat", "dblp_acm"})

#: Cluster-bootstrap resamples per cell.
BOOTSTRAP_RESAMPLES = 1000

#: Percentile interval width.
CI_ALPHA = 0.05

#: A cell whose held-out corpus carries fewer blocked gold pairs than this is
#: **measured and reported but excluded from constant selection**. On
#: ``tiny_fixture`` the held-out corpus is 3 records containing 1 gold pair, so
#: recall is binary and a single pair decides the metric -- PR #250 documented
#: exactly this. The rule is a data-driven floor rather than a name blacklist so
#: it cannot rot open when the portfolio grows.
MIN_SELECTION_GOLD = 10

#: The pre-registered ship/no-ship rule, stated before the numbers were seen and
#: evaluated by :func:`to_verdict_markdown` from the artifact. A family ships a
#: new constant only if ALL THREE hold over the selection-eligible benchmarks:
#:
#: 1. **Stable**: the LOBO-selected constants span <= 0.05. If dropping one
#:    dataset moves the choice further than that, the "constant" is really a
#:    per-dataset tuning in disguise.
#: 2. **Never significantly worse**: no eligible benchmark has a majority of its
#:    seeds with a 95% cluster-bootstrap CI on Delta-F1 lying entirely below 0.
#: 3. **Positive in the middle**: the median across eligible benchmarks of the
#:    per-benchmark mean Delta-F1 is > 0.
SHIP_RULE = (
    "ship iff (1) LOBO constants span <= 0.05, (2) no eligible benchmark is "
    "significantly worse (95% CI entirely < 0) in a majority of its seeds, and "
    "(3) the median per-benchmark mean delta-F1 is > 0"
)
SHIP_MAX_LOBO_SPREAD = 0.05

#: The CHECKPOINT half of the ship rule, pre-registered before any variant
#: artifact existed (commit history is the proof) and evaluated by
#: :func:`to_transfer_markdown`.
#:
#: A ``sim_cos`` cut lives on a cosine scale that belongs to the *encoder*, so a
#: constant selected on one checkpoint has to earn its place on the one the
#: library actually ships. The rule is deliberately about **harm relative to the
#: incumbent**, not about hitting the variant's own optimum: a constant that is
#: off-optimum for e5 but still far better than ``0.5`` is a strict improvement
#: for that user, and refusing it would protect nobody. So the movement of the
#: argmax is reported as *diagnostic context*, never as an automatic veto.
#:
#: Ship across checkpoints iff, on the variant's cells: (1) no benchmark is
#: significantly worse than 0.5 (95% CI entirely < 0) in a majority of its seeds,
#: and (2) the median per-benchmark mean delta-F1 is > 0. Same shape as the
#: dataset rule, applied to the checkpoint axis.
TRANSFER_RULE = (
    "transfers iff, on the variant checkpoint, (1) no benchmark is significantly "
    "worse than 0.5 in a majority of its seeds and (2) the median per-benchmark "
    "mean delta-F1 is > 0"
)

#: The tracked artifact a FULL run refreshes.
CANONICAL_OUT = Path("examples/research/results/threshold_constant_sweep.json")


class _StudyBenchmark(Benchmark[Any], BlockingBenchmark, Protocol):
    """A benchmark satisfying both the harness and the method-registry contracts."""


class _EmbedderOverride:
    """A benchmark whose vector blocker runs a DIFFERENT sentence-transformer.

    Why this exists. A ``sim_cos`` threshold is a cut on a **cosine scale**, and
    the scale belongs to the checkpoint, not to the family tag: two encoders can
    both emit "cosine similarity" and disagree about what 0.9 means. Every
    benchmark loader in this repo pins ``all-MiniLM-L6-v2`` for its blocker,
    while ``DEFAULT_EMBEDDING_MODEL`` is now ``intfloat/e5-base-v2``. So a
    constant selected from the portfolio is a constant *for MiniLM cosine*, and
    whether it is also a constant for ``sim_cos`` is an open question this study
    would otherwise answer by assumption.

    Swapping the checkpoint is a **resource** variant, not a new architecture
    (``.claude/rules/component-design.md``), so it belongs behind a flag on this
    harness rather than in a second script: the split, the oracle, the bootstrap
    and the LOBO selection stay bit-identical, and only the encoder moves. That
    is what makes the two artifacts comparable.
    """

    def __init__(self, bench: _StudyBenchmark, embedder: str) -> None:
        self._bench = bench
        self._embedder = embedder

    def __getattr__(self, name: str) -> Any:
        """Forward everything else (``schema``, ``blocking_k``, ``load``, ``split``)."""
        return getattr(self._bench, name)

    def build_blocker(self, k_neighbors: int) -> Any:
        """The benchmark's own blocker, re-pointed at ``self._embedder``.

        Raises:
            SystemExit: If the blocker exposes no ``vector_index.embedder``.
                Silently measuring the pinned checkpoint while the artifact
                claims another one is the failure mode worth crashing over.
        """
        from langres.core.embeddings import SentenceTransformerEmbedder

        blocker = self._bench.build_blocker(k_neighbors)
        index = getattr(blocker, "vector_index", None)
        if index is None or not hasattr(index, "embedder"):
            raise SystemExit(
                f"--embedder cannot apply to {type(blocker).__name__}: no "
                "vector_index.embedder to re-point. Refusing to measure the "
                "benchmark's pinned checkpoint under another checkpoint's name."
            )
        # Safe to mutate: build_blocker returns a FRESH blocker whose index has
        # not embedded anything yet (VectorBlocker embeds in prepare/stream).
        index.embedder = SentenceTransformerEmbedder(self._embedder)
        return blocker


class CellResult(BaseModel):
    """One (benchmark, method, seed) held-out grid sweep.

    Attributes:
        benchmark: Registry name.
        method: The $0 scorer.
        score_family: ``MethodSpec.score_type`` for ``method`` -- read from the
            registry, never hard-coded here, so the study and the thing it would
            change cannot disagree about which family a scorer belongs to.
        seed: Seed for the benchmark's own cluster-disjoint corpus split.
        n_train_records / n_test_records: The two disjoint corpora.
        n_test_pairs: Distinct blocked candidate pairs on the held-out corpus.
        n_test_gold_blocked: Gold pairs among them (0 makes F1 vacuous).
        n_test_gold_all: All gold pairs in the held-out corpus.
        n_units: Gold clusters in the held-out corpus -- the bootstrap's
            resampling unit count.
        selection_eligible: ``n_test_gold_blocked >= MIN_SELECTION_GOLD``.
        f1_blocked / f1_all_gold: Pair-F1 at each constant in :data:`GRID`.
            ``blocked`` restricts gold to pairs blocking proposed (``fit``'s own
            ``held_out_f1`` convention, and the primary series here -- blocking
            recall is a threshold-independent ceiling). ``all_gold`` charges
            blocking's misses to recall (the end-to-end view).
        ci_lo_blocked / ci_hi_blocked: 95% paired cluster-bootstrap interval on
            ``f1_blocked[i] - f1_blocked[SHIPPED_INDEX]`` at each constant.
        ci_lo_all_gold / ci_hi_all_gold: the same for ``f1_all_gold``.
        oracle_threshold_blocked / oracle_f1_blocked: Exact held-out argmax over
            every distinct score -- the ceiling, unattainable without test labels.
        oracle_threshold_all_gold / oracle_f1_all_gold: the same, end-to-end.
        derived_threshold: Youden's J on the DISJOINT train corpus's labeled
            blocked candidates -- what a user with full labels gets.
        derived_f1_blocked / derived_f1_all_gold: that cut, graded held-out.
        embedder: The sentence-transformer checkpoint the blocker ran, when
            ``--embedder`` overrode the benchmark's own pin; ``None`` means the
            benchmark's pinned model. Recorded per cell because a ``sim_cos``
            constant is a property of a *cosine scale*, and the checkpoint sets
            the scale -- a cell that does not say which checkpoint produced it
            cannot support a shipped constant.
        seconds: Wall clock for the cell.
    """

    benchmark: str
    method: str
    score_family: str
    seed: int
    n_train_records: int
    n_test_records: int
    n_test_pairs: int
    n_test_gold_blocked: int
    n_test_gold_all: int
    n_units: int
    selection_eligible: bool
    f1_blocked: list[float]
    f1_all_gold: list[float]
    ci_lo_blocked: list[float]
    ci_hi_blocked: list[float]
    ci_lo_all_gold: list[float]
    ci_hi_all_gold: list[float]
    oracle_threshold_blocked: float
    oracle_f1_blocked: float
    oracle_threshold_all_gold: float
    oracle_f1_all_gold: float
    derived_threshold: float
    derived_f1_blocked: float
    derived_f1_all_gold: float
    embedder: str | None = None
    seconds: float

    @property
    def shipped_f1_blocked(self) -> float:
        """Pair-F1 at the constant the architectures hard-code today."""
        return self.f1_blocked[SHIPPED_INDEX]

    @property
    def shipped_f1_all_gold(self) -> float:
        """End-to-end pair-F1 at the shipped constant."""
        return self.f1_all_gold[SHIPPED_INDEX]


class SweepReport(BaseModel):
    """The tracked artifact: the grid the cells are indexed by, plus the cells.

    The grid lives at the top level rather than on every cell so a reader cannot
    end up with rows indexed by two different grids in one file.
    """

    grid: list[float]
    shipped_threshold: float
    bootstrap_resamples: int
    #: Identity of the code that produced these cells -- see
    #: :func:`_source_fingerprint`. Optional so artifacts written before it
    #: existed still load; ``None`` means "unknown", which resume treats as a
    #: reason to refuse rather than to assume a match.
    source_fingerprint: str | None = None
    cells: list[CellResult]


def _source_fingerprint() -> str:
    """Identify the code a cell was measured with, for the resume guard.

    ``--resamples`` and ``--embedder`` are validated on resume, but they are only
    the *arguments*. If the harness, a matcher, a benchmark loader or the input
    data changed between an interrupted run and its resume, the old cells stay
    and the remaining ones are measured with different code -- then both are
    pooled into one artifact attributed to the current source, and the ship rule
    can select a constant from measurements that were never comparable. Raised in
    review; this is the identity that makes that detectable.

    Two components, because either alone has a blind spot: the harness file's own
    hash (catches edits to this script, including uncommitted ones, which a git
    SHA would miss) and the repo state (catches changes to `langres` itself,
    which the harness hash would miss).

    The repo half hashes the **content** of the uncommitted diff, not a
    ``+dirty`` boolean. An earlier version used the flag, and review pointed out
    it defeats the purpose: two *different* uncommitted matcher edits at one
    commit both render ``<HEAD>+dirty`` with an unchanged harness hash, so the
    guard would call them the same revision and pool exactly the incomparable
    cells it exists to separate.

    **What it cannot see, stated rather than implied:** untracked files (a
    brand-new loader module) and, outside a git checkout, anything but this
    file -- ``nogit`` compares equal to ``nogit``. It is a tripwire for the
    common case, not a proof of identity.

    Returns:
        A short opaque string. Never parsed -- only compared for equality.
    """
    root = Path(__file__).resolve().parent
    harness = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()[:12]

    def _git(*argv: str) -> str:
        return subprocess.run(
            ["git", *argv], cwd=root, capture_output=True, text=True, check=True
        ).stdout

    try:
        head = _git("rev-parse", "--short", "HEAD").strip()
        diff = _git("diff", "HEAD")
        dirty = f"+{hashlib.sha256(diff.encode()).hexdigest()[:8]}" if diff.strip() else ""
        repo = f"{head}{dirty}"
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover - no git
        repo = "nogit"
    return f"{harness}/{repo}"


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------


def dedupe_scores(judgements: Sequence[PairwiseJudgement]) -> dict[frozenset[str], float]:
    """Collapse judgements to one score per unordered pair, keeping the max.

    ``classify_pairs`` identifies a pair order-independently and predicts it when
    *any* judgement for it clears the cut, so the max is the score that reproduces
    its counts exactly. :func:`_assert_matches_classify_pairs` checks that claim
    rather than trusting it.

    Args:
        judgements: Held-out candidate judgements from a ranker.

    Returns:
        ``{frozenset({left, right}): score}``.

    Raises:
        ValueError: If any judgement carries no score (this study sweeps rankers;
            a decider would make the threshold inert and every number meaningless).
    """
    scores: dict[frozenset[str], float] = {}
    for judgement in judgements:
        if judgement.score is None:
            raise ValueError(
                f"{judgement.left_id}/{judgement.right_id}: score is None. This study "
                "sweeps the match cut, which a decider's `decision` would override "
                "(see core.models.predicted_match), so every F1 below would be a "
                "constant. Refusing to report it as a threshold sweep."
            )
        key = frozenset({judgement.left_id, judgement.right_id})
        previous = scores.get(key)
        if previous is None or judgement.score > previous:
            scores[key] = judgement.score
    return scores


def _unit_index(clusters: list[set[str]]) -> dict[str, int]:
    """Map every held-out record id to its gold cluster's index (the bootstrap unit)."""
    return {record_id: index for index, cluster in enumerate(clusters) for record_id in cluster}


def _suffix_counts(unit: np.ndarray, bin_index: np.ndarray, n_units: int) -> np.ndarray:
    """Per-unit counts of pairs predicted at each grid constant.

    ``bin_index[i]`` is ``#{t in GRID : t <= score_i}``, so pair ``i`` is
    predicted at ``GRID[t]`` exactly when ``t < bin_index[i]``. The result is
    therefore a per-unit suffix sum of the ``bin_index`` histogram.

    Args:
        unit: Bootstrap unit index per pair.
        bin_index: ``np.searchsorted(GRID, score, side="right")`` per pair.
        n_units: Number of bootstrap units.

    Returns:
        ``(n_units, len(GRID))`` float32 counts.
    """
    n_grid = len(GRID)
    flat = np.bincount(unit * (n_grid + 1) + bin_index, minlength=n_units * (n_grid + 1)).reshape(
        n_units, n_grid + 1
    )
    # counts[u, t] = sum over k > t  ->  reverse cumulative sum, dropping k == 0.
    reverse = np.cumsum(flat[:, ::-1], axis=1)[:, ::-1]
    return reverse[:, 1:].astype(np.float32)


def _f1(tp: np.ndarray, predicted: np.ndarray, gold: np.ndarray) -> np.ndarray:
    """Pair-F1 from counts: ``2*TP / (predicted + gold)``, 0 where both are 0.

    Algebraically identical to ``classify_pairs``' precision/recall form (and
    checked against it by :func:`_assert_matches_classify_pairs`), but written on
    the totals so a whole bootstrap replicate is one vectorized expression.
    """
    denominator = predicted + gold
    return np.divide(
        2.0 * tp, denominator, out=np.zeros_like(tp, dtype=np.float64), where=denominator > 0
    )


def _exact_oracle(scores: np.ndarray, is_gold: np.ndarray, n_gold: int) -> tuple[float, float]:
    """The best achievable pair-F1 on this held-out set, and the cut that attains it.

    Sweeps every distinct score rather than a grid, so the "capture" ratios later
    are measured against the true ceiling. Ties are collapsed: a threshold is only
    realizable at a value where the predicted set actually changes.

    Args:
        scores: One score per distinct candidate pair.
        is_gold: Gold flag aligned with ``scores``.
        n_gold: Gold pairs in the denominator (blocked-only or all-gold).

    Returns:
        ``(threshold, f1)``.
    """
    if scores.size == 0 or n_gold == 0:
        return (float("nan"), 0.0)
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    tp = np.cumsum(is_gold[order]).astype(np.float64)
    predicted = np.arange(1, scores.size + 1, dtype=np.float64)
    f1 = _f1(tp, predicted, np.full(scores.size, float(n_gold)))
    # Only positions where the next score differs are realizable cut points.
    realizable = np.empty(scores.size, dtype=bool)
    realizable[-1] = True
    realizable[:-1] = sorted_scores[:-1] != sorted_scores[1:]
    f1 = np.where(realizable, f1, -1.0)
    best = int(np.argmax(f1))
    return (float(sorted_scores[best]), float(f1[best]))


def _bootstrap_delta_ci(
    tp_units: np.ndarray,
    predicted_units: np.ndarray,
    gold_units: np.ndarray,
    *,
    rng: np.random.Generator,
    resamples: int,
) -> tuple[list[float], list[float]]:
    """Paired cluster-bootstrap CI on ``F1(t) - F1(0.5)`` at every grid constant.

    Resamples **gold clusters** with replacement (multinomial multiplicities), not
    pair rows: pairs touching one entity are dependent, and resampling them
    independently would understate the interval badly.

    Args:
        tp_units: ``(n_units, n_grid)`` true positives per unit per constant.
        predicted_units: ``(n_units, n_grid)`` predicted pairs per unit per constant.
        gold_units: ``(n_units,)`` gold pairs per unit (threshold-independent).
        rng: Seeded generator.
        resamples: Replicate count.

    Returns:
        ``(ci_lo, ci_hi)``, each ``n_grid`` long.
    """
    n_units = gold_units.size
    deltas = np.empty((resamples, len(GRID)), dtype=np.float64)
    chunk = 100
    probabilities = np.full(n_units, 1.0 / n_units)
    for start in range(0, resamples, chunk):
        size = min(chunk, resamples - start)
        weights = rng.multinomial(n_units, probabilities, size=size).astype(np.float32)
        tp = weights @ tp_units
        predicted = weights @ predicted_units
        gold = (weights @ gold_units.astype(np.float32))[:, None]
        f1 = _f1(tp.astype(np.float64), predicted.astype(np.float64), gold.astype(np.float64))
        deltas[start : start + size] = f1 - f1[:, SHIPPED_INDEX : SHIPPED_INDEX + 1]
    lo = np.percentile(deltas, 100 * CI_ALPHA / 2, axis=0)
    hi = np.percentile(deltas, 100 * (1 - CI_ALPHA / 2), axis=0)
    return (lo.tolist(), hi.tolist())


def _assert_matches_classify_pairs(
    judgements: list[PairwiseJudgement],
    gold: set[frozenset[str]],
    curve: list[float],
    label: str,
) -> None:
    """Cross-check the vectorized curve against the function ``fit`` itself grades with.

    A gate is only worth having if it can observe the failure it exists to catch,
    so this recomputes three grid points -- including the shipped 0.5 -- through
    ``classify_pairs`` and refuses any disagreement. Three, not all 99, because
    ``classify_pairs`` builds a Python set per call and the large benchmarks
    carry over a million pairs.

    Raises:
        RuntimeError: If any checked point disagrees beyond float noise.
    """
    for index in (9, SHIPPED_INDEX, 89):
        reference = classify_pairs(judgements, gold, GRID[index]).f1
        if abs(reference - curve[index]) > 1e-9:
            raise RuntimeError(
                f"{label}: vectorized F1 at t={GRID[index]} is {curve[index]!r} but "
                f"classify_pairs says {reference!r}. The sweep is not measuring the "
                "same metric the library grades with."
            )


def run_cell(
    *,
    benchmark: str,
    method: str,
    seed: int,
    bench: _StudyBenchmark,
    corpus: list[Any],
    gold_clusters: list[set[str]],
    gold_pairs: set[frozenset[str]],
    resamples: int,
    embedder: str | None = None,
) -> CellResult:
    """Score the held-out corpus once, then evaluate every constant on it.

    Args:
        benchmark: Registry name.
        method: The $0 scorer.
        seed: Seed for ``bench.split`` and for the bootstrap.
        bench: The benchmark adapter (already wrapped if a checkpoint override
            is in force -- this argument records it, it does not apply it).
        corpus: The full record list.
        gold_clusters: The closed-world gold partition.
        gold_pairs: The closed-world gold pair set.
        resamples: Bootstrap replicates.
        embedder: Checkpoint override in force, recorded on the cell.

    Returns:
        The assembled :class:`CellResult`.

    Raises:
        RuntimeError: If the held-out corpus contains no blocked gold pair (every
            F1 would be a vacuous 0.0 a reader would mistake for a measurement).
    """
    started = time.monotonic()
    tag = f"{benchmark}/{method}/seed={seed}"
    train_records, test_records, _train_clusters, test_clusters = bench.split(
        corpus, gold_clusters, seed=seed
    )
    factory = make_resolver_factory(method, bench)

    # 1. The label-derived reference cut, fitted on the DISJOINT train corpus.
    progress(f"  {tag}: block+score TRAIN ({len(train_records):,} records) ...")
    train_judgements = factory(SHIPPED_THRESHOLD).predict(
        [record.model_dump() for record in train_records]
    )
    train_scores = dedupe_scores(train_judgements)
    del train_judgements
    progress(f"  {tag}: derive cut from {len(train_scores):,} labeled train pairs ...")
    derived = derive_threshold(
        list(train_scores.values()),
        [pair in gold_pairs for pair in train_scores],
    )
    del train_scores

    # 2. Score the held-out corpus once.
    progress(f"  {tag}: block+score TEST ({len(test_records):,} records) ...")
    judgements = factory(SHIPPED_THRESHOLD).predict(
        [record.model_dump() for record in test_records]
    )
    pair_scores = dedupe_scores(judgements)
    gold_all = gold_pairs_from_clusters(test_clusters)
    gold_blocked = gold_all & set(pair_scores)
    if not gold_blocked:
        raise RuntimeError(
            f"{benchmark}/{method}/seed={seed}: the held-out corpus has no blocked "
            f"gold pair ({len(gold_all)} gold, {len(pair_scores)} blocked), so every "
            "F1 would be a vacuous 0.0. Refusing to report it as a measurement."
        )

    unit_of = _unit_index(test_clusters)
    pairs = list(pair_scores)
    scores = np.fromiter((pair_scores[p] for p in pairs), dtype=np.float64, count=len(pairs))
    is_gold = np.fromiter((p in gold_all for p in pairs), dtype=bool, count=len(pairs))
    units = np.fromiter(
        (min(unit_of[a], unit_of[b]) for a, b in (tuple(p) for p in pairs)),
        dtype=np.int64,
        count=len(pairs),
    )
    n_units = len(test_clusters)
    bins = np.searchsorted(np.asarray(GRID), scores, side="right")

    # 3. Per-unit sufficient statistics -> every constant's F1, and the CIs.
    tp_units = _suffix_counts(units[is_gold], bins[is_gold], n_units)
    predicted_units = _suffix_counts(units, bins, n_units)
    gold_blocked_units = np.bincount(units[is_gold], minlength=n_units).astype(np.float32)
    unblocked = gold_all - gold_blocked
    unblocked_units = np.bincount(
        np.fromiter(
            (unit_of[next(iter(pair))] for pair in unblocked), dtype=np.int64, count=len(unblocked)
        ),
        minlength=n_units,
    ).astype(np.float32)
    gold_all_units = gold_blocked_units + unblocked_units

    tp_total = tp_units.sum(axis=0, dtype=np.float64)
    predicted_total = predicted_units.sum(axis=0, dtype=np.float64)
    f1_blocked = _f1(tp_total, predicted_total, np.full(len(GRID), float(len(gold_blocked))))
    f1_all_gold = _f1(tp_total, predicted_total, np.full(len(GRID), float(len(gold_all))))
    _assert_matches_classify_pairs(
        judgements, gold_blocked, f1_blocked.tolist(), f"{benchmark}/{method}/seed={seed} blocked"
    )
    _assert_matches_classify_pairs(
        judgements, gold_all, f1_all_gold.tolist(), f"{benchmark}/{method}/seed={seed} all_gold"
    )

    progress(f"  {tag}: cluster bootstrap ({resamples} resamples over {n_units:,} units) ...")
    rng = np.random.default_rng(seed)
    ci_lo_blocked, ci_hi_blocked = _bootstrap_delta_ci(
        tp_units, predicted_units, gold_blocked_units, rng=rng, resamples=resamples
    )
    ci_lo_all, ci_hi_all = _bootstrap_delta_ci(
        tp_units, predicted_units, gold_all_units, rng=rng, resamples=resamples
    )

    oracle_t_blocked, oracle_f1_blocked = _exact_oracle(scores, is_gold, len(gold_blocked))
    oracle_t_all, oracle_f1_all = _exact_oracle(scores, is_gold, len(gold_all))
    derived_blocked = classify_pairs(judgements, gold_blocked, derived).f1
    derived_all = classify_pairs(judgements, gold_all, derived).f1

    return CellResult(
        benchmark=benchmark,
        method=method,
        score_family=get_method(method).score_type,
        seed=seed,
        n_train_records=len(train_records),
        n_test_records=len(test_records),
        n_test_pairs=len(pairs),
        n_test_gold_blocked=len(gold_blocked),
        n_test_gold_all=len(gold_all),
        n_units=n_units,
        selection_eligible=len(gold_blocked) >= MIN_SELECTION_GOLD,
        f1_blocked=f1_blocked.tolist(),
        f1_all_gold=f1_all_gold.tolist(),
        ci_lo_blocked=ci_lo_blocked,
        ci_hi_blocked=ci_hi_blocked,
        ci_lo_all_gold=ci_lo_all,
        ci_hi_all_gold=ci_hi_all,
        oracle_threshold_blocked=oracle_t_blocked,
        oracle_f1_blocked=oracle_f1_blocked,
        oracle_threshold_all_gold=oracle_t_all,
        oracle_f1_all_gold=oracle_f1_all,
        derived_threshold=derived,
        derived_f1_blocked=derived_blocked,
        derived_f1_all_gold=derived_all,
        embedder=embedder,
        seconds=time.monotonic() - started,
    )


def run_benchmark(
    name: str,
    *,
    methods: tuple[str, ...],
    seeds: tuple[int, ...],
    resamples: int,
    embedder: str | None = None,
) -> Iterator[CellResult]:
    """Run every (method, seed) cell for one registered benchmark.

    Yields rather than returns so a cell raising midway does not take the cells
    already measured down with it -- the large benchmarks cost minutes each.

    Args:
        name: Registry name (must be ``loadable``).
        methods: $0 method names.
        seeds: Corpus-split seeds.
        resamples: Bootstrap replicates.
        embedder: Sentence-transformer checkpoint to run instead of the
            benchmark's pinned one (see :class:`_EmbedderOverride`).

    Yields:
        One :class:`CellResult` per (method, seed), as each completes.
    """
    progress(f"  {name}: loading corpus ...")
    bench = cast(_StudyBenchmark, get_benchmark(name))
    if embedder is not None:
        progress(f"  {name}: blocker re-pointed at {embedder}")
        bench = cast(_StudyBenchmark, _EmbedderOverride(bench, embedder))
    corpus, gold_clusters, gold_pairs = bench.load()
    progress(f"  {name}: {len(corpus):,} records, {len(gold_clusters):,} gold clusters")
    for method in methods:
        for seed in seeds:
            result = run_cell(
                benchmark=name,
                method=method,
                seed=seed,
                bench=bench,
                corpus=corpus,
                gold_clusters=gold_clusters,
                gold_pairs=gold_pairs,
                resamples=resamples,
                embedder=embedder,
            )
            best = int(np.argmax(result.f1_blocked))
            print(
                f"       {method} seed={seed}: 0.50 F1={result.shipped_f1_blocked:.4f} | "
                f"best grid t={GRID[best]:.2f} F1={result.f1_blocked[best]:.4f} | "
                f"oracle t={result.oracle_threshold_blocked:.4f} "
                f"F1={result.oracle_f1_blocked:.4f} | derived t={result.derived_threshold:.4f} "
                f"F1={result.derived_f1_blocked:.4f} ({result.seconds:.1f}s)",
                flush=True,
            )
            yield result


# --------------------------------------------------------------------------
# Analysis (pure functions over the artifact -- the write-up's tables)
# --------------------------------------------------------------------------


def _mean_by_benchmark(cells: list[CellResult]) -> dict[str, np.ndarray]:
    """Per benchmark, the seed-mean F1 curve. A *selection* statistic only."""
    by_benchmark: dict[str, list[list[float]]] = {}
    for cell in cells:
        by_benchmark.setdefault(cell.benchmark, []).append(cell.f1_blocked)
    return {name: np.mean(np.asarray(curves), axis=0) for name, curves in by_benchmark.items()}


def _select_constant(curves: dict[str, np.ndarray]) -> float:
    """The constant maximizing the MEDIAN across benchmarks of the seed-mean F1.

    Median, not mean: one benchmark with a wide F1 range would otherwise choose
    the constant for all the others.

    This is a **selection** rule, never a reported performance number. It is not,
    however, the *only* cross-benchmark aggregation in this file -- an earlier
    version of this docstring claimed that and review caught it surviving here
    after the same claim had been corrected in the write-up. :func:`to_verdict_markdown`
    and :func:`to_transfer_markdown` also take a median over per-benchmark means,
    as the quantity their pre-registered rules test. The invariant that actually
    holds is: aggregation appears only in statistics that **decide**, never in one
    that **reports**.
    """
    stacked = np.median(np.asarray(list(curves.values())), axis=0)
    return GRID[int(np.argmax(stacked))]


def lobo_constants(cells: list[CellResult]) -> dict[str, float]:
    """Leave-one-benchmark-out constants: ``{held_out_benchmark: constant}``.

    Only selection-eligible cells vote; the held-out benchmark is excluded from
    its own selection, so the F1 reported at the returned constant is out-of-sample
    with respect to the *dataset*, which is the generalization that matters for a
    shipped default.
    """
    eligible = [c for c in cells if c.selection_eligible]
    curves = _mean_by_benchmark(eligible)
    return {
        held_out: _select_constant({n: c for n, c in curves.items() if n != held_out})
        for held_out in curves
        if len(curves) > 1
    }


def to_cell_markdown(report: SweepReport) -> str:
    """The full per-cell table. Never an average across benchmarks."""
    header = (
        "| benchmark | family | method | seed | test pairs | held-out gold | units | "
        "F1@0.50 | best grid t | F1@best | oracle t | oracle F1 | derived t | derived F1 |"
    )
    lines = [header, "|" + "---|" * 14]
    for cell in report.cells:
        best = int(np.argmax(cell.f1_blocked))
        lines.append(
            f"| {cell.benchmark} | `{cell.score_family}` | {cell.method} | {cell.seed} | "
            f"{cell.n_test_pairs:,} | {cell.n_test_gold_blocked:,} | {cell.n_units:,} | "
            f"{cell.shipped_f1_blocked:.4f} | {GRID[best]:.2f} | {cell.f1_blocked[best]:.4f} | "
            f"{cell.oracle_threshold_blocked:.4f} | {cell.oracle_f1_blocked:.4f} | "
            f"{cell.derived_threshold:.4f} | {cell.derived_f1_blocked:.4f} |"
        )
    # Generated, not assumed: a `sim_cos` cut is a cut on a cosine scale, so which
    # encoder produced the scores is part of what the number means.
    checkpoints = sorted(
        {cell.embedder or "all-MiniLM-L6-v2 (benchmark pin)" for cell in report.cells}
    )
    lines.append("")
    lines.append(f"Blocker checkpoint(s) in this artifact: {', '.join(checkpoints)}.")
    return "\n".join(lines)


def to_lobo_markdown(report: SweepReport) -> str:
    """The headline table: a LOBO-selected fixed constant, graded per benchmark/seed.

    ``capture`` is the share of the oracle's headroom the constant recovers,
    ``(F1@c - F1@0.5) / (oracle - F1@0.5)``, blank where the shipped constant is
    already at the ceiling.
    """
    lines = [
        "| family | benchmark | seed | LOBO t | F1@0.50 | F1@LOBO | Δ | 95% CI | "
        "oracle F1 | capture |",
        "|" + "---|" * 10,
    ]
    for family in sorted({c.score_family for c in report.cells}):
        family_cells = [c for c in report.cells if c.score_family == family]
        constants = lobo_constants(family_cells)
        for cell in family_cells:
            constant = constants.get(cell.benchmark)
            if constant is None:
                continue
            index = GRID.index(constant)
            delta = cell.f1_blocked[index] - cell.shipped_f1_blocked
            headroom = cell.oracle_f1_blocked - cell.shipped_f1_blocked
            capture = f"{delta / headroom:.0%}" if headroom > 1e-9 else "n/a"
            lines.append(
                f"| `{family}` | {cell.benchmark} | {cell.seed} | {constant:.2f} | "
                f"{cell.shipped_f1_blocked:.4f} | {cell.f1_blocked[index]:.4f} | "
                f"{delta:+.4f} | [{cell.ci_lo_blocked[index]:+.4f}, "
                f"{cell.ci_hi_blocked[index]:+.4f}] | {cell.oracle_f1_blocked:.4f} | {capture} |"
            )
    excluded = sorted({c.benchmark for c in report.cells if not c.selection_eligible})
    lines.append("")
    lines.append(
        f"Selection-ineligible and therefore absent above: "
        f"{', '.join(f'`{n}`' for n in excluded) if excluded else 'none'} "
        f"(fewer than {MIN_SELECTION_GOLD} blocked gold pairs held out, so one pair "
        "swings the metric). They are still measured -- see the per-cell table."
    )
    return "\n".join(lines)


def to_all_gold_markdown(report: SweepReport) -> str:
    """The same LOBO comparison under the end-to-end (all-gold) convention.

    Present so the conclusion cannot rest on the choice of denominator: if the two
    conventions disagree about whether a constant helps, that is the finding.
    """
    lines = [
        "| family | benchmark | seed | LOBO t | F1@0.50 | F1@LOBO | Δ | 95% CI |",
        "|" + "---|" * 8,
    ]
    for family in sorted({c.score_family for c in report.cells}):
        family_cells = [c for c in report.cells if c.score_family == family]
        constants = lobo_constants(family_cells)
        for cell in family_cells:
            constant = constants.get(cell.benchmark)
            if constant is None:
                continue
            index = GRID.index(constant)
            delta = cell.f1_all_gold[index] - cell.shipped_f1_all_gold
            lines.append(
                f"| `{family}` | {cell.benchmark} | {cell.seed} | {constant:.2f} | "
                f"{cell.shipped_f1_all_gold:.4f} | {cell.f1_all_gold[index]:.4f} | "
                f"{delta:+.4f} | [{cell.ci_lo_all_gold[index]:+.4f}, "
                f"{cell.ci_hi_all_gold[index]:+.4f}] |"
            )
    return "\n".join(lines)


def to_stability_markdown(report: SweepReport) -> str:
    """Per family: how far the LOBO constant moves when one dataset is dropped.

    This is the honest read on "is there ONE constant?": a wide spread means the
    choice is really per-dataset tuning wearing a constant's clothes.
    """
    lines = [
        "| family | eligible benchmarks | in-sample argmax | LOBO min | LOBO max | spread |",
        "|" + "---|" * 6,
    ]
    for family in sorted({c.score_family for c in report.cells}):
        family_cells = [
            c for c in report.cells if c.score_family == family and c.selection_eligible
        ]
        if not family_cells:
            continue
        in_sample = _select_constant(_mean_by_benchmark(family_cells))
        constants = sorted(
            lobo_constants([c for c in report.cells if c.score_family == family]).values()
        )
        count = len({c.benchmark for c in family_cells})
        # Empty on a narrowed run: leaving one benchmark out of ONE benchmark
        # leaves nothing to select from. Report that rather than indexing [0].
        span = (
            f"{constants[0]:.2f} | {constants[-1]:.2f} | {constants[-1] - constants[0]:.2f}"
            if constants
            else "n/a | n/a | n/a"
        )
        lines.append(f"| `{family}` | {count} | {in_sample:.2f} | {span} |")
    return "\n".join(lines)


def to_ladder_markdown(report: SweepReport) -> str:
    """Per benchmark: shipped constant vs LOBO constant vs labels vs oracle.

    The decision question in one row -- how much of what labels buy you does a
    free constant buy you, and how much is left on the table after both.
    """
    lines = [
        "| family | benchmark | F1@0.50 | F1@LOBO | F1@derived (labels) | oracle F1 |",
        "|" + "---|" * 6,
    ]
    # Counted here rather than in prose. The write-up's interpretation leans on
    # "the constant beats the derived cut on N of M benchmarks", and a
    # hand-typed N is exactly the drift this file's generated tables exist to
    # prevent -- review pointed out the guarantee was being broken by the
    # sentence explaining it.
    scoreboard: dict[str, tuple[int, int]] = {}
    for family in sorted({c.score_family for c in report.cells}):
        family_cells = [c for c in report.cells if c.score_family == family]
        constants = lobo_constants(family_cells)
        constant_wins = 0
        graded = 0
        for benchmark in sorted({c.benchmark for c in family_cells}):
            rows = [c for c in family_cells if c.benchmark == benchmark]
            constant = constants.get(benchmark)
            if constant is None:
                continue
            index = GRID.index(constant)
            at_lobo = statistics.mean(c.f1_blocked[index] for c in rows)
            at_derived = statistics.mean(c.derived_f1_blocked for c in rows)
            graded += 1
            constant_wins += at_lobo > at_derived
            lines.append(
                f"| `{family}` | {benchmark} | "
                f"{statistics.mean(c.shipped_f1_blocked for c in rows):.4f} | "
                f"{at_lobo:.4f} | "
                f"{at_derived:.4f} | "
                f"{statistics.mean(c.oracle_f1_blocked for c in rows):.4f} |"
            )
        scoreboard[family] = (constant_wins, graded)
    lines.append("")
    lines.append(
        "Constant vs. labels, counted from the rows above: "
        + "; ".join(
            f"for `{family}` the LOBO constant scores higher than the derived cut on "
            f"**{wins} of {graded}** benchmarks"
            for family, (wins, graded) in sorted(scoreboard.items())
        )
        + ". Neither approach dominates."
    )
    lines.append("")
    lines.append(
        "Seed-mean per benchmark, never pooled across benchmarks. `F1@derived` is what "
        "a user **with full labels** gets (PR #250's seam); `F1@LOBO` is what a user "
        "with **no labels at all** would get from a shipped constant."
    )
    return "\n".join(lines)


def to_verdict_markdown(report: SweepReport) -> str:
    """Evaluate the pre-registered ship rule, per family, from the artifact."""
    lines = [f"Pre-registered rule: **{SHIP_RULE}**.", ""]
    lines += [
        "| family | LOBO spread | benchmarks significantly worse | median per-benchmark Δ | "
        "verdict | constant |",
        "|" + "---|" * 6,
    ]
    for family in sorted({c.score_family for c in report.cells}):
        family_cells = [c for c in report.cells if c.score_family == family]
        constants = lobo_constants(family_cells)
        eligible = [c for c in family_cells if c.selection_eligible and c.benchmark in constants]
        if not eligible:
            continue
        values = sorted(constants[c.benchmark] for c in eligible)
        spread = values[-1] - values[0]
        worse: list[str] = []
        per_benchmark_delta: list[float] = []
        for benchmark in sorted({c.benchmark for c in eligible}):
            rows = [c for c in eligible if c.benchmark == benchmark]
            index = GRID.index(constants[benchmark])
            deltas = [c.f1_blocked[index] - c.shipped_f1_blocked for c in rows]
            per_benchmark_delta.append(statistics.mean(deltas))
            significant_losses = sum(1 for c in rows if c.ci_hi_blocked[index] < 0)
            if significant_losses * 2 > len(rows):
                worse.append(benchmark)
        median_delta = statistics.median(per_benchmark_delta)
        passes = spread <= SHIP_MAX_LOBO_SPREAD + 1e-9 and not worse and median_delta > 0
        in_sample = _select_constant(_mean_by_benchmark(eligible))
        lines.append(
            f"| `{family}` | {spread:.2f} | {', '.join(worse) if worse else 'none'} | "
            f"{median_delta:+.4f} | {'**SHIP**' if passes else '**DO NOT SHIP**'} | "
            f"{in_sample:.2f} |"
        )
    lines.append("")
    lines.append(
        "`constant` is the in-sample argmax -- the value to ship **only** when the "
        "verdict is SHIP. It is deliberately printed for both outcomes so a DO NOT "
        "SHIP row cannot be read as 'no number was found'; the number exists, it just "
        "does not generalize."
    )
    return "\n".join(lines)


def to_transfer_markdown(baseline: SweepReport, variant: SweepReport) -> str:
    """Does a constant selected on ONE checkpoint still help on ANOTHER?

    ``sim_cos`` is a family tag over cosine scores, but the *scale* belongs to the
    encoder — two models can both emit "cosine similarity" and disagree about
    what 0.9 means. So the constant this study would ship is selected on the
    portfolio's pinned ``all-MiniLM-L6-v2`` while the library's own
    ``DEFAULT_EMBEDDING_MODEL`` is ``intfloat/e5-base-v2``. This grades the
    baseline's constant on the variant's cells, which is the question a user of
    the shipped default is actually asking.

    No re-measurement is needed: the variant artifact's stored interval is
    already the paired cluster-bootstrap CI on ``F1(t) - F1(0.5)``, which is
    exactly the quantity that decides whether shipping the constant helps or
    hurts that user.

    Args:
        baseline: The artifact the constant is selected FROM.
        variant: The artifact it is graded ON (a different checkpoint).

    Returns:
        A markdown table, one row per variant cell, plus a per-family verdict.
    """
    lines = [
        "| family | benchmark | seed | baseline t | F1@0.50 | F1@baseline t | Δ | 95% CI | "
        "variant's own argmax |",
        "|" + "---|" * 9,
    ]
    verdicts: list[str] = []
    for family in sorted({c.score_family for c in variant.cells}):
        base_eligible = [
            c for c in baseline.cells if c.score_family == family and c.selection_eligible
        ]
        var_cells = [c for c in variant.cells if c.score_family == family]
        var_eligible = [c for c in var_cells if c.selection_eligible]
        if not base_eligible or not var_eligible:
            continue
        constant = _select_constant(_mean_by_benchmark(base_eligible))
        variant_argmax = _select_constant(_mean_by_benchmark(var_eligible))
        index = GRID.index(constant)
        # Counted over the SELECTION-ELIGIBLE cells only, because that is the set
        # the rule below judges. Counting all cells here and eligible ones there
        # would put two different denominators in one sentence.
        harmed = sum(1 for c in var_eligible if c.ci_hi_blocked[index] < 0)
        for cell in var_cells:
            delta = cell.f1_blocked[index] - cell.shipped_f1_blocked
            lines.append(
                f"| `{family}` | {cell.benchmark} | {cell.seed} | {constant:.2f} | "
                f"{cell.shipped_f1_blocked:.4f} | {cell.f1_blocked[index]:.4f} | "
                f"{delta:+.4f} | [{cell.ci_lo_blocked[index]:+.4f}, "
                f"{cell.ci_hi_blocked[index]:+.4f}] | {variant_argmax:.2f} |"
            )
        # The pre-registered TRANSFER_RULE, evaluated from the artifact.
        worse: list[str] = []
        per_benchmark_delta: list[float] = []
        for benchmark in sorted({c.benchmark for c in var_eligible}):
            rows = [c for c in var_eligible if c.benchmark == benchmark]
            deltas = [c.f1_blocked[index] - c.shipped_f1_blocked for c in rows]
            per_benchmark_delta.append(statistics.mean(deltas))
            if sum(1 for c in rows if c.ci_hi_blocked[index] < 0) * 2 > len(rows):
                worse.append(benchmark)
        median_delta = statistics.median(per_benchmark_delta)
        transfers = not worse and median_delta > 0
        moved = abs(variant_argmax - constant)
        verdicts.append(
            f"- `{family}`: **{'TRANSFERS' if transfers else 'DOES NOT TRANSFER'}**. "
            f"Median per-benchmark Δ on the variant checkpoint {median_delta:+.4f}; "
            f"significantly worse than 0.5 on "
            f"{', '.join(worse) if worse else 'no benchmark'} "
            f"({harmed} of {len(var_eligible)} eligible cells). "
            f"Diagnostic, not a veto: the "
            f"argmax moves {moved:.2f} across checkpoints "
            f"({constant:.2f} -> {variant_argmax:.2f})."
        )
    checkpoints = sorted({c.embedder or "all-MiniLM-L6-v2 (pin)" for c in variant.cells})
    lines.append("")
    lines.append(f"Variant checkpoint(s): {', '.join(checkpoints)}.")
    lines.extend(["", f"Pre-registered rule: **{TRANSFER_RULE}**.", "", *verdicts])
    return "\n".join(lines)


def to_families_markdown() -> str:
    """Every registered score family, and whether this study can speak to it."""
    from langres.core.method_registry import list_methods

    lines = ["| family | methods | measurable at $0? | why |", "|" + "---|" * 4]
    families: dict[str, list[str]] = {}
    for name in list_methods():
        families.setdefault(get_method(name).score_type, []).append(name)
    reasons = {
        "heuristic": ("yes", "pure string similarity -- no model, no spend"),
        "sim_cos": ("yes", "local sentence-transformer -- free after one download"),
        "prob_llm": ("no", "every score costs a paid completion"),
        "prob_group_llm": ("no", "every score costs a paid completion"),
        # NOT "labels required" for prob_fs -- FellegiSunterMatcher.fit_unlabeled
        # does an unsupervised u-estimate + EM, so a label-free user CAN run it.
        # (Review caught the earlier wording; it was simply wrong about the
        # library.) The real reason it is out of scope is that its scale is
        # re-fitted per dataset, so "one shipped constant" is not the question.
        "prob_fs": ("no", "FITTED per dataset (EM) -- its scale is re-estimated, not shared"),
        "prob_rf": ("no", "a FITTED matcher, labels required -- no label-free path at all"),
    }
    for family in sorted(families):
        answer, why = reasons.get(family, ("?", "unclassified"))
        lines.append(f"| `{family}` | {', '.join(sorted(families[family]))} | {answer} | {why} |")
    return "\n".join(lines)


def print_tables(report: SweepReport) -> None:
    """Print every table the write-up uses, in order. 'Every' is meant literally."""
    print("Score families, and which this study can speak to:")
    print(to_families_markdown())
    print()
    print("Per-cell held-out sweep (blocked-gold convention):")
    print(to_cell_markdown(report))
    print()
    print("Leave-one-benchmark-out constant, graded on the held-out benchmark:")
    print(to_lobo_markdown(report))
    print()
    print("The same, under the end-to-end (all-gold) convention:")
    print(to_all_gold_markdown(report))
    print()
    print("Is the constant stable when a dataset is dropped?")
    print(to_stability_markdown(report))
    print()
    print("The ladder: shipped constant -> free constant -> labels -> oracle:")
    print(to_ladder_markdown(report))
    print()
    print("Verdict against the pre-registered rule:")
    print(to_verdict_markdown(report))


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def select_benchmarks(*, fast: bool, only: list[str] | None) -> list[str]:
    """The loadable benchmark names this run sweeps, name-sorted.

    Raises:
        SystemExit: If ``--only`` names something unregistered or external-only.
    """
    loadable = {entry.name for entry in list_benchmarks() if entry.loadable}
    if only:
        unknown = sorted(set(only) - loadable)
        if unknown:
            raise SystemExit(
                f"not loadable benchmark(s): {', '.join(unknown)}. "
                f"Choose from: {', '.join(sorted(loadable))}."
            )
        # De-duplicated: `--only abt_buy abt_buy` would otherwise measure an
        # expensive benchmark twice, keep one copy of its cells (slice
        # replacement is per benchmark), and still count both toward
        # `expected_cells` -- so the run would refuse to publish at the very end,
        # after paying for the duplicate work (raised in review).
        return sorted(set(only))
    if fast:
        return sorted(loadable & FAST_SUBSET)
    return sorted(loadable)


def write_report(report: SweepReport, path: Path) -> None:
    """Write the artifact atomically (staging file, then ``os.replace``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + ".writing")
    staging.write_text(report.model_dump_json(indent=2) + "\n")
    os.replace(staging, path)


def read_report(path: Path) -> SweepReport:
    """Load an artifact and refuse one whose grid is not the grid this module indexes by.

    Raises:
        RuntimeError: If the stored grid or shipped constant differs from this
            module's, which would silently reindex every curve.
    """
    report = SweepReport.model_validate_json(path.read_text())
    if tuple(report.grid) != GRID or report.shipped_threshold != SHIPPED_THRESHOLD:
        raise RuntimeError(
            f"{path} was measured on a different grid/incumbent than this module defines; "
            "every curve index would shift. Re-run the sweep."
        )
    return report


def _worker_command(name: str, args: argparse.Namespace, out: Path) -> list[str]:
    """The argv that measures ONE benchmark in a fresh interpreter."""
    override = ["--embedder", args.embedder] if args.embedder else []
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--in-process",
        # ALWAYS resume the worker, even on a first attempt. A worker that died
        # mid-benchmark leaves its own `<worker_out>.partial`, and the parent
        # clears only `worker_out` before relaunching -- so without this the
        # fresh worker hits the "an earlier sweep was interrupted" guard and
        # refuses, forever, with no way out but manual file surgery. (Found in
        # review; the guard meant to protect unsaved cells was deadlocking the
        # retry that would have re-measured them.) With no partial present
        # `--resume` is a no-op, and a partial written under different flags now
        # fails the metadata check with a message instead of being pooled.
        "--resume",
        *override,
        # The parent prints the tables once, over the whole matrix. A worker's
        # single-benchmark tables would be noise ten times over.
        "--no-tables",
        "--only",
        name,
        "--methods",
        *args.methods,
        "--seeds",
        *[str(seed) for seed in args.seeds],
        "--resamples",
        str(args.resamples),
        "--out",
        str(out),
    ]


def run_benchmark_isolated(
    name: str, args: argparse.Namespace, scratch: Path, fingerprint: str
) -> list[CellResult]:
    """Measure one benchmark in a SUBPROCESS and return its cells.

    Process isolation is a memory bound, not tidiness. torch's MPS caching
    allocator does not release between encodes, so a long single-process sweep
    grows until the OS kills it -- measured on this machine at **42 GiB of MPS
    allocations for a 0.6B model**, while the process's RSS still read 0.8 GB
    (MPS allocations do not appear in RSS, so ``ps`` says the process is
    innocent; ``sysctl vm.swapusage`` is the signal that isn't lying). Every cell
    here builds a ``VectorBlocker`` (sentence-transformers + FAISS), so the same
    accumulation applies -- smaller per cell, because the portfolio pins
    ``all-MiniLM-L6-v2``, but unbounded over 60 cells all the same.

    One benchmark per process caps the pool at what one benchmark needs and costs
    nothing: the corpus and the embedding cache are on disk, and the interpreter
    start is seconds against minutes of blocking. ``--in-process`` opts out.

    Args:
        name: Registry benchmark name.
        args: Parsed CLI namespace (methods/seeds/resamples are forwarded).
        scratch: The parent's checkpoint path -- the worker's own file is named
            beside it so an abandoned one is visible to ``git status -uall``.

    Returns:
        The worker's cells.

    Raises:
        RuntimeError: If the worker exits non-zero, writes no artifact, or writes
            one whose ``source_fingerprint`` differs from ``fingerprint``. The
            caller downgrades this to a per-benchmark failure; a *killed* worker
            (OOM, SIGKILL) lands here rather than taking the sweep with it.
    """
    worker_out = scratch.with_name(f"{scratch.name}.{name}.json")

    # A finished worker artifact already sitting here means the previous run
    # completed this benchmark and then died before the parent committed its
    # cells -- the window the caller's deferred unlink deliberately leaves open.
    # ADOPT it. The earlier code unlinked unconditionally right here, which threw
    # away the very hour of compute the deferred unlink was protecting (review
    # caught that the two halves of that fix contradicted each other).
    if worker_out.exists():
        adopted = read_report(worker_out)
        # Matching CODE is not matching INVOCATION. An artifact that survived a
        # parent crash may have been measured with different --resamples, a
        # different --embedder, or a different method/seed matrix; adopting it on
        # the fingerprint alone republishes those measurements under this run's
        # header, and if the cell COUNT happens to agree the final length check
        # waves it through (raised in review). Every field the header will claim
        # is checked here.
        wanted = {(name, m, s) for m in args.methods for s in args.seeds}
        found = {(c.benchmark, c.method, c.seed) for c in adopted.cells}
        stale_embedder = sorted({c.embedder for c in adopted.cells if c.embedder != args.embedder})
        reasons = []
        if adopted.source_fingerprint != fingerprint:
            reasons.append(f"source {adopted.source_fingerprint!r} != {fingerprint!r}")
        if adopted.bootstrap_resamples != args.resamples:
            reasons.append(f"--resamples {adopted.bootstrap_resamples} != {args.resamples}")
        if stale_embedder:
            reasons.append(f"embedder {stale_embedder} != {args.embedder!r}")
        if found != wanted:
            reasons.append(f"cells {sorted(found)} != {sorted(wanted)}")
        if not reasons:
            print(f"[adopt] {name}: reusing the completed worker artifact", flush=True)
            return adopted.cells
        # Rejected for THIS invocation is not worthless: those cells are still
        # valid measurements of the invocation that produced them, and in the
        # parent-crash window this file can be their only durable copy. Deleting
        # it here and then failing to re-measure loses an hour of compute for
        # nothing (raised in review). Move it aside instead -- it stays visible to
        # `git status -uall`, so an operator decides rather than a crash deciding.
        # A COLLISION-FREE name. `replace()` onto a fixed `.rejected` silently
        # overwrites an artifact rescued by an earlier invocation -- which may
        # itself be the only durable copy of an hour-long measurement, so the
        # rescue would become the second data loss (raised in review). Find the
        # first free slot instead; nothing here is ever overwritten.
        rejected = worker_out.with_name(worker_out.name + ".rejected")
        serial = 1
        while rejected.exists():
            rejected = worker_out.with_name(f"{worker_out.name}.rejected.{serial}")
            serial += 1
        worker_out.replace(rejected)
        print(
            f"[stale] {name}: worker artifact rejected ({'; '.join(reasons)}) -- "
            f"kept at {rejected.name}, re-measuring",
            flush=True,
        )

    completed = subprocess.run(_worker_command(name, args, worker_out), check=False)
    if completed.returncode != 0 or not worker_out.exists():
        raise RuntimeError(
            f"worker for {name} exited {completed.returncode} "
            f"(artifact {'written' if worker_out.exists() else 'missing'})"
        )
    # Read the report WHOLE, not just `.cells`. Every benchmark relaunches the
    # current on-disk script, so a commit or edit part-way through a multi-hour
    # sweep makes later workers run different code -- and the parent would stamp
    # all of it with the fingerprint it captured at startup, silently pooling
    # revisions under one label.
    report = read_report(worker_out)
    if report.source_fingerprint != fingerprint:
        raise RuntimeError(
            f"worker for {name} ran source {report.source_fingerprint!r}, but the "
            f"parent is {fingerprint!r} -- the script or repo changed mid-sweep. "
            "Refusing to pool cells from two revisions into one artifact."
        )
    # Deliberately NOT unlinked here. Between this read and the parent's
    # checkpoint() there is a window in which the worker's artifact would be the
    # only durable copy of an hour of compute -- and the successful worker has
    # already removed its own .partial. Deleting it here means a parent that dies
    # (or a checkpoint() that raises OSError) in that window loses the benchmark
    # outright. The caller unlinks once the cells are committed; this is the same
    # "commit before it can disappear" rule the repo learned by losing a paid run.
    return report.cells


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true", help="only the small in-repo subset")
    parser.add_argument("--only", nargs="+", help="explicit benchmark names")
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    parser.add_argument(
        "--embedder",
        default=None,
        help=(
            "run the blocker on this sentence-transformer instead of the benchmark's "
            "pinned all-MiniLM-L6-v2. Use it to test whether a sim_cos constant is a "
            "property of the FAMILY or of the CHECKPOINT -- e.g. --embedder "
            "intfloat/e5-base-v2, the current DEFAULT_EMBEDDING_MODEL"
        ),
    )
    parser.add_argument(
        "--in-process",
        action="store_true",
        help=(
            "measure in THIS interpreter instead of one subprocess per benchmark. "
            "Also what each subprocess runs. Opting out removes the memory bound -- "
            "see run_benchmark_isolated"
        ),
    )
    parser.add_argument(
        "--no-tables",
        action="store_true",
        help="write the artifact but print no tables (what each subprocess worker runs)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "continue an interrupted sweep: read the .partial checkpoint and skip "
            "the (benchmark, method, seed) cells it already holds"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "where to write the machine-readable findings (a TRACKED path). "
            f"Defaults to {CANONICAL_OUT} for a FULL run; REQUIRED with any narrowing, "
            "which would otherwise replace the full-portfolio artifact with a subset"
        ),
    )
    parser.add_argument(
        "--render",
        type=Path,
        default=None,
        help="print every table from an EXISTING artifact and exit -- measures nothing",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        type=Path,
        default=None,
        metavar=("BASELINE", "VARIANT"),
        help=(
            "print ONLY the checkpoint-transfer table: grade the constant selected "
            "on BASELINE against VARIANT's cells (measures nothing). Use with an "
            "artifact produced by --embedder"
        ),
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.compare is not None:
        print("Does the constant transfer across embedding checkpoints?")
        print(to_transfer_markdown(read_report(args.compare[0]), read_report(args.compare[1])))
        return

    if args.render is not None:
        print_tables(read_report(args.render))
        return

    # This script advertises "$0 spend" everywhere: --methods is free text, and a
    # paid scorer would reach make_resolver_factory with llm_client=None,
    # whereupon LLMMatcher lazily builds a real, billed client. Refuse instead.
    paid = [m for m in args.methods if m not in ZERO_SPEND_METHODS]
    if paid:
        parser.error(
            f"{', '.join(paid)} is not a zero-spend method and this study claims $0. "
            f"Choose from: {', '.join(ZERO_SPEND_METHODS)}."
        )

    # The canonical artifact may only be written by a run that measured the WHOLE
    # matrix -- checked on the four inputs that DEFINE it, so a future narrowing
    # flag cannot rot the guard open the way an enumerated flag list would.
    is_full_sweep = (
        not args.fast
        and not args.only
        and tuple(args.methods) == DEFAULT_METHODS
        and tuple(args.seeds) == DEFAULT_SEEDS
        and args.resamples == BOOTSTRAP_RESAMPLES
        # A checkpoint override measures a DIFFERENT cosine scale. Letting it
        # write the canonical artifact would relabel one encoder's cells as the
        # portfolio's, which is the precise confusion the flag exists to expose.
        and args.embedder is None
    )
    out: Path = args.out if args.out is not None else CANONICAL_OUT
    if out.resolve() == CANONICAL_OUT.resolve() and not is_full_sweep:
        parser.error(
            "a narrowed run measures a subset and the write replaces the whole file, "
            f"which would reduce {CANONICAL_OUT} to just what was measured. "
            "Send it elsewhere (e.g. --out tmp/threshold_subset.json)."
        )

    # Durability and publication are two different files: cells are checkpointed
    # as they land (a sweep costs ~an hour and usually runs in a transient
    # worktree), but `out` is written only once the whole matrix succeeded. The
    # checkpoint is deliberately NOT gitignored -- an interrupted run must show up
    # in `git status -uall` and force a decision rather than vanish with the
    # worktree.
    scratch = out.with_name(out.name + ".partial")
    fingerprint = _source_fingerprint()
    cells: list[CellResult] = []
    # A surviving `.writing` means a checkpoint was fully staged but crashed
    # before `os.replace()` committed it, so it holds MORE completed cells than
    # `.partial` does. Resuming past it loads the older file and the next
    # checkpoint overwrites the staging one, silently discarding that extra work
    # (raised in review). Refuse and let an operator compare -- picking
    # automatically would mean guessing which file is authoritative, and the
    # wrong guess is the data loss.
    staging = scratch.with_name(scratch.name + ".writing")
    if args.resume and staging.exists():
        parser.error(
            f"{staging} exists alongside {scratch}: a checkpoint was staged but never "
            "committed, so the staging file likely holds MORE cells than the "
            "checkpoint. Refusing to resume past it. Compare the two "
            "(`cells` length and identities), keep the one you want as "
            f"{scratch.name}, and delete the other."
        )
    if args.resume and scratch.exists():
        partial = read_report(scratch)
        # A resumed cell is REUSED, not recomputed, but the published report's
        # header is rewritten from THIS invocation's flags. So a resume under
        # different flags silently mints an artifact whose header describes
        # neither half of it: cells bootstrapped at the old --resamples while the
        # report claims the new count, or MiniLM and e5 cosine cells -- two
        # different score scales -- pooled under one embedder label. That is the
        # "a label decoupled from what it describes" failure this repo keeps
        # hitting, so the header is checked against the cells before they count
        # as done.
        if partial.bootstrap_resamples != args.resamples:
            parser.error(
                f"{scratch} was measured with --resamples {partial.bootstrap_resamples}, "
                f"but this run passes {args.resamples}. Resuming would publish those "
                "cells' intervals under the new number. Re-run with "
                f"--resamples {partial.bootstrap_resamples}, or start a fresh --out."
            )
        stale_embedder = sorted({c.embedder for c in partial.cells if c.embedder != args.embedder})
        if stale_embedder:
            parser.error(
                f"{scratch} holds cells measured with embedder(s) {stale_embedder}, but "
                f"this run passes {args.embedder!r}. A cosine cut is a cut on an "
                "encoder's scale, so pooling them would compare different scales in one "
                "table. Re-run with the original --embedder, or start a fresh --out."
            )
        if partial.source_fingerprint != fingerprint:
            parser.error(
                f"{scratch} was measured by source {partial.source_fingerprint!r}, but "
                f"this run is {fingerprint!r}. The flags match, but the CODE does not -- "
                "resuming would measure the remaining cells with a different harness, "
                "matcher, loader or dataset and pool both into one artifact attributed "
                "to the current source. Re-run the whole sweep to a fresh --out (or "
                "check out the original revision to finish this one)."
            )
        cells = partial.cells
        print(f"[resume] {len(cells)} cell(s) already measured in {scratch}", flush=True)
    else:
        for stale in (scratch, scratch.with_name(scratch.name + ".writing")):
            if stale.exists():
                parser.error(
                    f"{stale} exists: an earlier sweep was interrupted and those cells are "
                    "its only copy. Pass --resume to continue it, or read it and then "
                    "move, commit or delete it (or pick a different --out)."
                )

    def checkpoint() -> None:
        try:
            write_report(
                SweepReport(
                    grid=list(GRID),
                    shipped_threshold=SHIPPED_THRESHOLD,
                    bootstrap_resamples=args.resamples,
                    source_fingerprint=fingerprint,
                    cells=cells,
                ),
                scratch,
            )
        except OSError as exc:
            # A benchmark that fails is recoverable -- skip it and carry on. A
            # checkpoint that cannot be written is NOT: every later cell would run
            # for minutes with nothing catching it. Abort while the loss is small.
            raise CheckpointError(f"cannot write checkpoint {scratch}: {exc}") from exc

    selected = select_benchmarks(fast=args.fast, only=args.only)
    expected_cells = len(selected) * len(args.methods) * len(args.seeds)
    done = {(c.benchmark, c.method, c.seed) for c in cells}
    failures: list[str] = []
    for name in selected:
        wanted = {(name, m, s) for m in args.methods for s in args.seeds}
        if wanted <= done:
            print(f"[skip] {name}: already in the checkpoint", flush=True)
            continue
        where = "in-process" if args.in_process else "subprocess"
        print(f"[benchmark] {name} ({where}): {' '.join(args.methods)} ($0 spend)", flush=True)
        try:
            # DO NOT drop this benchmark's existing cells up front. They are
            # durable data, and dropping them before the replacement exists means
            # a worker that then fails leaves the filtered list in memory -- the
            # `except` below continues, and the NEXT benchmark's checkpoint()
            # writes that loss to disk permanently. Old cells are swapped out only
            # once replacements are in hand (raised in review).
            kept = [c for c in cells if c.benchmark != name]
            previous = [c for c in cells if c.benchmark == name]
            if args.in_process:
                # Per-cell checkpointing is load-bearing here: an --in-process run
                # IS the worker, so this loop is what writes `<worker_out>.partial`
                # for the parent's resume.
                #
                # Reconcile PER IDENTITY, not per benchmark. Swapping the whole
                # slice for `fresh` on the first yielded cell truncates it: with
                # seeds 0 and 1 already durable, a retry that yields seed 0 and
                # then fails on seed 1 would drop seed 1, and the next successful
                # benchmark's checkpoint() would persist that loss (raised in
                # review -- my previous fix bounded this window but did not close
                # it). An old cell survives until its own replacement exists.
                fresh: dict[tuple[str, str, int], CellResult] = {}
                for cell in run_benchmark(
                    name,
                    methods=tuple(args.methods),
                    seeds=tuple(args.seeds),
                    resamples=args.resamples,
                    embedder=args.embedder,
                ):
                    fresh[(cell.benchmark, cell.method, cell.seed)] = cell
                    not_yet_replaced = [
                        c for c in previous if (c.benchmark, c.method, c.seed) not in fresh
                    ]
                    cells = kept + not_yet_replaced + list(fresh.values())
                    checkpoint()
            else:
                # The worker is atomic from here, so the swap is too: measure
                # first, and only then replace the slice. A raise leaves `cells`
                # exactly as it was.
                measured = run_benchmark_isolated(name, args, scratch, fingerprint)
                cells = kept + measured
                checkpoint()
                # Only NOW is the worker's artifact redundant: its cells are in
                # the parent checkpoint on disk. Unlinking it any earlier opens a
                # window where a parent crash loses the benchmark entirely.
                scratch.with_name(f"{scratch.name}.{name}.json").unlink(missing_ok=True)
        except CheckpointError:
            raise
        except Exception as exc:  # noqa: BLE001 - a broken loader must not kill the sweep
            logger.exception("%s: run_benchmark raised", name)
            print(f"[fail] {name}: {type(exc).__name__}: {exc}", flush=True)
            failures.append(f"{name} ({type(exc).__name__})")
            continue

    report = SweepReport(
        grid=list(GRID),
        shipped_threshold=SHIPPED_THRESHOLD,
        bootstrap_resamples=args.resamples,
        source_fingerprint=fingerprint,
        cells=cells,
    )
    print()
    if cells and not args.no_tables:
        print_tables(report)
        print()
    if failures:
        print(f"[fail] {len(failures)} benchmark(s) did not complete: {', '.join(failures)}")
        print(f"[partial] {len(cells)} cell(s) kept at {scratch}; {out} NOT updated")
        raise SystemExit(1)
    if len(cells) != expected_cells:
        print(f"[fail] measured {len(cells)} cells, expected {expected_cells}")
        print(f"[partial] kept at {scratch}; {out} NOT updated")
        raise SystemExit(1)

    write_report(report, out)
    scratch.unlink(missing_ok=True)
    print(f"[out] {out} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
