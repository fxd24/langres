"""The named score-family / confidence-source type aliases — one source of truth.

``PairwiseJudgement`` spelled these two ``Literal``s inline, and the ``Pairs``
carrier (:mod:`langres.core.pairs`) needs the *same* seven score families and
five confidence sources. Rather than duplicate the literals in two files (a
guaranteed drift point the moment one grows a value), they live here once and
both modules import them.

The aliases are a byte-identical **extraction**, not a widening: they resolve to
exactly the historical inline literals, so ``PairwiseJudgement``'s emitted JSON
schema is unchanged (asserted in ``tests/core/test_pairs.py``).

:data:`DEFAULT_THRESHOLDS` lives here for the same one-source-of-truth reason: a
family's out-of-the-box match cut is a property *of the family*, and the front
doors, the method registry and the docs must not each carry their own copy.

A strict stdlib leaf — it imports nothing from ``langres`` — so both
``langres.core.models`` and the ``langres.core.pairs`` leaf can depend on it
without adding an edge that could knot the import graph.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Literal, TypeAlias

#: The seven score families a judge may tag a score with. ``score_type`` doubles
#: as the judge-family tag even when the score itself is ``None`` (a decider that
#: only emits a ``decision`` still names its family, e.g. ``"prob_llm"``). This
#: is the frozen 7-value set — do **not** widen it; a lifecycle "not yet scored"
#: state is spelled ``ScoreType | None`` at the field, never a new member here.
ScoreType: TypeAlias = Literal[
    "sim_cos",
    "prob_llm",
    "heuristic",
    "calibrated_prob",
    "prob_fs",
    "prob_rf",
    "prob_group_llm",
]

#: Provenance of a judgement's ``confidence``. ``"none"`` means the judge
#: structurally has no confidence to give; ``"unrequested"`` means it could but
#: was not asked. The set is provisional (expected to grow in Wave 2), unlike the
#: frozen :data:`ScoreType`.
ConfidenceSource: TypeAlias = Literal[
    "none",
    "unrequested",
    "logprob",
    "calibrated",
    "heuristic",
]

#: The out-of-the-box match cut per score family — the ONE place a shipped
#: default threshold is written down.
#:
#: Score scales are not comparable across families (a rapidfuzz similarity, a
#: cosine, and an LLM's probability all live in ``[0, 1]`` and mean different
#: things), so a single global constant cannot be right for all of them. That is
#: what :attr:`~langres.core.method_registry.MethodSpec.default_threshold` was
#: introduced to express; this mapping is where the value it defaults to lives,
#: so the registry, the named architectures and the research recipes cannot
#: drift apart by each hard-coding their own literal.
#:
#: **Every entry is in exactly one of three states. Do not read them alike.**
#: (The measurement is ``examples/research/threshold_constant_sweep.py``, swept on
#: held-out splits of the benchmark portfolio; write-up
#: ``docs/research/20260728_threshold_constant.md``.)
#:
#: 1. **Measured, and changed** — ``sim_cos`` = ``0.90``. Selected
#:    leave-one-benchmark-out, so it is out-of-sample w.r.t. the dataset, and it
#:    beat ``0.5`` on *every* eligible benchmark and seed with 95% intervals
#:    entirely above zero. ``0.5`` was not mildly mistuned on a cosine scale, it
#:    was catastrophic: normalized embeddings put nearly every candidate pair
#:    above it, so it accepted almost everything the blocker proposed.
#: 2. **Measured, and deliberately left alone** — ``heuristic`` stays ``0.5``.
#:    A better constant exists *on the median* and its leave-one-out selection is
#:    stable, but it reliably **damages** ``abt_buy`` (every seed, interval
#:    entirely below zero, and ``febrl_dedup`` mildly too) while helping the other
#:    seven. A default that improves the median while predictably harming a known
#:    data class is a recommendation with an undisclosed victim, so the
#:    pre-registered ship rule rejects it. The honest guidance for this family is
#:    to derive the cut from labels
#:    (``langres.training.calibration.derive_threshold``) — **not** because
#:    deriving beats any constant everywhere (measured: it does not; the rejected
#:    constant actually scores higher on 5 of 9 benchmarks) but because deriving
#:    is what rescues the case a constant cannot. On ``abt_buy`` — the very
#:    benchmark that vetoes the constant — the derived cut beats both the
#:    constant and ``0.5``.
#: 3. **Not measured — status quo, not a finding.** ``prob_llm``,
#:    ``prob_group_llm`` (a paid completion per score, so a portfolio grid sweep
#:    is a real invoice), and ``calibrated_prob`` / ``prob_fs`` / ``prob_rf``
#:    (*fitted* matchers a label-free user cannot run at all, so "best
#:    out-of-the-box constant" is not even the same question). Change one of
#:    these only with a measurement behind it.
#:
#: **A ``sim_cos`` cut is a cut on an encoder's scale, not on the family tag.**
#: ``0.90`` was selected on the ``all-MiniLM-L6-v2`` every benchmark loader pins,
#: so it was re-measured on ``DEFAULT_EMBEDDING_MODEL`` (``intfloat/e5-base-v2``)
#: under the same protocol, in **both** directions. It never harms there — but
#: that checkpoint's own selection is *higher*, and carrying **that** value back
#: significantly harms ``abt_buy``. So ``0.90`` is the constant that is safe on
#: both, which is what a default owes you; it is a floor that stops the
#: out-of-the-box path being catastrophically wrong, **not** a tuned value for
#: your encoder. Tune with labels if you need the rest.
DEFAULT_THRESHOLDS: Mapping[ScoreType, float] = MappingProxyType(
    {
        "heuristic": 0.5,
        "sim_cos": 0.90,
        "calibrated_prob": 0.5,
        "prob_fs": 0.5,
        "prob_rf": 0.5,
        "prob_llm": 0.7,
        "prob_group_llm": 0.7,
    }
)
# Completeness (every ``ScoreType`` has an entry) is gated by
# ``tests/core/test_score_type_defaults.py`` — a missing family would otherwise
# surface as a ``KeyError`` in a user's first ``dedupe()``.


def resolve_threshold(threshold: float | None, score_type: ScoreType) -> float:
    """Resolve a caller's ``threshold=`` against its score family's shipped default.

    The seam every front door with a ``threshold: float | None = None`` parameter
    goes through, so "what does ``None`` mean here?" has one answer and one place
    to change it. A caller's explicit value is returned untouched — including one
    that happens to equal the default, which is *not* the same statement as
    omitting it.

    Args:
        threshold: The caller's value, or ``None`` to take the family default.
        score_type: The family the scores being cut belong to.

    Returns:
        The threshold to cut at.

    Raises:
        KeyError: If ``score_type`` is not a known family. Deliberately not
            softened to a fallback constant: silently cutting an unknown score
            scale at 0.5 is the failure mode this whole mapping exists to end.
    """
    return DEFAULT_THRESHOLDS[score_type] if threshold is None else threshold
