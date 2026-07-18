"""The named score-family / confidence-source type aliases — one source of truth.

``PairwiseJudgement`` spelled these two ``Literal``s inline, and the ``Pairs``
carrier (:mod:`langres.core.pairs`) needs the *same* seven score families and
five confidence sources. Rather than duplicate the literals in two files (a
guaranteed drift point the moment one grows a value), they live here once and
both modules import them.

This is a byte-identical **extraction**, not a widening: the aliases resolve to
exactly the historical inline literals, so ``PairwiseJudgement``'s emitted JSON
schema is unchanged (asserted in ``tests/core/test_pairs.py``).

A strict stdlib leaf — it imports nothing from ``langres`` — so both
``langres.core.models`` and the ``langres.core.pairs`` leaf can depend on it
without adding an edge that could knot the import graph.
"""

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
