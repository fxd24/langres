"""Frozen fixture for the mixed-family cascade byte-parity net (epic #193, W3-d).

A ``$0``, offline, deterministic scripted cascade whose sole purpose is to
exercise the **mixed-family, decision-vs-score** path that the W3-d match-cut
rewire is most likely to break. W3-d moves the match cut out of the
:class:`~langres.core.clusterer.Clusterer` into a ``Select(THRESHOLD)`` and runs
the clusterer as pure transitive closure, on the claim::

    ThresholdSelect(t) -> Clusterer(0.0)  ==  Clusterer(t)

That equivalence hinges entirely on :func:`~langres.core.models.predicted_match`
being applied identically before and after the move -- and ``predicted_match``
gives a judge's boolean ``decision`` **precedence over its numeric ``score``**
(a decider already decided; the threshold never overrides it). A rewrite that
naively re-implements the cut as ``score >= t`` -- dropping that precedence --
would silently break exactly the rows where ``decision`` and ``score`` disagree.

This fixture builds a real :class:`~langres.core.matchers.cascade_judge.CascadeMatcher`
(the production mixed-family matcher) over a scripted, no-network student and
escalation, wired into a plain :class:`~langres.core.resolver.ERModel` over an
:class:`~langres.core.blockers.AllPairsBlocker`. The student emits a ``sim_cos``
numeric score for every pair; pairs whose student score lands inside the band
escalate to a scripted decider that emits a ``prob_llm`` **decision**. The result
is a genuinely mixed ``sim_cos``/``prob_llm`` stream flowing through
``resolve()`` / ``dedupe()`` / ``predict()`` -- the exact shape W3-d rewires.

Seven disjoint pairs pin the seven cases that matter (``r{2k-1}``, ``r{2k}``):

======  ============================  =================================================
 pair    what the cascade emits        why it is here (the W3-d guard)
======  ============================  =================================================
r01/r02  student ``sim_cos`` 0.95      score-decided MATCH (score > band, score >= thr)
r03/r04  student ``sim_cos`` 0.10      score-decided NON-match (score < band, score < thr)
r05/r06  escalated ``decision=True``   decider MATCH (score=None -> decision drives)
r07/r08  escalated ``decision=False``  decider NON-match (score=None -> decision drives)
r09/r10  escalated **abstain**         no decision, no score -> EXCLUDED (not a "no")
r11/r12  ``decision=True``, score 0.20 **decision wins over a LOW score** -> MATCH
r13/r14  ``decision=False``, score 0.90 **decision suppresses a HIGH score** -> NON-match
======  ============================  =================================================

``r11/r12`` and ``r13/r14`` are the crux: a ``score >= t`` cut would *drop*
``r11/r12`` (0.20 < 0.5) and *wrongly add* ``r13/r14`` (0.90 >= 0.5). Only a
faithful ``predicted_match`` produces the clustering this fixture freezes. Every
other pair the ``AllPairsBlocker`` generates is a background ``sim_cos`` 0.05
student non-match (below band and threshold -> never escalated, never merged).

Do not edit the records, scripts, band, or threshold without regenerating the
golden -- a change here is a deliberate re-baseline, never an accident.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from pydantic import BaseModel

from langres.core.blockers import AllPairsBlocker
from langres.core.clusterer import Clusterer
from langres.core.matcher import Matcher, SchemaT
from langres.core.matchers.cascade_judge import CascadeMatcher
from langres.core.models import ERCandidate, PairwiseJudgement
from langres.core.reports import ScoreInspectionReport, _inspect_scores_impl
from langres.core.resolver import Resolver
from langres.testing import ScriptedJudge

#: Clusterer / match-cut threshold. 0.20 falls below it and 0.90 above it, so the
#: two decision-vs-score conflict pairs genuinely straddle the cut.
CASCADE_THRESHOLD = 0.5

#: The student's uncertainty band (over its ``sim_cos`` scores). A 0.50 student
#: score lands inside it and escalates; 0.95 / 0.10 / 0.05 fall outside.
CASCADE_BAND: tuple[float, float] = (0.3, 0.7)

#: Background score for any pair not deliberately scripted below: a ``sim_cos``
#: non-match, out of band (never escalated) and under threshold (never merged).
_BACKGROUND_SCORE = 0.05


class CascadeBusinessW3D(BaseModel):
    """A tiny entity: ``id`` + ``name``. The scripts key on ``id`` only."""

    id: str
    name: str


class ScriptedDecider(Matcher[SchemaT]):
    """A ``$0`` escalation double that emits a scripted ``(decision, score)`` per pair.

    :class:`~langres.testing.ScriptedJudge` can only *score* or abstain -- it has
    no way to emit a boolean ``decision``, which is precisely the signal the
    decision-vs-score path turns on. This minimal decider fills that gap: it maps
    each unordered pair (``frozenset({left_id, right_id})``) to a
    ``(decision, score)`` verdict and stamps it ``score_type="prob_llm"`` (a
    probability family, so the cascade's shared-scale contract stays quiet for
    the escalation tier). A pair absent from the map defaults to a confident
    ``decision=False`` -- but with this fixture the escalation is only ever
    reached by the five deliberately in-band pairs, so the default is never hit.
    """

    def __init__(self, verdicts: dict[frozenset[str], tuple[bool | None, float | None]]) -> None:
        self.verdicts = verdicts
        self.seen: list[frozenset[str]] = []

    def forward(self, candidates: Iterator[ERCandidate[SchemaT]]) -> Iterator[PairwiseJudgement]:
        for candidate in candidates:
            key = frozenset(
                {candidate.left.id, candidate.right.id}  # type: ignore[attr-defined]
            )
            self.seen.append(key)
            decision, score = self.verdicts.get(key, (False, None))
            yield PairwiseJudgement(
                left_id=candidate.left.id,  # type: ignore[attr-defined]
                right_id=candidate.right.id,  # type: ignore[attr-defined]
                decision=decision,
                score=score,
                score_type="prob_llm",
                decision_step="scripted_frontier",
                provenance={},
            )

    def inspect_scores(
        self, judgements: list[PairwiseJudgement], sample_size: int = 10
    ) -> ScoreInspectionReport:
        return _inspect_scores_impl(judgements, sample_size)


#: 14 records = 7 disjoint pairs (see the module docstring's table).
RECORDS: list[dict[str, object]] = [
    {"id": f"r{i:02d}", "name": f"Entity {i}"} for i in range(1, 15)
]


def _pair(left_id: str, right_id: str) -> frozenset[str]:
    return frozenset({left_id, right_id})


#: Student ``sim_cos`` scores. Only these seven pairs are scripted; every other
#: pair the AllPairsBlocker generates falls to ``_BACKGROUND_SCORE`` (0.05).
_STUDENT_SCORES: dict[frozenset[str], float] = {
    _pair("r01", "r02"): 0.95,  # > band high, > threshold -> student MATCH
    _pair("r03", "r04"): 0.10,  # < band low,  < threshold -> student NON-match
    _pair("r05", "r06"): 0.50,  # in band -> escalate
    _pair("r07", "r08"): 0.50,  # in band -> escalate
    _pair("r09", "r10"): 0.50,  # in band -> escalate
    _pair("r11", "r12"): 0.50,  # in band -> escalate
    _pair("r13", "r14"): 0.50,  # in band -> escalate
}

#: Escalation verdicts for the five in-band pairs (decision drives the outcome).
_ESCALATION_VERDICTS: dict[frozenset[str], tuple[bool | None, float | None]] = {
    _pair("r05", "r06"): (True, None),  # decider MATCH
    _pair("r07", "r08"): (False, None),  # decider NON-match
    _pair("r09", "r10"): (None, None),  # ABSTAIN -> excluded (not a "no")
    _pair("r11", "r12"): (True, 0.20),  # decision wins over a LOW score -> MATCH
    _pair("r13", "r14"): (False, 0.90),  # decision suppresses a HIGH score -> NON-match
}

#: The pairs whose behavior the golden is really about (for focused assertions).
INTERESTING_PAIRS: dict[str, tuple[str, str]] = {
    "student_match": ("r01", "r02"),
    "student_nonmatch": ("r03", "r04"),
    "escalated_decision_match": ("r05", "r06"),
    "escalated_decision_nonmatch": ("r07", "r08"),
    "escalated_abstain": ("r09", "r10"),
    "decision_over_low_score": ("r11", "r12"),
    "decision_over_high_score": ("r13", "r14"),
}

#: The three multi-record clusters the fixture must produce (everything else is a
#: singleton the Clusterer drops). Pinned as a semantic backstop so a blind
#: ``LANGRES_PARITY_UPDATE=1`` re-baseline still has to reproduce these.
EXPECTED_CLUSTERS: list[set[str]] = [
    {"r01", "r02"},  # student sim_cos match
    {"r05", "r06"},  # escalated decision=True
    {"r11", "r12"},  # decision=True beats a below-threshold score
]

#: Ids that must NOT be in any cluster (the decision/score conflict + abstain
#: NON-matches). ``r13``/``r14`` are the load-bearing ones: a ``score >= t`` cut
#: would merge them (0.90 >= 0.5); a faithful ``predicted_match`` does not.
EXPECTED_UNMERGED_IDS: set[str] = {"r03", "r04", "r07", "r08", "r09", "r10", "r13", "r14"}


def build_cascade_model() -> tuple[Resolver, ScriptedDecider[CascadeBusinessW3D]]:
    """Build the scripted mixed-family cascade pipeline.

    Returns the wired :class:`~langres.core.resolver.ERModel` and the escalation
    decider (so a test can assert on its ``seen`` escalation-laziness spy). A
    fresh instance per call keeps the ``seen`` spy and the cascade's one-time
    ``score_type`` warning flag un-shared across tests.
    """
    student: ScriptedJudge[CascadeBusinessW3D] = ScriptedJudge(
        _STUDENT_SCORES,
        score_type="sim_cos",
        decision_step="scripted_student",
        default_score=_BACKGROUND_SCORE,
    )
    escalation: ScriptedDecider[CascadeBusinessW3D] = ScriptedDecider(_ESCALATION_VERDICTS)
    cascade: CascadeMatcher[CascadeBusinessW3D] = CascadeMatcher(
        student=student, escalation=escalation, band=CASCADE_BAND
    )
    model = Resolver(
        blocker=AllPairsBlocker(schema=CascadeBusinessW3D),
        comparator=None,
        matcher=cascade,
        clusterer=Clusterer(threshold=CASCADE_THRESHOLD),
    )
    return model, escalation


def record(record_id: str) -> dict[str, Any]:
    """The one fixture record with ``id == record_id`` (fixture ids are unique)."""
    return next(r for r in RECORDS if r["id"] == record_id)
