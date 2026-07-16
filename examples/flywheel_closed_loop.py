"""The data flywheel, closed end to end, at $0 -- bootstrap -> review -> train -> cascade.

langres closes a loop most ER tools leave open: an expensive judge bootstraps
*silver* labels, a human reviews only the uncertain margin, those labels train a
*cheap* student, and a **cascade** runs the student everywhere while escalating
only the still-uncertain pairs back to the expensive judge. This example runs
the whole loop against committed Fodors-Zagat fixtures with **zero real API
calls** -- the "frontier" judge is a deterministic local simulation
(:class:`SimulatedFrontierJudge`) whose ``cost_usd`` stamps are *fictional* (no
network, no spend).

.. warning::
   This is a **plumbing** demo, not an economics claim. The simulated teacher
   and gold-derived human answers make the outcome favorable *by construction*.
   Real escalation rates and real dollar savings only mean something with a real
   frontier model on hard data -- those live in the paid validation runs, never
   here. What this demo (and its exit-criteria test) proves is that the *wiring*
   holds: every seam composes, one threshold cuts the mixed student/teacher
   stream correctly, and corrected pairs are never re-asked.

The eight stages (all $0, all deterministic)::

    1. bootstrap   SimulatedFrontierJudge scores every candidate under a
                   LoggingMatcher -> judgements.jsonl (the flywheel inlet).
    2. select      select_for_review(uncertainty) -> a review queue; prints the
                   equivalent `uv run langres review queue.jsonl` command.
    3. answer      a simulated human answers the queue from gold -> corrections.
    4. harvest     BEFORE: harvest verdicts alone -> derive_threshold fires the
                   silver-only circularity warning. AFTER: overlay corrections.
    5. train       fit a RandomForestMatcher student on the harvested labels; calibrate its
                   threshold on the STUDENT's OWN scores (never the teacher's --
                   different scale).
    6. cascade     CascadeMatcher(student, teacher, band=...) where the band is
                   DERIVED from calibration-split student scores (widen around
                   the student threshold until ~20% of pairs fall inside) -- not
                   a magic +/-0.15 constant.
    7. next loop   select_for_review(disagreement) seeds the next queue (already
                   corrected pairs are never re-asked); prints the exhaustion
                   message when nothing is left.
    8. report      pairwise P/R/F1 vs gold for teacher / student / cascade, plus
                   the audit-slice disagreement rate (the governance/trust metric
                   that catches confident false merges), the escalation rate,
                   the frontier-call reduction, and the (simulated) dollars saved.

Run it (needs the ``trained`` extra for scikit-learn behind ``RandomForestMatcher``; no
network, no spend)::

    uv run python examples/flywheel_closed_loop.py

Regenerate the fixtures with ``examples/data/flywheel_loop/generate_fixtures.py``.
"""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
import warnings
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from langres.clients.openrouter import BudgetExceeded, SpendMonitor
from langres.core.calibration import derive_threshold
from langres.core.comparators import StringComparator
from langres.core.harvest import (
    Correction,
    CorrectionLog,
    LabeledPair,
    derive_threshold_from_pairs,
    harvest_labeled_pairs,
)
from langres.core.judgement_log import JudgementLog, LoggingMatcher
from langres.core.models import ERCandidate, PairwiseJudgement
from langres.core.matcher import Matcher
from langres.core.matchers.cascade_judge import CASCADE_ESCALATED_STEP, CascadeMatcher
from langres.core.matchers.random_forest_judge import RandomForestMatcher
from langres.core.reports import ScoreInspectionReport, _inspect_scores_impl
from langres.core.review import ReviewQueue, select_for_review

#: Committed fixtures (regenerate with ``data/flywheel_loop/generate_fixtures.py``).
DATA_DIR = Path(__file__).resolve().parent / "data" / "flywheel_loop"

#: Seed for every split/sample here -- matches the fixture generator so the
#: train half is the one it asserted spans both label classes.
_SEED = 7

#: FICTIONAL per-call cost stamped on each simulated frontier judgement. No API
#: is called; this exists only to make the "dollars saved" arithmetic concrete.
_FICTIONAL_COST_PER_CALL = 0.002

#: Share of calibration-split pairs the escalation band should capture (stage 6).
_BAND_TARGET_FRACTION = 0.20

#: Name-led feature weights for the simulated teacher's signal (mirrors the
#: fixture generator's negative-mining weights).
_TEACHER_WEIGHTS = {"name": 0.45, "addr": 0.30, "phone": 0.20, "city": 0.05}


class FZRecord(BaseModel):
    """A Fodors-Zagat restaurant record with a stable, source-prefixed id.

    Stable ids are load-bearing for the flywheel: the judgement log stores ids
    only, and every downstream join (review queue, corrections, harvest) keys on
    them -- so positional ids (what a schema-less ``dedupe()`` assigns) could not
    survive a fresh run. All fields are strings; the Comparator treats an empty
    value as MISSING.
    """

    id: str
    name: str = ""
    addr: str = ""
    city: str = ""
    phone: str = ""


class SimulatedFrontierJudge(Matcher[FZRecord]):
    """A deterministic, $0 stand-in for an expensive frontier LLM judge.

    Maps a name-led rapidfuzz signal (read off the pair's ``ComparisonVector``)
    through a steep logistic into a confident probability, plus a small, stable
    per-pair perturbation so the scores spread realistically instead of snapping
    to two values. Emits ``score_type="prob_llm"`` on a shared ``[0, 1]``
    probability scale (so it cascades with a ``RandomForestMatcher`` student without
    tripping the scale-mismatch warning), and stamps a **fictional** ``cost_usd``
    -- no network call is ever made.

    It is a strong-but-imperfect oracle: on Fodors-Zagat, name/addr/phone
    similarity separates true duplicates from hard non-matches well, which is
    exactly why a cheap trained student can eventually imitate it on the easy
    majority and leave only the margin to escalate.
    """

    def __init__(
        self,
        *,
        center: float = 0.55,
        steepness: float = 12.0,
        jitter: float = 0.04,
        seed: int = _SEED,
        cost_usd: float = _FICTIONAL_COST_PER_CALL,
    ) -> None:
        self.center = center
        self.steepness = steepness
        self.jitter = jitter
        self.seed = seed
        self.cost_usd = cost_usd

    def _signal(self, candidate: ERCandidate[FZRecord]) -> float:
        """Name-led weighted rapidfuzz similarity from the pair's comparison vector."""
        if candidate.comparison is None:
            raise ValueError(
                "SimulatedFrontierJudge needs candidates carrying a comparison vector "
                "-- run them through a StringComparator first."
            )
        similarities = candidate.comparison.similarities
        total = sum(
            weight * similarities.get(field, 0.0) for field, weight in _TEACHER_WEIGHTS.items()
        )
        return total / sum(_TEACHER_WEIGHTS.values())

    def _perturbation(self, left_id: str, right_id: str) -> float:
        """A stable per-pair jitter in ``[-jitter, +jitter]`` (no per-process hash salt)."""
        key = "|".join(sorted((left_id, right_id))) + f"|{self.seed}"
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        unit = int.from_bytes(digest[:8], "big") / float(1 << 64)  # in [0, 1)
        return (unit * 2.0 - 1.0) * self.jitter

    def forward(self, candidates: Iterator[ERCandidate[FZRecord]]) -> Iterator[PairwiseJudgement]:
        """Score each pair with a confident, deterministic probability (prob_llm)."""
        for candidate in candidates:
            left_id = candidate.left.id
            right_id = candidate.right.id
            signal = self._signal(candidate)
            prob = 1.0 / (1.0 + math.exp(-self.steepness * (signal - self.center)))
            prob += self._perturbation(left_id, right_id)
            score = min(0.98, max(0.02, prob))
            yield PairwiseJudgement(
                left_id=left_id,
                right_id=right_id,
                score=score,
                score_type="prob_llm",
                decision_step="simulated_frontier",
                reasoning="simulated frontier judge (rapidfuzz signal + seeded perturbation)",
                provenance={
                    "model": "simulated-frontier/glm-oracle",
                    "cost_usd": self.cost_usd,
                    "note": "FICTIONAL cost -- no API was called",
                },
            )

    def inspect_scores(
        self, judgements: list[PairwiseJudgement], sample_size: int = 10
    ) -> ScoreInspectionReport:
        """Explore scores without ground truth (shared Matcher utility)."""
        return _inspect_scores_impl(judgements, sample_size)


class _OuterSpendCap(Matcher[FZRecord]):
    """Wrap the teacher judge in ONE cumulative spend cap for the whole run.

    This is the correct place for a cascade's spend cap: on the OUTSIDE of the
    whole (teacher) judge, holding ONE
    :class:`~langres.clients.openrouter.SpendMonitor` for the wrapper's lifetime
    so cost accumulates across EVERY ``forward`` call -- the stage-1 bootstrap
    batch AND the cascade's per-pair escalations (the same teacher instance is
    reused as the cascade's escalation tier). Core's
    :class:`~langres.core.presets._SpendCappedMatcher` deliberately opens a FRESH
    monitor per ``forward`` call (right for a single verb call), which -- passed
    as a cascade tier -- would reset the budget on every one-pair escalation and
    so bound nothing (see
    :class:`~langres.core.matchers.cascade_judge.CascadeMatcher`'s docstring). One
    shared monitor is what makes the outer cap real.

    On a breach it raises :class:`~langres.clients.openrouter.BudgetExceeded`
    carrying the judgements already produced (and paid for) on
    ``.partial_judgements`` -- the same recovery contract ``_SpendCappedMatcher``,
    ``LoggingMatcher``, and the cascade's escalation-side re-raise all rely on.
    """

    def __init__(self, judge: Matcher[FZRecord], *, budget_usd: float) -> None:
        self._judge = judge
        self._monitor = SpendMonitor(budget_usd=budget_usd)

    @property
    def spent(self) -> float:
        """Cumulative metered cost so far (real USD for a real teacher)."""
        return self._monitor.spent

    def forward(self, candidates: Iterator[ERCandidate[FZRecord]]) -> Iterator[PairwiseJudgement]:
        """Yield each teacher judgement, hard-stopping once cumulative cost passes budget."""
        produced: list[PairwiseJudgement] = []
        for judgement in self._judge.forward(candidates):
            produced.append(judgement)
            cost = judgement.provenance.get("cost_usd", 0.0)
            self._monitor.add(float(cost) if cost is not None else 0.0)
            try:
                self._monitor.check()
            except BudgetExceeded as exc:
                exc.partial_judgements = list(produced)
                raise
            yield judgement

    def inspect_scores(
        self, judgements: list[PairwiseJudgement], sample_size: int = 10
    ) -> ScoreInspectionReport:
        """Delegate score inspection to the wrapped judge."""
        return self._judge.inspect_scores(judgements, sample_size)


class PairMetrics(BaseModel):
    """Pairwise classification quality of one judge at one decision threshold."""

    label: str
    threshold: float
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int


class ClosedLoopReport(BaseModel):
    """The closed-loop run's structured outcome (what ``format_report`` renders)."""

    n_records: int
    n_candidates: int
    n_train: int
    n_heldout: int
    n_review_items: int
    n_corrections: int
    circularity_warning_fired: bool
    before_threshold: float
    student_threshold: float
    band_low: float
    band_high: float
    band_fraction: float
    escalation_rate: float
    frontier_call_reduction: float
    simulated_dollars_saved: float
    #: Cumulative cost metered by the outer spend cap (``spend_cap_usd``): REAL
    #: USD for an injected real teacher, fictional dollars for the simulated
    #: default, and ``0.0`` when no cap is applied.
    teacher_spend_usd: float = 0.0
    teacher: PairMetrics
    student: PairMetrics
    cascade: PairMetrics
    n_escalated: int
    escalated_correct: int
    audit_sample_size: int
    audit_disagreement_rate: float
    next_queue_size: int
    next_queue_has_corrected_pair: bool

    @property
    def escalated_accuracy(self) -> float:
        """Fraction of escalated pairs the cascade verdict got right (1.0 if none)."""
        return 1.0 if self.n_escalated == 0 else self.escalated_correct / self.n_escalated


# ----------------------------------------------------------------------------
# Fixtures + candidates
# ----------------------------------------------------------------------------


def _load_records(data_dir: Path) -> dict[str, FZRecord]:
    """Load ``records.json`` into ``{id: FZRecord}``."""
    rows = json.loads((data_dir / "records.json").read_text(encoding="utf-8"))
    return {row["id"]: FZRecord.model_validate(row) for row in rows}


def _load_candidate_specs(data_dir: Path) -> list[tuple[str, str, bool]]:
    """Load ``gold_pairs.json`` into ``(left_id, right_id, gold_label)`` tuples."""
    payload = json.loads((data_dir / "gold_pairs.json").read_text(encoding="utf-8"))
    return [
        (pair["left_id"], pair["right_id"], bool(pair["label"]))
        for pair in payload["candidate_pairs"]
    ]


def _pair_key(left_id: str, right_id: str) -> frozenset[str]:
    """Order-independent pair key (mirrors the core selectors/harvest)."""
    return frozenset({left_id, right_id})


def _build_candidates(
    records: Mapping[str, FZRecord],
    specs: Sequence[tuple[str, str, bool]],
    comparator: StringComparator[FZRecord],
) -> list[ERCandidate[FZRecord]]:
    """Attach a ComparisonVector to every candidate pair (the two-phase pipeline)."""
    candidates: list[ERCandidate[FZRecord]] = []
    for left_id, right_id, _label in specs:
        left, right = records[left_id], records[right_id]
        candidates.append(
            ERCandidate(
                left=left,
                right=right,
                blocker_name="fixture",
                comparison=comparator.compare(left, right),
            )
        )
    return candidates


def _seeded_split(n: int, seed: int) -> tuple[list[int], list[int]]:
    """Deterministic 50/50 train/held-out index split (matches the fixture generator)."""
    import random

    order = list(range(n))
    random.Random(seed).shuffle(order)
    return order[::2], order[1::2]


# ----------------------------------------------------------------------------
# Metrics + band derivation
# ----------------------------------------------------------------------------


def _score_verdicts(
    judgements: Sequence[PairwiseJudgement],
    threshold: float,
    gold: Mapping[frozenset[str], bool],
    *,
    label: str,
) -> PairMetrics:
    """Pairwise P/R/F1 of ``score >= threshold`` against gold, over ``judgements``."""
    tp = fp = fn = 0
    for judgement in judgements:
        predicted = judgement.score >= threshold
        actual = gold[_pair_key(judgement.left_id, judgement.right_id)]
        if predicted and actual:
            tp += 1
        elif predicted and not actual:
            fp += 1
        elif not predicted and actual:
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return PairMetrics(
        label=label,
        threshold=threshold,
        precision=precision,
        recall=recall,
        f1=f1,
        tp=tp,
        fp=fp,
        fn=fn,
    )


def derive_escalation_band(
    scores: Sequence[float],
    threshold: float,
    *,
    target_fraction: float = _BAND_TARGET_FRACTION,
    step: float = 0.01,
    max_half_width: float = 0.5,
) -> tuple[tuple[float, float], float]:
    """Widen a symmetric band around ``threshold`` until it captures ``target_fraction``.

    Returns ``((low, high), achieved_fraction)``. This replaces a hand-set
    ``+/-0.15`` constant: the band is read straight off the calibration-split
    student score distribution, so it adapts to how confident the student
    actually is. ``low``/``high`` are clipped into ``[0, 1]`` with ``low < high``
    (the :class:`CascadeMatcher` contract).
    """
    if not scores:
        raise ValueError("derive_escalation_band needs at least one calibration score.")
    n = len(scores)
    half = step
    achieved = 0.0
    while half <= max_half_width:
        low = max(0.0, threshold - half)
        high = min(1.0, threshold + half)
        achieved = sum(1 for s in scores if low <= s <= high) / n
        if achieved >= target_fraction:
            break
        half += step
    low = max(0.0, threshold - half)
    high = min(1.0, threshold + half)
    if low >= high:  # threshold pinned at an edge -- keep a non-empty band
        low = max(0.0, high - step)
    return (low, high), achieved


# ----------------------------------------------------------------------------
# The loop
# ----------------------------------------------------------------------------


def _log_rows(judgements: Sequence[PairwiseJudgement], threshold: float) -> list[dict[str, Any]]:
    """Render judgements as JudgementLog-shaped rows (for the disagreement selector)."""
    return [
        {
            "left_id": j.left_id,
            "right_id": j.right_id,
            "score": j.score,
            "verdict": j.score >= threshold,
            "model": j.provenance.get("model"),
            "decision_step": j.decision_step,
        }
        for j in judgements
    ]


def run_closed_loop(
    data_dir: Path = DATA_DIR,
    *,
    seed: int = _SEED,
    work_dir: Path | None = None,
    verbose: bool = False,
    teacher: Matcher[FZRecord] | None = None,
    spend_cap_usd: float | None = None,
) -> ClosedLoopReport:
    """Run the full bootstrap -> review -> harvest -> train -> cascade loop.

    Defaults to the deterministic **$0 simulation** (the ``SimulatedFrontierJudge``
    teacher). Inject a REAL judge via ``teacher=`` (the paid FZ/AG validation
    scripts do) and pass ``spend_cap_usd=`` to hard-cap its cumulative spend --
    the same loop, no duplication.

    Args:
        data_dir: Directory holding ``records.json`` and ``gold_pairs.json``.
        seed: Seed for every split/sample (matches the fixture generator).
        work_dir: Where the loop writes its JSONL artifacts (queue, corrections,
            logs). A temporary directory is used when omitted.
        verbose: When ``True``, print the ``uv run langres review`` command and
            the next-queue / exhaustion message as the loop runs.
        teacher: The frontier/escalation judge scored on every pair (bootstrap)
            and inside the cascade band. ``None`` -> the deterministic $0
            :class:`SimulatedFrontierJudge`. Must score :class:`FZRecord`
            candidates carrying a comparison vector.
        spend_cap_usd: When set, wraps ``teacher`` in ONE outer
            :class:`~langres.clients.openrouter.SpendMonitor` cap
            (:class:`_OuterSpendCap`) spanning the whole run -- the bootstrap
            batch AND the cascade's per-pair escalations. Raises
            :class:`~langres.clients.openrouter.BudgetExceeded` if cumulative
            (real) cost crosses it. ``None`` (default) applies no cap. The cap
            wraps the OUTSIDE of the teacher, never a cascade tier.

    Returns:
        A :class:`ClosedLoopReport` with every metric the demo reports (including
        ``teacher_spend_usd``, the cap's cumulative metered cost).

    Raises:
        BudgetExceeded: If ``spend_cap_usd`` is set and the teacher's cumulative
            metered cost crosses it (carries ``.partial_judgements``).
    """
    if work_dir is None:
        with tempfile.TemporaryDirectory(prefix="flywheel_loop_") as tmp:
            return run_closed_loop(
                data_dir,
                seed=seed,
                work_dir=Path(tmp),
                verbose=verbose,
                teacher=teacher,
                spend_cap_usd=spend_cap_usd,
            )

    records = _load_records(data_dir)
    specs = _load_candidate_specs(data_dir)
    gold = {_pair_key(left, right): label for left, right, label in specs}
    comparator = StringComparator.from_schema(FZRecord)
    candidates = _build_candidates(records, specs, comparator)

    # --- Stage 1: bootstrap -- the frontier teacher scores every pair, logged. ---
    # Default teacher = the deterministic $0 simulation; a caller (the paid FZ/AG
    # scripts) injects a REAL judge instead. An optional spend cap wraps the
    # OUTSIDE of the whole teacher (ONE shared monitor across the bootstrap batch
    # AND the cascade's per-pair escalations, since the same instance is reused as
    # the escalation tier) -- never a cascade tier, which would reset the budget
    # per pair (see _OuterSpendCap / CascadeMatcher's docstring).
    teacher = teacher if teacher is not None else SimulatedFrontierJudge(seed=seed)
    spend_cap = (
        _OuterSpendCap(teacher, budget_usd=spend_cap_usd) if spend_cap_usd is not None else None
    )
    if spend_cap is not None:
        teacher = spend_cap
    teacher_threshold = 0.5  # the teacher's natural cut (its logistic is centered near here)
    boot_log = JudgementLog(work_dir / "judgements.jsonl")
    boot_log.path.unlink(missing_ok=True)
    logged_teacher = LoggingMatcher(teacher, log=boot_log, threshold=teacher_threshold)
    teacher_judgements = list(logged_teacher.forward(iter(candidates)))
    teacher_by_pair = {_pair_key(j.left_id, j.right_id): j for j in teacher_judgements}
    judgement_rows = boot_log.read()

    # --- Stage 2: select the uncertain margin for human review. ---
    review_items = select_for_review(
        judgement_rows,
        strategy="uncertainty",
        threshold=teacher_threshold,
        margin=0.2,
        records=list(records.values()),
        limit=30,
        audit_fraction=0.1,
        seed=seed,
    )
    queue_path = work_dir / "queue.jsonl"
    ReviewQueue(queue_path).write(review_items)
    if verbose:
        print(f"\n[stage 2] wrote {len(review_items)} pairs to a review queue. Review them with:")
        print(f"    uv run langres review {queue_path}\n")

    # --- Stage 3: a simulated human answers the queue from gold. ---
    corrections_path = work_dir / "corrections.jsonl"
    corrections_path.unlink(missing_ok=True)
    correction_log = CorrectionLog(corrections_path)
    corrections: list[Correction] = []
    for item in review_items:
        correction = Correction(
            left_id=item.left_id,
            right_id=item.right_id,
            label=gold[_pair_key(item.left_id, item.right_id)],
            original_score=item.score,
            original_verdict=item.verdict,
            reviewer="simulated-human",
        )
        correction_log.append(correction)
        corrections.append(correction)

    # --- Stage 4: harvest -- BEFORE (silver-only, circular) then AFTER. ---
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        before_pairs = harvest_labeled_pairs(judgement_rows, corrections=[])
        before_threshold = derive_threshold_from_pairs(before_pairs)
    circularity_fired = any(issubclass(w.category, UserWarning) for w in caught)
    after_pairs = harvest_labeled_pairs(judgement_rows, corrections)
    label_by_pair = {_pair_key(p.left_id, p.right_id): p.label for p in after_pairs}

    # --- Stage 5: train the student; calibrate ITS threshold on ITS OWN scores. ---
    train_idx, heldout_idx = _seeded_split(len(candidates), seed)
    train_candidates = [candidates[i] for i in train_idx]
    train_labels = [label_by_pair[_pair_key(c.left.id, c.right.id)] for c in train_candidates]
    student: RandomForestMatcher[FZRecord] = RandomForestMatcher(
        feature_specs=comparator.feature_specs, random_state=seed
    )
    student.fit(iter(train_candidates), train_labels)

    heldout_candidates = [candidates[i] for i in heldout_idx]
    heldout_gold = [gold[_pair_key(c.left.id, c.right.id)] for c in heldout_candidates]
    student_judgements = list(student.forward(iter(heldout_candidates)))
    student_scores = [j.score for j in student_judgements]
    # NEVER the teacher's scores here -- prob_rf and prob_llm are different scales.
    student_threshold = derive_threshold(student_scores, heldout_gold, method="youden")

    # --- Stage 6: derive the escalation band from the student's own score spread. ---
    (band_low, band_high), band_fraction = derive_escalation_band(student_scores, student_threshold)
    if verbose:
        print(
            f"[stage 6] band derived around student threshold {student_threshold:.3f}: "
            f"[{band_low:.3f}, {band_high:.3f}] captures {band_fraction:.0%} of "
            "calibration pairs (target ~20%) -- not a magic constant."
        )
    cascade: CascadeMatcher[FZRecord] = CascadeMatcher(
        student=student, escalation=teacher, band=(band_low, band_high)
    )
    cascade_judgements = list(cascade.forward(iter(heldout_candidates)))

    # --- Stage 7: disagreement selection seeds the next queue (skips corrected pairs). ---
    student_rows = _log_rows(student_judgements, student_threshold)
    teacher_heldout = [
        teacher_by_pair[_pair_key(c.left.id, c.right.id)] for c in heldout_candidates
    ]
    teacher_rows = _log_rows(teacher_heldout, student_threshold)
    next_items = select_for_review(
        student_rows,
        strategy="disagreement",
        against=teacher_rows,
        corrections=corrections,
        records=list(records.values()),
        limit=30,
        audit_fraction=0.0,
        seed=seed,
    )
    next_queue_path = work_dir / "next_queue.jsonl"
    ReviewQueue(next_queue_path).write(next_items)
    corrected_keys = {_pair_key(c.left_id, c.right_id) for c in corrections}
    next_has_corrected = any(_pair_key(i.left_id, i.right_id) in corrected_keys for i in next_items)
    if verbose:
        if next_items:
            print(f"\n[stage 7] {len(next_items)} student/teacher disagreements to review next:")
            print(f"    uv run langres review {next_queue_path}\n")
        else:
            print(
                "\n[stage 7] no student/teacher disagreements left to review -- the loop is "
                "exhausted (the stop signal).\n"
            )

    # --- Stage 8: metrics -- one threshold cuts every stream, on the held-out split. ---
    teacher_metrics = _score_verdicts(
        teacher_heldout, student_threshold, gold, label="teacher (frontier)"
    )
    student_metrics = _score_verdicts(
        student_judgements, student_threshold, gold, label="student (cheap)"
    )
    cascade_metrics = _score_verdicts(cascade_judgements, student_threshold, gold, label="cascade")

    escalated = [j for j in cascade_judgements if j.decision_step == CASCADE_ESCALATED_STEP]
    escalated_correct = sum(
        1
        for j in escalated
        if (j.score >= student_threshold) == gold[_pair_key(j.left_id, j.right_id)]
    )
    escalation_rate = len(escalated) / len(cascade_judgements)
    reduction = 1.0 - escalation_rate
    dollars_saved = (len(cascade_judgements) - len(escalated)) * _FICTIONAL_COST_PER_CALL

    # Audit slice: a seeded governance sample; disagreement = cascade verdict vs gold.
    cascade_rows = _log_rows(cascade_judgements, student_threshold)
    audit_items = select_for_review(cascade_rows, strategy="audit", seed=seed, limit=15)
    audit_disagreements = sum(
        1 for i in audit_items if i.verdict != gold[_pair_key(i.left_id, i.right_id)]
    )
    audit_rate = audit_disagreements / len(audit_items) if audit_items else 0.0

    return ClosedLoopReport(
        n_records=len(records),
        n_candidates=len(candidates),
        n_train=len(train_candidates),
        n_heldout=len(heldout_candidates),
        n_review_items=len(review_items),
        n_corrections=len(corrections),
        circularity_warning_fired=circularity_fired,
        before_threshold=before_threshold,
        student_threshold=student_threshold,
        band_low=band_low,
        band_high=band_high,
        band_fraction=band_fraction,
        escalation_rate=escalation_rate,
        frontier_call_reduction=reduction,
        simulated_dollars_saved=dollars_saved,
        teacher_spend_usd=spend_cap.spent if spend_cap is not None else 0.0,
        teacher=teacher_metrics,
        student=student_metrics,
        cascade=cascade_metrics,
        n_escalated=len(escalated),
        escalated_correct=escalated_correct,
        audit_sample_size=len(audit_items),
        audit_disagreement_rate=audit_rate,
        next_queue_size=len(next_items),
        next_queue_has_corrected_pair=next_has_corrected,
    )


def format_report(report: ClosedLoopReport) -> str:
    """Render a human-readable summary of the closed-loop run."""

    def metric_row(m: PairMetrics) -> str:
        return (
            f"  {m.label:<20} F1={m.f1:.3f}  precision={m.precision:.3f}  "
            f"recall={m.recall:.3f}  (tp={m.tp} fp={m.fp} fn={m.fn})"
        )

    # A real paid run (teacher injected with a spend cap) metered real cost; the
    # default simulated run did not. Keep the banner/spend/footer honest either way
    # -- a script that prints "$0 / no API called" while spending money is a footgun.
    real = report.teacher_spend_usd > 0.0
    banner = (
        f"  REAL PAID RUN -- ${report.teacher_spend_usd:.4f} spent on a live frontier "
        "teacher (spend-capped)."
        if real
        else "  ALL SIMULATED / $0 -- the 'frontier' judge is a local stand-in; costs are fictional."
    )
    spend_line = (
        f"Teacher spend (REAL):     ${report.teacher_spend_usd:.4f} on live API calls (spend-capped)"
        if real
        else (
            f"Simulated dollars saved:  ${report.simulated_dollars_saved:.4f} "
            "(FICTIONAL -- no API was called)"
        )
    )
    # The "easy fixture" caveat is specific to THIS example's simulated FZ run; on a
    # real run the dataset-specific caveats live in the committed result doc instead.
    easy_note = (
        []
        if real
        else [
            "  (On this easy fixture the cheap student already resolves the held-out split, so",
            "   the cascade's win here is the frontier-call reduction below -- NOT an F1 gain;",
            "   the quality gain from escalation only shows up on harder data. See the paid runs.)",
            "",
        ]
    )
    footer = (
        [
            "  Measured on a live model -- these numbers are real, not simulated.",
            "  Dataset-specific caveats are in the committed result doc.",
        ]
        if real
        else [
            "  Plumbing demo -- favorable BY CONSTRUCTION. Real economics need a real model",
            "  on hard data; see the paid validation runs, never these simulated numbers.",
        ]
    )
    lines = [
        "=" * 82,
        "The data flywheel, closed: bootstrap -> review -> harvest -> train -> cascade",
        "=" * 82,
        banner,
        "",
        f"Dataset:   {report.n_records} records, {report.n_candidates} candidate pairs "
        f"({report.n_train} train / {report.n_heldout} held-out).",
        f"Review:    {report.n_review_items} uncertain pairs queued, "
        f"{report.n_corrections} answered from gold.",
        f"Harvest:   silver-only calibration warning fired = {report.circularity_warning_fired} "
        f"(BEFORE threshold={report.before_threshold:.3f}, circular).",
        "",
        f"Student threshold (calibrated on the student's OWN scores): "
        f"{report.student_threshold:.3f}",
        f"Escalation band (derived, not a magic constant): "
        f"[{report.band_low:.3f}, {report.band_high:.3f}] "
        f"-- captures {report.band_fraction:.0%} of calibration pairs.",
        "",
        "Pairwise quality vs gold on the held-out split (one threshold cuts every stream):",
        metric_row(report.teacher),
        metric_row(report.student),
        metric_row(report.cascade),
        "",
        *easy_note,
        f"Escalation rate:          {report.escalation_rate:.1%} "
        f"({report.n_escalated}/{report.n_heldout} held-out pairs escalated to the frontier)",
        f"Frontier-call reduction:  {report.frontier_call_reduction:.1%} "
        "vs judging every pair with the frontier",
        f"Escalated-pair accuracy:  {report.escalated_accuracy:.1%} "
        f"({report.escalated_correct}/{report.n_escalated} escalated verdicts correct)",
        spend_line,
        f"Audit-slice disagreement: {report.audit_disagreement_rate:.1%} "
        f"of a {report.audit_sample_size}-pair governance sample (cascade vs gold)",
        f"Next review queue:        {report.next_queue_size} student/teacher disagreements "
        f"(already-corrected pairs included: {report.next_queue_has_corrected_pair})",
        "=" * 82,
        *footer,
        "=" * 82,
    ]
    return "\n".join(lines)


def main() -> None:
    """Run the closed loop and print its report (verbose narration + summary)."""
    report = run_closed_loop(verbose=True)
    print(format_report(report))


if __name__ == "__main__":
    main()
