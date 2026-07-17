"""CascadeMatcher: a two-tier student -> escalation judge over the Matcher contract.

The flywheel's cost lever (T3): a cheap trained ``student`` scores every pair;
only pairs whose student score falls inside an uncertainty ``band`` are
escalated to an expensive ``escalation`` judge (typically a frontier LLM).
Unlike the deprecated :class:`~langres.core.matchers.cascade.CascadeChainMatcher`
(which hard-wires embeddings + an OpenAI client), CascadeMatcher composes ANY two
pairwise :class:`~langres.core.matcher.Matcher` instances and round-trips through
``Resolver.save``/``load`` via the component registry.

**Shared probability-scale contract:** both tiers must emit
probability-calibrated scores on a shared ``[0, 1]`` scale (e.g. ``prob_rf``,
``prob_llm``, ``calibrated_prob``) -- the ``band`` cuts *student* scores and a
single downstream :class:`~langres.core.clusterer.Clusterer`/verdict threshold
cuts the *mixed* student/escalation stream, so a raw similarity mixed with a
probability makes both cuts meaningless. The API cannot prevent misuse, so
:meth:`CascadeMatcher.forward` emits a one-time ``UserWarning`` when a child
judgement's ``score_type`` falls outside the known probability set.

Deliberately imports nothing from ``modules/cascade.py`` (that module pulls
litellm + sentence-transformers at import time; this one must stay eager-safe
for plain ``import langres`` -- see ``tests/test_import_budget.py``).
"""

import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar

from langres.core.models import ERCandidate, PairwiseJudgement
from langres.core.matcher import GroupwiseMatcher, Matcher, SchemaT
from langres.core.registry import get_component, register
from langres.core.reports import ScoreInspectionReport, _inspect_scores_impl
from langres.core.serialization import SerializableState
from langres.core.spend import BudgetExceeded

#: ``decision_step`` stamped on pairs the student answered (score outside the
#: band). The tier is rewritten into ``decision_step`` -- not only provenance --
#: because default :class:`~langres.tracking.judgement_log.JudgementLog` lines carry
#: ``decision_step`` but not ``provenance``, and disagreement selection (T2)
#: must be able to tell the tiers apart from the log alone.
CASCADE_STUDENT_STEP = "cascade_student"

#: ``decision_step`` stamped on pairs the escalation judge answered (student
#: score inside the band). See :data:`CASCADE_STUDENT_STEP` for why the tier
#: lives in ``decision_step``.
CASCADE_ESCALATED_STEP = "cascade_escalated"

#: ``score_type`` values understood as probability-calibrated on a [0, 1]
#: scale (see the module docstring's shared-scale contract). Anything else
#: (``sim_cos``, ``heuristic``) triggers the one-time misuse warning.
_PROBABILITY_SCORE_TYPES = frozenset(
    {"prob_llm", "prob_rf", "prob_fs", "prob_group_llm", "calibrated_prob"}
)

__all__ = ["CASCADE_ESCALATED_STEP", "CASCADE_STUDENT_STEP", "CascadeMatcher"]


@register("cascade_judge")
class CascadeMatcher(Matcher[SchemaT]):
    """Two-tier judge: a cheap student everywhere, escalation only in the band.

    The student scores each pair; a score in ``[low, high]`` (both edges
    INCLUSIVE) escalates that single pair to the escalation judge, whose
    judgement wins (score, score_type, reasoning) with
    ``decision_step=CASCADE_ESCALATED_STEP``. Everything else passes through
    as the student's judgement with ``decision_step=CASCADE_STUDENT_STEP``.
    Either way the child's original ``decision_step`` and the student score are
    preserved in ``provenance`` (the cascade keys MERGE INTO the child's
    provenance dict, never replace it -- ``provenance["cost_usd"]`` feeds the
    verbs' spend cap and ``provenance["model"]`` feeds the JudgementLog's
    ``model`` column, so both must survive escalation).

    Escalation is lazy and per-pair (``escalation.forward(iter([pair]))``):
    the escalation judge is only ever pulled for band pairs -- explicit over
    clever, and LLM judges are per-pair calls anyway, so no cost is lost.

    Both tiers must emit probability-calibrated scores on a shared ``[0, 1]``
    scale (e.g. ``prob_rf`` / ``prob_llm``): one ``band`` cuts student scores
    and one downstream threshold cuts the mixed output stream. A child
    judgement with a non-probability ``score_type`` triggers a one-time
    ``UserWarning`` per CascadeMatcher instance.

    Spend caps belong on the OUTSIDE of a cascade, not on a tier. A cap around
    the whole cascade meters BOTH tiers -- the cheap student's cost and the
    escalation's -- against one budget, which is what a caller means by "spend
    at most $X on this matcher"; a cap on the escalation child alone bounds only
    that tier and silently ignores the rest. ``langres.link``/``dedupe`` and
    :class:`~langres.core.resolver.Resolver` all apply their ``budget_usd`` cap
    around the whole resolved judge, which sees the escalated cost via
    ``provenance["cost_usd"]``; keep it there. Pass bare judges as the tiers.

    (Before B1 there was a sharper reason: the cap rebuilt its ledger on every
    ``forward()`` call, and ``forward`` runs the escalation judge once per band
    pair, so a tier-level cap reset its budget every pair and bounded nothing.
    :class:`~langres.core.spend_cap.SpendCappedMatcher`'s ledger is per instance
    now, so a tier-level cap does at least accumulate -- it is merely the wrong
    scope, no longer inert.)

    Serialization mirrors :class:`~langres.core.blockers.composite.CompositeBlocker`:
    children serialize as ``{"type_name", "config"}`` registry specs, and
    out-of-band child state (e.g. a fitted
    :class:`~langres.core.matchers.random_forest_judge.RandomForestMatcher` forest) persists via
    :class:`~langres.core.serialization.SerializableState` into per-child
    subdirectories (``<state_dir>/student``, ``<state_dir>/escalation``) --
    so a fitted student survives ``Resolver.save``/``load`` with zero Resolver
    changes.

    Args:
        student: The cheap pairwise judge run on every pair.
        escalation: The expensive pairwise judge run only on band pairs.
        band: ``(low, high)`` uncertainty band over *student* scores, with
            ``0 <= low < high <= 1``; both edges escalate (inclusive).

    Raises:
        ValueError: If ``band`` is not ``0 <= low < high <= 1``, or a child is
            a :class:`~langres.core.matcher.GroupwiseMatcher` (its group-call
            cost contract is incompatible with per-pair escalation -- pass a
            pairwise ``Matcher``, or keep the group-wise judge as a standalone
            tier outside the cascade).

    Example:
        >>> student = RandomForestMatcher(feature_specs=comparator.feature_specs)
        >>> student.fit(iter(train_candidates), train_labels)   # prob_rf scores
        >>> frontier = LLMMatcher(client=client, model="gpt-4o")  # prob_llm scores
        >>> judge = CascadeMatcher(student=student, escalation=frontier, band=(0.35, 0.65))
        >>> judgements = list(judge.forward(iter(candidates)))
        >>> # cheap student everywhere; the frontier only inside the band:
        >>> clusters = Clusterer(threshold=0.5).cluster(judgements)
    """

    type_name: ClassVar[str] = "cascade_judge"

    def __init__(
        self,
        student: Matcher[SchemaT],
        escalation: Matcher[SchemaT],
        *,
        band: tuple[float, float],
    ) -> None:
        low, high = band
        if not 0.0 <= low < high <= 1.0:
            raise ValueError(
                f"band must satisfy 0 <= low < high <= 1, got ({low}, {high}). "
                "The band is an uncertainty interval over the student's [0, 1] "
                "probability scores; both edges are inclusive."
            )
        for slot, child in (("student", student), ("escalation", escalation)):
            if isinstance(child, GroupwiseMatcher):
                raise ValueError(
                    f"CascadeMatcher does not accept a GroupwiseMatcher as its {slot} "
                    f"(got {type(child).__name__}): escalation is per-pair, which "
                    "conflicts with the group-call cost contract (one call priced "
                    "across a whole group, stamp_group_cost). Pass a pairwise "
                    "Matcher instead, or run the group-wise judge standalone."
                )
        self.student = student
        self.escalation = escalation
        self.band = (low, high)
        self._score_type_warned = False

    # ------------------------------------------------------------------
    # Scoring (Matcher)
    # ------------------------------------------------------------------

    def forward(self, candidates: Iterator[ERCandidate[SchemaT]]) -> Iterator[PairwiseJudgement]:
        """Score each pair with the student; escalate band pairs to the escalation judge.

        Yields:
            One PairwiseJudgement per candidate: the escalation judgement
            (``decision_step=CASCADE_ESCALATED_STEP``) for pairs whose student
            score (or, absent a score, ``confidence``) falls inside the band
            (inclusive), and for pairs where the student *abstained*
            (``is_abstain`` -- neither a decision nor a score), since an
            abstention is maximally uncertain. Otherwise the student judgement
            is trusted (``decision_step=CASCADE_STUDENT_STEP``) -- including a
            binary decider that confidently decided with no score and no
            confidence, which is trusted rather than escalated. Cascade
            provenance keys (``cascade_tier``, ``student_score``, the inner
            ``decision_step`` values) merge into the winning child's provenance.

        Raises:
            RuntimeError: If a child judge yields anything but exactly one
                judgement for a single candidate pair (contract violation).
            BudgetExceeded: Propagated from a spend-capped escalation child,
                re-raised with ``partial_judgements`` reset to the cascade's
                own full produced list (everything already yielded plus the
                paid, in-flight escalation judgements in cascade form) --
                otherwise an outer ``LoggingMatcher``'s ``[logged:]`` slice
                would drop exactly the paid judgements.
            Exception: Any OTHER exception raised by the escalation child
                (e.g. a network error, rate limit, or API failure) propagates
                unchanged, WITHOUT partial-judgement preservation -- only
                ``BudgetExceeded`` is langres's own exception and carries the
                ``partial_judgements`` field the ``LoggingMatcher`` slice needs.
        """
        low, high = self.band
        produced: list[PairwiseJudgement] = []
        for candidate in candidates:
            student_judgement = self._one(
                list(self.student.forward(iter([candidate]))), tier="student"
            )
            self._check_score_type(student_judgement)
            # The band is applied to the student's confidence-ordered value: its
            # ``score`` if it ranked, else its ``confidence``. A student that
            # abstained (is_abstain) is maximally uncertain -> escalate. But a
            # student that *confidently decided* with no score and no confidence
            # (a binary decider: decision set, score=None, confidence=None) has
            # band_value=None yet is NOT uncertain -- trust its decision rather
            # than escalate every pair and erase the cascade's cost savings.
            band_value = (
                student_judgement.score
                if student_judgement.score is not None
                else student_judgement.confidence
            )
            if student_judgement.is_abstain or (
                band_value is not None and low <= band_value <= high
            ):
                try:
                    raw = list(self.escalation.forward(iter([candidate])))
                except BudgetExceeded as exc:
                    # The child's partial_judgements only cover ITS OWN forward
                    # call, while an outer LoggingMatcher counts everything the
                    # cascade yielded -- its `partial_judgements[logged:]` slice
                    # (judgement_log.py) would come up empty and silently drop
                    # the paid judgements. Reset to the cascade's full produced
                    # list: everything yielded so far plus the in-flight paid
                    # judgements, rewritten to their escalated cascade form so
                    # the log rows keep cost_usd/model AND the tier step.
                    exc.partial_judgements = produced + [
                        self._escalated(paid, student_judgement) for paid in exc.partial_judgements
                    ]
                    raise
                escalation_judgement = self._one(raw, tier="escalation")
                self._check_score_type(escalation_judgement)
                judgement = self._escalated(escalation_judgement, student_judgement)
            else:
                judgement = student_judgement.model_copy(
                    update={
                        "decision_step": CASCADE_STUDENT_STEP,
                        "provenance": {
                            **student_judgement.provenance,
                            "cascade_tier": "student",
                            "student_score": student_judgement.score,
                            "student_decision_step": student_judgement.decision_step,
                        },
                    }
                )
            produced.append(judgement)
            yield judgement

    @staticmethod
    def _one(judgements: list[PairwiseJudgement], *, tier: str) -> PairwiseJudgement:
        """Enforce the exactly-one-judgement-per-pair contract (mirrors verbs.link)."""
        if len(judgements) != 1:
            raise RuntimeError(
                f"the cascade's {tier} judge produced {len(judgements)} judgements "
                "for a single candidate pair; every candidate must yield exactly "
                f"one PairwiseJudgement. This indicates a bug in the injected "
                f"{tier} Matcher."
            )
        return judgements[0]

    @staticmethod
    def _escalated(
        escalation_judgement: PairwiseJudgement, student_judgement: PairwiseJudgement
    ) -> PairwiseJudgement:
        """The escalation judgement wins, with cascade keys MERGED INTO its provenance.

        Merging (never replacing) keeps the child's ``cost_usd`` (feeds
        ``_SpendCappedMatcher``) and ``model`` (feeds the JudgementLog ``model``
        column) intact on escalated pairs.
        """
        return escalation_judgement.model_copy(
            update={
                "decision_step": CASCADE_ESCALATED_STEP,
                "provenance": {
                    **escalation_judgement.provenance,
                    "cascade_tier": "escalated",
                    "student_score": student_judgement.score,
                    "student_decision_step": student_judgement.decision_step,
                    "escalation_decision_step": escalation_judgement.decision_step,
                },
            }
        )

    def _check_score_type(self, judgement: PairwiseJudgement) -> None:
        """One-time soft enforcement of the shared probability-scale contract."""
        if self._score_type_warned or judgement.score_type in _PROBABILITY_SCORE_TYPES:
            return
        self._score_type_warned = True
        warnings.warn(
            f"CascadeMatcher received a child judgement with score_type="
            f"{judgement.score_type!r}, outside the known probability set "
            f"{sorted(_PROBABILITY_SCORE_TYPES)}. Both tiers must emit "
            "probability-calibrated scores on a shared [0, 1] scale: the band "
            "cuts student scores and one downstream threshold cuts the mixed "
            "student/escalation stream, so a raw similarity (e.g. 'sim_cos', "
            "'heuristic') mixed in makes those cuts meaningless. Calibrate the "
            "child's scores (e.g. train a RandomForestMatcher, or map similarities through "
            "derive_threshold-calibrated probabilities) before cascading.",
            stacklevel=2,
        )

    def inspect_scores(
        self, judgements: list[PairwiseJudgement], sample_size: int = 10
    ) -> ScoreInspectionReport:
        """Explore scores without ground truth (shared Matcher utility)."""
        return _inspect_scores_impl(judgements, sample_size)

    # ------------------------------------------------------------------
    # Serialization (registry config + SerializableState sidecars)
    # ------------------------------------------------------------------

    @property
    def config(self) -> dict[str, object]:
        """Serializable construction config for the registry.

        Returns:
            ``{"band": [low, high], "student": {"type_name", "config"},
            "escalation": {"type_name", "config"}}`` (children serialized the
            same way ``CompositeBlocker`` serializes its children).

        Raises:
            ValueError: If a child has no registry ``type_name``.
        """
        return {
            "band": [self.band[0], self.band[1]],
            "student": self._child_spec("student", self.student),
            "escalation": self._child_spec("escalation", self.escalation),
        }

    @staticmethod
    def _child_spec(slot: str, child: Matcher[Any]) -> dict[str, object]:
        child_type_name = getattr(child, "type_name", None)
        if child_type_name is None:
            raise ValueError(
                f"CascadeMatcher {slot} {child!r} has no registry 'type_name'; "
                "construct with a registered Matcher subclass to persist."
            )
        child_config: Any = child.config  # type: ignore[attr-defined]
        return {"type_name": child_type_name, "config": child_config}

    @classmethod
    def from_config(cls, config: dict[str, object]) -> "CascadeMatcher[SchemaT]":
        """Rebuild a CascadeMatcher from its serialized config.

        Children are rebuilt through the component registry by ``type_name``
        (a fitted child's out-of-band state is restored separately via
        :meth:`load_state`).
        """
        band_raw: Any = config["band"]
        return cls(
            student=cls._child_from_spec(config["student"]),
            escalation=cls._child_from_spec(config["escalation"]),
            band=(float(band_raw[0]), float(band_raw[1])),
        )

    @staticmethod
    def _child_from_spec(spec: Any) -> Matcher[Any]:
        child_cls: Any = get_component(str(spec["type_name"]))
        child: Matcher[Any] = child_cls.from_config(spec["config"])
        return child

    def save_state(self, state_dir: Path) -> None:
        """Persist each stateful child's out-of-band state into a named subdir.

        Delegates to any child implementing
        :class:`~langres.core.serialization.SerializableState`
        (``<state_dir>/student``, ``<state_dir>/escalation``). A stateful
        child with nothing to persist yet (e.g. an unfit ``RandomForestMatcher``) writes
        no files; its empty subdir is dropped so :meth:`load_state` never
        tries to read a missing state file (mirrors ``Resolver.save``'s
        empty-sidecar handling).
        """
        for slot, child in (("student", self.student), ("escalation", self.escalation)):
            if not isinstance(child, SerializableState):
                continue
            child_dir = state_dir / slot
            child_dir.mkdir(parents=True, exist_ok=True)
            child.save_state(child_dir)
            if not any(child_dir.iterdir()):
                child_dir.rmdir()

    def load_state(self, state_dir: Path) -> None:
        """Restore child state previously written by :meth:`save_state`.

        Tolerates stateless children and absent/empty subdirs: only a child
        that implements ``SerializableState`` AND has a populated subdir gets
        ``load_state`` called (an absent subdir means the child had nothing to
        persist -- e.g. it was never fitted).
        """
        for slot, child in (("student", self.student), ("escalation", self.escalation)):
            child_dir = state_dir / slot
            if (
                isinstance(child, SerializableState)
                and child_dir.is_dir()
                and any(child_dir.iterdir())
            ):
                child.load_state(child_dir)
