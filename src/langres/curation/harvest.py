"""The flywheel's harvest half: logged verdicts + human corrections -> labeled pairs.

``JudgementLog`` (W0.2, :mod:`langres.tracking.judgement_log`) is the flywheel's
*inlet* -- an opt-in JSONL log of every judge call (ids, score, decision, verdict,
model, cost, ``"v": 3``). This module is the *outlet*: it turns those logged
judgements,
plus a review tool's human corrections, into **labeled pairs** that feed
:func:`langres.training.calibration.derive_threshold` (its first production caller)
and, where applicable, a judge's ``fit()``.

The division of labor is deliberate: langres owns the **contract**, the
**harvest**, and the headless + terminal review surfaces (the ``langres review``
CLI and its CSV round-trip); anything with a rendering loop (a web review UI)
stays in the downstream application. :class:`Correction` is the
stable line schema a review tool writes to ``corrections.jsonl``; :class:`CorrectionLog`
is the reference reader/writer for that file (mirroring ``JudgementLog`` so the
two flywheel files are handled the same way); :func:`harvest_labeled_pairs` is
the merge -- verdicts as weak labels, corrections overriding them where a human
has reviewed the pair.

Weak-label provenance survives onto every :class:`LabeledPair` via ``source``
(``"verdict"`` vs ``"correction"``), so a caller can weight, filter, or audit the
human-reviewed subset instead of trusting all labels equally.

Positioning: like :mod:`langres.training.calibration`, this is eval/calibration-tier
code, not part of the ``link()``/``dedupe()`` runtime. Importing the module is
cheap (Pydantic only); the one function that needs scikit-learn,
:func:`derive_threshold_from_pairs`, imports it lazily so the *contract* models
(:class:`Correction`, :class:`LabeledPair`) a review tool depends on never pull a
heavy dependency. Held-out evaluation of a derived threshold is intentionally
*not* here -- it belongs to :mod:`langres.core.metrics` (``classify_pairs``),
which pulls ``ranx``; keeping harvest light means emitting a ``corrections.jsonl``
never drags eval tooling in.
"""

from __future__ import annotations

import json
import logging
import random
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar

from pydantic import BaseModel

if TYPE_CHECKING:
    from langres.training.calibration import ThresholdMethod
    from langres.core.models import ERCandidate

__all__ = [
    "AlignedPairs",
    "AlignedSplit",
    "Correction",
    "CorrectionLog",
    "GoldCoverage",
    "LabeledPair",
    "PairLabel",
    "align_pairs",
    "derive_threshold_from_pairs",
    "harvest_labeled_pairs",
]

#: Schema type variable for the entity schema an ``ERCandidate`` carries. Defined
#: locally to keep this module import-light (no ``models`` import at runtime).
SchemaT = TypeVar("SchemaT", bound=BaseModel)

logger = logging.getLogger(__name__)

#: Schema-version tag written into every ``corrections.jsonl`` line -- mirrors
#: ``JudgementLog``'s ``"v": 1`` so a future format migration can branch on it
#: instead of guessing.
_CORRECTION_SCHEMA_VERSION = 1


class Correction(BaseModel):
    """One human verdict-correction for a judged pair: the ``corrections.jsonl`` line.

    This is the stable contract an external review queue writes
    when a reviewer overrides a judge's verdict. Only ``left_id``/``right_id`` and
    the corrected ``label`` are required; the rest is optional audit context. A
    pair is identified order-independently on harvest (see
    :func:`harvest_labeled_pairs`), so which record is ``left`` need not match the
    judgement log's ordering.

    Attributes:
        v: Schema-version tag (mirrors ``JudgementLog``'s ``"v"``). Default ``1``.
        left_id: Identifier of one entity in the corrected pair.
        right_id: Identifier of the other entity in the corrected pair.
        label: The reviewer's corrected verdict -- ``True`` for a match,
            ``False`` for a non-match. This overrides the judge's logged verdict.
        original_score: The judge's score for this pair, if the review tool
            carried it through (audit only; harvest reads the score from the
            judgement log, not from here).
        original_verdict: The judge's verdict being overridden, if recorded
            (audit only).
        reviewer: Who made the correction, if recorded (audit only).
        timestamp: ISO-8601 timestamp of the correction, if recorded (audit only).
    """

    v: int = _CORRECTION_SCHEMA_VERSION
    left_id: str
    right_id: str
    label: bool
    original_score: float | None = None
    original_verdict: bool | None = None
    reviewer: str | None = None
    timestamp: str | None = None


class LabeledPair(BaseModel):
    """A harvested labeled pair: a score plus a label ready for calibration/fit.

    The output unit of :func:`harvest_labeled_pairs`. ``score`` comes from the
    judgement log; ``label`` is the judge's verdict unless a human correction
    overrode it, which ``source`` records so a caller can weight or audit the
    human-reviewed subset.

    Attributes:
        left_id: Identifier of the left entity (as logged by the judge).
        right_id: Identifier of the right entity (as logged by the judge).
        score: The judge's score for the pair (from the judgement log), or
            ``None`` for a decision-only judge that logged no score. The label is
            still usable, but a score-less pair cannot contribute to a *score*
            threshold -- :func:`derive_threshold_from_pairs` rejects such input
            rather than calibrating on the self-selected scored subset.
        label: The weak/corrected match label -- ``True`` for a match.
        source: Where ``label`` came from -- ``"verdict"`` (the judge's logged
            verdict, a weak label) or ``"correction"`` (a human override).
    """

    left_id: str
    right_id: str
    score: float | None
    label: bool
    source: Literal["verdict", "correction"]


class CorrectionLog:
    """JSONL-file-backed reader/writer for ``corrections.jsonl`` -- the harvest inlet.

    The reference implementation of the :class:`Correction` file contract,
    mirroring :class:`~langres.tracking.judgement_log.JudgementLog` so both flywheel
    files are appended and reloaded the same way. A downstream review tool may
    write the file itself (any JSONL of :class:`Correction`-shaped lines is
    valid); this class is what langres-side tooling and tests use to produce and
    consume it.

    Args:
        path: The ``corrections.jsonl`` file to append to / read from. Parent
            directories are created on first :meth:`append` if missing.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, correction: Correction) -> None:
        """Append one JSON line for ``correction`` (creating parent dirs if needed)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(correction.model_dump()) + "\n")

    def read(self) -> list[Correction]:
        """Reload every correction written so far, in write order.

        Returns ``[]`` if the file was never created (no corrections yet). Blank
        lines are skipped; each non-blank line is validated into a
        :class:`Correction`.
        """
        if not self.path.exists():
            return []
        corrections: list[Correction] = []
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if stripped:
                    corrections.append(Correction.model_validate(json.loads(stripped)))
        return corrections


def harvest_labeled_pairs(
    judgement_rows: Sequence[Mapping[str, Any]],
    corrections: Sequence[Correction],
) -> list[LabeledPair]:
    """Merge logged judgements with human corrections into labeled pairs.

    Each judgement row (a ``JudgementLog``-format dict, e.g. from
    :meth:`JudgementLog.read <langres.tracking.judgement_log.JudgementLog.read>`)
    yields one :class:`LabeledPair`. Its label is the logged ``verdict`` (a weak
    label) unless a :class:`Correction` covers the same pair, in which case the
    human ``label`` overrides it and ``source`` is ``"correction"``. Pairs are
    matched order-independently by ``frozenset({left_id, right_id})``, so a
    correction need not repeat the log's left/right ordering.

    Corrections referencing a pair that appears in no judgement row are skipped
    with a warning: a correction carries no score, so without a logged score for
    the pair it cannot become a calibration example.

    A decision-only row (``score: null`` -- a judge that decided but did not
    rank) yields a :class:`LabeledPair` with ``score=None``: the label is still
    usable, but the pair carries no score for a *score* threshold, so
    :func:`derive_threshold_from_pairs` rejects it rather than dropping it into a
    biased scored-only subset. The score is carried as-is (never coerced to
    ``0.0``), so a genuine ``score == 0.0`` and a missing score stay distinct.

    An *abstention* row (``verdict: null`` -- the judge neither decided nor
    scored) carries no usable label and is **skipped** unless a human correction
    supplies one. Coercing a null verdict to a ``False`` non-match would seed
    silver-label training with a label the judge never gave -- the label-side
    twin of never coercing a null score to ``0.0``. So the output has one
    :class:`LabeledPair` per *labeled* row, not necessarily per input row.

    Args:
        judgement_rows: Logged judge calls as mappings with at least ``left_id``,
            ``right_id``, ``score`` and ``verdict`` keys. ``score`` may be
            ``None`` for a decision-only judge.
        corrections: Human corrections to overlay. Later corrections for the same
            pair win (last-write-wins).

    Returns:
        One :class:`LabeledPair` per *labeled* judgement row, in row order. An
        abstention row (``verdict`` null) with no correction carries no usable
        label and is omitted, so the output may be shorter than the input.
    """
    corrections_by_pair: dict[frozenset[str], Correction] = {}
    for correction in corrections:
        key = frozenset({correction.left_id, correction.right_id})
        if key in corrections_by_pair:
            logger.debug("Duplicate correction for pair %s; last one wins", tuple(key))
        corrections_by_pair[key] = correction

    matched: set[frozenset[str]] = set()
    pairs: list[LabeledPair] = []
    for row in judgement_rows:
        left_id = str(row["left_id"])
        right_id = str(row["right_id"])
        key = frozenset({left_id, right_id})
        override = corrections_by_pair.get(key)
        if override is not None:
            matched.add(key)
            label = override.label
            source: Literal["verdict", "correction"] = "correction"
        else:
            verdict = row["verdict"]
            if verdict is None:
                # An abstention (decision=None -> verdict=None): the judge gave
                # no verdict, so there is no label to harvest. Skip it rather
                # than coerce None to a False non-match -- a fabricated
                # "not a match" would poison silver labels exactly as a
                # fabricated 0.0 would poison a score threshold.
                continue
            label = bool(verdict)
            source = "verdict"
        raw_score = row["score"]
        pairs.append(
            LabeledPair(
                left_id=left_id,
                right_id=right_id,
                # Carry a decision-only row's null score as None (never coerce to
                # 0.0): the label is usable, but calibration must see the score is
                # absent, not a real 0.0.
                score=None if raw_score is None else float(raw_score),
                label=label,
                source=source,
            )
        )

    unmatched = len(corrections_by_pair) - len(matched)
    if unmatched:
        logger.warning(
            "%d correction(s) reference a pair absent from the judgement log and "
            "were skipped (no logged score to attach a label to).",
            unmatched,
        )
    return pairs


def derive_threshold_from_pairs(
    pairs: Sequence[LabeledPair],
    *,
    method: ThresholdMethod = "youden",
    percentile: float | None = None,
) -> float:
    """Derive a decision threshold from harvested labeled pairs.

    The flywheel's payoff and :func:`~langres.training.calibration.derive_threshold`'s
    first production caller: it reads the score distribution and labels straight
    off :func:`harvest_labeled_pairs`' output and returns a data-driven cut,
    replacing a hand-set constant.

    Calibrating on silver labels alone is circular: the judge's own verdicts
    were produced *by* a cut, so they are the only signal a threshold search
    can recover. A :class:`UserWarning` flags that case -- overlay human
    corrections (``source="correction"``) before calibrating. (Training a
    *different* model on silver labels is legitimate, which is why
    :func:`harvest_labeled_pairs` itself stays warning-free.)

    Args:
        pairs: Harvested labeled pairs (from :func:`harvest_labeled_pairs`).
        method: Passed through to
            :func:`~langres.training.calibration.derive_threshold`.
        percentile: Passed through (required only for ``method="percentile"``).

    Returns:
        The derived threshold as a plain ``float``.

    Raises:
        ValueError: If any pair has ``score is None`` (a decision-only judge has
            no scores to derive a *score* threshold from) -- raised here, naming
            the offending pair, rather than dropping the score-less pairs into a
            biased scored-only subset. Also propagated from
            :func:`~langres.training.calibration.derive_threshold` (empty input,
            single-class labels under ``"youden"``, bad ``percentile``, ...).

    Warns:
        UserWarning: If ``pairs`` is non-empty and every label is silver
            (``source == "verdict"``) -- silver-only calibration is circular
            (see above). Suppress deliberately via :mod:`warnings` filters.
    """
    # A score-less pair (decision-only judge) has no score to calibrate on;
    # collect the scores while checking, so mypy sees a list[float] below.
    scores: list[float] = []
    for pair in pairs:
        if pair.score is None:
            raise ValueError(
                "cannot derive a score threshold: pair "
                f"{pair.left_id}/{pair.right_id} has no score (a decision-only "
                "judge logged score=null); a decision-only judge has no scores to "
                "derive a score threshold from. Drop these pairs or calibrate on "
                "a scoring judge's output."
            )
        scores.append(pair.score)

    if pairs and all(pair.source == "verdict" for pair in pairs):
        warnings.warn(
            "silver-only calibration is circular -- deriving a threshold from "
            "a judge's own verdicts can only recover the cut that produced "
            "them; overlay human corrections (source='correction') before "
            "calibrating. (Training a DIFFERENT model on silver labels is "
            "fine.)",
            UserWarning,
            stacklevel=2,
        )

    from langres.training.calibration import derive_threshold

    return derive_threshold(
        scores,
        [pair.label for pair in pairs],
        method=method,
        percentile=percentile,
    )


# ---------------------------------------------------------------------------
# align_pairs: the id-join bridge from labeled pairs to fit-ready candidates
# ---------------------------------------------------------------------------

#: Public spelling of the pair-label record. :class:`LabeledPair` already IS the
#: pair-label schema (a score + a label + provenance), so ``PairLabel`` is a
#: thin alias rather than a forked, duplicated model -- callers/docs may prefer
#: the ``PairLabel`` name without a second class to keep in sync.
PairLabel = LabeledPair


class GoldCoverage(BaseModel):
    """How many labeled *positive* pairs survived blocking (the honest-numbers guardrail).

    A positive label whose pair the blocker never proposed is silently
    unrecoverable downstream -- no matcher can match a pair it never sees. This
    model makes that leak visible instead of letting held-out metrics quietly
    absorb it.

    Attributes:
        gold_coverage: Fraction of labeled positive pairs that appeared among the
            candidates -- blocking pair-completeness restricted to the labeled
            positives (from :func:`~langres.core.metrics.evaluate_blocking`, not
            reimplemented). ``1.0`` when there are no positive labels (nothing to
            miss).
        dropped_positives: The lexicographically-ordered ``(left_id, right_id)``
            id-pairs of positive labels with no matching candidate -- the pairs
            blocking dropped. ``len(dropped_positives)`` is the dropped count.
        n_labeled: Distinct labeled pairs after order-independent de-duplication.
        n_aligned: Labeled pairs that matched a candidate (the fit-set size).
        n_positive_labels: Positive labels among ``n_labeled`` (the coverage
            denominator).
    """

    gold_coverage: float
    dropped_positives: list[tuple[str, str]]
    n_labeled: int
    n_aligned: int
    n_positive_labels: int


@dataclass(frozen=True)
class AlignedSplit(Generic[SchemaT]):
    """One split's positionally-aligned candidates and their boolean labels.

    ``candidates[i]`` and ``labels[i]`` describe the same pair, ready to hand to
    :meth:`SupervisedFitMixin.fit(candidates, labels)
    <langres.core.fit.SupervisedFitMixin>`.
    """

    candidates: list[ERCandidate[SchemaT]]
    labels: list[bool]


@dataclass(frozen=True)
class AlignedPairs(Generic[SchemaT]):
    """Result of :func:`align_pairs`: fit-ready train/valid splits + coverage.

    A small named result (not a bare 4-tuple) so downstream code reads
    ``aligned.train.candidates`` / ``aligned.coverage.gold_coverage`` instead of
    positional unpacking.

    Attributes:
        train: The training split (all labeled candidates when ``split is None``).
        valid: The held-out split (empty when ``split is None``); entity-disjoint
            from ``train`` so no entity id appears in both.
        coverage: The :class:`GoldCoverage` guardrail for the whole label set.
    """

    train: AlignedSplit[SchemaT]
    valid: AlignedSplit[SchemaT]
    coverage: GoldCoverage

    @property
    def labels(self) -> list[bool]:
        """The training split's labels -- convenience for the common no-split case."""
        return self.train.labels


class _UnionFind:
    """Minimal union-find over string ids for the entity-disjoint split."""

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        """Root of ``x``'s component, with path compression."""
        self._parent.setdefault(x, x)
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        node = x
        while self._parent[node] != root:
            self._parent[node], node = root, self._parent[node]
        return root

    def union(self, a: str, b: str) -> None:
        """Merge the components of ``a`` and ``b``."""
        self._parent[self.find(a)] = self.find(b)


def _labels_to_map(
    labels: str | Path | Sequence[LabeledPair] | Sequence[Correction],
) -> dict[frozenset[str], bool]:
    """Normalize any labels input to ``{frozenset({left_id, right_id}): label}``.

    A ``corrections.jsonl`` path is read via :class:`CorrectionLog`. Pairs are
    keyed order-independently (mirroring :func:`harvest_labeled_pairs`), and a
    later label for the same pair wins (last-write-wins) -- so duplicate and
    conflicting labels collapse deterministically to the last one seen.
    """
    resolved: Sequence[LabeledPair] | Sequence[Correction]
    if isinstance(labels, (str, Path)):
        resolved = CorrectionLog(labels).read()
    else:
        resolved = labels
    label_map: dict[frozenset[str], bool] = {}
    for item in resolved:
        label_map[frozenset({str(item.left_id), str(item.right_id)})] = bool(item.label)
    return label_map


def _candidate_key(candidate: ERCandidate[Any]) -> frozenset[str]:
    """Order-independent id key for a candidate (mirrors the label key)."""
    return frozenset({str(candidate.left.id), str(candidate.right.id)})


def _gold_coverage(
    candidates: list[ERCandidate[Any]],
    label_map: dict[frozenset[str], bool],
    candidate_keys: set[frozenset[str]],
) -> GoldCoverage:
    """Build the :class:`GoldCoverage` guardrail, reusing ``evaluate_blocking``."""
    positive_keys = {key for key, label in label_map.items() if label and len(key) == 2}

    dropped: list[tuple[str, str]] = []
    for key in positive_keys - candidate_keys:
        ordered = sorted(key)
        dropped.append((ordered[0], ordered[1]))
    dropped.sort()

    if positive_keys:
        # Reuse evaluate_blocking (do NOT reimplement pair-completeness): each
        # positive pair is its own 2-id cluster, so candidate_recall is exactly
        # the fraction of labeled positive pairs captured by blocking -- with no
        # transitive closure fabricating pairs that were never labeled.
        from langres.core.metrics import evaluate_blocking

        gold_clusters = [set(key) for key in positive_keys]
        gold_coverage = evaluate_blocking(candidates, gold_clusters).candidate_recall
    else:
        gold_coverage = 1.0  # No positives -> nothing to miss.

    return GoldCoverage(
        gold_coverage=gold_coverage,
        dropped_positives=dropped,
        n_labeled=len(label_map),
        n_aligned=sum(1 for key in label_map if key in candidate_keys),
        n_positive_labels=len(positive_keys),
    )


def _entity_disjoint_split(
    aligned: list[tuple[ERCandidate[Any], bool]],
    *,
    split: float | None,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Partition aligned indices into ``(train, valid)`` with NO entity in both.

    Groups aligned pairs into connected entity-components (union-find over the
    two ids each pair touches), then assigns whole components to valid -- in a
    seed-shuffled order -- until about ``split`` of the pairs are held out. A
    row-random split would leak an entity across the boundary and inflate
    held-out metrics; assigning whole components cannot. Indices within each
    returned split keep their original candidate order.
    """
    n = len(aligned)
    if split is None or n == 0:
        return list(range(n)), []

    uf = _UnionFind()
    for candidate, _ in aligned:
        uf.union(str(candidate.left.id), str(candidate.right.id))

    components: dict[str, list[int]] = {}
    for index, (candidate, _) in enumerate(aligned):
        components.setdefault(uf.find(str(candidate.left.id)), []).append(index)

    # Deterministic base order (by smallest member index), then a seeded shuffle
    # so the split is reproducible but not coupled to component discovery order.
    component_lists = sorted(components.values(), key=min)
    random.Random(seed).shuffle(component_lists)

    target_valid = round(split * n)
    valid_set: set[int] = set()
    for members in component_lists:
        if len(valid_set) >= target_valid:
            break
        # Never empty train: skip a component that would swallow EVERY pair. A
        # single all-connected component cannot be split entity-disjointly, so
        # the honest outcome is an empty valid (train keeps everything), not an
        # empty train. With >1 component this still fills valid toward target.
        if len(valid_set) + len(members) >= n:
            continue
        valid_set.update(members)

    train_indices = [i for i in range(n) if i not in valid_set]
    valid_indices = [i for i in range(n) if i in valid_set]
    return train_indices, valid_indices


def align_pairs(
    candidates: Iterable[ERCandidate[SchemaT]],
    labels: str | Path | Sequence[LabeledPair] | Sequence[Correction],
    *,
    split: float | None = None,
    seed: int = 0,
) -> AlignedPairs[SchemaT]:
    """Join labeled pairs to blocked candidates, split entity-disjointly, report coverage.

    The id-join bridge between the two halves of a supervised fit: raw labels
    (by id, in any left/right order) on one side, the blocker's candidate stream
    on the other. Each candidate whose ``{left_id, right_id}`` matches a label is
    emitted with that label, positionally aligned for
    :meth:`SupervisedFitMixin.fit <langres.core.fit.SupervisedFitMixin>`.

    Args:
        candidates: The blocked (and, if a comparator is configured,
            comparison-attached) candidate stream to align labels onto. Consumed
            once and materialized.
        labels: A ``corrections.jsonl`` path (``str``/``Path``), or an in-memory
            ``Sequence`` of :class:`LabeledPair`/:class:`Correction`. Pairs are
            keyed order-independently; a later label for the same pair wins.
        split: ``None`` (default) puts every labeled candidate in ``train`` and
            leaves ``valid`` empty. Otherwise the held-out fraction
            (``0 < split < 1``): whole entity-components are assigned to ``valid``
            until about ``split`` of the labeled pairs are held out.
        seed: Seed for the deterministic component shuffle.

    Returns:
        An :class:`AlignedPairs` with ``train``/``valid`` splits and a
        :class:`GoldCoverage` guardrail.

    Raises:
        ValueError: If ``split`` is given but not in the open interval ``(0, 1)``.
    """
    if split is not None and not 0.0 < split < 1.0:
        raise ValueError(f"split must be in the open interval (0, 1) or None; got {split!r}")

    label_map = _labels_to_map(labels)
    materialized = list(candidates)

    aligned: list[tuple[ERCandidate[SchemaT], bool]] = []
    candidate_keys: set[frozenset[str]] = set()
    for candidate in materialized:
        key = _candidate_key(candidate)
        candidate_keys.add(key)
        if key in label_map:
            aligned.append((candidate, label_map[key]))

    coverage = _gold_coverage(materialized, label_map, candidate_keys)

    train_indices, valid_indices = _entity_disjoint_split(aligned, split=split, seed=seed)
    train = AlignedSplit(
        candidates=[aligned[i][0] for i in train_indices],
        labels=[aligned[i][1] for i in train_indices],
    )
    valid = AlignedSplit(
        candidates=[aligned[i][0] for i in valid_indices],
        labels=[aligned[i][1] for i in valid_indices],
    )
    return AlignedPairs(train=train, valid=valid, coverage=coverage)
