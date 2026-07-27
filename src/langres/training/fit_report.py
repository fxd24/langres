"""FitReport: the human-facing digest of one ``Resolver.fit()`` call (W1.x).

``bootstrap/report.py:BootstrapReport`` is the template: a plain Pydantic model
composed of the sub-stat models it needs, a ``@classmethod build(...)``, and a
``to_markdown()`` digest. FitReport answers "what just trained, on how much
data, and how well does it hold out?" for a single ``fit()``:

- **what trained** -- the component + fit role (``trainable``), and whether a
  fit hook actually ran (``trained``) -- an honest no-op is not a silent success;
- **on how much** -- train/valid sizes and the split provenance;
- **did blocking keep the positives** -- the
  :class:`~langres.curation.harvest.GoldCoverage` from
  :func:`~langres.curation.harvest.align_pairs`;
- **what cut did it settle on** -- the decision threshold, plus whether that
  number was *derived* from the labels or is the constructor's no-data default
  (:class:`ThresholdFit`);
- **how well does it hold out** -- pair P/R/F1 on the entity-disjoint ``valid``
  split (from :func:`~langres.core.metrics.classify_pairs`), when a split was
  given.

Import-light on purpose (Pydantic + the two light leaves
:mod:`langres.curation.harvest`/:mod:`langres.core.metrics` only): it must NEVER
pull sklearn/torch, so a report built right after a heavy fit stays cheap to
import, dump, and render (locked by ``tests/test_import_budget.py``). Lineage is
*referenced*, not duplicated: ``run_ref`` carries the enclosing
:class:`~langres.tracking.runs.RunRecord`'s ``attempt_id`` (the machine record),
while this model is the human-facing digest.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from langres.curation.harvest import GoldCoverage
from langres.core.metrics import PairMetrics


class ThresholdCandidate(BaseModel):
    """One cut that was considered, with pair-F1 wherever it was measured.

    Attributes:
        threshold: The cut itself.
        selection_f1: Pair-F1 on the split that *chose* between the candidates
            (``train``). This is the number the decision was made on, so it is
            fitted to that split by construction.
        held_out_f1: Pair-F1 on the entity-disjoint ``valid`` split, or ``None``
            when no split was given. **This one is a clean estimate:** selection
            happens on ``train``, so nothing ever tuned against ``valid``. The
            honest before/after of a threshold fit is ``previous.held_out_f1``
            vs ``candidate.held_out_f1``.
    """

    threshold: float
    selection_f1: float | None = None
    held_out_f1: float | None = None


class ThresholdFit(BaseModel):
    """How the decision threshold in :attr:`FitReport.threshold` was arrived at.

    The provenance half of the number: a cut derived from 8 labeled pairs and a
    cut derived from 800 are the same ``float`` and are *not* the same claim, and
    a cut that was never derived at all is a constructor default masquerading as
    a measurement. This model makes the difference readable.

    **Deriving is not the same as keeping.** A derived cut is not automatically
    better: Youden's J maximizes ``tpr - fpr`` on ``train``, and on a wide,
    cleanly-separated margin that can pick a cut which scores identically there
    and *worse* on unseen pairs (measured on a toy fixture during development,
    not hypothesized: held-out pair-F1 1.00 -> 0.80). So a threshold fit derives
    a candidate, scores it against the incumbent **on the same ``train`` split it
    was derived from**, and keeps the incumbent unless the candidate is strictly
    better. :attr:`source` records which happened.

    Selecting on ``train`` rather than ``valid`` is deliberate. Choosing the
    winner on ``valid`` would make ``valid`` a selection set and quietly turn
    :attr:`FitReport.metrics` from a held-out estimate into an optimistic one.
    ``train`` was already spent on the derivation, so selecting there costs no
    additional honesty and both :attr:`ThresholdCandidate.held_out_f1` values
    stay clean. Residual gap, stated rather than hidden: a cut chosen on
    ``train`` can still fail to generalize -- selecting there bounds that risk,
    it does not remove it. A three-way derive/select/report split would close it,
    and is deliberately left as follow-on work: at the label counts this feature
    targets (a handful of corrections out of a review loop) three splits would
    make every number noise.

    **What survives ``save``/``load``, and what does not.** The threshold *value*
    does -- it lives on the clusterer (or the chain's
    :class:`~langres.core.op.ThresholdSelect`), both of which serialize into
    ``resolver.json``. This provenance does **not**: it hangs off
    ``ERModel.fit_report_``, which is deliberately never serialized (a fit-time
    artifact, see ``langres.core._model_state``). So a reloaded model carries a
    measured threshold with no record that it was measured -- keep the
    :class:`FitReport` (``model_dump_json()``) beside the artifact if that
    provenance matters.

    Attributes:
        source: What happened to the threshold. ``"derived"`` -- a cut was
            derived and kept. ``"declined"`` -- a cut was derived, lost to the
            incumbent on the selection split, and was **not** applied; the
            threshold is unchanged. ``"not_fitted"`` -- **this** fit did not fit
            the threshold (``derive_threshold=False``), so the cut is whatever
            the model already carried.

            Note what ``"not_fitted"`` deliberately does **not** claim: that the
            threshold is a constructor default. This report cannot know that. A
            model whose earlier ``fit(derive_threshold=True)`` derived its cut,
            re-fitted later without the flag, would be relabelled a "default" by
            such a claim -- a false provenance record in the one field whose
            entire job is provenance. ``fit_report_`` is not serialized, so a
            later fit has no way to recover the earlier one's origin; saying
            "this fit did not touch it" is the strongest true statement
            available.
        method: The derivation method (``"youden"``), or ``None`` when
            ``source="not_fitted"``.
        n_pairs: Labeled pairs the cut was derived from (``0`` when defaulted).
            Read this before trusting the threshold.
        held_out: Whether :attr:`FitReport.metrics` grades the threshold in force
            on pairs it was *not* derived from. ``False`` means the cut and the
            score share rows (in-sample), so any improvement is optimistic.
            Deriving without an entity-disjoint ``split`` always yields ``False``.
        applied_to: Which seam the threshold was written to --
            ``"clusterer"`` (a classic four-slot model) or ``"threshold_select"``
            (an explicit ``_ops`` chain's ``ThresholdSelect``). ``None`` when
            nothing was written, i.e. for ``"not_fitted"`` and ``"declined"``.
        selected_on: The split that chose between the candidates (``"train"``),
            or ``None`` when no derivation ran. Named explicitly so a reader can
            see it was *not* ``valid``.
        previous: The incumbent cut -- the one in force before this fit.
        candidate: The cut the derivation produced. Present for ``"declined"``
            too: a rejected candidate is evidence, not noise, and reporting it
            lets a caller judge the margin instead of trusting the rule.
    """

    source: Literal["derived", "declined", "not_fitted"]
    method: str | None = None
    n_pairs: int = 0
    held_out: bool = False
    applied_to: Literal["clusterer", "threshold_select"] | None = None
    selected_on: Literal["train"] | None = None
    previous: ThresholdCandidate | None = None
    candidate: ThresholdCandidate | None = None


class CalibrationDelta(BaseModel):
    """Before-vs-after calibration quality for a ``method="calibrate"`` fit.

    Measured on the held-out ``valid`` split (the honest test): the matcher's raw
    scores vs the scores mapped through the fitted
    :class:`~langres.training.calibration.Calibrator`. Lower is better for both -- a
    real calibrator drives ``brier``/``ece`` down.

    Attributes:
        method: The calibrator map fitted (``"platt"`` / ``"isotonic"``).
        brier_before / brier_after: :func:`~langres.core.metrics.brier_score` on
            the raw vs calibrated valid scores (the headline number).
        ece_before / ece_after: :func:`~langres.core.metrics.expected_calibration_error`
            on the same (a secondary, binning-dependent diagnostic).
    """

    method: str
    brier_before: float
    brier_after: float
    ece_before: float
    ece_after: float


class FitReport(BaseModel):
    """Digest of one ``Resolver.fit()`` call. Build with :meth:`build`.

    Attributes:
        trainable: What trained -- ``"<Matcher> (<FitRole>)"`` (e.g.
            ``"RandomForestMatcher (SupervisedFitMixin)"``), or the matcher name
            tagged ``"(no fit hook)"`` for the no-op case. ``None`` is reserved
            for a genuinely empty pipeline.
        trained: Whether a fit hook actually ran (``False`` for the no-op case).
        n_train: Aligned training pairs the fit consumed.
        n_valid: Held-out validation pairs (``0`` when no split was given).
        split: The held-out fraction requested, or ``None`` for no split.
        seed: Seed for the entity-disjoint split.
        entity_disjoint: Whether a split was applied entity-disjointly (``True``
            iff ``split`` is not ``None`` -- the only split algorithm).
        coverage: Blocking coverage of the labeled positives, or ``None`` when
            fit was given pre-aligned labels (no id-join, so no coverage) or
            nothing trained.
        threshold: The decision threshold in force after this fit, or ``None``.
        threshold_fit: Where that threshold came from -- derived from labels or
            left at its constructed default (see :class:`ThresholdFit`), or
            ``None`` for a fit that reports no threshold at all.
        metrics: Held-out pair P/R/F1 on ``valid``, or ``None`` when no split.
        cost: The derived dollar cost of a paid/GPU fit (tokens→$ or
            GPU-seconds→$), else ``None``. For a local fine-tune with no
            ``$/GPU-hour`` rate configured this is ``0.0`` (honest, like the
            in-process serve path).
        gpu_seconds: Wall-clock training seconds for a fine-tune fit (the
            GPU-seconds cost *fact* ``cost`` is derived from), else ``None``.
        model_ref: The weightless model reference a fine-tune produced (a base id
            / local dir string, or a ``{base, adapter}`` dict whose shape also
            encodes merge status), else ``None``. Serialize with
            :func:`~langres.core.model_ref.to_config`.
        calibration: Before-vs-after Brier/ECE for a ``method="calibrate"`` fit
            (on the ``valid`` split), else ``None``.
        run_ref: The enclosing run's ``attempt_id`` (lineage reference to the
            machine :class:`~langres.tracking.runs.RunRecord`), or ``None``.
    """

    trainable: str | None
    trained: bool
    n_train: int
    n_valid: int
    split: float | None
    seed: int
    entity_disjoint: bool
    coverage: GoldCoverage | None
    threshold: float | None = None
    threshold_fit: ThresholdFit | None = None
    metrics: PairMetrics | None = None
    cost: float | None = None
    gpu_seconds: float | None = None
    model_ref: str | dict[str, str] | None = None
    calibration: CalibrationDelta | None = None
    run_ref: str | None = None

    @classmethod
    def build(
        cls,
        *,
        trainable: str | None,
        trained: bool,
        n_train: int,
        n_valid: int = 0,
        split: float | None = None,
        seed: int = 0,
        coverage: GoldCoverage | None = None,
        threshold: float | None = None,
        threshold_fit: ThresholdFit | None = None,
        metrics: PairMetrics | None = None,
        cost: float | None = None,
        gpu_seconds: float | None = None,
        model_ref: str | dict[str, str] | None = None,
        calibration: CalibrationDelta | None = None,
        run_ref: str | None = None,
    ) -> FitReport:
        """Assemble a FitReport from the artefacts of one ``fit()`` call.

        ``entity_disjoint`` is derived (``split is not None``) rather than passed:
        the entity-disjoint union-find split is the only algorithm, so a split
        always implies it.
        """
        return cls(
            trainable=trainable,
            trained=trained,
            n_train=n_train,
            n_valid=n_valid,
            split=split,
            seed=seed,
            entity_disjoint=split is not None,
            coverage=coverage,
            threshold=threshold,
            threshold_fit=threshold_fit,
            metrics=metrics,
            cost=cost,
            gpu_seconds=gpu_seconds,
            model_ref=model_ref,
            calibration=calibration,
            run_ref=run_ref,
        )

    @classmethod
    def nothing_trainable(cls, matcher_name: str) -> FitReport:
        """A minimal report for the no-op branch: nothing in the pipeline trained.

        Names the matcher so the digest is still informative ("this pipeline had
        nothing to train") rather than an anonymous empty report.
        """
        return cls.build(trainable=f"{matcher_name} (no fit hook)", trained=False, n_train=0)

    def _threshold_suffix(self) -> str:
        """The provenance clause appended to the rendered threshold line.

        Says *how* the number was reached in the same breath as the number, so a
        reader cannot take a constructor default for a measurement, nor an
        in-sample cut for a held-out one.
        """
        fit = self.threshold_fit
        if fit is None:
            return ""
        if fit.source == "not_fitted":
            return " (not fitted by this fit — pass derive_threshold=True to fit it from labels)"
        if fit.source == "declined":
            candidate = "" if fit.candidate is None else f" {fit.candidate.threshold:.4f}"
            return (
                f" (kept: the cut derived from {fit.n_pairs} labeled pairs"
                f"{candidate} did not beat it on {fit.selected_on})"
            )
        moved = "" if fit.previous is None else f" from {fit.previous.threshold:.4f}"
        sample = "held-out" if fit.held_out else "IN-SAMPLE"
        return (
            f" (derived{moved} by {fit.method} on {fit.n_pairs} labeled pairs, "
            f"chosen on {fit.selected_on}, {sample}, applied to the {fit.applied_to})"
        )

    def _threshold_choice_lines(self) -> list[str]:
        """The before/after block: both candidate cuts and how each scored.

        ``held-out`` here is a genuinely clean estimate -- selection ran on
        ``train``, so nothing was ever tuned against ``valid``.
        """
        fit = self.threshold_fit
        if fit is None or fit.source == "not_fitted":
            return []
        rows = [("incumbent", fit.previous), ("derived", fit.candidate)]
        lines = [f"## Threshold selection (chosen on {fit.selected_on})", ""]
        for label, candidate in rows:
            if candidate is None:
                continue
            kept = (label == "derived") == (fit.source == "derived")
            parts = [f"- {label} {candidate.threshold:.4f}{' — KEPT' if kept else ''}"]
            if candidate.selection_f1 is not None:
                parts.append(f"selection F1 {candidate.selection_f1:.4f}")
            if candidate.held_out_f1 is not None:
                parts.append(f"held-out F1 {candidate.held_out_f1:.4f}")
            lines.append(": ".join([parts[0], ", ".join(parts[1:])]) if parts[1:] else parts[0])
        lines.append("")
        return lines

    def to_markdown(self) -> str:
        """Render a human-readable Markdown digest of the report.

        The model itself is the source of truth and is JSON-serializable; this is
        for quick eyeballing after a ``fit()``.
        """
        lines: list[str] = ["# Fit Report", ""]

        split_line = f"- Split: {self.split if self.split is not None else 'none'}"
        if self.entity_disjoint:
            split_line += f" (entity-disjoint, seed={self.seed})"
        lines += [
            "## What trained",
            f"- Trainable: {self.trainable if self.trainable is not None else 'nothing'}",
            f"- Trained: {self.trained}",
            f"- Train pairs: {self.n_train}",
            f"- Valid pairs: {self.n_valid}",
            split_line,
        ]
        if self.threshold is not None:
            lines.append(f"- Threshold: {self.threshold:.4f}{self._threshold_suffix()}")
        if self.model_ref is not None:
            lines.append(f"- Model ref: {self.model_ref}")
        if self.gpu_seconds is not None:
            lines.append(f"- GPU-seconds: {self.gpu_seconds:.1f}")
        if self.cost is not None:
            lines.append(f"- Cost: ${self.cost}")
        if self.run_ref is not None:
            lines.append(f"- Run: {self.run_ref}")
        lines.append("")

        lines += self._threshold_choice_lines()

        lines.append("## Gold coverage (labeled positives kept by blocking)")
        if self.coverage is None:
            lines.append("- Not computed (fit received pre-aligned labels or nothing trained).")
        else:
            c = self.coverage
            lines += [
                f"- Gold coverage: {c.gold_coverage:.4f}",
                f"- Positive labels: {c.n_positive_labels}",
                f"- Dropped positives: {len(c.dropped_positives)}",
                f"- Labeled pairs: {c.n_labeled} (aligned to candidates: {c.n_aligned})",
            ]
            if c.dropped_positives:
                preview = ", ".join(f"({a}, {b})" for a, b in c.dropped_positives[:5])
                extra = len(c.dropped_positives) - 5
                more = f", +{extra} more" if extra > 0 else ""
                lines.append(f"  - Dropped: {preview}{more}")
        lines.append("")

        lines.append("## Held-out pair metrics (valid split)")
        if self.metrics is None:
            if self.n_valid > 0:
                # A held-out split exists but this fit reports no pair P/R/F1 (e.g.
                # a calibrate fit, whose held-out signal is the calibration delta).
                lines.append("- No held-out pair P/R/F1 computed for this fit.")
            elif self.split is not None:
                lines.append(
                    "- The requested split produced no held-out pairs (all labeled "
                    "entities are connected -- no entity-disjoint valid is possible)."
                )
            else:
                lines.append("- No split was given (no held-out evaluation).")
        else:
            m = self.metrics
            lines += [
                f"- Precision: {m.precision:.4f}",
                f"- Recall: {m.recall:.4f}",
                f"- F1: {m.f1:.4f}",
                f"- TP/FP/FN: {m.tp}/{m.fp}/{m.fn} @ threshold {m.threshold:.4f}",
            ]
        lines.append("")

        if self.calibration is not None:
            cal = self.calibration
            lines += [
                f"## Calibration ({cal.method}, valid split — lower is better)",
                f"- Brier: {cal.brier_before:.4f} → {cal.brier_after:.4f}",
                f"- ECE:   {cal.ece_before:.4f} → {cal.ece_after:.4f}",
                "",
            ]

        return "\n".join(lines)

    def render(self) -> str:
        """Render the report as Markdown.

        Provided alongside :meth:`to_markdown` so callers expecting either the
        generic ``render()`` name or the format-specific one get the same output
        (mirrors ``BootstrapReport``).
        """
        return self.to_markdown()
