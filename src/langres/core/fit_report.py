"""FitReport: the human-facing digest of one ``Resolver.fit()`` call (W1.x).

``bootstrap/report.py:BootstrapReport`` is the template: a plain Pydantic model
composed of the sub-stat models it needs, a ``@classmethod build(...)``, and a
``to_markdown()`` digest. FitReport answers "what just trained, on how much
data, and how well does it hold out?" for a single ``fit()``:

- **what trained** -- the component + fit role (``trainable``), and whether a
  fit hook actually ran (``trained``) -- an honest no-op is not a silent success;
- **on how much** -- train/valid sizes and the split provenance;
- **did blocking keep the positives** -- the
  :class:`~langres.core.harvest.GoldCoverage` from
  :func:`~langres.core.harvest.align_pairs`;
- **how well does it hold out** -- pair P/R/F1 on the entity-disjoint ``valid``
  split (from :func:`~langres.core.metrics.classify_pairs`), when a split was
  given.

Import-light on purpose (Pydantic + the two light leaves
:mod:`langres.core.harvest`/:mod:`langres.core.metrics` only): it must NEVER
pull sklearn/torch, so a report built right after a heavy fit stays cheap to
import, dump, and render (locked by ``tests/test_import_budget.py``). Lineage is
*referenced*, not duplicated: ``run_ref`` carries the enclosing
:class:`~langres.core.runs.RunRecord`'s ``attempt_id`` (the machine record),
while this model is the human-facing digest.
"""

from __future__ import annotations

from pydantic import BaseModel

from langres.core.harvest import GoldCoverage
from langres.core.metrics import PairMetrics


class CalibrationDelta(BaseModel):
    """Before-vs-after calibration quality for a ``method="calibrate"`` fit.

    Measured on the held-out ``valid`` split (the honest test): the matcher's raw
    scores vs the scores mapped through the fitted
    :class:`~langres.core.calibration.Calibrator`. Lower is better for both -- a
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
        threshold: The clusterer decision threshold in force, or ``None``.
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
            machine :class:`~langres.core.runs.RunRecord`), or ``None``.
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
            lines.append(f"- Threshold: {self.threshold:.4f}")
        if self.model_ref is not None:
            lines.append(f"- Model ref: {self.model_ref}")
        if self.gpu_seconds is not None:
            lines.append(f"- GPU-seconds: {self.gpu_seconds:.1f}")
        if self.cost is not None:
            lines.append(f"- Cost: ${self.cost}")
        if self.run_ref is not None:
            lines.append(f"- Run: {self.run_ref}")
        lines.append("")

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
