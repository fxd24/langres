"""Coverage and calibration report for cold-start gold-set bootstrapping (M1).

:class:`BootstrapReport` turns the artefacts of a bootstrap run -- the
teacher-labeled pairs, the post-blocking candidates, and whatever ground truth
is available -- into a single, honest health check answering three questions:

1. **Did blocking keep the true matches?** -- blocking pair-completeness
   (candidate recall) from :func:`langres.core.metrics.evaluate_blocking`.
2. **Does the teacher agree with truth?** -- accuracy / precision / recall / F1
   plus Cohen's kappa **and** MCC on the pairs that have a ground-truth label.
   Under the ~2% positive prevalence of bootstrap labeling kappa collapses
   (prevalence paradox), so the headline gate is F1/MCC, not kappa alone (W5).
3. **Are the teacher's confidences trustworthy?** -- Brier (primary, no binning)
   and a verbalized-confidence ECE with equal-mass bins, plus reliability bins
   for a diagram (W6).

It also reports an agreement-convergence curve (teacher F1-vs-truth as the
evaluated sample grows, by deterministically subsampling already-collected
labels -- no re-labeling, zero extra spend; W8) and honest routing/cost
coverage read from the gold-set metadata and pair provenance.

This is a PLAIN Pydantic model: not ``@register``-ed, not part of the
``SerializableState`` protocol. The builder is pure and deterministic, so it is
fully testable from synthetic :class:`~langres.bootstrap.models.GoldPair` data
with no LLM and no embeddings.
"""

from typing import Any

from pydantic import BaseModel

from langres.bootstrap.models import GoldPair, GoldSet
from langres.core.metrics import (
    ReliabilityBin,
    brier_score,
    cohens_kappa,
    evaluate_blocking,
    expected_calibration_error,
    matthews_corrcoef,
    pairs_from_clusters,
    reliability_bins,
)
from langres.core.models import ERCandidate

# Metadata/provenance keys searched, in order, for a USD cost figure.
_COST_KEYS = ("total_cost_usd", "cost_usd", "cost")


def _f1(precision: float, recall: float) -> float:
    """Harmonic mean of precision and recall; ``0.0`` when both are zero."""
    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


class BlockingCoverage(BaseModel):
    """How well blocking kept the true matches (pair-completeness).

    Attributes:
        pair_completeness: Fraction of ground-truth match pairs that appear among
            the candidates (a.k.a. candidate recall / pair-completeness). The
            single most important blocking number -- matches dropped here cannot
            be recovered downstream.
        candidate_precision: Fraction of candidate pairs that are true matches.
        total_candidates: Number of unique candidate pairs after blocking.
        missed_matches: Number of true match pairs *not* captured by blocking.
    """

    pair_completeness: float
    candidate_precision: float
    total_candidates: int
    missed_matches: int


class AgreementStats(BaseModel):
    """Teacher-vs-truth agreement on pairs that have a ground-truth label.

    The positive class is "match" (``label is True``). ``cohens_kappa`` and
    ``mcc`` are reported together: kappa is the familiar chance-corrected number
    but degrades under low positive prevalence (W5), so MCC is the robust
    companion and, with F1, the headline agreement gate.

    Attributes:
        n_evaluated: Number of teacher pairs that had a ground-truth label.
        accuracy: Fraction of teacher labels matching truth.
        precision: Match precision (TP / (TP + FP)).
        recall: Match recall (TP / (TP + FN)).
        f1: Harmonic mean of precision and recall.
        cohens_kappa: Chance-corrected agreement (see caveat above).
        mcc: Matthews correlation coefficient (prevalence-robust).
    """

    n_evaluated: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    cohens_kappa: float
    mcc: float


class CalibrationStats(BaseModel):
    """Calibration of teacher confidence against label-correctness.

    ``confidence`` is the teacher's verbalized self-confidence in its own label;
    the outcome being calibrated is whether that label was actually correct
    (``teacher_label == truth_label``). So this answers "when the teacher says it
    is 80% sure, is it right 80% of the time?".

    Attributes:
        n_evaluated: Number of teacher pairs with both a confidence and a
            ground-truth label.
        brier: Brier score -- the primary, binning-free proper score (lower is
            better) (W6).
        ece: Verbalized-confidence Expected Calibration Error with equal-mass
            bins (secondary, binning-dependent diagnostic).
        n_bins: Number of bins requested for ECE / reliability.
        reliability: Per-bin reliability points (lowest to highest confidence).
    """

    n_evaluated: int
    brier: float
    ece: float
    n_bins: int
    reliability: list[ReliabilityBin]


class ConvergencePoint(BaseModel):
    """One point on the agreement-convergence curve (W8).

    Attributes:
        n_labeled: Number of already-collected labels evaluated (a prefix).
        f1: Teacher F1-vs-truth computed on that prefix.
    """

    n_labeled: int
    f1: float


class CoverageStats(BaseModel):
    """Routing / coverage counts and the honest run cost.

    Attributes:
        total_candidates: Candidate pairs produced by blocking.
        mined: Candidates selected for labeling (from metadata, else ``labeled``).
        labeled: Teacher-labeled pairs in the gold set.
        skipped: Candidates not labeled (``total_candidates - labeled``, floored
            at 0).
        with_ground_truth: Labeled pairs that had a ground-truth label and so
            contributed to the agreement/calibration numbers.
        total_cost_usd: Honest total spend, from gold-set metadata if present,
            else summed from per-pair provenance.
    """

    total_candidates: int
    mined: int
    labeled: int
    skipped: int
    with_ground_truth: int
    total_cost_usd: float


class BootstrapReport(BaseModel):
    """Coverage + calibration report for one bootstrap run.

    Build it with :meth:`build` and serialize it with the usual Pydantic
    ``model_dump_json``; render a human-readable digest with :meth:`to_markdown`.

    Attributes:
        blocking: Blocking pair-completeness coverage.
        agreement: Teacher-vs-truth agreement, or ``None`` when no labeled pair
            had a ground-truth label.
        calibration: Teacher-confidence calibration, or ``None`` when no labeled
            pair had both a confidence and a ground-truth label.
        convergence: Agreement-convergence curve (possibly empty).
        coverage: Routing counts and honest cost.
    """

    blocking: BlockingCoverage
    agreement: AgreementStats | None
    calibration: CalibrationStats | None
    convergence: list[ConvergencePoint]
    coverage: CoverageStats

    @classmethod
    def build(
        cls,
        gold: GoldSet | list[GoldPair],
        candidates: list[ERCandidate[Any]],
        truth_clusters: list[set[str]],
        *,
        n_bins: int = 8,
    ) -> "BootstrapReport":
        """Build a report from teacher labels, candidates, and ground-truth clusters.

        Pure and deterministic: identical inputs always produce an identical
        report (the convergence curve walks labels in a fixed ``(left_id,
        right_id)`` order). No network, LLM, or embedding calls.

        Args:
            gold: The teacher-labeled pairs -- either a
                :class:`~langres.bootstrap.models.GoldSet` (metadata is used for
                cost/mined counts) or a bare ``list[GoldPair]``.
            candidates: Post-blocking candidate pairs, used only for blocking
                pair-completeness via
                :func:`~langres.core.metrics.evaluate_blocking` (not reimplemented).
            truth_clusters: Ground-truth entity clusters (sets of record ids). A
                teacher pair has a ground-truth label only when *both* of its ids
                appear in these clusters; its truth label is then whether the two
                ids share a cluster.
            n_bins: Bins for the calibration ECE / reliability diagram.

        Returns:
            The assembled :class:`BootstrapReport`.
        """
        pairs = gold.pairs if isinstance(gold, GoldSet) else gold
        metadata = gold.metadata if isinstance(gold, GoldSet) else {}

        # --- Blocking pair-completeness (reuse evaluate_blocking) ---------------
        blocking_stats = evaluate_blocking(candidates, truth_clusters)
        blocking = BlockingCoverage(
            pair_completeness=blocking_stats.candidate_recall,
            candidate_precision=blocking_stats.candidate_precision,
            total_candidates=blocking_stats.total_candidates,
            missed_matches=blocking_stats.missed_matches_count,
        )

        # --- Resolve ground truth for each teacher pair ------------------------
        truth_entities = {e for cluster in truth_clusters for e in cluster}
        truth_match_pairs = pairs_from_clusters(truth_clusters)

        # Deterministic order so the convergence curve is reproducible.
        ordered_pairs = sorted(pairs, key=lambda p: (p.left_id, p.right_id))

        teacher_labels: list[bool] = []
        truth_labels: list[bool] = []
        confidences: list[float] = []
        correctness: list[bool] = []
        for pair in ordered_pairs:
            if pair.left_id not in truth_entities or pair.right_id not in truth_entities:
                continue  # No ground truth for this pair -> excluded, honestly.
            key = tuple(sorted((pair.left_id, pair.right_id)))
            truth_label = key in truth_match_pairs
            teacher_labels.append(pair.label)
            truth_labels.append(truth_label)
            if pair.confidence is not None:
                confidences.append(pair.confidence)
                correctness.append(pair.label == truth_label)

        agreement = cls._build_agreement(teacher_labels, truth_labels)
        calibration = cls._build_calibration(confidences, correctness, n_bins)
        convergence = cls._build_convergence(teacher_labels, truth_labels)

        # --- Routing / coverage + honest cost ----------------------------------
        labeled = len(pairs)
        total_candidates = len(candidates)
        mined_value = metadata.get("mined")
        mined = (
            int(mined_value)
            if isinstance(mined_value, (int, float)) and not isinstance(mined_value, bool)
            else labeled
        )
        coverage = CoverageStats(
            total_candidates=total_candidates,
            mined=mined,
            labeled=labeled,
            skipped=max(total_candidates - labeled, 0),
            with_ground_truth=len(teacher_labels),
            total_cost_usd=_resolve_cost(metadata, pairs),
        )

        return cls(
            blocking=blocking,
            agreement=agreement,
            calibration=calibration,
            convergence=convergence,
            coverage=coverage,
        )

    @staticmethod
    def _build_agreement(
        teacher_labels: list[bool], truth_labels: list[bool]
    ) -> AgreementStats | None:
        """Compute teacher-vs-truth agreement, or ``None`` if no labeled pair has truth."""
        if not teacher_labels:
            return None
        tp = fp = fn = tn = 0
        for truth, teacher in zip(truth_labels, teacher_labels, strict=True):
            if truth and teacher:
                tp += 1
            elif not truth and teacher:
                fp += 1
            elif truth and not teacher:
                fn += 1
            else:
                tn += 1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        accuracy = (tp + tn) / len(teacher_labels)
        return AgreementStats(
            n_evaluated=len(teacher_labels),
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=_f1(precision, recall),
            cohens_kappa=cohens_kappa(truth_labels, teacher_labels),
            mcc=matthews_corrcoef(truth_labels, teacher_labels),
        )

    @staticmethod
    def _build_calibration(
        confidences: list[float], correctness: list[bool], n_bins: int
    ) -> CalibrationStats | None:
        """Compute confidence calibration, or ``None`` if no pair has confidence + truth."""
        if not confidences:
            return None
        return CalibrationStats(
            n_evaluated=len(confidences),
            brier=brier_score(confidences, correctness),
            ece=expected_calibration_error(
                confidences, correctness, n_bins=n_bins, strategy="quantile"
            ),
            n_bins=n_bins,
            reliability=reliability_bins(
                confidences, correctness, n_bins=n_bins, strategy="quantile"
            ),
        )

    @staticmethod
    def _build_convergence(
        teacher_labels: list[bool], truth_labels: list[bool]
    ) -> list[ConvergencePoint]:
        """Teacher F1-vs-truth over growing prefixes of the (already-collected) labels.

        Subsamples the labels already in hand in a fixed order -- no re-labeling,
        no held-out model, no leakage (W8). One point per prefix length.
        """
        points: list[ConvergencePoint] = []
        tp = fp = fn = 0
        for n, (truth, teacher) in enumerate(
            zip(truth_labels, teacher_labels, strict=True), start=1
        ):
            if truth and teacher:
                tp += 1
            elif teacher and not truth:
                fp += 1
            elif truth and not teacher:
                fn += 1
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            points.append(ConvergencePoint(n_labeled=n, f1=_f1(precision, recall)))
        return points

    def to_markdown(self) -> str:
        """Render a human-readable Markdown digest of the report.

        Includes the headline blocking, agreement, calibration, and coverage
        numbers. Used for quick eyeballing; the model itself is the source of
        truth and is JSON-serializable.

        Returns:
            A Markdown string.
        """
        lines: list[str] = ["# Bootstrap Report", ""]

        lines += [
            "## Blocking (pair-completeness)",
            f"- Pair-completeness (candidate recall): {self.blocking.pair_completeness:.4f}",
            f"- Candidate precision: {self.blocking.candidate_precision:.4f}",
            f"- Total candidates: {self.blocking.total_candidates}",
            f"- Missed matches: {self.blocking.missed_matches}",
            "",
        ]

        lines.append("## Teacher-vs-truth agreement")
        if self.agreement is None:
            lines.append("- No labeled pair had a ground-truth label.")
        else:
            a = self.agreement
            lines += [
                f"- Evaluated pairs: {a.n_evaluated}",
                f"- Accuracy: {a.accuracy:.4f}",
                f"- Precision: {a.precision:.4f}",
                f"- Recall: {a.recall:.4f}",
                f"- F1: {a.f1:.4f}",
                f"- Cohen's kappa: {a.cohens_kappa:.4f}",
                f"- MCC: {a.mcc:.4f}",
            ]
        lines.append("")

        lines.append("## Calibration (teacher confidence vs. correctness)")
        if self.calibration is None:
            lines.append("- No labeled pair had both a confidence and a ground-truth label.")
        else:
            c = self.calibration
            lines += [
                f"- Evaluated pairs: {c.n_evaluated}",
                f"- Brier score (primary): {c.brier:.4f}",
                f"- ECE (equal-mass, {c.n_bins} bins): {c.ece:.4f}",
                "- Reliability bins (mean_conf -> observed_freq, count):",
            ]
            lines += [
                f"  - {b.mean_confidence:.4f} -> {b.observed_frequency:.4f} (n={b.count})"
                for b in c.reliability
            ]
        lines.append("")

        cov = self.coverage
        lines += [
            "## Routing / coverage",
            f"- Total candidates: {cov.total_candidates}",
            f"- Mined: {cov.mined}",
            f"- Labeled: {cov.labeled}",
            f"- Skipped: {cov.skipped}",
            f"- With ground truth: {cov.with_ground_truth}",
            f"- Total cost (USD): {cov.total_cost_usd:.4f}",
            "",
        ]

        if self.convergence:
            last = self.convergence[-1]
            lines += [
                "## Agreement convergence",
                f"- Points: {len(self.convergence)}",
                f"- Final F1 @ {last.n_labeled} labels: {last.f1:.4f}",
                "",
            ]

        return "\n".join(lines)

    def render(self) -> str:
        """Render the report as Markdown.

        Provided alongside :meth:`to_markdown` so callers expecting either the
        generic ``render()`` name or the format-specific one get the same output.
        """
        return self.to_markdown()


def _resolve_cost(metadata: dict[str, object], pairs: list[GoldPair]) -> float:
    """Resolve honest total USD cost from gold-set metadata, else pair provenance.

    Prefers a run-level cost in ``metadata`` (``total_cost_usd`` / ``cost_usd`` /
    ``cost``); otherwise sums the first such key found in each pair's
    ``provenance``. Non-numeric values are ignored.
    """
    for key in _COST_KEYS:
        if key in metadata:
            value = metadata[key]
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)

    total = 0.0
    for pair in pairs:
        for key in _COST_KEYS:
            if key in pair.provenance:
                value = pair.provenance[key]
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    total += float(value)
                break
    return total
