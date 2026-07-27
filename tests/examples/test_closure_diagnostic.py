"""Behavior tests for the B1 closure diagnostic example.

The genuinely new logic here is the *reconstruction* -- rebuilding the clusterer's
edge set from judgement-log rows -- and the counting on top of it. Both traps the
example exists to avoid are pinned as tests:

* a ``verdict = null`` row is never counted as a rejection (trap 1);
* a decider row with ``verdict = True`` and a below-threshold score is counted as
  an ACCEPT, because ``predicted_match`` gives ``decision`` precedence -- the
  naive ``score < threshold`` scan would call it a rejection and manufacture the
  finding (trap 2).

Plus one slow end-to-end over the 12-record fixture, proving the whole path
(dedupe -> log -> reconstruct -> both clusterers) runs and self-verifies.
"""

from __future__ import annotations

from typing import Any

import pytest

from examples.research.closure_diagnostic import (
    BenchmarkFinding,
    SweepPoint,
    diagnose,
    judgements_from_log,
    select_benchmarks,
    to_markdown,
    tune_threshold,
    worst_sweep_point,
)


def _row(
    left: str,
    right: str,
    *,
    score: float | None,
    verdict: bool | None,
    decision: bool | None = None,
) -> dict[str, Any]:
    """One judgement-log row, in the shape ``JudgementLog.read()`` returns."""
    return {
        "left_id": left,
        "right_id": right,
        "score": score,
        "decision": decision,
        "verdict": verdict,
        "decision_step": "test",
    }


class TestReconstruction:
    def test_rebuilds_one_judgement_per_judged_row(self) -> None:
        rows = [
            _row("a", "b", score=0.9, verdict=True),
            _row("b", "c", score=0.1, verdict=False),
        ]
        judgements = judgements_from_log(rows)
        assert [(j.left_id, j.right_id, j.score) for j in judgements] == [
            ("a", "b", 0.9),
            ("b", "c", 0.1),
        ]

    def test_trap_1_verdict_null_rows_are_excluded(self) -> None:
        # A retrieval/reranking stage (or an abstention) logs verdict=null. It was
        # never a rejection, so it must not enter the reconstruction at all.
        rows = [
            _row("a", "b", score=0.9, verdict=True),
            _row("c", "d", score=0.2, verdict=None),
        ]
        assert len(judgements_from_log(rows)) == 1

    def test_trap_2_decider_row_reconstructs_as_a_match_not_a_rejection(self) -> None:
        # verdict=True with a below-threshold score: predicted_match must follow
        # the DECISION. A naive score<threshold scan would call this a rejection.
        rows = [_row("a", "b", score=0.1, verdict=True, decision=True)]
        judgement = judgements_from_log(rows)[0]
        assert judgement.decision is True
        finding = diagnose(
            "closure",
            [{"a", "b"}],
            [judgement],
            threshold=0.5,
            truth_clusters=[{"a", "b"}],
            all_ids=["a", "b"],
        )
        # Zero rejections at all -- so nothing can be "rejected inside".
        assert finding.n_rejected_inside == 0
        assert finding.rejected_inside_rate is None


class TestDiagnose:
    #: A-B and B-C accepted, A-C rejected: closure chains all three into one
    #: cluster, so exactly one rejected pair sits inside it.
    _CHAIN = [
        _row("a", "b", score=0.9, verdict=True),
        _row("b", "c", score=0.9, verdict=True),
        _row("a", "c", score=0.1, verdict=False),
    ]

    def test_counts_a_rejected_pair_inside_a_chained_cluster(self) -> None:
        finding = diagnose(
            "closure",
            [{"a", "b", "c"}],
            judgements_from_log(self._CHAIN),
            threshold=0.5,
            truth_clusters=[{"a", "b", "c"}],
            all_ids=["a", "b", "c"],
        )
        assert finding.n_rejected_inside == 1
        assert finding.rejected_inside_rate == 1.0  # 1 of 1 rejection
        assert finding.n_incluster_pairs == 3  # C(3, 2)
        assert finding.incluster_contamination == pytest.approx(1 / 3)

    def test_reports_the_distribution_over_cluster_size(self) -> None:
        finding = diagnose(
            "closure",
            [{"a", "b", "c"}, {"d", "e"}],
            judgements_from_log(self._CHAIN),
            threshold=0.5,
            truth_clusters=[{"a", "b", "c"}, {"d", "e"}],
            all_ids=["a", "b", "c", "d", "e"],
        )
        # (size, n_clusters, n_rejected_inside): only the size-3 cluster carries one.
        assert finding.by_cluster_size == [(2, 1, 0), (3, 1, 1)]
        assert finding.largest_cluster == 3

    def test_a_rejected_pair_across_two_clusters_is_not_counted(self) -> None:
        finding = diagnose(
            "closure",
            [{"a", "b"}, {"c"}],
            judgements_from_log(self._CHAIN),
            threshold=0.5,
            truth_clusters=[{"a", "b"}, {"c"}],
            all_ids=["a", "b", "c"],
        )
        assert finding.n_rejected_inside == 0
        assert finding.rejected_inside_rate == 0.0

    def test_size_two_clusters_force_a_zero(self) -> None:
        # The caveat the sweep exists for: a size-2 cluster's only in-cluster pair
        # is the accepted edge, so 0 here is structural, not measured.
        finding = diagnose(
            "closure",
            [{"a", "b"}],
            judgements_from_log([_row("a", "b", score=0.9, verdict=True)]),
            threshold=0.5,
            truth_clusters=[{"a", "b"}],
            all_ids=["a", "b"],
        )
        assert finding.n_incluster_pairs == 1
        assert finding.n_rejected_inside == 0

    def test_empty_clustering_reports_none_rather_than_zero(self) -> None:
        finding = diagnose(
            "closure",
            [],
            [],
            threshold=0.5,
            truth_clusters=[{"a"}],
            all_ids=["a"],
        )
        assert finding.n_clusters == 0
        assert finding.largest_cluster == 0
        assert finding.incluster_contamination is None
        assert finding.rejected_inside_rate is None


class TestTuneThreshold:
    def test_picks_the_best_bcubed_point_and_breaks_ties_low(self) -> None:
        judgements = judgements_from_log(
            [
                _row("a", "b", score=0.9, verdict=True),
                _row("b", "c", score=0.4, verdict=False),
            ]
        )
        # Gold says a-b match and c is a singleton, so 0.5 (which drops the 0.4
        # edge) beats 0.3 (which chains c in).
        best = tune_threshold(judgements, [{"a", "b"}, {"c"}], ["a", "b", "c"], (0.3, 0.5, 0.7))
        assert best == 0.5


class TestReporting:
    def _finding(self, sweep: list[SweepPoint]) -> BenchmarkFinding:
        empty = diagnose("closure", [], [], threshold=0.5, truth_clusters=[{"a"}], all_ids=["a"])
        return BenchmarkFinding(
            benchmark="demo",
            method="rapidfuzz",
            threshold=0.5,
            n_test_records=1,
            n_logged=0,
            n_judged=0,
            n_accepted=0,
            n_rejected=0,
            n_abstained=0,
            decider_override_rows=0,
            verdict_agreement=None,
            reconstruction_exact=True,
            seconds=0.1,
            closure=empty,
            correlation=empty,
            sweep=sweep,
        )

    def _point(self, threshold: float, rejected_inside: int) -> SweepPoint:
        return SweepPoint(
            threshold=threshold,
            n_rejected=1,
            closure_clusters=1,
            closure_largest=2,
            closure_rejected_inside=rejected_inside,
            closure_bcubed_f1=0.5,
            correlation_clusters=1,
            correlation_largest=2,
            correlation_rejected_inside=0,
            correlation_bcubed_f1=0.5,
        )

    def test_worst_point_is_the_largest_count(self) -> None:
        finding = self._finding([self._point(0.3, 5), self._point(0.8, 1)])
        worst = worst_sweep_point(finding)
        assert worst is not None and worst.threshold == 0.3

    def test_worst_point_breaks_ties_on_the_lower_threshold(self) -> None:
        finding = self._finding([self._point(0.8, 2), self._point(0.3, 2)])
        worst = worst_sweep_point(finding)
        assert worst is not None and worst.threshold == 0.3

    def test_worst_point_of_an_empty_sweep_is_none(self) -> None:
        assert worst_sweep_point(self._finding([])) is None

    def test_markdown_renders_a_header_plus_one_row_per_finding(self) -> None:
        table = to_markdown([self._finding([self._point(0.3, 5)])])
        lines = table.splitlines()
        assert len(lines) == 3  # header + separator + one row
        assert "demo" in table


class TestSelection:
    def test_skips_the_external_only_entry_with_a_note(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        names = select_benchmarks(fast=False, only=None)
        assert "opensanctions" not in names
        assert "fodors_zagat" in names
        assert "[skip] opensanctions" in capsys.readouterr().out

    def test_only_wins_over_fast(self) -> None:
        assert select_benchmarks(fast=True, only=["abt_buy"]) == ["abt_buy"]

    def test_fast_is_a_strict_subset(self) -> None:
        assert set(select_benchmarks(fast=True, only=None)) < set(
            select_benchmarks(fast=False, only=None)
        )

    def test_an_only_that_selects_nothing_raises_instead_of_warning(self) -> None:
        """A typo must stop the run, not produce an empty result set.

        This previously printed a warning and carried on, so the run reached
        ``write_findings`` with ``[]`` and replaced the tracked artifact with an
        empty list -- a warning that named the failure it could not prevent.
        """
        with pytest.raises(SystemExit, match="selected no loadable benchmark"):
            select_benchmarks(fast=False, only=["definitely_not_a_benchmark"])


class TestCanonicalArtifactIsProtected:
    """A narrowed run must not be able to shrink the tracked full-portfolio JSON."""

    def _main_with(self, argv: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
        import sys

        from examples.research import closure_diagnostic

        monkeypatch.setattr(sys, "argv", ["closure_diagnostic.py", *argv])
        closure_diagnostic.main()

    @pytest.mark.parametrize(
        "argv", [["--fast"], ["--only", "tiny_fixture"], ["--fast", "--only", "tiny_fixture"]]
    )
    def test_a_narrowed_run_without_out_exits_before_measuring(
        self, argv: list[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # argparse's error() exits 2. The point is that it happens BEFORE any
        # benchmark runs, so the canonical file is never opened for writing.
        with pytest.raises(SystemExit) as excinfo:
            self._main_with(argv, monkeypatch)
        assert excinfo.value.code == 2

    def test_a_full_run_still_defaults_to_the_canonical_path(self) -> None:
        from examples.research.closure_diagnostic import CANONICAL_OUT

        assert CANONICAL_OUT.as_posix() == "examples/research/results/closure_diagnostic.json"

    def test_writing_an_empty_result_over_a_real_one_is_refused(self, tmp_path) -> None:
        from examples.research.closure_diagnostic import write_findings

        out = tmp_path / "results.json"
        out.write_text('[{"benchmark": "abt_buy"}]\n')
        with pytest.raises(SystemExit, match="refusing to overwrite"):
            write_findings([], out)
        # The real results are still there -- that is the whole point.
        assert "abt_buy" in out.read_text()

    def test_writing_an_empty_result_over_nothing_is_allowed(self, tmp_path) -> None:
        from examples.research.closure_diagnostic import write_findings

        out = tmp_path / "fresh.json"
        write_findings([], out)
        assert out.read_text().strip() == "[]"


@pytest.mark.slow
def test_end_to_end_on_the_tiny_fixture_self_verifies(tmp_path) -> None:
    """The whole path runs and both instrument checks pass (loads MiniLM)."""
    from examples.research.closure_diagnostic import run_benchmark

    finding = run_benchmark("tiny_fixture", method="rapidfuzz", seed=0, log_dir=tmp_path)
    assert finding.reconstruction_exact is True
    assert finding.verdict_agreement == 1.0
    assert finding.n_judged == finding.n_accepted + finding.n_rejected
    assert len(finding.sweep) > 0
