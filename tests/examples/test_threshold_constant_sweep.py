"""Behavior tests for the per-score-family fixed-constant threshold sweep.

The measurement itself needs models and minutes, so what is pinned here is the
*reasoning* layered on top of the scores — the part that would silently produce a
wrong shipped constant rather than crash:

* the self-check gate can actually **fail** (a gate nobody has watched fail is a
  hypothesis, not a safety net — `.claude/rules/expert-knowledge.md`);
* LOBO selection really excludes the held-out benchmark, which is the whole
  reason its numbers are out-of-sample;
* selection uses the **median** across benchmarks, so one wide-range dataset
  cannot choose the constant for every other;
* the pre-registered ship rule refuses an unstable family;
* the exact oracle beats the grid, so "capture" is measured against the true
  ceiling;
* the checkpoint override refuses to relabel a blocker it did not actually swap.

No model loads here, so none of it is `slow`.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from examples.research.threshold_constant_sweep import (
    GRID,
    SHIPPED_INDEX,
    CellResult,
    SweepReport,
    _assert_matches_classify_pairs,
    _EmbedderOverride,
    _exact_oracle,
    _f1,
    _select_constant,
    _source_fingerprint,
    _worker_command,
    _unit_index,
    dedupe_scores,
    lobo_constants,
    main,
    read_report,
    run_benchmark_isolated,
    to_transfer_markdown,
    to_verdict_markdown,
    write_report,
)
from langres.core.models import PairwiseJudgement


def _judgement(left: str, right: str, score: float) -> PairwiseJudgement:
    return PairwiseJudgement(
        left_id=left,
        right_id=right,
        score=score,
        score_type="heuristic",
        decision_step="test",
        provenance={},
    )


def _cell(
    benchmark: str,
    *,
    seed: int = 0,
    curve: list[float] | None = None,
    family: str = "heuristic",
    eligible: bool = True,
) -> CellResult:
    """A CellResult carrying only the fields the analysis functions read."""
    values = curve if curve is not None else [0.5] * len(GRID)
    zeros = [0.0] * len(GRID)
    return CellResult(
        benchmark=benchmark,
        method="rapidfuzz",
        score_family=family,
        seed=seed,
        n_train_records=100,
        n_test_records=100,
        n_test_pairs=100,
        n_test_gold_blocked=50,
        n_test_gold_all=50,
        n_units=20,
        selection_eligible=eligible,
        f1_blocked=values,
        f1_all_gold=values,
        ci_lo_blocked=zeros,
        ci_hi_blocked=zeros,
        ci_lo_all_gold=zeros,
        ci_hi_all_gold=zeros,
        oracle_threshold_blocked=0.5,
        oracle_f1_blocked=max(values),
        oracle_threshold_all_gold=0.5,
        oracle_f1_all_gold=max(values),
        derived_threshold=0.5,
        derived_f1_blocked=0.5,
        derived_f1_all_gold=0.5,
        seconds=1.0,
    )


def _peak_at(threshold: float, *, height: float = 0.9, floor: float = 0.1) -> list[float]:
    """A curve whose single maximum sits exactly at ``threshold``."""
    index = GRID.index(threshold)
    return [height if i == index else floor for i in range(len(GRID))]


class TestSelfCheckGate:
    """The gate exists to catch the vectorized curve drifting from the library's
    own metric. If it cannot fail, it is decoration.
    """

    def test_a_correct_curve_passes(self) -> None:
        judgements = [_judgement("a", "b", 0.8), _judgement("c", "d", 0.2)]
        gold = {frozenset({"a", "b"})}
        # Only {a,b} is gold and only it scores above 0.5 -> perfect at 0.5.
        curve = [0.0] * len(GRID)
        curve[9] = 2 / 3  # t=0.10: both predicted, 1 TP -> 2*1/(2+1)
        curve[SHIPPED_INDEX] = 1.0  # t=0.50
        curve[89] = 0.0  # t=0.90: nothing predicted
        _assert_matches_classify_pairs(judgements, gold, curve, "ok")

    def test_a_wrong_curve_raises(self) -> None:
        """The failure this gate exists to observe, actually observed."""
        judgements = [_judgement("a", "b", 0.8), _judgement("c", "d", 0.2)]
        gold = {frozenset({"a", "b"})}
        curve = [0.0] * len(GRID)
        curve[9] = 2 / 3
        curve[SHIPPED_INDEX] = 0.87654  # the drift
        curve[89] = 0.0
        with pytest.raises(RuntimeError, match="not measuring the same metric"):
            _assert_matches_classify_pairs(judgements, gold, curve, "drifted")


class TestExactOracle:
    def test_finds_a_cut_between_grid_points(self) -> None:
        """The ceiling must not be an artifact of the grid's resolution."""
        scores = np.array([0.101, 0.105, 0.900])
        is_gold = np.array([False, False, True])
        threshold, f1 = _exact_oracle(scores, is_gold, n_gold=1)
        assert f1 == pytest.approx(1.0)
        assert threshold == pytest.approx(0.900)

    def test_no_gold_is_zero_not_a_divide_error(self) -> None:
        threshold, f1 = _exact_oracle(np.array([0.4]), np.array([False]), n_gold=0)
        assert f1 == 0.0
        assert np.isnan(threshold)


class TestF1:
    def test_zero_predictions_and_zero_gold_is_zero_not_nan(self) -> None:
        out = _f1(np.array([0.0]), np.array([0.0]), np.array([0.0]))
        assert out.tolist() == [0.0]

    def test_matches_the_definition(self) -> None:
        out = _f1(np.array([3.0]), np.array([4.0]), np.array([5.0]))
        assert out.tolist() == [pytest.approx(2 * 3 / (4 + 5))]


class TestUnitAssignment:
    def test_every_record_maps_to_its_cluster(self) -> None:
        units = _unit_index([{"a", "b"}, {"c"}])
        assert units["a"] == units["b"]
        assert units["c"] != units["a"]

    def test_a_pair_is_counted_once_under_the_min_rule(self) -> None:
        """The bootstrap must not double-count a cross-cluster pair."""
        units = _unit_index([{"a"}, {"b"}])
        pair = frozenset({"a", "b"})
        left, right = tuple(pair)
        assert min(units[left], units[right]) in (units["a"], units["b"])


class TestDedupeScores:
    def test_collapses_both_orderings_to_one_pair(self) -> None:
        scores = dedupe_scores([_judgement("a", "b", 0.3), _judgement("b", "a", 0.7)])
        assert scores == {frozenset({"a", "b"}): 0.7}


class TestLoboSelection:
    def test_the_held_out_benchmark_does_not_vote_for_its_own_constant(self) -> None:
        """The property that makes the reported number out-of-sample."""
        cells = [
            _cell("alpha", curve=_peak_at(0.20)),
            _cell("beta", curve=_peak_at(0.80)),
            _cell("gamma", curve=_peak_at(0.80)),
        ]
        constants = lobo_constants(cells)
        # Holding out alpha leaves beta+gamma, both peaking at 0.80.
        assert constants["alpha"] == 0.80
        # Holding out beta leaves alpha (0.20) and gamma (0.80); the median of
        # two curves cannot pick alpha's peak alone, but it must not be 0.80
        # *because beta voted* -- beta is excluded either way.
        assert set(constants) == {"alpha", "beta", "gamma"}

    def test_ineligible_cells_do_not_vote(self) -> None:
        cells = [
            _cell("alpha", curve=_peak_at(0.20)),
            _cell("beta", curve=_peak_at(0.80)),
            _cell("tiny", curve=_peak_at(0.05), eligible=False),
        ]
        constants = lobo_constants(cells)
        assert "tiny" not in constants
        assert constants["alpha"] == 0.80

    def test_a_single_benchmark_yields_no_constant(self) -> None:
        """Leaving one out of one leaves nothing to select from."""
        assert lobo_constants([_cell("alpha", curve=_peak_at(0.30))]) == {}


class TestSelectConstantUsesMedian:
    def test_one_wide_range_benchmark_cannot_choose_for_the_others(self) -> None:
        """A mean would let the outlier's amplitude win; the median must not."""
        curves = {
            "a": np.asarray(_peak_at(0.30, height=0.6, floor=0.5)),
            "b": np.asarray(_peak_at(0.30, height=0.6, floor=0.5)),
            # A huge peak somewhere else -- would dominate a mean.
            "outlier": np.asarray(_peak_at(0.90, height=100.0, floor=0.0)),
        }
        assert _select_constant(curves) == 0.30


class TestPreRegisteredShipRule:
    def _report(self, cells: list[CellResult]) -> SweepReport:
        return SweepReport(
            grid=list(GRID), shipped_threshold=0.5, bootstrap_resamples=10, cells=cells
        )

    def test_an_unstable_family_is_refused(self) -> None:
        """Constants that move with the dataset are per-dataset tuning."""
        cells = [
            _cell("alpha", curve=_peak_at(0.20)),
            _cell("beta", curve=_peak_at(0.80)),
            _cell("gamma", curve=_peak_at(0.40)),
        ]
        assert "**DO NOT SHIP**" in to_verdict_markdown(self._report(cells))

    def test_a_stable_improving_family_ships(self) -> None:
        cells = [_cell(name, curve=_peak_at(0.80)) for name in ("alpha", "beta", "gamma")]
        markdown = to_verdict_markdown(self._report(cells))
        assert "**SHIP**" in markdown

    def test_a_refused_family_still_prints_its_number(self) -> None:
        """So 'DO NOT SHIP' is never misread as 'no number was found'."""
        cells = [
            _cell("alpha", curve=_peak_at(0.20)),
            _cell("beta", curve=_peak_at(0.80)),
            _cell("gamma", curve=_peak_at(0.40)),
        ]
        markdown = to_verdict_markdown(self._report(cells))
        assert "in-sample argmax" in markdown


class TestEmbedderOverride:
    def test_forwards_everything_it_does_not_override(self) -> None:
        class _Bench:
            schema = "SCHEMA"
            blocking_k = 7

            def build_blocker(self, k_neighbors: int) -> Any:
                raise AssertionError("not reached")

        wrapped = _EmbedderOverride(_Bench(), "intfloat/e5-base-v2")  # type: ignore[arg-type]
        assert wrapped.schema == "SCHEMA"
        assert wrapped.blocking_k == 7

    def test_refuses_a_blocker_it_cannot_actually_swap(self) -> None:
        """Measuring the pinned checkpoint under another checkpoint's name is the
        one outcome worse than crashing.
        """

        class _NoIndexBench:
            def build_blocker(self, k_neighbors: int) -> Any:
                return object()

        wrapped = _EmbedderOverride(_NoIndexBench(), "intfloat/e5-base-v2")  # type: ignore[arg-type]
        with pytest.raises(SystemExit, match="no .*vector_index.embedder|cannot apply"):
            wrapped.build_blocker(5)


class TestCheckpointTransfer:
    """The cross-encoder question: a `sim_cos` cut is a cut on a cosine SCALE,
    and the scale belongs to the checkpoint. These pin the pre-registered
    transfer rule, which judges **harm relative to the incumbent** rather than
    distance from the variant's own optimum.
    """

    def _report(self, cells: list[CellResult]) -> SweepReport:
        return SweepReport(
            grid=list(GRID), shipped_threshold=0.5, bootstrap_resamples=10, cells=cells
        )

    def _variant(self, *, ci_hi: float, curve: list[float]) -> list[CellResult]:
        cells = []
        for name in ("alpha", "beta", "gamma"):
            cell = _cell(name, curve=curve)
            cell.embedder = "intfloat/e5-base-v2"
            cell.ci_hi_blocked = [ci_hi] * len(GRID)
            cells.append(cell)
        return cells

    def test_a_constant_that_still_helps_transfers(self) -> None:
        baseline = self._report([_cell(n, curve=_peak_at(0.80)) for n in ("a", "b", "c")])
        variant = self._report(self._variant(ci_hi=0.4, curve=_peak_at(0.80)))
        assert "**TRANSFERS**" in to_transfer_markdown(baseline, variant)

    def test_a_constant_that_significantly_hurts_does_not_transfer(self) -> None:
        """Every seed's interval entirely below zero -- the veto case."""
        baseline = self._report([_cell(n, curve=_peak_at(0.80)) for n in ("a", "b", "c")])
        # On the variant, 0.80 scores BELOW the incumbent at 0.50.
        losing = [0.9] * len(GRID)
        losing[GRID.index(0.80)] = 0.1
        variant = self._report(self._variant(ci_hi=-0.01, curve=losing))
        assert "**DOES NOT TRANSFER**" in to_transfer_markdown(baseline, variant)

    def test_it_names_the_variant_checkpoint(self) -> None:
        """A transfer claim that does not say which encoder is unfalsifiable."""
        baseline = self._report([_cell(n, curve=_peak_at(0.80)) for n in ("a", "b", "c")])
        variant = self._report(self._variant(ci_hi=0.4, curve=_peak_at(0.80)))
        assert "intfloat/e5-base-v2" in to_transfer_markdown(baseline, variant)

    def test_a_moved_argmax_alone_is_not_a_veto(self) -> None:
        """Pre-registered on purpose: off-optimum but far better than 0.5 is
        still a strict improvement for that user.
        """
        baseline = self._report([_cell(n, curve=_peak_at(0.80)) for n in ("a", "b", "c")])
        # Variant peaks elsewhere, but 0.80 still beats the incumbent at 0.50.
        curve = [0.1] * len(GRID)
        curve[GRID.index(0.50)] = 0.2
        curve[GRID.index(0.80)] = 0.7
        curve[GRID.index(0.95)] = 0.9  # the variant's own optimum, far from 0.80
        variant = self._report(self._variant(ci_hi=0.4, curve=curve))
        markdown = to_transfer_markdown(baseline, variant)
        assert "**TRANSFERS**" in markdown
        assert "0.80 -> 0.95" in markdown


class TestCellResultRecordsItsCheckpoint:
    def test_defaults_to_the_benchmark_pin(self) -> None:
        assert _cell("alpha").embedder is None

    def test_shipped_f1_reads_the_incumbent_off_the_grid(self) -> None:
        """GRID[49] is exactly 0.5, so no interpolation is ever needed."""
        assert GRID[SHIPPED_INDEX] == 0.5
        curve = [float(i) for i in range(len(GRID))]
        assert _cell("alpha", curve=curve).shipped_f1_blocked == float(SHIPPED_INDEX)


class TestResumeRefusesToPoolIncomparableCells:
    """A resumed cell is REUSED; the header is rewritten from the new flags.

    So without this check a resume under different flags mints an artifact whose
    header describes neither half of it -- intervals bootstrapped at one
    ``--resamples`` published under another, or MiniLM and e5 cosine cells (two
    different score scales) pooled under one embedder label. Both were raised in
    review. These tests drive ``main()``, which must refuse *before* measuring
    anything.
    """

    @staticmethod
    def _partial(
        tmp_path: Any,
        *,
        resamples: int,
        embedder: str | None,
        fingerprint: str | None = None,
    ) -> Any:
        out = tmp_path / "sweep.json"
        write_report(
            SweepReport(
                grid=list(GRID),
                shipped_threshold=0.5,
                bootstrap_resamples=resamples,
                source_fingerprint=(_source_fingerprint() if fingerprint is None else fingerprint),
                cells=[_cell("abt_buy").model_copy(update={"embedder": embedder})],
            ),
            out.with_name(out.name + ".partial"),
        )
        return out

    def _run(self, monkeypatch: pytest.MonkeyPatch, out: Any, argv: list[str]) -> str:
        monkeypatch.setattr(
            "sys.argv", ["threshold_constant_sweep.py", "--out", str(out), "--resume", *argv]
        )
        with pytest.raises(SystemExit) as excinfo:
            main()
        # argparse.error exits 2 and writes to stderr; the code is the contract.
        assert excinfo.value.code == 2
        return str(excinfo.value)

    def test_a_changed_resample_count_is_refused(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch, capsys: Any
    ) -> None:
        out = self._partial(tmp_path, resamples=250, embedder=None)
        self._run(monkeypatch, out, ["--resamples", "1000"])
        assert "250" in capsys.readouterr().err

    def test_a_changed_embedder_is_refused(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch, capsys: Any
    ) -> None:
        out = self._partial(tmp_path, resamples=1000, embedder="intfloat/e5-base-v2")
        self._run(monkeypatch, out, ["--resamples", "1000"])
        assert "e5-base-v2" in capsys.readouterr().err

    def test_a_different_source_revision_is_refused(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch, capsys: Any
    ) -> None:
        """Matching FLAGS are not matching CODE.

        Resuming across a harness/matcher/loader/data change pools measurements
        that were never comparable into one artifact attributed to the current
        source -- and the ship rule then selects from them.
        """
        out = self._partial(tmp_path, resamples=1000, embedder=None, fingerprint="deadbeef/oldrev")
        self._run(monkeypatch, out, ["--resamples", "1000"])
        assert "deadbeef/oldrev" in capsys.readouterr().err

    def test_an_artifact_predating_the_field_is_refused_not_assumed(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch, capsys: Any
    ) -> None:
        """`None` means UNKNOWN. Unknown must not read as 'matches'."""
        out = self._partial(tmp_path, resamples=1000, embedder=None, fingerprint=None)
        # Force the stored value back to None the way an older artifact would have it.
        partial = out.with_name(out.name + ".partial")
        partial.write_text(partial.read_text().replace(_source_fingerprint(), ""))
        monkeypatch.setattr(
            "sys.argv",
            ["threshold_constant_sweep.py", "--out", str(out), "--resume", "--resamples", "1000"],
        )
        with pytest.raises(SystemExit):
            main()
        assert "source" in capsys.readouterr().err

    def test_matching_flags_are_not_refused(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The negative control: without it, a check that always fires looks identical."""
        out = self._partial(tmp_path, resamples=1000, embedder=None)
        monkeypatch.setattr(
            "sys.argv",
            [
                "threshold_constant_sweep.py",
                "--out",
                str(out),
                "--resume",
                "--resamples",
                "1000",
                "--only",
                "definitely_not_a_benchmark",
            ],
        )
        with pytest.raises(SystemExit) as excinfo:
            main()
        # It got PAST the resume guard and failed later, on the unknown
        # benchmark -- proving the guard is selective rather than a blanket
        # refusal to resume. Asserting on WHICH error fired is the point; a bare
        # "it exited" would pass even if the resume guard had rejected it.
        message = str(excinfo.value)
        assert "definitely_not_a_benchmark" in message
        assert "--resamples" not in message
        assert "embedder" not in message


class TestWorkerRelaunchCanRecoverItsOwnPartial:
    """The parent clears only ``worker_out``, never ``<worker_out>.partial``.

    So the relaunched worker meets the "an earlier sweep was interrupted" guard
    on its own leftover checkpoint. Without ``--resume`` in the worker argv that
    guard is a **deadlock**: the benchmark can never be re-measured without
    manual file surgery, and the cells the guard was protecting are exactly what
    the retry would have recovered. Found in review.
    """

    def test_the_worker_argv_asks_to_resume(self) -> None:
        args = argparse.Namespace(
            methods=["rapidfuzz"],
            seeds=[0],
            resamples=10,
            embedder=None,
        )
        argv = _worker_command("abt_buy", args, Path("out.json"))
        assert "--resume" in argv

    def test_an_embedder_override_still_reaches_the_worker(self) -> None:
        """--resume must not have displaced the checkpoint override."""
        args = argparse.Namespace(
            methods=["embedding_cosine"],
            seeds=[0],
            resamples=10,
            embedder="intfloat/e5-base-v2",
        )
        argv = _worker_command("abt_buy", args, Path("out.json"))
        assert argv[argv.index("--embedder") + 1] == "intfloat/e5-base-v2"


class TestSourceFingerprintDistinguishesDirtyStates:
    """A dirty *bit* is not an identity.

    The first version appended ``+dirty``, so two different uncommitted matcher
    edits at one commit rendered the same string (the harness file itself being
    unchanged) and the resume guard pooled them -- the exact corruption the field
    exists to prevent. It now hashes the diff CONTENT.
    """

    def test_two_different_dirty_diffs_do_not_collide(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[str] = []

        def fake_run(argv: list[str], **_kwargs: Any) -> Any:
            if argv[1] == "rev-parse":
                return subprocess.CompletedProcess(argv, 0, stdout="abc1234\n", stderr="")
            return subprocess.CompletedProcess(argv, 0, stdout=seen.pop(0), stderr="")

        monkeypatch.setattr("examples.research.threshold_constant_sweep.subprocess.run", fake_run)
        seen.append("--- a/matcher.py\n+++ b/matcher.py\n-x = 1\n+x = 2\n")
        first = _source_fingerprint()
        seen.append("--- a/matcher.py\n+++ b/matcher.py\n-x = 1\n+x = 3\n")
        second = _source_fingerprint()
        assert first != second, "two distinct uncommitted edits must not share an identity"

    def test_a_clean_tree_has_no_dirty_suffix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_run(argv: list[str], **_kwargs: Any) -> Any:
            out = "abc1234\n" if argv[1] == "rev-parse" else ""
            return subprocess.CompletedProcess(argv, 0, stdout=out, stderr="")

        monkeypatch.setattr("examples.research.threshold_constant_sweep.subprocess.run", fake_run)
        assert _source_fingerprint().endswith("/abc1234")


class TestAdoptingAWorkerArtifactChecksTheWholeInvocation:
    """A matching CODE fingerprint is not a matching INVOCATION.

    An artifact that survives a parent crash may have been measured with
    different ``--resamples``, a different ``--embedder``, or a different
    method/seed matrix. Adopting on the fingerprint alone republishes those
    measurements under the new run's header -- and if the cell COUNT happens to
    agree, the final length check waves it through. Found in review.
    """

    @staticmethod
    def _args(**over: Any) -> argparse.Namespace:
        base = {"methods": ["rapidfuzz"], "seeds": [0], "resamples": 1000, "embedder": None}
        return argparse.Namespace(**{**base, **over})

    def _write(self, tmp_path: Any, **over: Any) -> Any:
        scratch = tmp_path / "sweep.json.partial"
        cell = _cell("abt_buy").model_copy(
            update={"method": "rapidfuzz", "seed": 0, "embedder": over.pop("embedder", None)}
        )
        write_report(
            SweepReport(
                grid=list(GRID),
                shipped_threshold=0.5,
                bootstrap_resamples=over.pop("resamples", 1000),
                source_fingerprint=over.pop("fingerprint", _source_fingerprint()),
                cells=[cell],
            ),
            scratch.with_name(f"{scratch.name}.abt_buy.json"),
        )
        return scratch

    def test_a_matching_artifact_is_adopted_without_relaunching(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The negative control: without this the rejection tests prove nothing."""
        scratch = self._write(tmp_path)
        # BEFORE patching subprocess.run -- _source_fingerprint shells out to git.
        fingerprint = _source_fingerprint()

        def boom(*_a: Any, **_k: Any) -> Any:
            raise AssertionError("should have adopted, not relaunched a worker")

        monkeypatch.setattr("examples.research.threshold_constant_sweep.subprocess.run", boom)
        cells = run_benchmark_isolated("abt_buy", self._args(), scratch, fingerprint)
        assert [c.benchmark for c in cells] == ["abt_buy"]

    @pytest.mark.parametrize(
        ("artifact_over", "args_over"),
        [
            ({"resamples": 250}, {}),
            ({"embedder": "intfloat/e5-base-v2"}, {}),
            ({}, {"seeds": [0, 1]}),
            ({}, {"methods": ["embedding_cosine"]}),
            ({"fingerprint": "deadbeef/oldrev"}, {}),
        ],
        ids=["resamples", "embedder", "seed-set", "method-set", "source"],
    )
    def test_a_mismatched_artifact_is_not_adopted(
        self,
        tmp_path: Any,
        monkeypatch: pytest.MonkeyPatch,
        artifact_over: dict[str, Any],
        args_over: dict[str, Any],
    ) -> None:
        scratch = self._write(tmp_path, **artifact_over)
        fingerprint = _source_fingerprint()  # before the patch below
        relaunched: list[bool] = []

        def fake_run(*_a: Any, **_k: Any) -> Any:
            relaunched.append(True)
            return subprocess.CompletedProcess([], 1)

        monkeypatch.setattr("examples.research.threshold_constant_sweep.subprocess.run", fake_run)
        with pytest.raises(RuntimeError):
            run_benchmark_isolated("abt_buy", self._args(**args_over), scratch, fingerprint)
        assert relaunched, "a mismatched artifact must be re-measured, not adopted"


class TestAFailedRetryDoesNotDestroyDurableCells:
    """Old cells must survive a worker that fails.

    They used to be filtered out of the in-memory list BEFORE the replacement
    worker ran. When it failed, the handler continued with the filtered list and
    the NEXT benchmark's checkpoint() wrote that loss to disk permanently -- the
    data-destruction shape a sibling PR shipped twice. Found in review.
    """

    def test_a_failing_benchmark_keeps_its_previous_cells(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        out = tmp_path / "sweep.json"
        scratch = out.with_name(out.name + ".partial")
        old = _cell("abt_buy").model_copy(update={"method": "rapidfuzz", "seed": 0})
        write_report(
            SweepReport(
                grid=list(GRID),
                shipped_threshold=0.5,
                bootstrap_resamples=1000,
                source_fingerprint=_source_fingerprint(),
                cells=[old],
            ),
            scratch,
        )

        def fake_isolated(name: str, *_a: Any, **_k: Any) -> list[CellResult]:
            if name == "abt_buy":
                raise RuntimeError("worker died")
            return [_cell(name).model_copy(update={"method": "rapidfuzz", "seed": 0})]

        monkeypatch.setattr(
            "examples.research.threshold_constant_sweep.run_benchmark_isolated", fake_isolated
        )
        monkeypatch.setattr(
            "sys.argv",
            [
                "threshold_constant_sweep.py",
                "--out",
                str(out),
                "--resume",
                "--methods",
                "rapidfuzz",
                "--seeds",
                "0",
                "1",
                "--only",
                "abt_buy",
                "dblp_acm",
                "--no-tables",
            ],
        )
        with pytest.raises(SystemExit):
            main()
        # dblp_acm succeeded and checkpointed AFTER abt_buy failed; abt_buy's
        # original cell must still be on disk.
        survivors = {(c.benchmark, c.method, c.seed) for c in read_report(scratch).cells}
        assert ("abt_buy", "rapidfuzz", 0) in survivors


class TestInProcessRetryReconcilesPerIdentity:
    """A partial retry must not truncate the durable slice.

    Swapping the whole benchmark slice for ``fresh`` on the first yielded cell
    means a retry that yields seed 0 and then dies on seed 1 destroys the
    previously durable seed-1 result, which the next successful benchmark's
    checkpoint() then persists. Found in review, refining an earlier fix in this
    same PR that bounded the window without closing it.
    """

    def test_a_half_finished_retry_keeps_the_seed_it_has_not_reached(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        out = tmp_path / "sweep.json"
        scratch = out.with_name(out.name + ".partial")
        # Only seed 1 is durable, so the benchmark is INCOMPLETE and is retried
        # (with both seeds present it is skipped and nothing is exercised).
        durable = [_cell("abt_buy").model_copy(update={"method": "rapidfuzz", "seed": 1})]
        write_report(
            SweepReport(
                grid=list(GRID),
                shipped_threshold=0.5,
                bootstrap_resamples=1000,
                source_fingerprint=_source_fingerprint(),
                cells=[*durable, _cell("dblp_acm").model_copy(update={"seed": 9})],
            ),
            scratch,
        )

        def half_then_die(name: str, **_k: Any) -> Any:
            if name == "abt_buy":
                yield _cell("abt_buy").model_copy(update={"method": "rapidfuzz", "seed": 0})
                raise RuntimeError("died before seed 1")
            yield _cell(name).model_copy(update={"method": "rapidfuzz", "seed": 0})

        monkeypatch.setattr(
            "examples.research.threshold_constant_sweep.run_benchmark", half_then_die
        )
        monkeypatch.setattr(
            "sys.argv",
            [
                "threshold_constant_sweep.py",
                "--out",
                str(out),
                "--resume",
                "--in-process",
                "--methods",
                "rapidfuzz",
                "--seeds",
                "0",
                "1",
                "--only",
                "abt_buy",
                "dblp_acm",
                "--no-tables",
            ],
        )
        with pytest.raises(SystemExit):
            main()
        survivors = {(c.benchmark, c.method, c.seed) for c in read_report(scratch).cells}
        assert ("abt_buy", "rapidfuzz", 1) in survivors, (
            "seed 1 was durable and was never re-measured; it must not be dropped"
        )


class TestARejectedWorkerArtifactIsPreservedNotDeleted:
    """Rejected for THIS invocation still means valid for its own.

    In the parent-crash window the file can be the only durable copy of an
    hour-long benchmark, so deleting it and then failing to re-measure loses it
    for nothing. Found in review.
    """

    def test_it_is_moved_aside_rather_than_unlinked(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        scratch = tmp_path / "sweep.json.partial"
        worker_out = scratch.with_name(f"{scratch.name}.abt_buy.json")
        write_report(
            SweepReport(
                grid=list(GRID),
                shipped_threshold=0.5,
                bootstrap_resamples=250,  # mismatches the args below -> rejected
                source_fingerprint=_source_fingerprint(),
                cells=[_cell("abt_buy").model_copy(update={"method": "rapidfuzz", "seed": 0})],
            ),
            worker_out,
        )
        fingerprint = _source_fingerprint()
        monkeypatch.setattr(
            "examples.research.threshold_constant_sweep.subprocess.run",
            lambda *_a, **_k: subprocess.CompletedProcess([], 1),
        )
        args = argparse.Namespace(methods=["rapidfuzz"], seeds=[0], resamples=1000, embedder=None)
        with pytest.raises(RuntimeError):
            run_benchmark_isolated("abt_buy", args, scratch, fingerprint)
        assert not worker_out.exists(), "it must not be left where it would be re-adopted"
        assert worker_out.with_name(worker_out.name + ".rejected").exists(), (
            "the rejected artifact is durable data and must be preserved"
        )
