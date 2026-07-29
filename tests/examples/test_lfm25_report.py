"""Behaviour tests for the LFM2.5 write-up generator.

The generator exists so no measured number is ever typed into prose. These
tests cover what that promise actually rests on: that the tables are read from
the rows files, that the two studies stay separate, that a degenerate bootstrap
bound is called out rather than counted as a win, and that the licence blocker
survives a re-render.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).parents[2]
REPORT_PATH = ROOT / "examples" / "research" / "lfm25_report.py"


def _load() -> ModuleType:
    name = "example_lfm25_report"
    spec = importlib.util.spec_from_file_location(name, REPORT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


REPORT = _load()


def _row(model: str, benchmark: str, **overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "model": model,
        "benchmark": benchmark,
        "prompt_arm": "none",
        "k": 20,
        "status": "ok",
        "parameter_count": 109_000_000,
        # The cohort key. The noise floor subtracts study B's control from study
        # A's best tuned score, which is only meaningful within one cohort.
        "metric_revision": 1,
        "candidate_recall": 0.80,
        "reachable_recall_ceiling": 0.9,
        "recall_of_reachable": 0.888,
        "ci_clusters": 400,
    }
    row.update(overrides)
    return row


@pytest.fixture
def artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the generator at synthetic artifacts, leaving the real ones alone."""
    tuned = [
        _row("intfloat/e5-base-v2", "abt_buy"),
        _row("intfloat/e5-base-v2", "fodors_zagat", candidate_recall=0.99),
        _row(
            "LiquidAI/LFM2.5-Embedding-350M",
            "abt_buy",
            parameter_count=354_483_968,
            candidate_recall=0.87,
            vs_reference_delta=0.07,
            vs_reference_ci_low=0.02,
            vs_reference_ci_high=0.11,
            reference_model="intfloat/e5-base-v2",
        ),
        _row(
            "LiquidAI/LFM2.5-Embedding-350M",
            "fodors_zagat",
            parameter_count=354_483_968,
            candidate_recall=0.99,
            vs_reference_delta=0.0,
            vs_reference_ci_low=0.0,
            vs_reference_ci_high=0.03,
            reference_model="intfloat/e5-base-v2",
        ),
        # The checkpoint's own documented-prompt arm. The baseline does not run it,
        # so it has NO paired interval and cannot appear in a win/loss verdict --
        # while still scoring ABOVE the baseline's best (0.80). That is the real
        # study's shape on abt_buy and the reason the headline is qualified by arm.
        # Deliberately below this model's own paired best (0.87) so it does not
        # move the best-arm cell that unrelated assertions here depend on.
        _row(
            "LiquidAI/LFM2.5-Embedding-350M",
            "abt_buy",
            prompt_arm="documented",
            parameter_count=354_483_968,
            candidate_recall=0.85,
            reference_model="intfloat/e5-base-v2",
        ),
        # The narrow-but-real benchmark. Its paired interval CONTAINS zero on
        # purpose, so it is neither a win nor a loss and cannot perturb the
        # win/loss assertions -- it exists to give the noise floor a third shape.
        _row("intfloat/e5-base-v2", "walmart_amazon", candidate_recall=0.88),
        _row(
            "LiquidAI/LFM2.5-Embedding-350M",
            "walmart_amazon",
            parameter_count=354_483_968,
            candidate_recall=0.875,
            vs_reference_delta=-0.005,
            vs_reference_ci_low=-0.02,
            vs_reference_ci_high=0.01,
            reference_model="intfloat/e5-base-v2",
        ),
    ]
    base = [
        _row("LiquidAI/LFM2.5-Embedding-350M", "abt_buy", parameter_count=354_483_968),
        _row(
            "LiquidAI/LFM2.5-Encoder-350M",
            "abt_buy",
            parameter_count=354_483_968,
            candidate_recall=0.31,
            vs_reference_delta=-0.49,
            vs_reference_ci_low=-0.55,
            vs_reference_ci_high=-0.43,
            reference_model="LiquidAI/LFM2.5-Embedding-350M",
        ),
        # The noise floor, in all THREE shapes the verdict distinguishes. The
        # control's PAIRED INTERVAL decides whether a benchmark separates a trained
        # retriever from random weights; the gap only says how much room there is.
        # Conflating the two is what produced a published wrong claim, so each
        # branch has a fixture.
        #
        # 1. Separates, with room: interval excludes zero, gap 0.67.
        _row(
            "random-init-control-350M",
            "abt_buy",
            parameter_count=354_483_968,
            candidate_recall=0.20,
            vs_reference_delta=-0.60,
            vs_reference_ci_low=-0.70,
            vs_reference_ci_high=-0.50,
            reference_model="LiquidAI/LFM2.5-Embedding-350M",
        ),
        # 2. Cannot separate: interval CONTAINS zero. The gap (0.0020) is small
        #    too, but it is the interval that decides.
        _row(
            "random-init-control-350M",
            "fodors_zagat",
            parameter_count=354_483_968,
            candidate_recall=0.988,
            vs_reference_delta=-0.002,
            vs_reference_ci_low=-0.01,
            vs_reference_ci_high=0.01,
            reference_model="LiquidAI/LFM2.5-Embedding-350M",
        ),
        # 3. Separates, but narrowly -- the walmart_amazon shape, and the branch
        #    whose absence let a point-estimate cutoff call a benchmark blind when
        #    its own measured interval excluded zero. Gap 0.02 (< NARROW_RANGE),
        #    interval clear of zero.
        _row(
            "random-init-control-350M",
            "walmart_amazon",
            parameter_count=354_483_968,
            candidate_recall=0.86,
            vs_reference_delta=-0.02,
            vs_reference_ci_low=-0.03,
            vs_reference_ci_high=-0.01,
            reference_model="LiquidAI/LFM2.5-Embedding-350M",
        ),
    ]
    probe = {
        "transformers_version": "4.57.6",
        "lfm2_natively_implemented": True,
        "probe_texts": ["a", "b"],
        "remote_code": {
            "trust_remote_code=True": {
                "trust_remote_code": True,
                "instantiated_class": "transformers_modules.x.Lfm2BidirectionalModel",
                "from_checkpoint_code": True,
                "declared_auto_model": "modeling_lfm2_bidirectional.Lfm2BidirectionalModel",
                "cosine_between_unrelated_records": 0.169354,
                "max_abs_prompt_shift": 0.0763013,
            },
            "trust_remote_code=False": {
                "trust_remote_code": False,
                "instantiated_class": "transformers.models.lfm2.modeling_lfm2.Lfm2Model",
                "from_checkpoint_code": False,
                "declared_auto_model": "modeling_lfm2_bidirectional.Lfm2BidirectionalModel",
                "cosine_between_unrelated_records": 1.0,
                "max_abs_prompt_shift": 0.0,
            },
        },
        "weight_loading": {
            "LiquidAI/LFM2.5-Encoder-350M": {
                "AutoModel": {
                    "class": "Lfm2BidirectionalModel",
                    "missing_keys": 148,
                    "unexpected_keys": 148,
                    "two_load_max_drift": 0.168,
                },
                "AutoModelForMaskedLM": {
                    "class": "Lfm2BidirectionalForMaskedLM",
                    "missing_keys": 0,
                    "unexpected_keys": 0,
                },
            }
        },
    }

    tuned_path = tmp_path / "tuned.jsonl"
    base_path = tmp_path / "base.jsonl"
    tuned_path.write_text("".join(json.dumps(r) + "\n" for r in tuned))
    base_path.write_text("".join(json.dumps(r) + "\n" for r in base))
    (tmp_path / "probe.json").write_text(json.dumps(probe))

    monkeypatch.setattr(REPORT, "TUNED_ROWS", tuned_path)
    monkeypatch.setattr(REPORT, "BASE_ROWS", base_path)
    monkeypatch.setattr(REPORT, "LOAD_PROBE", tmp_path / "probe.json")
    return tmp_path


class TestGeneratedTables:
    def test_recall_comes_from_the_rows_file(self, artifacts: Path) -> None:
        report = REPORT.render()

        assert "0.8700" in report  # the LFM cell
        assert "0.8000" in report  # the baseline cell

    def test_a_missing_artifact_stops_rather_than_rendering_a_gap(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # monkeypatch, not a bare assignment: a leaked module-level path would
        # make every later test in this file depend on collection order.
        monkeypatch.setattr(REPORT, "TUNED_ROWS", tmp_path / "absent.jsonl")

        with pytest.raises(SystemExit, match="missing artifact"):
            REPORT.render()

    def test_the_baseline_is_labelled_and_sorts_first(self, artifacts: Path) -> None:
        models = REPORT._models(
            [
                _row("LiquidAI/LFM2.5-Embedding-350M", "b", parameter_count=354_483_968),
                _row("intfloat/e5-base-v2", "b"),
            ],
            "intfloat/e5-base-v2",
        )

        assert models[0] == "intfloat/e5-base-v2"

    def test_benchmarks_are_reported_separately_never_averaged(self, artifacts: Path) -> None:
        report = REPORT.render()

        assert "`abt_buy`" in report and "`fodors_zagat`" in report
        assert "never averaged across them" in report
        assert "mean across benchmarks" not in report


class TestDegenerateBounds:
    def test_a_bound_exactly_on_zero_is_not_counted_as_excluding_zero(
        self, artifacts: Path
    ) -> None:
        report = REPORT.render()

        assert "bound **exactly** 0" in report
        assert "does NOT exclude zero" in report

    def test_a_bound_exactly_on_zero_is_not_a_win(self, artifacts: Path) -> None:
        rows = REPORT._read_rows(REPORT.TUNED_ROWS)
        ahead, _behind = REPORT._wins(rows, "LiquidAI/LFM2.5-Embedding-350M", "intfloat/e5-base-v2")

        # abt_buy has low=0.02 (a real win); fodors_zagat has low=0.0 exactly.
        # Verdicts are ARM-qualified: an arm the baseline never runs has no paired
        # interval, so an unqualified benchmark name would claim more than was tested.
        assert ahead == ["`abt_buy` (none)"]


class TestHeadlineScope:
    """The headline must not claim more than the paired test covered."""

    def test_an_arm_without_a_counterpart_is_surfaced_not_dropped(self, artifacts: Path) -> None:
        """Regression: the checkpoint's own best arm was invisible to the verdict.

        A paired interval needs both models' per-record vectors, so an arm the
        baseline never runs cannot be tested. ``_wins`` skips it. On the real
        study that skipped arm is the documented-prompt arm, which on ``abt_buy``
        scores ABOVE the baseline's best while every testable arm scores below —
        so an unqualified "behind on abt_buy" asserted a checkpoint-level verdict
        the data does not support.
        """
        rows = REPORT._read_rows(REPORT.TUNED_ROWS)

        unpaired = REPORT._unpaired_arms(
            rows, "LiquidAI/LFM2.5-Embedding-350M", "intfloat/e5-base-v2"
        )

        assert unpaired, "an arm with no paired interval must be reported, not skipped"
        assert all(len(entry) == 4 for entry in unpaired)

    def test_the_write_up_qualifies_the_verdict_by_arm(self, artifacts: Path) -> None:
        report = REPORT.render()

        assert "per-ARM verdicts, not a checkpoint-level one" in report
        assert "cannot be tested at all" in report


class TestMultiplicity:
    def test_the_family_wise_rate_is_stated_and_not_claimed_away(self, artifacts: Path) -> None:
        report = REPORT.render()

        assert "nominal, per-benchmark and uncorrected" in report
        assert "family-wise error rate" in report
        assert "is **not** supported" in report


class TestStudySeparation:
    def test_the_base_encoders_carry_the_not_like_for_like_warning(self, artifacts: Path) -> None:
        report = REPORT.render()

        assert "Not a like-for-like comparison" in report
        assert "untrained" in report and "mean-pooling head" in report

    def test_the_differing_backbone_of_the_230m_is_stated(self, artifacts: Path) -> None:
        report = REPORT.render()

        assert "LiquidAI/LFM2.5-230M-Base" in report
        assert "confounds tuning with model size" in report

    def test_saturated_benchmark_is_marked_as_carrying_no_signal(self, artifacts: Path) -> None:
        report = REPORT.render()

        assert "uninformative in the strict sense" in report
        assert "must never be cited as evidence" in report


class TestNoiseFloor:
    """The random-init control is a first-class result, not an incident report."""

    def test_a_benchmark_whose_control_interval_spans_zero_cannot_separate(
        self, artifacts: Path
    ) -> None:
        # fodors_zagat: control 0.9880 vs best tuned 0.9900 -> margin 0.0020, and
        # the control's paired interval [-0.01, +0.01] CONTAINS zero.
        table, uninformative, informative, _narrow, _inv = REPORT._noise_floor_table(
            REPORT._read_rows(REPORT.TUNED_ROWS), REPORT._read_rows(REPORT.BASE_ROWS)
        )

        assert "fodors_zagat" in uninformative
        assert "fodors_zagat" not in informative
        assert "**cannot separate trained from random**" in table

    def test_a_benchmark_with_real_range_is_called_usable(self, artifacts: Path) -> None:
        # abt_buy: control 0.2000 vs best tuned 0.8700 -> margin 0.6700.
        _table, uninformative, informative, narrow, _inv = REPORT._noise_floor_table(
            REPORT._read_rows(REPORT.TUNED_ROWS), REPORT._read_rows(REPORT.BASE_ROWS)
        )

        assert "abt_buy" in informative
        assert "abt_buy" not in uninformative
        assert "abt_buy" not in narrow

    def test_a_narrow_benchmark_still_separates_and_is_not_called_blind(
        self, artifacts: Path
    ) -> None:
        """The published-wrong-claim regression, pinned.

        ``walmart_amazon``'s control sits only 0.02 below the best tuned model, but
        its paired interval [-0.03, -0.01] excludes zero. An earlier version used
        the 0.05 gap as the test and called the benchmark unable to tell a trained
        retriever from random weights -- a point estimate standing in for a
        variance estimate, contradicting the measured interval in the same rows.
        Narrow is a statement about RESOLUTION; blind is a statement about
        SIGNIFICANCE, and only the interval can make it.
        """
        table, uninformative, informative, narrow, _inv = REPORT._noise_floor_table(
            REPORT._read_rows(REPORT.TUNED_ROWS), REPORT._read_rows(REPORT.BASE_ROWS)
        )

        assert "walmart_amazon" in narrow
        assert "walmart_amazon" in informative
        assert "walmart_amazon" not in uninformative
        narrow_line = next(ln for ln in table.splitlines() if "walmart_amazon" in ln)
        assert "separates, but narrow" in narrow_line

    def test_the_margin_is_computed_not_asserted(self, artifacts: Path) -> None:
        table, *_ = REPORT._noise_floor_table(
            REPORT._read_rows(REPORT.TUNED_ROWS), REPORT._read_rows(REPORT.BASE_ROWS)
        )

        assert "+0.6700" in table  # abt_buy
        assert "+0.0020" in table  # fodors_zagat

    def test_the_write_up_names_the_unusable_benchmarks(self, artifacts: Path) -> None:
        report = REPORT.render()

        assert "cannot tell a trained retriever from random weights" in report
        assert "is not evidence" in report

    def test_the_write_up_separates_resolution_from_significance(self, artifacts: Path) -> None:
        report = REPORT.render()

        assert "resolution, not significance" in report
        assert "That separation is real" in report

    def test_a_tie_credits_a_deterministic_model(self, artifacts: Path) -> None:
        """A tie must not hand the credit to whichever row was hashed first.

        Regression: the table iterated a *set* of model names, and ``max``
        returns the FIRST maximum -- so on a benchmark where several models
        reach the same recall, the model named as "best tuned model" flipped
        between processes (set order varies with ``PYTHONHASHSEED``) while the
        number stayed correct. The file then failed its own
        reproduce-the-committed-table check.

        Reversing the input cannot reproduce this in-process, because set order
        for the same elements is stable within one interpreter. So this pins the
        guarantee instead: ties break on model name, and the fixture's
        ``fodors_zagat`` is a real tie at 0.99.
        """
        table, *_ = REPORT._noise_floor_table(
            REPORT._read_rows(REPORT.TUNED_ROWS), REPORT._read_rows(REPORT.BASE_ROWS)
        )

        tied_line = next(line for line in table.splitlines() if "fodors_zagat" in line)
        # max() on (recall, model) -> the alphabetically LAST name wins the tie,
        # and "intfloat/..." sorts after "LiquidAI/..." (lowercase 'i' > 'L').
        assert "intfloat/e5-base-v2" in tied_line
        assert "LiquidAI/LFM2.5-Embedding-350M" not in tied_line

    def test_rendering_twice_gives_the_same_bytes(self, artifacts: Path) -> None:
        """The harness must reproduce its own committed table on a re-run."""
        assert REPORT.render() == REPORT.render()

    def test_the_gap_is_called_observed_not_a_dynamic_range(self, artifacts: Path) -> None:
        """One seeded control does not establish a benchmark's range.

        Neither endpoint is a bound: a different seed moves the floor by an
        amount never measured (no replicates), and a better model moves the
        ceiling. Calling it "the entire usable dynamic range" made the
        percentage-of-range figures look calibrated when they are ratios to a
        single observed interval.
        """
        report = REPORT.render()

        assert "observed gap" in report
        assert "neither endpoint is a bound" in report
        assert "entire usable dynamic range" not in report.split("earlier wording")[0]

    def test_the_correction_quotes_intervals_read_from_the_rows(self, artifacts: Path) -> None:
        """The retraction's numbers must come from the artifact, not the prose.

        A supported re-run replaces the JSONL and regenerates this document; a
        hard-coded interval would then contradict the table directly above it.
        """
        base = REPORT._read_rows(REPORT.BASE_ROWS)

        intervals = REPORT._control_intervals(base, "walmart_amazon")

        # The fixture's control interval on walmart_amazon is [-0.03, -0.01].
        assert intervals == ["`[-0.0300, -0.0100]`"]
        assert "".join(intervals).strip("`") in REPORT.render()

    def test_no_correction_is_printed_when_nothing_was_misclassified(self, artifacts: Path) -> None:
        """Do not retract a claim about a benchmark this run did not measure."""
        base = REPORT._read_rows(REPORT.BASE_ROWS)

        assert REPORT._correction_paragraph(base, ["fodors_zagat"], []) == []


class TestCohortSafety:
    """The noise floor is the one figure computed ACROSS the two studies."""

    def test_mismatched_metric_revisions_refuse_to_render(self, artifacts: Path) -> None:
        """`LFM25_STUDY=a|b` makes drifting cohorts a supported workflow.

        A fresh study-A score minus a stale study-B control is not a
        like-for-like gap, and publishing it as one would repeat the
        cross-cohort error this document already had to retract.
        """
        tuned = REPORT._read_rows(REPORT.TUNED_ROWS)
        base = [dict(row, metric_revision=999) for row in REPORT._read_rows(REPORT.BASE_ROWS)]

        with pytest.raises(SystemExit, match="different metric revisions"):
            REPORT._assert_comparable_cohorts(tuned, base)

    def test_matching_cohorts_are_accepted(self, artifacts: Path) -> None:
        """The check must not block every legitimate render."""
        tuned = REPORT._read_rows(REPORT.TUNED_ROWS)
        base = REPORT._read_rows(REPORT.BASE_ROWS)

        assert REPORT._assert_comparable_cohorts(tuned, base) == "1"

    def test_the_report_states_the_comparison_is_cross_study(self, artifacts: Path) -> None:
        report = REPORT.render()

        assert "across the two studies" in report
        assert "metric revision" in report


class TestMatchedArms:
    """A pretraining claim cannot rest on a comparison where the prompt varies."""

    def test_the_control_encoder_comparison_holds_the_arm_fixed(self, artifacts: Path) -> None:
        """Regression: each side picked its OWN best arm.

        On the real rows that compared control `none` against encoder
        `instruct` on ``abt_buy``, and the reverse on ``amazon_google`` — so
        prompt configuration varied alongside the weights while the conclusion
        claimed only pretraining differed.
        """
        report = REPORT.render()

        assert "comparing the same prompt arm on both sides" in report
        assert "matched backbone and matched prompt arm" in report

    def test_only_shared_arms_are_compared(self, artifacts: Path) -> None:
        base = REPORT._read_rows(REPORT.BASE_ROWS)

        control_arms = REPORT._arm_cells(base, REPORT.CONTROL, "abt_buy")
        encoder_arms = REPORT._arm_cells(base, "LiquidAI/LFM2.5-Encoder-350M", "abt_buy")

        assert set(control_arms) & set(encoder_arms), "the fixture must share an arm"

    def test_best_arm_and_matched_arm_disagree_and_matched_arm_wins(self) -> None:
        """The regression itself: the two methods reach OPPOSITE verdicts here.

        Control is far ahead on ``instruct`` and behind on ``none``; the encoder
        is the other way round. Comparing each side's own best arm hands the
        benchmark to the control (0.90 vs 0.60) purely because they were
        measured under different prompts. Holding the arm fixed, the control
        loses ``none`` and so does not sweep the benchmark — which is the honest
        answer and the one a pretraining claim has to rest on.
        """
        encoder = REPORT.MATCHED_BACKBONE_ENCODER
        base = [
            _row(REPORT.CONTROL, "abt_buy", prompt_arm="instruct", candidate_recall=0.90),
            _row(REPORT.CONTROL, "abt_buy", prompt_arm="none", candidate_recall=0.50),
            _row(encoder, "abt_buy", prompt_arm="instruct", candidate_recall=0.40),
            _row(encoder, "abt_buy", prompt_arm="none", candidate_recall=0.60),
        ]

        # Best-arm-vs-best-arm, the old comparison, would call this a sweep.
        assert REPORT._best_arm(base, REPORT.CONTROL, "abt_buy")["candidate_recall"] == 0.90
        assert REPORT._best_arm(base, encoder, "abt_buy")["candidate_recall"] == 0.60

        lines = REPORT._control_vs_base_encoders(base)

        assert "**0 of 1**" in lines[0], lines[0]


class TestProvenance:
    """Provenance names the code that MEASURED, not the code checked out now."""

    def test_the_blob_comes_from_the_sidecar_not_from_head(self) -> None:
        """Regression: the first version read ``git rev-parse HEAD:<path>``.

        That made the line move with every commit, so the first change to the
        harness silently reattributed every measured row to code that never ran
        -- a false statement, not merely a stale one -- and it also broke the
        reproduce-the-committed-table property.
        """
        recorded = json.loads(REPORT.PROVENANCE.read_text())
        section = "\n".join(REPORT._provenance_section())

        for path, meta in recorded["blobs"].items():
            assert path in section
            assert meta["blob"][:12] in section

    def test_a_missing_sidecar_says_so_instead_of_guessing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Silence is the honest answer; inventing a hash from the checkout is not."""
        monkeypatch.setattr(REPORT, "PROVENANCE", tmp_path / "absent.json")

        section = "\n".join(REPORT._provenance_section())

        assert "Not recorded" in section
        assert "blob" not in section.split("Not recorded")[1]


class TestLicenceBlocker:
    def test_the_threshold_clause_is_quoted_from_the_committed_licence(self) -> None:
        clauses = REPORT._licence_clauses()

        assert any("$10,000,000" in clause for clause in clauses)
        assert any("Commercial Use Limitation" in clause for clause in clauses)
        assert any("not exceeding the Threshold" in clause for clause in clauses)

    def test_the_write_up_forbids_making_it_the_default(self, artifacts: Path) -> None:
        report = REPORT.render()

        assert "must not** become `DEFAULT_EMBEDDING_MODEL`" in report
        assert "opt-in named model with a licence note" in report

    def test_the_licence_artifact_is_committed_beside_the_report(self) -> None:
        """The claim has to be re-checkable without the network."""
        assert REPORT.LICENSE.exists()
        assert '"Threshold" shall mean' in REPORT.LICENSE.read_text()

    def test_the_nonprofit_exemption_is_quoted_with_the_restriction(self) -> None:
        """§5(c) exempts a qualifying non-profit; a restriction quoted without it is a misquote.

        The extraction keyed on ``^\\(a\\) ``/``^\\(b\\) ``, which is the shape of
        every lettered sub-clause in the document — so it silently dropped
        §5(c), and the prose built on it said every user above $10M was outside
        the licence. False for exactly the readers the exemption is written for.
        """
        clauses = REPORT._licence_clauses()

        assert any("Qualified Non-Profit Organization's use of the Work" in c for c in clauses)
        assert any('"Qualified Non-Profit Organization" shall mean' in c for c in clauses)

    def test_no_clause_from_another_section_is_quoted(self) -> None:
        """§4's redistribution conditions have nothing to do with being a default."""
        clauses = REPORT._licence_clauses()

        assert not any("You must give any other recipients" in c for c in clauses)
        assert not any("prominent notices stating that You changed" in c for c in clauses)

    def test_the_consequence_is_scoped_to_commercial_use(self, artifacts: Path) -> None:
        """The restriction is on Commercial Use, not on all use."""
        report = REPORT.render()

        assert "restriction is on **Commercial Use**, not on all use" in report
        assert "put the commercial use of every langres user" in report


class TestLoadVerification:
    def test_the_collapse_under_the_native_class_is_shown(self, artifacts: Path) -> None:
        report = REPORT.render()

        assert "1.000000" in report
        assert "collapse" in report

    def test_the_randomised_backbone_is_shown_as_missing_keys(self, artifacts: Path) -> None:
        report = REPORT.render()

        assert "**148**" in report
        assert "**no**" in report  # not deterministic across two loads


class TestPretrainingConclusionIsGated:
    """A conclusion about pretrained-vs-random needs rows that carry it.

    The harness records a failure row and continues when a model dies, so a
    rendered report can contain no control cells at all. The paragraph was
    unconditional: it asserted "the pretrained checkpoint scores worse than
    random weights" over an empty bullet list, with the failures table right
    there naming the missing data.
    """

    def test_no_comparable_rows_conclude_nothing(self) -> None:
        """Control died: no `ok` cell, so no pair shares an arm."""
        base = [
            _row(REPORT.MATCHED_BACKBONE_ENCODER, "abt_buy", candidate_recall=0.55),
            _row(REPORT.CONTROL, "abt_buy", status="process_failure", candidate_recall=None),
        ]

        section = "\n".join(REPORT._pretraining_section(base))

        assert "No pretrained-versus-random comparison is available" in section
        assert "scores worse than random weights" not in section

    def test_a_missing_matched_backbone_blocks_the_causal_claim(self) -> None:
        """Only the 350M encoder shares the control's backbone.

        With it absent, the 230M row still renders as an observation, but the
        sentence that isolates pretraining must not be printed.
        """
        base = [
            _row("LiquidAI/LFM2.5-Encoder-230M", "abt_buy", candidate_recall=0.40),
            _row(REPORT.CONTROL, "abt_buy", candidate_recall=0.60),
        ]

        section = "\n".join(REPORT._pretraining_section(base))

        assert "produced no comparable cell in this run" in section
        assert "no causal claim" in section
        assert "scores worse than random weights" not in section

    def test_a_control_that_loses_does_not_print_the_finding(self) -> None:
        """The rows going the other way must flip the sentence, not be ignored."""
        base = [
            _row(REPORT.MATCHED_BACKBONE_ENCODER, "abt_buy", candidate_recall=0.70),
            _row(REPORT.CONTROL, "abt_buy", candidate_recall=0.30),
        ]

        section = "\n".join(REPORT._pretraining_section(base))

        assert "does not outscore the real base encoders" in section
        assert "is therefore **not** reproduced here" in section

    def test_supported_rows_still_state_the_finding(self) -> None:
        """The gate must not silence a conclusion the rows do support."""
        base = [
            _row(REPORT.MATCHED_BACKBONE_ENCODER, "abt_buy", candidate_recall=0.31),
            _row(REPORT.CONTROL, "abt_buy", candidate_recall=0.60),
        ]

        section = "\n".join(REPORT._pretraining_section(base))

        assert "scores worse than random weights" in section
        assert "matched backbone and matched prompt arm" in section
        assert "**1 of 1**" in section

    def test_the_fodors_zagat_reframing_is_dropped_when_it_is_not_uninformative(
        self, artifacts: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """It names one benchmark, so it is only true while that row says so.

        Here the control's interval on ``fodors_zagat`` excludes zero, so the
        benchmark *does* separate a trained retriever from random weights and
        the reframing paragraph is simply false.
        """
        base = [
            row
            if row["model"] != REPORT.CONTROL or row["benchmark"] != "fodors_zagat"
            else dict(
                row,
                candidate_recall=0.40,
                vs_reference_delta=-0.59,
                vs_reference_ci_low=-0.65,
                vs_reference_ci_high=-0.53,
            )
            for row in REPORT._read_rows(REPORT.BASE_ROWS)
        ]
        path = tmp_path / "separating_base.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in base))
        monkeypatch.setattr(REPORT, "BASE_ROWS", path)

        report = REPORT.render()

        assert "This reframes `fodors_zagat` specifically" not in report

    def test_the_fodors_zagat_reframing_is_kept_when_it_is_uninformative(
        self, artifacts: Path
    ) -> None:
        """The committed fixture's control interval there DOES contain zero."""
        assert "This reframes `fodors_zagat` specifically" in REPORT.render()


class TestInvertedControl:
    """A control that BEATS every tuned model is not a 'narrow' benchmark."""

    def test_a_negative_margin_is_not_filed_as_narrow(self) -> None:
        """`margin <= 0` also satisfies `margin < NARROW_RANGE`.

        So an inverted benchmark landed in the narrow bucket, whose prose states
        that the control *is* significantly below the tuned models — the exact
        opposite of the rows. Constructed here because the committed sweep has
        no inverted benchmark, which is why the branch was never exercised.
        """
        tuned = [_row("intfloat/e5-base-v2", "inverted_bm", candidate_recall=0.85)]
        base = [
            _row(
                REPORT.CONTROL,
                "inverted_bm",
                candidate_recall=0.90,
                vs_reference_delta=0.05,
                vs_reference_ci_low=0.02,
                vs_reference_ci_high=0.08,
            )
        ]

        table, uninformative, informative, narrow, inverted = REPORT._noise_floor_table(tuned, base)

        assert "inverted_bm" in uninformative
        assert "inverted_bm" not in narrow
        assert "inverted_bm" not in informative
        line = next(ln for ln in table.splitlines() if "inverted_bm" in ln)
        assert "control BEATS every tuned model" in line

    def test_an_inverted_benchmark_is_reported_separately_from_a_blind_one(self) -> None:
        """One list, two OPPOSITE reasons, one justifying sentence.

        `uninformative` means "no ranking may be read off this", and the prose
        justified the whole list with "the control's paired interval there does
        not exclude zero" — which the inverted branch specifically *requires* to
        be false. The interval `[+0.02, +0.08]` excludes zero cleanly.
        """
        tuned = [
            _row("intfloat/e5-base-v2", "inverted_bm", candidate_recall=0.85),
            _row("intfloat/e5-base-v2", "blind_bm", candidate_recall=0.85),
        ]
        base = [
            _row(
                REPORT.CONTROL,
                "inverted_bm",
                candidate_recall=0.90,
                vs_reference_delta=0.05,
                vs_reference_ci_low=0.02,
                vs_reference_ci_high=0.08,
            ),
            _row(
                REPORT.CONTROL,
                "blind_bm",
                candidate_recall=0.84,
                vs_reference_delta=-0.01,
                vs_reference_ci_low=-0.05,
                vs_reference_ci_high=0.03,
            ),
        ]

        _, uninformative, _, _, inverted = REPORT._noise_floor_table(tuned, base)

        assert inverted == ["inverted_bm"]
        assert sorted(uninformative) == ["blind_bm", "inverted_bm"]
        # The retraction's closing sentence names what "genuinely fails to
        # separate" -- true of the blind one, false of the inverted one.
        blind = [b for b in uninformative if b not in inverted]
        assert blind == ["blind_bm"]


class TestCrossedMetricRevisions:
    """Two matching SETS of revisions are not one shared revision."""

    def test_crossed_cohorts_are_refused(self) -> None:
        """Both sides hold `{1, 2}`, so the set comparison passed.

        Every per-benchmark subtraction still crosses revisions: study A holds
        `abt_buy` at 1 and `fodors_zagat` at 2, study B the reverse.
        """
        tuned = [
            _row("intfloat/e5-base-v2", "abt_buy", metric_revision=1),
            _row("intfloat/e5-base-v2", "fodors_zagat", metric_revision=2),
        ]
        base = [
            _row(REPORT.CONTROL, "abt_buy", metric_revision=2),
            _row(REPORT.CONTROL, "fodors_zagat", metric_revision=1),
        ]

        assert {r["metric_revision"] for r in tuned} == {r["metric_revision"] for r in base}
        with pytest.raises(SystemExit, match="different metric revisions"):
            REPORT._assert_comparable_cohorts(tuned, base)

    def test_one_shared_revision_is_accepted(self) -> None:
        tuned = [_row("intfloat/e5-base-v2", "abt_buy")]
        base = [_row(REPORT.CONTROL, "abt_buy")]

        assert REPORT._assert_comparable_cohorts(tuned, base) == "1"


class TestCohortIsNotMetricRevision:
    """`metric_revision` tracks how a metric is COMPUTED, not over what."""

    def test_different_populations_are_refused(self) -> None:
        """A dataset refresh keeps revision 1 and changes the denominator.

        The revision guard passes and the noise floor then subtracts recalls
        measured over different record counts — in a document whose own text
        says cross-cohort subtraction is refused.
        """
        tuned = [_row("intfloat/e5-base-v2", "abt_buy", n_records=2173, n_gold_pairs=1044)]
        base = [_row(REPORT.CONTROL, "abt_buy", n_records=2100, n_gold_pairs=1044)]

        assert {r["metric_revision"] for r in tuned} == {r["metric_revision"] for r in base}
        with pytest.raises(SystemExit, match="different populations"):
            REPORT._assert_comparable_cohorts(tuned, base)

    def test_a_split_population_inside_ONE_study_is_refused(self) -> None:
        """`setdefault` kept only the first tuple, so this passed.

        A partial model re-run across a dataset change leaves both populations
        in one file; the noise floor then picks a refreshed best-tuned row and
        subtracts a stale control.
        """
        tuned = [
            _row("intfloat/e5-base-v2", "abt_buy", n_records=2173, n_gold_pairs=1044),
            _row("LiquidAI/LFM2.5-Embedding-350M", "abt_buy", n_records=2100, n_gold_pairs=1044),
        ]
        base = [_row(REPORT.CONTROL, "abt_buy", n_records=2173, n_gold_pairs=1044)]

        with pytest.raises(SystemExit, match="DIFFERENT"):
            REPORT._assert_comparable_cohorts(tuned, base)

    def test_a_benchmark_present_in_only_one_study_is_not_compared(self) -> None:
        """Study B measures fewer benchmarks; that is not a cohort mismatch."""
        tuned = [
            _row("intfloat/e5-base-v2", "abt_buy", n_records=2173, n_gold_pairs=1044),
            _row("intfloat/e5-base-v2", "wdc_computers", n_records=999, n_gold_pairs=100),
        ]
        base = [_row(REPORT.CONTROL, "abt_buy", n_records=2173, n_gold_pairs=1044)]

        assert REPORT._assert_comparable_cohorts(tuned, base) == "1"

    def test_the_committed_artifacts_agree(self, artifacts: Path) -> None:
        """The real rows must pass, or the guard is unshippable."""
        tuned = REPORT._read_rows(REPORT.TUNED_ROWS)
        base = REPORT._read_rows(REPORT.BASE_ROWS)

        assert REPORT._assert_comparable_cohorts(tuned, base)


class TestLoadProbeStaleness:
    """The loading section is a property of the INSTALLED transformers."""

    def test_a_probe_from_another_environment_is_disclosed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(REPORT, "_installed_transformers", lambda: "5.0.0")

        lines = REPORT._probe_staleness({"transformers_version": "4.57.6"})

        assert any("captured under transformers `4.57.6`" in line for line in lines)
        assert any("`5.0.0`" in line for line in lines)

    def test_a_matching_environment_says_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(REPORT, "_installed_transformers", lambda: "4.57.6")
        monkeypatch.setattr(
            REPORT,
            "_window_bounds",
            lambda: ("2026-07-29T05:00:00+02:00", "2026-07-29T07:00:00+02:00"),
        )

        probe = {"transformers_version": "4.57.6", "captured_at": "2026-07-29T06:00:00+02:00"}

        assert REPORT._probe_staleness(probe) == []

    def test_transformers_absent_makes_no_version_claim(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Silence beats asserting a mismatch against a version that is not there."""
        monkeypatch.setattr(REPORT, "_installed_transformers", lambda: None)
        monkeypatch.setattr(REPORT, "_window_bounds", lambda: None)

        probe = {"transformers_version": "4.57.6", "captured_at": "2026-07-29T05:00:00+02:00"}

        assert REPORT._probe_staleness(probe) == []

    def test_a_probe_older_than_the_window_is_disclosed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Version equality is NOT freshness.

        A refresh that failed offline leaves the old probe in place; if its
        transformers version happens to match, the version check says nothing
        while remote code and cache contents may have moved.
        """
        monkeypatch.setattr(REPORT, "_installed_transformers", lambda: "4.57.6")
        monkeypatch.setattr(
            REPORT,
            "_window_bounds",
            lambda: ("2026-07-29T09:00:00+02:00", "2026-07-29T11:00:00+02:00"),
        )

        lines = REPORT._probe_staleness(
            {"transformers_version": "4.57.6", "captured_at": "2026-07-28T22:00:00+02:00"}
        )

        assert any("predates the measurement window" in line for line in lines)

    def test_a_probe_inside_the_window_is_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(REPORT, "_installed_transformers", lambda: "4.57.6")
        monkeypatch.setattr(
            REPORT,
            "_window_bounds",
            lambda: ("2026-07-29T09:00:00+02:00", "2026-07-29T11:00:00+02:00"),
        )

        probe = {"transformers_version": "4.57.6", "captured_at": "2026-07-29T09:30:00+02:00"}

        assert REPORT._probe_staleness(probe) == []

    def test_a_probe_with_no_capture_time_says_so(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The committed probe predates the field; that is 'unknown', not 'fresh'."""
        monkeypatch.setattr(REPORT, "_installed_transformers", lambda: "4.57.6")

        lines = REPORT._probe_staleness({"transformers_version": "4.57.6"})

        assert any("capture time was not recorded" in line for line in lines)


class TestProbeTimestampsAreInstants:
    """Two ISO-8601 stamps at different UTC offsets do not sort chronologically.

    Every stamp in this study carries ``+02:00`` — one host, one sweep — so the
    original string comparison was right by accident. Both of these cases are
    inside this study's real window bounds and both come out **wrong** under a
    lexicographic test, in opposite directions.
    """

    WINDOW = ("2026-07-29T01:18:12+02:00", "2026-07-29T03:44:51+02:00")

    def test_a_probe_after_the_close_is_not_read_as_inside(self) -> None:
        """04:00 local, half an hour past the close — but ``"02" < "03"`` as text."""
        assert REPORT._outside_window("2026-07-29T02:00:00+00:00", self.WINDOW) == "postdates"

    def test_a_probe_inside_the_window_is_not_read_as_predating(self) -> None:
        """02:30 local, comfortably inside — but ``"00:30" < "01:18"`` as text."""
        assert REPORT._outside_window("2026-07-29T00:30:00+00:00", self.WINDOW) is None

    def test_z_suffix_and_offset_agree(self) -> None:
        """The same instant written two ways must give the same verdict."""
        assert REPORT._outside_window(
            "2026-07-29T02:00:00Z", self.WINDOW
        ) == REPORT._outside_window("2026-07-29T04:00:00+02:00", self.WINDOW)

    def test_an_offsetless_stamp_is_disclosed_not_silently_passed(self) -> None:
        """No offset means no instant. Comparing it would raise; ignoring it would lie."""
        assert REPORT._outside_window("2026-07-29T02:00:00", self.WINDOW) == "unanchored"

    def test_an_unparseable_stamp_is_disclosed(self) -> None:
        assert REPORT._outside_window("last Tuesday", self.WINDOW) == "unanchored"

    def test_the_unanchored_note_does_not_claim_staleness(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(REPORT, "_installed_transformers", lambda: "4.57.6")
        monkeypatch.setattr(REPORT, "_window_bounds", lambda: self.WINDOW)

        lines = REPORT._probe_staleness(
            {"transformers_version": "4.57.6", "captured_at": "2026-07-29T02:00:00"}
        )

        assert any("cannot be placed against the measurement window" in line for line in lines)
        assert not any("predates" in line or "postdates" in line for line in lines)


class TestSignificanceIsPairedWithOneBaseline:
    """The control's interval covers study B's baseline and no other model.

    The noise-floor table's *gap* column runs to the best model in study A. On
    every benchmark here those are different models, so "the control is
    significantly below the tuned models" asserted a comparison that was never
    computed.
    """

    def test_the_reference_is_read_from_the_rows(self, artifacts: Path) -> None:
        base = REPORT._read_rows(REPORT.BASE_ROWS)

        assert REPORT._control_reference(base) == REPORT.BASE_BASELINE

    def test_disagreeing_references_name_nothing(self) -> None:
        """Naming the wrong model is worse than naming none."""
        rows = [
            {"model": REPORT.CONTROL, "reference_model": "a", "vs_reference_ci_low": -0.1},
            {"model": REPORT.CONTROL, "reference_model": "b", "vs_reference_ci_low": -0.1},
        ]

        assert REPORT._control_reference(rows) is None

    def test_the_narrow_verdict_names_the_paired_model(self, artifacts: Path) -> None:
        report = REPORT.render()

        assert "significantly below the tuned models" not in report
        assert f"significantly below `{REPORT.BASE_BASELINE}`" in report

    def test_rows_whose_best_model_is_unpaired_are_marked(self, artifacts: Path) -> None:
        report = REPORT.render()

        assert "no paired interval was computed between the control" in report
        assert "Nothing in this table is a significance test against the named best model" in report


class TestPartialWindowCaveat:
    """`studies_measured` is intent; `window_complete` is what was reached."""

    def test_an_aborted_both_study_run_still_warns(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The caveat went silent in the one case it exists for.

        An aborted `LFM25_STUDY=both` sweep leaves `studies_measured == ['a',
        'b']`, so keying only on that list returned no warning beside the
        cross-study subtraction — over rows the window never touched.
        """
        sidecar = json.loads(REPORT.PROVENANCE.read_text())
        sidecar["studies_measured"] = ["a", "b"]
        sidecar["window_complete"] = False
        path = tmp_path / "prov.json"
        path.write_text(json.dumps(sidecar))
        monkeypatch.setattr(REPORT, "PROVENANCE", path)

        caveat = "\n".join(REPORT._cross_study_caveat())

        assert "aborted before finishing every planned study" in caveat

    def test_a_completed_both_study_run_stays_quiet(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sidecar = json.loads(REPORT.PROVENANCE.read_text())
        sidecar["studies_measured"] = ["a", "b"]
        sidecar["window_complete"] = True
        path = tmp_path / "prov.json"
        path.write_text(json.dumps(sidecar))
        monkeypatch.setattr(REPORT, "PROVENANCE", path)

        assert REPORT._cross_study_caveat() == []


class TestTiesAreNotWins:
    """`>=` counts an exact tie as a win; "outscores" is directional."""

    def test_an_all_tie_pair_makes_no_directional_claim(self) -> None:
        """Recall ties are ordinary on saturated cells."""
        base = [
            _row(REPORT.MATCHED_BACKBONE_ENCODER, "fodors_zagat", candidate_recall=1.0),
            _row(REPORT.CONTROL, "fodors_zagat", candidate_recall=1.0),
        ]

        section = "\n".join(REPORT._pretraining_section(base))

        assert "matches, but never outscores" in section
        assert "exact recall tie" in section
        assert "scores worse than random weights" not in section
        # The bullet's own wording -- "match or beat" -- is still true of a tie.
        assert "**1 of 1**" in section

    def test_a_strict_win_still_states_the_finding(self) -> None:
        base = [
            _row(REPORT.MATCHED_BACKBONE_ENCODER, "abt_buy", candidate_recall=0.31),
            _row(REPORT.CONTROL, "abt_buy", candidate_recall=0.60),
        ]

        section = "\n".join(REPORT._pretraining_section(base))

        assert "scores worse than random weights" in section
        assert "strictly ahead" in section


class TestDirtyTreeDisclosure:
    """A sweep measured on a modified tree cannot be reproduced from a commit."""

    def test_dirty_paths_are_surfaced_not_swallowed(
        self, artifacts: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sidecar = json.loads(REPORT.PROVENANCE.read_text())
        sidecar["_comment"] = ["live capture"]  # not the retrospective branch
        sidecar["verified_unchanged_during_run"] = True
        sidecar["dirty_at_start"] = ["src/langres/core/blockers/vector.py"]
        path = tmp_path / "prov.json"
        path.write_text(json.dumps(sidecar))
        monkeypatch.setattr(REPORT, "PROVENANCE", path)

        section = "\n".join(REPORT._provenance_section())

        assert "Measured on a MODIFIED tree" in section
        assert "exists in no commit" in section
        assert "`src/langres/core/blockers/vector.py`" in section
