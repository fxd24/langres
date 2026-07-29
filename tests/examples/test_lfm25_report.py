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
        # The noise floor: far below the tuned models on abt_buy, level with them
        # on the saturated benchmark.
        _row(
            "random-init-control-350M",
            "abt_buy",
            parameter_count=354_483_968,
            candidate_recall=0.20,
        ),
        _row(
            "random-init-control-350M",
            "fodors_zagat",
            parameter_count=354_483_968,
            candidate_recall=0.988,
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
        assert ahead == ["abt_buy"]


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

    def test_a_benchmark_the_control_matches_is_called_uninformative(self, artifacts: Path) -> None:
        # fodors_zagat: control 0.9880 vs best tuned 0.9900 -> margin 0.0020.
        table, uninformative, informative = REPORT._noise_floor_table(
            REPORT._read_rows(REPORT.TUNED_ROWS), REPORT._read_rows(REPORT.BASE_ROWS)
        )

        assert "fodors_zagat" in uninformative
        assert "fodors_zagat" not in informative
        assert "**uninformative**" in table

    def test_a_benchmark_with_real_range_is_called_usable(self, artifacts: Path) -> None:
        # abt_buy: control 0.2000 vs best tuned 0.8700 -> margin 0.6700.
        _table, uninformative, informative = REPORT._noise_floor_table(
            REPORT._read_rows(REPORT.TUNED_ROWS), REPORT._read_rows(REPORT.BASE_ROWS)
        )

        assert "abt_buy" in informative
        assert "abt_buy" not in uninformative

    def test_the_margin_is_computed_not_asserted(self, artifacts: Path) -> None:
        table, _uninformative, _informative = REPORT._noise_floor_table(
            REPORT._read_rows(REPORT.TUNED_ROWS), REPORT._read_rows(REPORT.BASE_ROWS)
        )

        assert "+0.6700" in table  # abt_buy
        assert "+0.0020" in table  # fodors_zagat

    def test_the_write_up_names_the_unusable_benchmarks(self, artifacts: Path) -> None:
        report = REPORT.render()

        assert "cannot support an embedder claim" in report
        assert "is not evidence" in report

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
        table, _, _ = REPORT._noise_floor_table(
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


class TestLoadVerification:
    def test_the_collapse_under_the_native_class_is_shown(self, artifacts: Path) -> None:
        report = REPORT.render()

        assert "1.000000" in report
        assert "collapse" in report

    def test_the_randomised_backbone_is_shown_as_missing_keys(self, artifacts: Path) -> None:
        report = REPORT.render()

        assert "**148**" in report
        assert "**no**" in report  # not deterministic across two loads
