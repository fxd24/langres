"""Tests for the ``FailureModeSection`` + ``profile_failure_mode``.

Covers the log<->gold join and the FP/FN/abstain split, the score-band /
field-emptiness / source slicing (error rate + lift), the error-vs-success score
overlay, slice ranking + the display cap, the render ladder (markdown / summary /
rows / panels), graceful degrade on missing judgements / gold / records, empty
and degenerate inputs (all-correct, all-abstain), and the section's wiring into
``DataProfileReport`` (``from_records(failure_mode=...)`` + the ``include=``
filter).
"""

from __future__ import annotations

from typing import Any

from langres.data.data_profile import (
    DataProfileReport,
    FailureModeSection,
    FailureSlice,
    from_records,
    profile_failure_mode,
)


def _row(left: str, right: str, score: float | None, verdict: bool | None) -> dict[str, Any]:
    """One JudgementLog-shaped row."""
    return {"left_id": left, "right_id": right, "score": score, "verdict": verdict}


def _records(*items: tuple[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """An id -> field-mapping table for the field/source slices."""
    return dict(items)


def _basic() -> FailureModeSection:
    """A small, deterministic profile exercising every branch (FP, FN, TP, TN, abstain)."""
    judgements = [
        _row("a1", "b1", 0.90, True),  # gold match, predicted match  -> correct
        _row("a2", "b2", 0.55, True),  # gold non-match, predicted match -> FP
        _row("a3", "b3", 0.45, False),  # gold match, predicted non-match -> FN
        _row("a4", "b4", 0.10, False),  # gold non-match, predicted non-match -> correct
        _row("a5", "b5", None, None),  # abstain
    ]
    gold = {frozenset({"a1", "b1"}), frozenset({"a3", "b3"})}
    records = _records(
        ("a1", {"id": "a1", "name": "Acme", "price": "10", "source": "abt"}),
        ("b1", {"id": "b1", "name": "Acme Inc", "price": "10", "source": "buy"}),
        ("a2", {"id": "a2", "name": "Zephyr", "price": "", "source": "abt"}),  # price empty
        ("b2", {"id": "b2", "name": "Zephyr Co", "price": "5", "source": "buy"}),
        ("a3", {"id": "a3", "name": "Nimbus", "price": None, "source": "abt"}),  # price empty
        ("b3", {"id": "b3", "name": "Nimbus LLC", "price": "8", "source": "buy"}),
        ("a4", {"id": "a4", "name": "Vertex", "price": "3", "source": "abt"}),
        ("b4", {"id": "b4", "name": "Orion", "price": "9", "source": "abt"}),  # same-source
        ("a5", {"id": "a5", "name": "Titan", "price": "1", "source": "abt"}),
        ("b5", {"id": "b5", "name": "Nova", "price": "2", "source": "buy"}),
    )
    section = profile_failure_mode(judgements, gold_pairs=gold, records=records, n_score_bands=5)
    assert section is not None
    return section


class TestJoinAndSplit:
    def test_confusion_counts(self) -> None:
        section = _basic()
        assert section.n_judged == 5
        assert section.n_correct == 2  # a1/b1, a4/b4
        assert section.n_errors == 2  # a2/b2 (FP), a3/b3 (FN)
        assert section.n_false_positive == 1
        assert section.n_false_negative == 1
        assert section.n_abstain == 1

    def test_derived_rates(self) -> None:
        section = _basic()
        assert section.n_confident == 4
        assert section.error_rate == 0.5  # 2 errors / 4 confident
        assert section.abstain_rate == 0.2  # 1 / 5

    def test_gold_normalizes_from_tuples_and_ignores_order(self) -> None:
        # gold given as (left, right) tuples in the *opposite* order to the log.
        judgements = [_row("a", "b", 0.9, True)]
        section = profile_failure_mode(judgements, gold_pairs=[("b", "a")])
        assert section is not None
        assert section.n_correct == 1 and section.n_errors == 0

    def test_gold_drops_malformed_pairs(self) -> None:
        # A self-pair (one distinct id) is not a valid gold pair and is dropped.
        judgements = [_row("a", "b", 0.9, True)]
        section = profile_failure_mode(judgements, gold_pairs=[("a", "a"), ("x", "y")])
        assert section is not None
        assert section.n_false_positive == 1  # (a,b) not gold -> predicted match is a FP

    def test_int_ids_are_stringified_on_join(self) -> None:
        # Log ids may arrive as ints; the join keys them as strings (like the log).
        judgements = [_row("1", "2", 0.9, True)]
        section = profile_failure_mode(judgements, gold_pairs=[(1, 2)])
        assert section is not None
        assert section.n_correct == 1


class TestScoreSlicesAndOverlay:
    def test_score_band_slice_rates_and_lift(self) -> None:
        section = _basic()
        bands = {s.value: s for s in section.slices if s.dimension == "score_band"}
        # the two errors (0.55 and 0.45) both fall in the 0.40-0.60 band -> 100% error
        mid = bands["0.40-0.60"]
        assert mid.n == 2 and mid.n_errors == 2
        assert mid.error_rate == 1.0
        assert mid.lift == 2.0  # 1.0 / overall 0.5

    def test_score_one_lands_in_last_band(self) -> None:
        judgements = [_row("a", "b", 1.0, True)]  # correct, score exactly 1.0
        section = profile_failure_mode(judgements, gold_pairs=[("a", "b")], n_score_bands=5)
        assert section is not None
        bands = {s.value for s in section.slices if s.dimension == "score_band"}
        assert "0.80-1.00" in bands

    def test_overlay_counts_split_error_vs_success(self) -> None:
        section = _basic()
        assert len(section.score_edges) == 6  # n_bands + 1
        assert sum(section.error_counts) == 2  # the 2 errors carried a score
        assert sum(section.success_counts) == 2  # the 2 correct verdicts carried a score

    def test_deciders_have_no_score_overlay(self) -> None:
        # Decision-only rows (score=None, verdict set) -> no scores to bin, no bands.
        judgements = [_row("a", "b", None, True), _row("c", "d", None, False)]
        section = profile_failure_mode(judgements, gold_pairs=[("a", "b")])
        assert section is not None
        assert section.score_edges == []
        assert section.error_counts == [] and section.success_counts == []
        assert not any(s.dimension == "score_band" for s in section.slices)


class TestCategoricalSlices:
    def test_field_empty_slice(self) -> None:
        section = _basic()
        empty_price = next(s for s in section.slices if s.dimension == "empty:price")
        # a2/b2 (empty) and a3/b3 (None) both have an empty price and are both errors
        assert empty_price.value == "either side empty"
        assert empty_price.n == 2 and empty_price.n_errors == 2
        assert empty_price.error_rate == 1.0 and empty_price.lift == 2.0

    def test_source_cross_vs_same(self) -> None:
        section = _basic()
        by_value = {s.value: s for s in section.slices if s.dimension == "source"}
        assert set(by_value) == {"cross-source", "same-source"}
        # a4/b4 is the only same-source confident pair, and it is correct.
        assert by_value["same-source"].n == 1 and by_value["same-source"].n_errors == 0

    def test_id_and_source_excluded_from_field_slices(self) -> None:
        section = _basic()
        dims = {s.dimension for s in section.slices}
        assert "empty:id" not in dims  # id is structural, never a content field
        assert "empty:source" not in dims  # source has its own dimension

    def test_no_records_yields_only_score_slices(self) -> None:
        judgements = [_row("a", "b", 0.9, True), _row("c", "d", 0.4, False)]
        section = profile_failure_mode(judgements, gold_pairs=[("c", "d")], records=None)
        assert section is not None
        assert {s.dimension for s in section.slices} == {"score_band"}

    def test_source_key_none_skips_source_slice(self) -> None:
        section = _basic_with(source_key=None)
        assert not any(s.dimension == "source" for s in section.slices)

    def test_source_absent_from_records_skips_source_slice(self) -> None:
        judgements = [_row("a", "b", 0.9, True)]
        records = _records(("a", {"id": "a", "name": "x"}), ("b", {"id": "b", "name": "y"}))
        section = profile_failure_mode(judgements, gold_pairs=[("a", "b")], records=records)
        assert section is not None
        assert not any(s.dimension == "source" for s in section.slices)


def _basic_with(**kwargs: Any) -> FailureModeSection:
    """A ``_basic``-shaped profile with overridable ``profile_failure_mode`` kwargs."""
    judgements = [
        _row("a1", "b1", 0.9, True),
        _row("a2", "b2", 0.55, True),
        _row("a3", "b3", 0.45, False),
    ]
    gold = {frozenset({"a1", "b1"}), frozenset({"a3", "b3"})}
    records = _records(
        ("a1", {"id": "a1", "name": "Acme", "source": "abt"}),
        ("b1", {"id": "b1", "name": "Acme Inc", "source": "buy"}),
        ("a2", {"id": "a2", "name": "Zephyr", "source": "abt"}),
        ("b2", {"id": "b2", "name": "Zephyr Co", "source": "buy"}),
        ("a3", {"id": "a3", "name": "Nimbus", "source": "abt"}),
        ("b3", {"id": "b3", "name": "Nimbus LLC", "source": "buy"}),
    )
    section = profile_failure_mode(judgements, gold_pairs=gold, records=records, **kwargs)
    assert section is not None
    return section


class TestRankingAndCap:
    def test_slices_ranked_by_lift_descending(self) -> None:
        section = _basic()
        lifts = [s.lift for s in section.slices if s.lift is not None]
        assert lifts == sorted(lifts, reverse=True)

    def test_max_slices_caps_and_counts_hidden(self) -> None:
        section = profile_failure_mode(
            _basic_judgements(), gold_pairs=_basic_gold(), records=_basic_records(), max_slices=2
        )
        assert section is not None
        assert len(section.slices) == 2
        assert section.n_slices_hidden >= 1

    def test_no_cap_hides_nothing(self) -> None:
        section = _basic()
        assert section.n_slices_hidden == 0


def _basic_judgements() -> list[dict[str, Any]]:
    return [
        _row("a1", "b1", 0.90, True),
        _row("a2", "b2", 0.55, True),
        _row("a3", "b3", 0.45, False),
        _row("a4", "b4", 0.10, False),
    ]


def _basic_gold() -> set[frozenset[str]]:
    return {frozenset({"a1", "b1"}), frozenset({"a3", "b3"})}


def _basic_records() -> dict[str, dict[str, Any]]:
    return _records(
        ("a1", {"id": "a1", "name": "Acme", "price": "10", "source": "abt"}),
        ("b1", {"id": "b1", "name": "Acme Inc", "price": "10", "source": "buy"}),
        ("a2", {"id": "a2", "name": "Zephyr", "price": "", "source": "abt"}),
        ("b2", {"id": "b2", "name": "Zephyr Co", "price": "5", "source": "buy"}),
        ("a3", {"id": "a3", "name": "Nimbus", "price": None, "source": "abt"}),
        ("b3", {"id": "b3", "name": "Nimbus LLC", "price": "8", "source": "buy"}),
        ("a4", {"id": "a4", "name": "Vertex", "price": "3", "source": "abt"}),
        ("b4", {"id": "b4", "name": "Orion", "price": "9", "source": "abt"}),
    )


class TestGracefulDegradation:
    def test_no_judgements_returns_none(self) -> None:
        assert profile_failure_mode([], gold_pairs=[("a", "b")]) is None

    def test_no_gold_returns_none(self) -> None:
        assert profile_failure_mode([_row("a", "b", 0.9, True)], gold_pairs=None) is None

    def test_all_correct_gives_zero_error_and_no_lift(self) -> None:
        # Every verdict matches gold: error rate 0, and no slice can have a lift.
        judgements = [_row("a", "b", 0.9, True), _row("c", "d", 0.1, False)]
        section = profile_failure_mode(judgements, gold_pairs=[("a", "b")])
        assert section is not None
        assert section.n_errors == 0
        assert section.error_rate == 0.0
        assert all(s.lift is None for s in section.slices)  # overall rate 0 -> lift undefined

    def test_all_abstain_gives_no_confident_and_no_slices(self) -> None:
        judgements = [_row("a", "b", None, None), _row("c", "d", None, None)]
        records = _records(
            ("a", {"id": "a", "name": "x"}),
            ("b", {"id": "b", "name": "y"}),
            ("c", {"id": "c", "name": "z"}),
            ("d", {"id": "d", "name": "w"}),
        )
        section = profile_failure_mode(judgements, gold_pairs=[("a", "b")], records=records)
        assert section is not None
        assert section.n_judged == 2
        assert section.n_confident == 0
        assert section.error_rate is None
        assert section.abstain_rate == 1.0
        assert section.slices == []
        assert "_(no slices)_" in section.to_markdown()  # n_judged>0 but no slice rows

    def test_record_missing_from_mapping_marks_field_empty_and_skips_source(self) -> None:
        # The pair's ids are absent from `records`: every content field reads empty
        # for the pair, and the source slice cannot classify a pair it has no records for.
        judgements = [_row("x", "y", 0.9, False)]  # gold match, predicted non-match -> FN
        records = _records(("z", {"id": "z", "name": "present", "source": "abt"}))
        section = profile_failure_mode(judgements, gold_pairs=[("x", "y")], records=records)
        assert section is not None
        assert any(s.dimension == "empty:name" for s in section.slices)
        assert not any(s.dimension == "source" for s in section.slices)

    def test_non_string_field_value_is_never_empty(self) -> None:
        # A present, non-string value (e.g. a number) is not "empty" -> no slice for it.
        judgements = [_row("a", "b", 0.4, False)]  # FN error to slice on
        records = _records(
            ("a", {"id": "a", "name": "x", "year": 2020}),
            ("b", {"id": "b", "name": "y", "year": 2021}),
        )
        section = profile_failure_mode(judgements, gold_pairs=[("a", "b")], records=records)
        assert section is not None
        assert not any(s.dimension == "empty:year" for s in section.slices)

    def test_empty_field_detection_treats_blank_and_missing_alike(self) -> None:
        # "", "   ", None, and an absent key all count as empty on that side.
        judgements = [_row("a", "b", 0.4, False)]  # FN so it is an error to slice on
        records = _records(
            ("a", {"id": "a", "name": "x", "extra": "   "}),  # whitespace -> empty
            ("b", {"id": "b", "name": "y"}),  # 'extra' absent -> empty
        )
        section = profile_failure_mode(judgements, gold_pairs=[("a", "b")], records=records)
        assert section is not None
        assert any(s.dimension == "empty:extra" for s in section.slices)


class TestRenderLadder:
    def test_markdown_headline_and_slice_table(self) -> None:
        md = _basic().to_markdown()
        assert md.startswith("## Failure modes")
        assert "false negatives" in md
        assert "Error concentration by slice" in md
        assert "| score_band |" in md
        assert "NaN" not in md and "Infinity" not in md

    def test_markdown_full_error_rate_renders_100_not_scientific(self) -> None:
        # A 100% slice error rate must render "100%", never ".2g" scientific "1e+02%".
        assert "1e+02" not in _basic().to_markdown()
        assert "100%" in _basic().to_markdown()

    def test_summary_is_title_namespaced(self) -> None:
        assert _basic().summary == {
            "Failure modes.n_judged": 5,
            "Failure modes.n_errors": 2,
            "Failure modes.error_rate": 0.5,
            "Failure modes.n_false_positive": 1,
            "Failure modes.n_false_negative": 1,
            "Failure modes.n_abstain": 1,
        }

    def test_rows_one_per_slice(self) -> None:
        section = _basic()
        rows = section.rows()
        assert len(rows) == len(section.slices)
        assert set(rows[0]) == {"dimension", "value", "n", "n_errors", "error_rate", "lift"}

    def test_panels_render_kv_overlay_and_slice_table(self) -> None:
        panel = _basic().panels()[0]
        assert panel.startswith("<section><h2>Failure modes</h2>")
        assert 'table class="kv"' in panel
        assert "<svg" in panel  # error-vs-success score overlay
        assert 'table class="errors"' in panel  # slice table
        assert "NaN" not in panel and "Infinity" not in panel

    def test_panels_without_scores_have_no_overlay(self) -> None:
        judgements = [_row("a", "b", None, True)]  # decider: no score
        section = profile_failure_mode(judgements, gold_pairs=[("a", "b")])
        assert section is not None
        panel = section.panels()[0]
        assert "<svg" not in panel

    def test_panels_without_slices_render_empty_note(self) -> None:
        judgements = [_row("a", "b", None, None)]  # abstain -> no confident, no slices
        section = profile_failure_mode(judgements, gold_pairs=[("a", "b")])
        assert section is not None
        panel = section.panels()[0]
        assert "No slices to report." in panel

    def test_zero_judged_section_renders_note(self) -> None:
        # n_judged == 0 is unreachable via the profiler (it returns None), but the
        # section's defensive branch still renders an honest note.
        section = FailureModeSection(
            title="Failure modes",
            n_judged=0,
            n_correct=0,
            n_errors=0,
            n_abstain=0,
            n_false_positive=0,
            n_false_negative=0,
            score_edges=[],
            error_counts=[],
            success_counts=[],
            slices=[],
            n_slices_hidden=0,
        )
        assert "nothing to analyze" in section.to_markdown()
        assert section.error_rate is None and section.abstain_rate is None

    def test_hidden_slices_noted_in_markdown(self) -> None:
        section = profile_failure_mode(
            _basic_judgements(), gold_pairs=_basic_gold(), records=_basic_records(), max_slices=2
        )
        assert section is not None
        assert "below the display cap" in section.to_markdown()

    def test_markdown_renders_null_rate_slice_as_na(self) -> None:
        # A null-rate slice (n=0) is unreachable via the profiler but the render
        # guards it: the error rate renders "n/a" rather than crashing on None.
        section = FailureModeSection(
            title="Failure modes",
            n_judged=3,
            n_correct=3,
            n_errors=0,
            n_abstain=0,
            n_false_positive=0,
            n_false_negative=0,
            score_edges=[],
            error_counts=[],
            success_counts=[],
            slices=[
                FailureSlice(
                    dimension="score_band",
                    value="0.00-0.20",
                    n=0,
                    n_errors=0,
                    error_rate=None,
                    lift=None,
                )
            ],
            n_slices_hidden=0,
        )
        md = section.to_markdown()
        assert "n/a" in md
        assert "NaN" not in md and "Infinity" not in md

    def test_failure_slice_is_frozen(self) -> None:
        s = FailureSlice(dimension="d", value="v", n=1, n_errors=1, error_rate=1.0, lift=2.0)
        assert isinstance(s, FailureSlice)


class TestReportWiring:
    def _records_and_gold(self) -> tuple[list[dict[str, str]], list[set[str]]]:
        records = [
            {"id": "1", "name": "Acme"},
            {"id": "2", "name": "Acme Inc"},
            {"id": "3", "name": "Zephyr"},
        ]
        gold = [{"1", "2"}, {"3"}]
        return records, gold

    def _section(self) -> FailureModeSection:
        section = profile_failure_mode(
            [_row("1", "2", 0.9, True), _row("1", "3", 0.6, True)],
            gold_pairs=[("1", "2")],
        )
        assert section is not None
        return section

    def test_section_included_when_supplied(self) -> None:
        records, gold = self._records_and_gold()
        report = from_records(records, gold=gold, failure_mode=self._section())
        assert isinstance(report["Failure modes"], FailureModeSection)

    def test_section_absent_when_not_supplied(self) -> None:
        records, gold = self._records_and_gold()
        report = from_records(records, gold=gold)
        assert "failure_mode" not in {s.kind for s in report.sections}

    def test_include_selects_the_kind(self) -> None:
        records, gold = self._records_and_gold()
        report = from_records(
            records, gold=gold, failure_mode=self._section(), include=["failure_mode"]
        )
        assert {s.kind for s in report.sections} == {"failure_mode"}

    def test_report_renders_with_section(self) -> None:
        records, gold = self._records_and_gold()
        report = from_records(records, gold=gold, failure_mode=self._section())
        assert "Failure modes" in report.to_markdown()
        assert "<section><h2>Failure modes</h2>" in report.to_html()

    def test_report_is_a_data_profile_report(self) -> None:
        records, gold = self._records_and_gold()
        report = from_records(records, gold=gold, failure_mode=self._section())
        assert isinstance(report, DataProfileReport)
