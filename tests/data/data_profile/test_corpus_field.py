"""Tests for the corpus-field profile section.

Behavior + edges: no records -> absent section, per-field completeness /
cardinality / length, the most-missing-first ordering, all-null fields
flagged-not-dropped, non-string presence, the top-N truncation (logged, never
silent), the length sparkline (including degenerate + all-null), and the render
invariants (no ``NaN``/``Infinity`` in HTML, Markdown/HTML escaping of field
names).
"""

from __future__ import annotations

import logging

import pytest

from langres.data.data_profile import ProfileSection
from langres.data.data_profile.corpus_field import (
    CorpusFieldSection,
    FieldStat,
    profile_corpus_fields,
)


def _fields_by_name(section: CorpusFieldSection) -> dict[str, FieldStat]:
    return {f.name: f for f in section.fields}


class TestGracefulDegradation:
    def test_none_records_returns_none(self) -> None:
        assert profile_corpus_fields(None) is None

    def test_empty_records_returns_none(self) -> None:
        assert profile_corpus_fields([]) is None

    def test_records_with_no_fields_render_empty_table(self) -> None:
        # Non-empty records but zero field keys: the section renders an empty
        # table rather than raising or returning None.
        section = profile_corpus_fields([{}, {}])
        assert section is not None
        assert section.n_records == 2
        assert section.n_fields_total == 0
        assert section.fields == []
        assert "no fields" in section.to_markdown()
        html = "".join(section.panels())
        assert "<table" in html
        assert "NaN" not in html and "Infinity" not in html


class TestFieldMetrics:
    def _section(self) -> CorpusFieldSection:
        records = [
            {"id": "1", "name": "Acme Corp", "city": "NYC", "note": None},
            {"id": "2", "name": "acme corporation", "city": "", "note": None},
            {"id": "3", "name": "Beta LLC", "note": None},  # missing 'city' key
        ]
        section = profile_corpus_fields(records)
        assert section is not None
        return section

    def test_non_null_rate(self) -> None:
        fields = _fields_by_name(self._section())
        assert fields["id"].non_null_rate == 1.0
        assert fields["name"].non_null_rate == 1.0
        # 'city' present only in record 1 (record 2 is "", record 3 missing key).
        assert fields["city"].non_null_rate == 1 / 3
        assert fields["note"].non_null_rate == 0.0

    def test_cardinality(self) -> None:
        fields = _fields_by_name(self._section())
        assert fields["name"].n_distinct == 3
        assert fields["name"].uniqueness == 1.0
        assert fields["city"].n_distinct == 1

    def test_value_lengths(self) -> None:
        fields = _fields_by_name(self._section())
        # names: len("Acme Corp")=9, len("acme corporation")=16, len("Beta LLC")=8.
        assert fields["name"].mean_len == pytest.approx((9 + 16 + 8) / 3)
        assert fields["name"].median_len == 9.0

    def test_most_missing_first_ordering(self) -> None:
        # note (0.0) < city (0.33) < id/name (1.0); id before name on the name
        # tie-break -> deterministic order.
        section = self._section()
        assert [f.name for f in section.fields] == ["note", "city", "id", "name"]

    def test_all_null_field_flagged_not_dropped(self) -> None:
        fields = _fields_by_name(self._section())
        note = fields["note"]
        assert note.all_null is True
        assert note.n_present == 0
        assert note.uniqueness is None
        assert note.mean_len is None
        assert note.median_len is None
        assert note.len_hist == []
        # Flagged in both render surfaces, not dropped.
        assert "all-null" in self._section().to_markdown()
        assert "all-null" in "".join(self._section().panels())

    def test_non_string_values_are_present(self) -> None:
        # ints / bools are populated values (not null); length is their str form.
        records = [{"count": 10, "ok": True}, {"count": 200, "ok": False}]
        section = profile_corpus_fields(records)
        assert section is not None
        fields = _fields_by_name(section)
        assert fields["count"].non_null_rate == 1.0
        assert fields["count"].n_distinct == 2
        assert fields["count"].mean_len == pytest.approx((2 + 3) / 2)  # "10", "200"

    def test_whitespace_only_string_is_null(self) -> None:
        records = [{"x": "   "}, {"x": "value"}]
        section = profile_corpus_fields(records)
        assert section is not None
        assert _fields_by_name(section)["x"].non_null_rate == 0.5


class TestTruncation:
    def test_top_n_truncates_and_logs(self, caplog: pytest.LogCaptureFixture) -> None:
        records = [{"a": "1", "b": "2", "c": "3", "d": "4", "e": "5"}]
        with caplog.at_level(logging.WARNING):
            section = profile_corpus_fields(records, top_n=2)
        assert section is not None
        assert section.n_fields_total == 5
        assert len(section.fields) == 2
        # The truncation is logged (never silent) and surfaced in the render.
        assert any("truncated" in r.message.lower() for r in caplog.records)
        assert "truncated" in section.to_markdown().lower()
        assert "truncated" in "".join(section.panels()).lower()

    def test_no_truncation_below_cap(self, caplog: pytest.LogCaptureFixture) -> None:
        records = [{"a": "1", "b": "2"}]
        with caplog.at_level(logging.WARNING):
            section = profile_corpus_fields(records, top_n=10)
        assert section is not None
        assert len(section.fields) == 2
        assert "truncated" not in section.to_markdown().lower()
        assert not any("truncated" in r.message.lower() for r in caplog.records)


class TestSparkline:
    def test_varied_lengths_produce_blocks(self) -> None:
        # Values of differing length -> a non-empty unicode sparkline.
        records = [{"x": "a"}, {"x": "abcd"}, {"x": "abcdefghij"}]
        section = profile_corpus_fields(records)
        assert section is not None
        field = _fields_by_name(section)["x"]
        assert field.len_hist != []
        assert sum(field.len_hist) == 3
        # The sparkline is rendered (any block glyph present).
        md = section.to_markdown()
        assert any(block in md for block in "▁▂▃▄▅▆▇█")

    def test_uniform_length_is_a_single_spike(self) -> None:
        records = [{"x": "abc"}, {"x": "xyz"}]  # both length 3
        section = profile_corpus_fields(records)
        assert section is not None
        field = _fields_by_name(section)["x"]
        # Degenerate min==max: all mass in the first bucket.
        assert field.len_hist[0] == 2
        assert sum(field.len_hist) == 2


class TestRenderInvariants:
    def test_html_has_no_nan_or_infinity(self) -> None:
        records = [{"a": "x", "b": None}, {"a": "y", "b": None}]
        section = profile_corpus_fields(records)
        assert section is not None
        html = "".join(section.panels())
        assert "NaN" not in html and "Infinity" not in html

    def test_field_name_escaped_in_html(self) -> None:
        section = profile_corpus_fields([{"a<b> & 'c'": "v"}])
        assert section is not None
        html = "".join(section.panels())
        assert "a&lt;b&gt;" in html
        assert "<b>" not in html

    def test_field_name_pipe_escaped_in_markdown(self) -> None:
        # A '|' in a field name must not corrupt the Markdown table.
        section = profile_corpus_fields([{"a|b": "v"}])
        assert section is not None
        md = section.to_markdown()
        assert "a\\|b" in md


class TestContract:
    def test_kind_and_type(self) -> None:
        section = profile_corpus_fields([{"a": "1"}])
        assert isinstance(section, CorpusFieldSection)
        assert isinstance(section, ProfileSection)
        assert section.kind == "corpus_field"

    def test_summary_is_title_namespaced(self) -> None:
        section = profile_corpus_fields(
            [{"a": "1", "b": None}, {"a": "2", "b": None}], title="Fields"
        )
        assert section is not None
        summary = section.summary
        assert summary["Fields.n_records"] == 2
        assert summary["Fields.n_fields"] == 2
        assert summary["Fields.n_all_null_fields"] == 1

    def test_rows_exclude_len_hist(self) -> None:
        section = profile_corpus_fields([{"a": "abc"}])
        assert section is not None
        rows = section.rows()
        assert rows[0]["field"] == "a"
        assert "len_hist" not in rows[0]

    def test_fieldstat_is_frozen(self) -> None:
        section = profile_corpus_fields([{"a": "1"}])
        assert section is not None
        with pytest.raises(Exception):  # noqa: B017 - frozen model
            section.fields[0].n_present = 5  # type: ignore[misc]
