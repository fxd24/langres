"""Tests for the composable seam: ``ProfileSection`` + ``DataProfileReport``.

Structure-based render asserts (doctype prefix, ``<svg>`` counts, escaping, no
literal ``NaN``/``Infinity``), plus the container contract (positional build,
graceful empty render, subset by title, ``SerializeAsAny`` keeping subclass
fields, thin delegation of the convenience constructors). No concrete profiler
section exists yet (Wave 2), so a tiny in-test subclass stands in for one.
"""

from __future__ import annotations

import sys
import types
from typing import Any, Literal

import pytest

from langres.core import _report_html, _svg
from langres.data.data_profile import DataProfileReport, ProfileSection


class _FakeSection(ProfileSection):
    """A minimal concrete section for exercising the container in Wave 0.

    Carries an ``extra`` field the base does not know about, so the
    ``SerializeAsAny`` round-trip can be observed.
    """

    kind: Literal["fake"] = "fake"
    value: float = 0.0
    extra: str = "subclass-only"

    def to_markdown(self) -> str:
        return f"## {self.title}\n\nvalue = {self.value:g}"

    @property
    def summary(self) -> dict[str, Any]:
        return {f"{self.title}.value": self.value}

    def rows(self) -> list[dict[str, Any]]:
        return [{"title": self.title, "value": self.value}]

    def panels(self) -> list[str]:
        chart = _svg.bar_chart([0.0, 1.0], [("v", "#0a0", [self.value])])
        return [_report_html.section(self.title, chart)]


class TestProfileSectionABC:
    def test_base_is_abstract_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError):
            ProfileSection(title="x", kind="base")  # type: ignore[abstract]

    def test_is_frozen(self) -> None:
        section = _FakeSection(title="s")
        with pytest.raises(Exception):  # noqa: B017 - pydantic ValidationError on frozen set
            section.value = 1.0  # type: ignore[misc]

    def test_to_dict_is_model_dump_and_keeps_subclass_fields(self) -> None:
        section = _FakeSection(title="s", value=0.5)
        dumped = section.to_dict()
        assert dumped["title"] == "s"
        assert dumped["kind"] == "fake"
        assert dumped["value"] == 0.5
        assert dumped["extra"] == "subclass-only"

    def test_repr_is_markdown(self) -> None:
        section = _FakeSection(title="s", value=2.0)
        assert repr(section) == section.to_markdown()
        assert repr(section).startswith("## s")

    def test_summary_and_rows(self) -> None:
        section = _FakeSection(title="s", value=3.0)
        assert section.summary == {"s.value": 3.0}
        assert section.rows() == [{"title": "s", "value": 3.0}]

    def test_panels_are_section_html(self) -> None:
        panels = _FakeSection(title="s", value=1.0).panels()
        assert len(panels) == 1
        assert panels[0].startswith("<section><h2>s</h2>")
        assert "<svg" in panels[0]


class TestContainerConstruction:
    def test_positional_list_of_sections(self) -> None:
        report = DataProfileReport([_FakeSection(title="a"), _FakeSection(title="b")])
        assert len(report.sections) == 2

    def test_keyword_sections_still_work(self) -> None:
        report = DataProfileReport(sections=[_FakeSection(title="a")])
        assert len(report.sections) == 1

    def test_empty_report_is_valid(self) -> None:
        report = DataProfileReport([])
        assert report.sections == []

    def test_no_args_raises_validation_error(self) -> None:
        # ``sections`` is required (pass ``[]`` for an empty report): omitting it
        # entirely leaves the field unset -> pydantic validation error. Exercises
        # the ``sections is None`` branch of the positional-friendly __init__.
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            DataProfileReport()

    def test_is_frozen(self) -> None:
        report = DataProfileReport([])
        with pytest.raises(Exception):  # noqa: B017 - frozen model
            report.sections = [_FakeSection(title="a")]  # type: ignore[misc]


class TestContainerTextSurfaces:
    def test_to_markdown_concatenates_sections_under_heading(self) -> None:
        report = DataProfileReport([_FakeSection(title="a", value=1.0), _FakeSection(title="b")])
        md = report.to_markdown()
        assert md.startswith("# Data profile")
        assert "## a" in md
        assert "## b" in md
        assert "NaN" not in md

    def test_to_markdown_empty_report(self) -> None:
        md = DataProfileReport([]).to_markdown()
        assert md.startswith("# Data profile")
        assert "_No sections._" in md

    def test_repr_is_markdown(self) -> None:
        report = DataProfileReport([_FakeSection(title="a")])
        assert repr(report) == report.to_markdown()

    def test_summary_flattens_section_summaries(self) -> None:
        report = DataProfileReport(
            [_FakeSection(title="a", value=1.0), _FakeSection(title="b", value=2.0)]
        )
        assert report.summary == {"a.value": 1.0, "b.value": 2.0}

    def test_summary_empty_report(self) -> None:
        assert DataProfileReport([]).summary == {}

    def test_getitem_pulls_section_by_title(self) -> None:
        wanted = _FakeSection(title="b", value=9.0)
        report = DataProfileReport([_FakeSection(title="a"), wanted])
        assert report["b"] is wanted

    def test_getitem_missing_raises_keyerror(self) -> None:
        report = DataProfileReport([_FakeSection(title="a")])
        with pytest.raises(KeyError):
            report["nope"]


class TestContainerToDict:
    def test_model_dump_keeps_subclass_only_fields(self) -> None:
        # SerializeAsAny: the container declares list[ProfileSection], but the
        # dump must retain each concrete subclass's own fields (value/extra),
        # not narrow to the base.
        report = DataProfileReport([_FakeSection(title="a", value=0.7)])
        dumped = report.to_dict()
        section = dumped["sections"][0]
        assert section["value"] == 0.7
        assert section["extra"] == "subclass-only"
        assert section["kind"] == "fake"

    def test_to_dict_is_model_dump(self) -> None:
        report = DataProfileReport([_FakeSection(title="a")])
        assert report.to_dict() == report.model_dump()


class TestContainerHtml:
    def test_empty_report_renders_valid_doctype(self) -> None:
        out = DataProfileReport([]).to_html()
        assert out.startswith("<!doctype html>")
        assert out.rstrip().endswith("</html>")
        assert out.count("<svg") == 0

    def test_renders_one_svg_per_section_panel(self) -> None:
        report = DataProfileReport([_FakeSection(title="a"), _FakeSection(title="b")])
        out = report.to_html()
        assert out.startswith("<!doctype html>")
        assert out.count("<svg") == 2
        assert out.count("<section") == 2
        assert "NaN" not in out and "Infinity" not in out

    def test_custom_title_is_escaped(self) -> None:
        out = DataProfileReport([]).to_html(title="a & <b>")
        assert "a &amp; &lt;b&gt;" in out
        assert "<title>a &amp; &lt;b&gt;</title>" in out

    def test_section_title_escaped_in_panel(self) -> None:
        out = DataProfileReport([_FakeSection(title="x & <y>")]).to_html()
        assert "x &amp; &lt;y&gt;" in out
        assert "<y>" not in out


class TestConvenienceConstructorsDelegate:
    """``from_benchmark`` / ``from_records`` are thin delegators to Wave 2 ``builders``.

    ``builders`` does not exist yet, so a fake module is injected to observe the
    delegation without pinning the (Wave 2) signature.
    """

    def _install_fake_builders(
        self, monkeypatch: pytest.MonkeyPatch, sentinel: DataProfileReport
    ) -> dict[str, Any]:
        calls: dict[str, Any] = {}
        fake = types.ModuleType("langres.data.data_profile.builders")

        def _from_benchmark(*args: Any, **kwargs: Any) -> DataProfileReport:
            calls["benchmark"] = (args, kwargs)
            return sentinel

        def _from_records(*args: Any, **kwargs: Any) -> DataProfileReport:
            calls["records"] = (args, kwargs)
            return sentinel

        fake.from_benchmark = _from_benchmark  # type: ignore[attr-defined]
        fake.from_records = _from_records  # type: ignore[attr-defined]
        # The delegators resolve builders dynamically (importlib.import_module),
        # which consults sys.modules first -- so injecting the fake here is enough.
        monkeypatch.setitem(sys.modules, "langres.data.data_profile.builders", fake)
        return calls

    def test_from_benchmark_delegates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sentinel = DataProfileReport([])
        calls = self._install_fake_builders(monkeypatch, sentinel)
        result = DataProfileReport.from_benchmark("bench", include={"labels"})
        assert result is sentinel
        assert calls["benchmark"] == (("bench",), {"include": {"labels"}})

    def test_from_records_delegates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sentinel = DataProfileReport([])
        calls = self._install_fake_builders(monkeypatch, sentinel)
        result = DataProfileReport.from_records([{"id": "1"}], gold_pairs=None)
        assert result is sentinel
        assert calls["records"] == (([{"id": "1"}],), {"gold_pairs": None})
