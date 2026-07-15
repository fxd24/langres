"""End-to-end tests: a full multi-section report rendered to HTML/markdown.

Exercises the whole Wave 2 assembly through ``from_records`` with gold, a schema,
and two embedding sources -- then asserts the render contracts the plan pins:
valid doctype, one ``<svg>`` per present chart panel, no literal ``NaN``/
``Infinity``, the pinned section order, and HTML/Markdown escaping.
"""

from __future__ import annotations

import numpy as np

from langres.data.data_profile import ArraySource, DataProfileReport
from langres.data.data_profile.builders import from_records
from langres.core.models import CompanySchema

_RECORDS = [
    {"id": "1", "name": "Acme Corporation"},
    {"id": "2", "name": "Acme Corp"},
    {"id": "3", "name": "Globex Inc"},
    {"id": "4", "name": "Globex Incorporated"},
    {"id": "5", "name": "Initech"},
    {"id": "6", "name": "Initech LLC"},
    {"id": "7", "name": "Unrelated Bakery"},
]
_CLUSTERS = [{"1", "2"}, {"3", "4"}, {"5", "6"}, {"7"}]
_IDS = [record["id"] for record in _RECORDS]


def _cluster_structured_matrix(dim: int, scale: float, seed: int) -> np.ndarray:
    """Per-cluster base direction + noise, so within-cluster cosine runs high."""
    rng = np.random.default_rng(seed)
    cluster_of = {rid: ci for ci, cluster in enumerate(_CLUSTERS) for rid in cluster}
    bases = {ci: rng.normal(size=dim) for ci in range(len(_CLUSTERS))}
    return np.asarray(
        [(bases[cluster_of[rid]] + rng.normal(scale=0.3, size=dim)) * scale for rid in _IDS]
    )


def _full_report() -> DataProfileReport:
    embeddings = [
        ArraySource("mini-8d", _IDS, _cluster_structured_matrix(dim=8, scale=1.0, seed=1)),
        ArraySource("large-16d", _IDS, _cluster_structured_matrix(dim=16, scale=2.5, seed=2)),
    ]
    return from_records(_RECORDS, gold=_CLUSTERS, schema=CompanySchema, embeddings=embeddings)


class TestEndToEndHtml:
    def test_pinned_section_order(self) -> None:
        report = _full_report()
        assert [section.kind for section in report.sections] == [
            "hero",
            "label_structure",
            "separability",  # string
            "separability",  # cosine · mini-8d
            "separability",  # cosine · large-16d
            "corpus_field",
            "embedding",
            "embedding",
            "embedding_comparison",
        ]

    def test_to_html_is_valid_self_contained_document(self) -> None:
        out = _full_report().to_html()
        assert out.startswith("<!doctype html>")
        assert out.rstrip().endswith("</html>")
        # Self-contained: inline styles, no external stylesheet/script/CDN. (An
        # inline SVG's xmlns="http://www.w3.org/2000/svg" is not an external asset.)
        assert "<style>" in out
        assert "<link" not in out
        assert "<script" not in out

    def test_svg_count_matches_present_panels(self) -> None:
        report = _full_report()
        out = report.to_html()
        expected_svg = sum(
            panel.count("<svg") for section in report.sections for panel in section.panels()
        )
        assert expected_svg > 0
        assert out.count("<svg") == expected_svg

    def test_section_count_matches_present_panels(self) -> None:
        report = _full_report()
        out = report.to_html()
        expected_sections = sum(len(section.panels()) for section in report.sections)
        assert out.count("<section") == expected_sections

    def test_no_literal_nan_or_infinity(self) -> None:
        report = _full_report()
        assert "NaN" not in report.to_html()
        assert "Infinity" not in report.to_html()
        assert "NaN" not in report.to_markdown()

    def test_markdown_has_report_heading_and_all_sections(self) -> None:
        report = _full_report()
        md = report.to_markdown()
        assert md.startswith("# Data profile")
        assert "## Overview" in md  # hero
        assert "## Label structure" in md
        assert "## Corpus fields" in md

    def test_summary_flattens_every_section(self) -> None:
        report = _full_report()
        summary = report.summary
        assert "Overview.n_records" in summary
        assert summary["Overview.n_records"] == 7
        assert "Label structure.prevalence" in summary


class TestEscaping:
    def test_html_and_markdown_escape_field_names(self) -> None:
        # A hostile field name must be HTML-escaped and pipe-safe in markdown.
        records = [{"id": "1", "na<me|x>": "v"}, {"id": "2", "na<me|x>": "w"}]
        report = DataProfileReport.from_records(records)
        html_out = report.to_html()
        assert "na&lt;me|x&gt;" in html_out  # < and > escaped
        assert "<me|x>" not in html_out  # never a raw tag-like sequence
        md = report.to_markdown()
        assert "na<me\\|x>" in md  # pipe escaped so the table stays aligned
