"""Tests for the vocabulary-overlap profile section.

Behavior + edges: the "no second source" absent-section contract, the two
overlap ratios (type Jaccard vs occurrence-weighted coverage) on hand-computed
inputs, the asymmetry that makes coverage worth reporting separately from
Jaccard, empty/degenerate sides yielding ``None`` rather than a fabricated
``0.0``, deterministic top-token ranking and its cap, tokenizer behavior
(case-folding, punctuation, Unicode), and the render invariants
(Markdown/HTML escaping, no ``NaN``/``Infinity`` in HTML).
"""

from __future__ import annotations

import pytest

from langres.data.data_profile import ProfileSection
from langres.data.data_profile.vocabulary import (
    VocabularyOverlapSection,
    profile_vocabulary_overlap,
    tokenize,
)


def _section(left: list[str], right: list[str], **kwargs: object) -> VocabularyOverlapSection:
    section = profile_vocabulary_overlap(left, right, **kwargs)  # type: ignore[arg-type]
    assert section is not None
    return section


class TestTokenize:
    def test_case_folds_and_splits_on_punctuation(self) -> None:
        assert tokenize("Canon EOS-400D, black!") == ["canon", "eos", "400d", "black"]

    def test_keeps_duplicates_in_order(self) -> None:
        assert tokenize("a b a") == ["a", "b", "a"]

    def test_unicode_word_characters_stay_one_token(self) -> None:
        assert tokenize("Zoë Müller") == ["zoë", "müller"]

    def test_empty_text_yields_no_tokens(self) -> None:
        assert tokenize("") == []


class TestGracefulDegradation:
    def test_none_left_returns_none(self) -> None:
        assert profile_vocabulary_overlap(None, ["a"]) is None

    def test_none_right_returns_none(self) -> None:
        assert profile_vocabulary_overlap(["a"], None) is None

    def test_two_empty_sides_render_with_none_ratios(self) -> None:
        # Present-but-empty is a valid degenerate input: the section renders,
        # and every undefined ratio is None rather than a fabricated 0.0.
        section = _section([], [])
        assert section.jaccard is None
        assert section.overlap_coefficient is None
        assert section.left_token_coverage is None
        assert section.right_token_coverage is None
        assert section.top_shared == []

    def test_one_empty_side_leaves_its_coverage_none(self) -> None:
        section = _section(["alpha beta"], [])
        assert section.left_token_coverage == 0.0  # nothing shared, but tokens exist
        assert section.right_token_coverage is None  # no tokens at all -> undefined
        assert section.overlap_coefficient is None  # min(types) == 0 -> undefined
        assert section.jaccard == 0.0


class TestOverlapMetrics:
    def test_identical_sides_are_fully_overlapping(self) -> None:
        section = _section(["alpha beta"], ["alpha beta"])
        assert section.jaccard == 1.0
        assert section.overlap_coefficient == 1.0
        assert section.left_token_coverage == 1.0
        assert section.right_token_coverage == 1.0

    def test_disjoint_sides_share_nothing(self) -> None:
        section = _section(["alpha"], ["beta"])
        assert section.jaccard == 0.0
        assert section.overlap_coefficient == 0.0
        assert section.left_token_coverage == 0.0
        assert section.right_token_coverage == 0.0
        assert section.n_shared_types == 0
        assert section.n_union_types == 2

    def test_partial_overlap_matches_hand_computed_jaccard(self) -> None:
        # left types {a, b}, right types {b, c} -> shared {b}, union {a, b, c}.
        section = _section(["a b"], ["b c"])
        assert section.n_shared_types == 1
        assert section.n_union_types == 3
        assert section.jaccard == pytest.approx(1 / 3)
        # min(|left|, |right|) == 2 -> 1/2.
        assert section.overlap_coefficient == pytest.approx(0.5)

    def test_coverage_is_occurrence_weighted_not_type_weighted(self) -> None:
        # The whole reason coverage is reported next to Jaccard: a shared common
        # core plus two disjoint long tails gives LOW Jaccard and HIGH coverage.
        left = ["the the the the alpha"]
        right = ["the the the the beta"]
        section = _section(left, right)
        # types: left {the, alpha}, right {the, beta}; shared {the} -> J = 1/3.
        assert section.jaccard == pytest.approx(1 / 3)
        # occurrences: 4 of 5 tokens are shared on each side.
        assert section.left_token_coverage == pytest.approx(0.8)
        assert section.right_token_coverage == pytest.approx(0.8)

    def test_coverage_is_asymmetric_when_sides_differ_in_size(self) -> None:
        section = _section(["alpha"], ["alpha beta gamma delta"])
        assert section.left_token_coverage == 1.0
        assert section.right_token_coverage == pytest.approx(0.25)

    def test_counts_documents_and_occurrences(self) -> None:
        section = _section(["a b", "c"], ["a"])
        assert section.n_left_docs == 2
        assert section.n_right_docs == 1
        assert section.n_left_tokens == 3
        assert section.n_right_tokens == 1
        assert section.n_left_types == 3
        assert section.n_right_types == 1


class TestTopSharedRanking:
    def test_ranks_by_the_smaller_of_the_two_counts(self) -> None:
        # "rare" is frequent on the left but nearly absent on the right, so the
        # token BOTH sides lean on ("common") must outrank it.
        left = ["rare " * 10 + "common common"]
        right = ["rare", "common common common"]
        section = _section(left, right)
        assert [t.token for t in section.top_shared] == ["common", "rare"]

    def test_ties_break_on_token_for_determinism(self) -> None:
        section = _section(["beta alpha"], ["alpha beta"])
        assert [t.token for t in section.top_shared] == ["alpha", "beta"]

    def test_top_n_caps_the_table(self) -> None:
        words = [f"w{i}" for i in range(30)]
        section = _section([" ".join(words)], [" ".join(words)], top_n=5)
        assert len(section.top_shared) == 5
        # The cap is a display cap only -- the counts stay complete.
        assert section.n_shared_types == 30

    def test_rows_mirror_top_shared(self) -> None:
        section = _section(["a b"], ["a b"])
        assert section.rows() == [
            {"token": t.token, "left_count": t.left_count, "right_count": t.right_count}
            for t in section.top_shared
        ]


class TestRender:
    def test_is_a_profile_section_with_a_stable_kind(self) -> None:
        section = _section(["a"], ["a"])
        assert isinstance(section, ProfileSection)
        assert section.kind == "vocabulary_overlap"

    def test_summary_keys_are_title_namespaced(self) -> None:
        section = _section(["a"], ["a"], title="Vocab X")
        assert set(section.summary) == {
            "Vocab X.jaccard",
            "Vocab X.overlap_coefficient",
            "Vocab X.left_token_coverage",
            "Vocab X.right_token_coverage",
            "Vocab X.n_shared_types",
        }

    def test_markdown_carries_source_names_and_metrics(self) -> None:
        section = _section(["a"], ["a"], left_name="abt", right_name="buy")
        md = section.to_markdown()
        assert "abt vs buy" in md
        assert "type Jaccard" in md
        assert "token coverage (abt)" in md

    def test_markdown_escapes_pipe_in_a_token(self) -> None:
        # A pipe cannot survive raw in a Markdown table cell.
        section = _section(["a|b"], ["a|b"])
        assert "|" in section.to_markdown()
        assert "a\\|b" not in section.to_markdown()  # tokenizer split it, no pipe token
        assert "a" in [t.token for t in section.top_shared]

    def test_markdown_renders_empty_shared_table(self) -> None:
        assert "_(none)_" in _section(["alpha"], ["beta"]).to_markdown()

    def test_panels_escape_html_in_source_names(self) -> None:
        section = _section(["a"], ["a"], left_name="<script>", right_name="b")
        panel = "".join(section.panels())
        assert "<script>" not in panel
        assert "&lt;script&gt;" in panel

    def test_panels_never_emit_nan_or_infinity(self) -> None:
        panel = "".join(_section([], []).panels())
        assert "NaN" not in panel
        assert "Infinity" not in panel
