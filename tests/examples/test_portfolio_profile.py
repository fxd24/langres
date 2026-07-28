"""Behavior tests for the B2 portfolio-profile example.

The two things worth pinning are the *rules* (they decide a published verdict, so
a silent change to one is a silent change to the annotation) and the *wheel
status*, which is checked against a genuinely independent source: the exact-file
manifest in ``tests/test_wheel_contents.py``. That manifest is derived by
building the real artifacts; this example derives the same fact by reading
``pyproject.toml``'s exclude globs. Two paths, one answer -- if they disagree,
one of them is wrong and the annotation would carry it.

Plus one slow end-to-end profiling the 12-record fixture.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from examples.research.portfolio_profile import (
    CANONICAL_OUT,
    CAPPED_RECALL_CEILING,
    LARGE_COMPONENT,
    LEXICAL_GAP_COVERAGE,
    PUBLISHED_SOTA,
    REPO_ROOT,
    SATURATION_MARGIN,
    TINY_GOLD_PAIRS,
    BenchmarkProfile,
    _verdict,
    cross_source_ceiling,
    structural_caveats,
    to_markdown,
    wheel_exclusions,
    wheel_status,
)
from langres.data.registry import list_benchmarks
from tests.test_wheel_contents import SHIPPED_NON_PY_FILES


def _profile(**kwargs: object) -> BenchmarkProfile:
    """A minimal profile with the mandatory fields filled in."""
    base: dict[str, object] = {
        "name": "demo",
        "task": "linkage",
        "domain": "product",
        "loadable": True,
        "loaded": True,
        "wheel_loadable": True,
        "n_files_shipped": 1,
        "excluded_files": [],
    }
    base.update(kwargs)
    return BenchmarkProfile(**base)  # type: ignore[arg-type]


class TestWheelStatus:
    def test_exclusions_are_read_from_pyproject(self) -> None:
        patterns = wheel_exclusions()
        assert patterns  # the list is non-empty and real
        assert all(p.startswith("src/langres/data/datasets/") for p in patterns)

    def test_agrees_with_the_independent_wheel_contents_manifest(self) -> None:
        """Two derivations of "can a pip user load this?" must agree.

        ``SHIPPED_NON_PY_FILES`` is asserted against the REAL built wheel by
        ``tests/test_wheel_contents.py``; this example derives the same answer
        from ``pyproject.toml``'s globs. A disagreement means one of them is
        stale -- exactly the "path literals that fail silently" failure the
        pyproject comment warns about.
        """
        patterns = wheel_exclusions()
        datasets_root = REPO_ROOT / "src" / "langres" / "data" / "datasets"
        for entry in list_benchmarks():
            directory = datasets_root / entry.name
            if not directory.is_dir():
                continue
            on_disk = {
                p.relative_to(REPO_ROOT / "src" / "langres").as_posix()
                for p in directory.rglob("*")
                if p.is_file() and p.suffix != ".md"
            }
            manifest_loadable = bool(on_disk) and on_disk <= SHIPPED_NON_PY_FILES
            computed_loadable, _, _ = wheel_status(entry.name, patterns)
            assert computed_loadable == manifest_loadable, entry.name

    def test_an_unvendored_dataset_ships_nothing(self) -> None:
        loadable, shipped, excluded = wheel_status("opensanctions", wheel_exclusions())
        assert (loadable, shipped, excluded) == (False, 0, [])

    def test_a_partially_shipped_dataset_is_not_loadable(self) -> None:
        # abt_buy ships peeters_sampled_test.csv but not its corpus tables: some
        # files ship, yet load() still raises -- so "some ship" is not the rule.
        loadable, shipped, excluded = wheel_status("abt_buy", wheel_exclusions())
        assert shipped > 0
        assert excluded
        assert loadable is False


class TestStructuralCaveats:
    def test_no_caveats_on_a_clean_profile(self) -> None:
        assert (
            structural_caveats(
                _profile(
                    positive_pairs=TINY_GOLD_PAIRS + 1,
                    max_cluster_size=3,
                    n_singletons=5,
                    n_clusters=10,
                    vocab_min_coverage=0.9,
                )
            )
            == []
        )

    def test_tiny_gold_is_flagged(self) -> None:
        assert "tiny-gold" in structural_caveats(_profile(positive_pairs=TINY_GOLD_PAIRS - 1))

    def test_one_to_one_needs_pairs_only_and_no_singletons(self) -> None:
        one_to_one = _profile(max_cluster_size=2, n_singletons=0, n_clusters=500)
        assert "one-to-one" in structural_caveats(one_to_one)
        # The same shape WITH singletons is an ordinary linkage set, not 1:1.
        with_singletons = _profile(max_cluster_size=2, n_singletons=640, n_clusters=752)
        assert "one-to-one" not in structural_caveats(with_singletons)

    def test_large_component_is_flagged(self) -> None:
        assert "large-component" in structural_caveats(_profile(max_cluster_size=LARGE_COMPONENT))

    def test_large_component_does_not_fire_on_a_dedup_corpus(self) -> None:
        # The rule reads a big cluster as a two-source closure artifact. In a
        # dedup set a 10-record entity is ordinary, so firing there would brand a
        # healthy dataset -- and `febrl_dedup` is now registered as task="dedup",
        # so this scoping guards a real entry rather than a hypothetical one.
        dedup = _profile(task="dedup", max_cluster_size=LARGE_COMPONENT * 5)
        assert "large-component" not in structural_caveats(dedup)

    def test_lexical_gap_is_flagged(self) -> None:
        assert "lexical-gap" in structural_caveats(_profile(vocab_min_coverage=0.1))

    def test_capped_recall_is_flagged(self) -> None:
        assert "capped-recall" in structural_caveats(_profile(cross_source_ceiling=0.84))
        assert "capped-recall" not in structural_caveats(_profile(cross_source_ceiling=1.0))


class TestCrossSourceCeiling:
    """The ceiling a cross-source candidate set can never exceed."""

    class _Rec:
        def __init__(self, rid: str, source: str) -> None:
            self.id = rid
            self.source = source

    def test_pure_one_to_one_labels_have_no_ceiling(self) -> None:
        corpus = [self._Rec("a1", "A"), self._Rec("b1", "B")]
        assert cross_source_ceiling(corpus, [{"a1", "b1"}]) == 1.0

    def test_a_three_record_cluster_emits_an_unreachable_pair(self) -> None:
        # {a1, b1, b2} -> 3 gold pairs, of which b1-b2 is intra-source and no
        # cross-source candidate set can ever contain it.
        corpus = [self._Rec("a1", "A"), self._Rec("b1", "B"), self._Rec("b2", "B")]
        assert cross_source_ceiling(corpus, [{"a1", "b1", "b2"}]) == pytest.approx(2 / 3)

    def test_a_corpus_without_a_source_field_has_no_ceiling_to_report(self) -> None:
        class _Plain:
            def __init__(self, rid: str) -> None:
                self.id = rid

        assert cross_source_ceiling([_Plain("1"), _Plain("2")], [{"1", "2"}]) is None

    def test_no_gold_pairs_is_none_not_zero(self) -> None:
        corpus = [self._Rec("a1", "A"), self._Rec("b1", "B")]
        assert cross_source_ceiling(corpus, [{"a1"}, {"b1"}]) is None

    def test_missing_measurements_produce_no_caveats(self) -> None:
        # An unloaded benchmark must not be silently branded; absent != bad.
        assert structural_caveats(_profile(loaded=False)) == []


class TestSaturationRule:
    def test_every_published_number_is_on_the_line_it_cites(self) -> None:
        """The citation must SUPPORT the number, not merely name a file that exists.

        Checking only ``path.exists()`` passes for a stale F1 or a drifted line
        number -- the two ways a citation actually rots. This reads the cited
        line and looks for the number on it.
        """
        for name, published in PUBLISHED_SOTA.items():
            assert published.source, name
            path_part, _, rest = published.source.partition(":")
            path = REPO_ROOT / path_part
            assert path.exists(), f"{name}: {published.source} does not exist"
            lineno = int(rest.split()[0])
            line = path.read_text().splitlines()[lineno - 1]
            # Both spellings: "0.893" and a "~0.98"-style approximation.
            assert f"{published.f1:g}" in line or f"{published.f1:.3f}" in line, (
                f"{name}: {published.f1} is not on {published.source}: {line!r}"
            )

    def test_the_published_rules_are_pinned_not_merely_referenced(self) -> None:
        """Every rule that decides a published verdict is pinned to its value.

        Referring to a constant symbolically (``TINY_GOLD_PAIRS - 1``) tests the
        comparison but not the number, so redefining it would silently rewrite
        the annotation with a green suite -- exactly what this module's docstring
        says it is here to prevent.
        """
        assert (
            SATURATION_MARGIN,
            TINY_GOLD_PAIRS,
            LARGE_COMPONENT,
            LEXICAL_GAP_COVERAGE,
            CAPPED_RECALL_CEILING,
        ) == (0.02, 200, 10, 0.5, 0.95)

    def test_verdict_renders_unknown_as_a_question_not_a_no(self) -> None:
        assert _verdict(None) == "?"
        assert _verdict(True) == "YES"
        assert _verdict(False) == "no"


class TestReporting:
    def test_markdown_renders_one_row_per_profile(self) -> None:
        table = to_markdown([_profile(name="one"), _profile(name="two")])
        lines = table.splitlines()
        assert len(lines) == 4  # header + separator + two rows
        assert "n/a" in table  # unmeasured cells are honest, not zeroed


def test_canonical_artifact_covers_every_registered_benchmark() -> None:
    """The tracked ``portfolio_profile.json`` must not fall behind the registry.

    ``docs/research/20260727_portfolio_annotation.md`` names this file as its raw
    data, so a benchmark registered without regenerating it leaves the published
    annotation describing a portfolio that no longer exists — silently, because
    nothing re-derives the file. That is exactly what happened when
    ``febrl_dedup`` landed: the artifact still listed ten linkage entries while
    the annotation's §7 concluded the portfolio had no dedup benchmark, and a
    reviewer caught it rather than CI.

    This gate can genuinely fail — it would have, before the regeneration — and
    the fix is one command: ``uv run python examples/research/portfolio_profile.py``.
    """
    import json

    profiled = {row["name"] for row in json.loads((REPO_ROOT / CANONICAL_OUT).read_text())}
    registered = {entry.name for entry in list_benchmarks()}
    missing = registered - profiled
    assert not missing, (
        f"{CANONICAL_OUT} is stale: {sorted(missing)} registered but not profiled. "
        "Regenerate it with `uv run python examples/research/portfolio_profile.py` "
        "and re-read docs/research/20260727_portfolio_annotation.md, which cites it "
        "as its raw data."
    )
    assert not profiled - registered, (
        f"{CANONICAL_OUT} profiles {sorted(profiled - registered)}, which are no "
        "longer registered; regenerate it."
    )


@pytest.mark.slow
def test_end_to_end_profiles_the_tiny_fixture(tmp_path: Path) -> None:
    """The full profile path runs over the 12-record fixture (loads no embedder)."""
    from examples.research.portfolio_profile import profile_entry

    entry = next(e for e in list_benchmarks() if e.name == "tiny_fixture")
    profile = profile_entry(entry, wheel_exclusions())
    assert profile.loaded is True
    assert profile.n_records == 12
    assert profile.wheel_loadable is True
    # Two sources -> the vocabulary-overlap section contributed.
    assert profile.vocab_jaccard is not None
