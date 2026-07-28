"""Offline tests that the prompt-axis harness's encoder-reach guard can actually fail.

$0 and model-free. The guard exists because a prompt that never reaches the
encoder produces *identical numbers*, which reads as "instructions do not help"
rather than as a bug — this repo shipped exactly that once (``search_all`` served
queries from cached corpus vectors). A guard that has never been seen to fail is
a hypothesis, not a safety net, so each failure shape is exercised here.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

ROOT = Path(__file__).parents[2]
HARNESS_PATH = ROOT / "examples" / "research" / "prompt_axis.py"


def _load() -> ModuleType:
    name = "example_prompt_axis"
    sys.path.insert(0, str(HARNESS_PATH.parent))
    try:
        spec = importlib.util.spec_from_file_location(name, HARNESS_PATH)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(HARNESS_PATH.parent))


harness = _load()


def _recipe(document: str | None, query: str | None, arm: str = "test") -> object:
    return harness.Recipe(
        arm=arm, document_prompt=document, query_prompt=query, kind="ours", note=""
    )


def _row(model: str, benchmark: str, arm: str = "none") -> object:
    """A Row carrying the provenance the harness would really have recorded."""
    spec = harness.MODELS_BY_NAME[model]
    recipe = next(r for r in spec.recipes if r.arm == arm)
    return harness.Row(
        model=model,
        benchmark=benchmark,
        arm=arm,
        kind=recipe.kind,
        k=20,
        document_prompt=recipe.document_prompt,
        query_prompt=recipe.query_prompt,
        note=recipe.note,
        candidate_recall=1.0,
        candidate_precision=1.0,
        reduction_ratio=1.0,
        total_candidates=1,
        reachable_ceiling=1.0,
        recall_of_reachable=1.0,
        doc_shift_vs_none=0.0,
        query_shift_vs_none=0.0,
        doc_query_cosine=1.0,
        pair_jaccard_vs_none=1.0,
        revision=spec.revision,
        recipe_fingerprint=harness._recipe_fingerprint(recipe),
    )


BASE = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
MOVED = np.array([[0.8, 0.6], [0.6, 0.8]], dtype=np.float32)
OTHER = np.array([[0.6, 0.8], [0.8, 0.6]], dtype=np.float32)


def test_document_prompt_that_never_moved_the_corpus_vectors_raises() -> None:
    """The index-build path silently dropping the prompt must not pass as a flat result."""
    with pytest.raises(RuntimeError, match="corpus vectors are bit-identical"):
        harness._prompt_reached_encoder(
            _recipe("doc: ", "query: "), BASE, MOVED, BASE, doc_query_cosine=0.8
        )


def test_query_prompt_that_never_moved_the_query_vectors_raises() -> None:
    """This is the exact shape of the search_all() bug #242 hardened."""
    with pytest.raises(RuntimeError, match="query vectors are bit-identical"):
        harness._prompt_reached_encoder(
            _recipe(None, "query: "), BASE, BASE, BASE, doc_query_cosine=1.0
        )


def test_symmetric_recipe_whose_two_encode_paths_disagree_raises() -> None:
    """One side applying the prompt and the other not, when both were told to."""
    with pytest.raises(RuntimeError, match="disagree"):
        harness._prompt_reached_encoder(
            _recipe("same: ", "same: "), MOVED, OTHER, BASE, doc_query_cosine=0.75
        )


def test_asymmetric_recipe_producing_identical_vectors_raises() -> None:
    """Two different prompts cannot legitimately yield the same vectors."""
    with pytest.raises(RuntimeError, match="both sides produced bit-identical"):
        harness._prompt_reached_encoder(
            _recipe("doc: ", "query: "), MOVED, MOVED, BASE, doc_query_cosine=1.0
        )


def test_float16_residual_does_not_hide_an_ignored_prompt() -> None:
    """The regression that motivated comparing arrays instead of thresholding shift.

    On a float16 checkpoint an *ignored* prompt does not produce ``shift == 0.0``
    -- the committed Qwen3 baseline rows carry residuals like ``-0.0002207``. The
    old exact-equality test would have let that through and published a flat
    result as evidence the prompt was applied. Array equality has no such hole:
    the vectors are bit-identical regardless of what the cosine rounds to.
    """
    with pytest.raises(RuntimeError, match="query vectors are bit-identical"):
        harness._prompt_reached_encoder(
            _recipe(None, "query: "), BASE, BASE, BASE, doc_query_cosine=1.000126
        )


def test_float16_round_off_above_one_is_tolerated_for_the_symmetric_check() -> None:
    """A cosine >1 is impossible for distinct unit vectors, so it is round-off.

    Measured on ``Qwen/Qwen3-Embedding-0.6B`` (float16): the no-prompt arm, where
    both sides are literally the same call, reported ``1.000126``. That must not
    abort a run.
    """
    harness._prompt_reached_encoder(
        _recipe(None, None, arm="none"), BASE, BASE, BASE, doc_query_cosine=1.000126
    )


def test_document_prompt_order_puts_the_bare_group_first() -> None:
    """The `none` arm establishes the baseline, so its group must run first."""
    recipes = [
        harness.Recipe("a", "doc: ", "doc: ", "ours", ""),
        harness.Recipe("none", None, None, "baseline", ""),
        harness.Recipe("b", None, "q: ", "ours", ""),
    ]
    assert harness._document_prompt_order(recipes)[0] is None


def test_a_cache_disagreeing_with_the_checkpoint_is_refused() -> None:
    """A stale cache is invisible to the prompt guards, so it needs its own check.

    Every prompt-reach guard compares arms *against each other*, so uniformly
    stale vectors look perfectly consistent. Only a comparison against the live
    checkpoint can catch it.
    """

    class _Base:
        def encode(self, texts, prompt=None):  # noqa: ANN001, ANN202, ARG002
            return np.array([[1.0, 0.0]], dtype=np.float32)

    class _StaleCache:
        """A cache with entries stored for both the document and a query partition."""

        db_path = "stale.db"

        def __init__(self, populated: set[str | None]) -> None:
            self.populated = populated

        def _hash_text(self, text, prompt=None):  # noqa: ANN001, ANN202
            return f"{text}|{prompt}"

        def _get_from_db(self, key):  # noqa: ANN001, ANN202
            prompt = key.split("|", 1)[1]
            stored = None if prompt == "None" else prompt
            return object() if stored in self.populated else None

        def encode(self, texts, prompt=None):  # noqa: ANN001, ANN202, ARG002
            return np.array([[0.0, 1.0]], dtype=np.float32)

    with pytest.raises(RuntimeError, match="document/default"):
        harness._assert_cache_matches_checkpoint(_Base(), _StaleCache({None}), ["a text"])

    # The document partition can be fine while a QUERY partition is stale. Checking
    # only the default partition missed exactly this, and the prompt-reach guards
    # cannot see it either -- stale prompted vectors still differ from the baseline.
    with pytest.raises(RuntimeError, match="'q: '"):
        harness._assert_cache_matches_checkpoint(_Base(), _StaleCache({"q: "}), ["a text"], ["q: "])


def test_an_unpopulated_cache_partition_is_not_a_vacuous_pass() -> None:
    """A partition with nothing stored must be skipped, never "verified".

    On a miss the wrapper delegates to the base embedder and stores the result,
    so comparing a miss against a fresh encode compares a value to itself -- a
    check that cannot fail. Probing only populated entries is what keeps the
    canary honest.
    """
    calls: list[str | None] = []

    class _Base:
        def encode(self, texts, prompt=None):  # noqa: ANN001, ANN202, ARG002
            calls.append(prompt)
            return np.array([[1.0, 0.0]], dtype=np.float32)

    class _EmptyCache:
        db_path = "empty.db"

        def _hash_text(self, text, prompt=None):  # noqa: ANN001, ANN202
            return f"{text}|{prompt}"

        def _get_from_db(self, key):  # noqa: ANN001, ANN202, ARG002
            return None  # nothing cached anywhere

        def encode(self, texts, prompt=None):  # noqa: ANN001, ANN202, ARG002
            raise AssertionError("must not encode through the cache on an empty partition")

    harness._assert_cache_matches_checkpoint(_Base(), _EmptyCache(), ["a text"], ["q: "])
    assert calls == [], "an empty cache must not be probed at all"


def test_tolerance_still_rejects_the_smallest_real_divergence_measured() -> None:
    """The loosened tolerance must stay far below any genuine one-sided failure.

    ``0.9446`` is the closest-to-1 asymmetric cosine observed anywhere in the
    sweep (``e5-base-v2``, ``official_asymmetric``). If the symmetric tolerance
    ever grew enough to swallow that, the guard would stop catching a dropped
    prompt.
    """
    assert harness._SYMMETRIC_TOLERANCE < (1.0 - 0.9446) / 10
    with pytest.raises(RuntimeError, match="disagree"):
        harness._prompt_reached_encoder(
            _recipe("x: ", "x: "), MOVED, OTHER, BASE, doc_query_cosine=0.9446
        )


def test_cell_complete_requires_every_arm_and_k() -> None:
    """Resume must not skip a cell that is only partly recorded."""
    spec = harness.MODELS_BY_NAME["sentence-transformers/all-MiniLM-L6-v2"]
    recipes = spec.recipes
    rows = [
        harness.Row(
            model=spec.name,
            benchmark="abt_buy",
            arm=recipe.arm,
            kind=recipe.kind,
            k=k,
            document_prompt=recipe.document_prompt,
            query_prompt=recipe.query_prompt,
            note="",
            candidate_recall=1.0,
            candidate_precision=1.0,
            reduction_ratio=1.0,
            total_candidates=1,
            reachable_ceiling=1.0,
            recall_of_reachable=1.0,
            doc_shift_vs_none=0.0,
            query_shift_vs_none=0.0,
            doc_query_cosine=1.0,
            pair_jaccard_vs_none=1.0,
            revision=spec.revision,
            recipe_fingerprint=harness._recipe_fingerprint(recipe),
        )
        for recipe in recipes
        for k in (20,)
    ]
    assert harness._cell_complete(rows, spec, "abt_buy", recipes, [20])
    # A k that was never measured leaves the cell incomplete.
    assert not harness._cell_complete(rows, spec, "abt_buy", recipes, [20, 50])
    # A different benchmark has nothing recorded at all.
    assert not harness._cell_complete(rows, spec, "amazon_google", recipes, [20])

    # Provenance is part of completeness. Rows measured on OTHER weights, or
    # under a since-edited prompt string, must not satisfy resume -- otherwise
    # the report keeps the previous study's numbers under the new definition.
    other_weights = [dataclasses.replace(r, revision="0" * 40) for r in rows]
    assert not harness._cell_complete(other_weights, spec, "abt_buy", recipes, [20])

    edited_prompt = [dataclasses.replace(r, recipe_fingerprint="deadbeef") for r in rows]
    assert not harness._cell_complete(edited_prompt, spec, "abt_buy", recipes, [20])

    # And a row recorded before provenance existed counts as unknown, not as a
    # match -- the committed rows are exactly this case.
    legacy = [dataclasses.replace(r, revision=None, recipe_fingerprint=None) for r in rows]
    assert not harness._cell_complete(legacy, spec, "abt_buy", recipes, [20])

    # Mixed provenance in one cell: a crashed rerun under NEW weights leaves
    # fresh rows beside stale ones. Provenance is what makes that visible --
    # the cell is incomplete because the stale rows no longer match, so resume
    # recomputes instead of publishing a table built from two runs. This is the
    # invariant that replaced deleting the cell up front.
    mixed = [
        rows[0],
        *[dataclasses.replace(r, revision="0" * 40) for r in rows[1:]],
    ]
    assert not harness._cell_complete(mixed, spec, "abt_buy", recipes, [20])


def test_a_typo_in_arms_is_refused_and_nothing_is_deleted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unknown `--arms` value must fail loudly and leave every row intact.

    This once cleared each requested cell, evaluated nothing and exited 0,
    because the cell was emptied before evaluation. Nothing deletes rows now,
    and the selector is rejected outright.
    """
    rows_path = tmp_path / "rows.jsonl"
    # Real committed-looking content, so the assertion is about DATA SURVIVING,
    # not merely about the exit code. An empty file would pass trivially.
    original = (ROOT / "docs" / "research" / "20260728_prompt_axis_rows.jsonl").read_text()
    rows_path.write_text(original)

    monkeypatch.setattr(
        sys, "argv", ["prompt_axis", "--arms", "offical_asymmetric", "--rows", str(rows_path)]
    )
    with pytest.raises(SystemExit) as excinfo:
        harness.main()
    assert excinfo.value.code != 0
    assert rows_path.read_text() == original, "a typo in --arms destroyed committed rows"


def test_merging_replaces_only_the_rows_it_actually_recomputed() -> None:
    """A narrow rerun must never cost an unselected measurement.

    Rows are now replaced key-by-key as replacements arrive, instead of the cell
    being cleared first. A selective run such as `--arms none --k 20` therefore
    updates that one key and leaves every other arm and k untouched -- the
    earlier clear-then-refill deleted all of them.
    """
    spec = harness.MODELS_BY_NAME["sentence-transformers/all-MiniLM-L6-v2"]
    other = harness.MODELS_BY_NAME["BAAI/bge-base-en-v1.5"]

    rows = [
        _row(spec.name, "abt_buy"),
        _row(spec.name, "amazon_google"),
        _row(other.name, "abt_buy"),
    ]
    fresh = dataclasses.replace(rows[0], candidate_recall=0.5)
    merged = harness.merge_rows(rows, [fresh])

    # Same number of rows: a rerun replaces, it never removes.
    assert len(merged) == len(rows)
    by_key = {(r.model, r.benchmark): r for r in merged}
    assert by_key[(spec.name, "abt_buy")].candidate_recall == 0.5
    # The measurements that were NOT recomputed survive untouched.
    assert by_key[(spec.name, "amazon_google")].candidate_recall == 1.0
    assert by_key[(other.name, "abt_buy")].candidate_recall == 1.0


def test_merging_two_checkpoints_of_one_model_raises_and_keeps_every_row() -> None:
    """A partial re-run must not leave two checkpoints under one model name.

    ``_cell_complete`` refuses to *resume* a cell whose revision moved, but it
    decides one cell at a time: a run narrowed with ``--benchmarks`` recomputes
    only what it was pointed at, and the rest keep the old checkpoint's numbers.
    The merged file then reports two sets of weights under one model heading with
    nothing on the surface to show it. Only a check over the whole merged set can
    see that, so this is where it lives.
    """
    spec = harness.MODELS_BY_NAME["sentence-transformers/all-MiniLM-L6-v2"]
    stale = dataclasses.replace(_row(spec.name, "amazon_google"), revision="0" * 40)
    rows = [_row(spec.name, "abt_buy"), stale]

    with pytest.raises(ValueError, match="different checkpoints"):
        harness.merge_rows(rows, [])

    # It reports the conflict; it does not resolve it by throwing a side away.
    # An earlier draft resolved a similar ambiguity by deleting, and destroyed
    # measured rows three review rounds running.
    assert rows[1].revision == "0" * 40


def test_rendering_refuses_numbers_measured_under_a_different_prompt() -> None:
    """The prose and the numbers must not be free to drift apart.

    The report prints "The arms" from today's ``MODELS`` and the results from the
    rows file. Nothing in the layout couples them, so an edited prompt string and
    a re-render -- with no re-measurement -- yields a document that describes one
    experiment and reports another.
    """
    spec = harness.MODELS_BY_NAME["sentence-transformers/all-MiniLM-L6-v2"]
    good = _row(spec.name, "abt_buy")
    assert harness.render_report([good])  # the check is not simply always-on

    edited = dataclasses.replace(good, recipe_fingerprint="deadbeefdeadbeef")
    with pytest.raises(ValueError, match="prompt fingerprint"):
        harness.render_report([edited])

    moved = dataclasses.replace(good, revision="0" * 40)
    with pytest.raises(ValueError, match="report describes"):
        harness.render_report([moved])

    unknown = dataclasses.replace(good, revision=None, recipe_fingerprint=None)
    with pytest.raises(ValueError, match="refusing to render"):
        harness.render_report([unknown])


def test_the_committed_rows_still_match_the_harness_that_describes_them() -> None:
    """The published artifact regenerates -- the guard passes on real data, not only toys."""
    rows_path = ROOT / "docs" / "research" / "20260728_prompt_axis_rows.jsonl"
    rows = harness.read_rows(rows_path)
    assert len(rows) == 400
    assert harness.render_report(rows)


def test_correcting_for_multiplicity_can_only_remove_claims() -> None:
    """An interval spanning zero must not become an effect under any correction.

    The recovered p-value is a function of the interval, so this is worth pinning
    rather than assuming: if a spanning interval could read p<=0.05 the whole
    section would be able to *invent* significance, which is the opposite of what
    a correction is for.
    """
    assert harness._approximate_p_value(0.01, -0.02, 0.04) > 0.05
    assert harness._approximate_p_value(-0.01, -0.04, 0.02) > 0.05
    # And a wide margin still reads as strong, so the test above is not vacuous.
    assert harness._approximate_p_value(0.10, 0.09, 0.11) < 1e-30


def test_a_bound_resting_exactly_on_zero_reads_exactly_alpha() -> None:
    """The knife-edge case is decided by arithmetic, not by a judgement call."""
    assert harness._approximate_p_value(0.0152, 0.0, 0.031) == pytest.approx(0.05, abs=1e-12)
    assert harness._approximate_p_value(-0.0152, -0.031, 0.0) == pytest.approx(0.05, abs=1e-12)
    # p = alpha fails every Holm threshold once the family has more than one member.
    assert harness._holm({"a": 0.05}) == {"a"}
    assert harness._holm({"a": 0.05, "b": 0.5}) == set()


def test_a_boundary_cell_is_found_structurally_not_by_comparing_p_to_alpha() -> None:
    """The first version of this section reported zero boundary cells. There are four.

    It classified them with ``p == 0.05``, and the recovered p lands on
    0.05000000000000004 -- so the count printed 0 and the list printed empty,
    which reads as "no boundary cases here" over four of them. The bound itself
    is stored exactly, so that is what the classifier reads now.
    """
    rows = harness.read_rows(ROOT / "docs" / "research" / "20260728_prompt_axis_rows.jsonl")
    testable = [
        row
        for row in rows
        if row.k == harness.HEADLINE_K
        and row.arm != "none"
        and harness._approximate_p_value(row.delta_per_record_recall, row.ci_low, row.ci_high)
        is not None
    ]
    structural = [row for row in testable if harness._on_boundary(row)]
    assert len(structural) == 4
    by_float_equality = [
        row
        for row in testable
        if harness._approximate_p_value(row.delta_per_record_recall, row.ci_low, row.ci_high)
        == harness.ALPHA
    ]
    assert by_float_equality == [], "the discarded classifier would have to be blind here"


def test_holm_stops_at_the_first_failure_instead_of_testing_each_alone() -> None:
    """Step-down, not per-hypothesis: a failure blocks every larger p behind it."""
    # m=3, thresholds 0.0167 / 0.025 / 0.05. The third p would pass its own
    # threshold in isolation; Holm must not reach it, because the second fails.
    assert harness._holm({"a": 0.001, "b": 0.03, "c": 0.04}) == {"a"}
    assert harness._holm({"a": 0.001, "b": 0.02, "c": 0.04}) == {"a", "b", "c"}
    # A family of one is simply the uncorrected test.
    assert harness._holm({"a": 0.049}) == {"a"}


def test_the_saturated_benchmark_produces_no_testable_comparison() -> None:
    """No variation means no test -- not a null result, and not a silent zero."""
    assert harness._approximate_p_value(0.0, 0.0, 0.0) is None
    rows = harness.read_rows(ROOT / "docs" / "research" / "20260728_prompt_axis_rows.jsonl")
    verdicts = harness._multiplicity(rows)
    untestable = {key for key, (p_value, _) in verdicts.items() if p_value is None}
    assert untestable, "the guard has to be seen firing on the real rows"
    assert {benchmark for _, _, benchmark in untestable} == {"fodors_zagat"}


def test_holm_withdraws_exactly_the_three_e5_documented_comparisons() -> None:
    """Pin the retraction: the correction changed the study's answer, and by how much.

    e5's documented arms are the only claims multiplicity costs us. Its whole
    family fails at Holm's *first* step (smallest p 0.0221 against a 0.0167
    threshold), so no e5 documented comparison survives -- which is why the
    headline is "three of four instruction-trained models", not four.
    """
    rows = harness.read_rows(ROOT / "docs" / "research" / "20260728_prompt_axis_rows.jsonl")
    verdicts = harness._multiplicity(rows)
    headline = [r for r in rows if r.k == harness.HEADLINE_K and r.arm != "none"]
    excluded_zero = {
        (r.model, r.arm, r.benchmark) for r in headline if not harness._spans_zero(r)
    }
    held = {key for key, (_, rejected) in verdicts.items() if rejected}
    assert len(excluded_zero) == 40
    assert len(held) == 37
    assert excluded_zero - held == {
        ("intfloat/e5-base-v2", "official_asymmetric", "abt_buy"),
        ("intfloat/e5-base-v2", "official_symmetric", "amazon_google"),
        ("intfloat/e5-base-v2", "official_symmetric", "wdc_computers"),
    }
    # Every documented-arm claim that survives belongs to one of the other three.
    documented = {"official_retrieval", "official_query_instruction", "official_query_instruct"}
    assert {model for model, arm, _ in held if arm in documented} == {
        "google/embeddinggemma-300m",
        "BAAI/bge-base-en-v1.5",
        "Qwen/Qwen3-Embedding-0.6B",
    }


def test_every_model_pins_the_revision_it_was_measured_on() -> None:
    """A published row must name the weights that produced it.

    Loading by bare repo name is what let the study cite one commit while
    `refs/main` resolved to another -- real here, not hypothetical: two Qwen3
    snapshots and three MiniLM snapshots sit in the local HF cache.
    """
    for spec in harness.MODELS:
        assert len(spec.revision) == 40, spec.name
        assert set(spec.revision) <= set("0123456789abcdef"), spec.name


def test_the_default_sweep_is_exactly_the_benchmarks_that_were_measured() -> None:
    """The documented command must reproduce the committed artifact, not extend it.

    `walmart_amazon` sat in this default while the committed rows covered four
    benchmarks, so a default `--resume` run would have silently added a fifth
    and rewritten the report.
    """
    rows_path = ROOT / "docs" / "research" / "20260728_prompt_axis_rows.jsonl"
    measured = {
        json.loads(line)["benchmark"] for line in rows_path.read_text().splitlines() if line.strip()
    }
    assert set(harness.BENCHMARKS) == measured
