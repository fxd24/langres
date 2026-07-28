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
        db_path = "stale.db"

        def encode(self, texts, prompt=None):  # noqa: ANN001, ANN202, ARG002
            return np.array([[0.0, 1.0]], dtype=np.float32)

    with pytest.raises(RuntimeError, match="disagrees with the live checkpoint"):
        harness._assert_cache_matches_checkpoint(_Base(), _StaleCache(), ["a text"])


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

    # Dropping the cell first is what stops a crashed rerun from LOOKING
    # complete: merging one fresh arm into the stale ones would leave every
    # (arm, k) key present, so --resume would skip a cell built from two runs.
    survivors = harness._drop_cell(rows, spec, "abt_buy")
    assert survivors == []
    assert not harness._cell_complete(survivors, spec, "abt_buy", recipes, [20])


def test_a_typo_in_arms_is_refused_before_anything_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--arms offical_asymmetric` must not silently erase committed cells.

    `_drop_cell` clears a cell up front so a crashed rerun looks partial. That
    made an unknown arm name destructive: it selected nothing, cleared every
    requested cell, evaluated nothing and exited 0.
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


def test_drop_cell_only_removes_the_named_cell() -> None:
    """Clearing one cell must not disturb another model's or benchmark's rows."""
    spec = harness.MODELS_BY_NAME["sentence-transformers/all-MiniLM-L6-v2"]
    other = harness.MODELS_BY_NAME["BAAI/bge-base-en-v1.5"]

    def row(model: str, benchmark: str) -> object:
        return harness.Row(
            model=model,
            benchmark=benchmark,
            arm="none",
            kind="baseline",
            k=20,
            document_prompt=None,
            query_prompt=None,
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
        )

    rows = [
        row(spec.name, "abt_buy"),
        row(spec.name, "amazon_google"),
        row(other.name, "abt_buy"),
    ]
    kept = harness._drop_cell(rows, spec, "abt_buy")
    assert {(r.model, r.benchmark) for r in kept} == {
        (spec.name, "amazon_google"),
        (other.name, "abt_buy"),
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
