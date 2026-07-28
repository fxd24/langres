"""Offline tests that the prompt-axis harness's encoder-reach guard can actually fail.

$0 and model-free. The guard exists because a prompt that never reaches the
encoder produces *identical numbers*, which reads as "instructions do not help"
rather than as a bug — this repo shipped exactly that once (``search_all`` served
queries from cached corpus vectors). A guard that has never been seen to fail is
a hypothesis, not a safety net, so each failure shape is exercised here.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

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


def test_document_prompt_that_never_moved_the_corpus_vectors_raises() -> None:
    """The index-build path silently dropping the prompt must not pass as a flat result."""
    with pytest.raises(RuntimeError, match="corpus vectors are identical"):
        harness._prompt_reached_encoder(
            _recipe("doc: ", "query: "), doc_shift=0.0, query_shift=0.2, doc_query_cosine=0.8
        )


def test_query_prompt_that_never_moved_the_query_vectors_raises() -> None:
    """This is the exact shape of the search_all() bug #242 hardened."""
    with pytest.raises(RuntimeError, match="query vectors are identical"):
        harness._prompt_reached_encoder(
            _recipe(None, "query: "), doc_shift=0.0, query_shift=0.0, doc_query_cosine=1.0
        )


def test_symmetric_recipe_whose_two_encode_paths_disagree_raises() -> None:
    """One side applying the prompt and the other not, when both were told to."""
    with pytest.raises(RuntimeError, match="disagree"):
        harness._prompt_reached_encoder(
            _recipe("same: ", "same: "), doc_shift=0.2, query_shift=0.2, doc_query_cosine=0.75
        )


def test_asymmetric_recipe_producing_identical_vectors_raises() -> None:
    """Two different prompts cannot legitimately yield the same vectors."""
    with pytest.raises(RuntimeError, match="both sides produced identical vectors"):
        harness._prompt_reached_encoder(
            _recipe("doc: ", "query: "), doc_shift=0.2, query_shift=0.2, doc_query_cosine=1.0
        )


def test_float16_round_off_above_one_is_tolerated() -> None:
    """A cosine >1 is impossible for distinct unit vectors, so it is round-off.

    Measured on ``Qwen/Qwen3-Embedding-0.6B`` (float16): the no-prompt arm, where
    both sides are literally the same call, reported ``1.000126``. That must not
    abort a run.
    """
    harness._prompt_reached_encoder(
        _recipe(None, None, arm="none"),
        doc_shift=0.0,
        query_shift=0.0,
        doc_query_cosine=1.000126,
    )


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
            _recipe("x: ", "x: "), doc_shift=0.1, query_shift=0.1, doc_query_cosine=0.9446
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
        )
        for recipe in recipes
        for k in (20,)
    ]
    assert harness._cell_complete(rows, spec, "abt_buy", recipes, [20])
    # A k that was never measured leaves the cell incomplete.
    assert not harness._cell_complete(rows, spec, "abt_buy", recipes, [20, 50])
    # A different benchmark has nothing recorded at all.
    assert not harness._cell_complete(rows, spec, "amazon_google", recipes, [20])
