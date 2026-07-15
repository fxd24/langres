"""Tests for the autoresearch :class:`SearchSpace` (the proposal substrate).

Core contract code -> high coverage tier. Covers the config Cartesian product,
its size (``__len__`` vs enumerated count), the **k_neighbors-innermost** ordering
guarantee the loop depends on for index reuse, fail-loud empty-axis validation,
frozenness, and import-lightness (no faiss/torch pulled by importing this module).
"""

import subprocess
import sys

import pytest

from langres.core.autoresearch.search_space import SearchSpace


def test_defaults_are_a_vector_k_sweep() -> None:
    space = SearchSpace()
    assert space.blocker == ("vector",)
    assert space.embedding_model == ("all-MiniLM-L6-v2",)
    assert space.metric == ("cosine",)
    assert space.text_field == ("name",)
    assert space.k_neighbors == (5, 10, 20)


def test_is_frozen() -> None:
    space = SearchSpace()
    with pytest.raises((AttributeError, TypeError)):
        space.k_neighbors = (1, 2)  # type: ignore[misc]


def test_len_is_product_of_axis_sizes() -> None:
    space = SearchSpace(
        blocker=("vector",),
        embedding_model=("m1", "m2"),
        metric=("cosine", "L2"),
        text_field=("a", "b", "c"),
        k_neighbors=(5, 10),
    )
    assert len(space) == 1 * 2 * 2 * 3 * 2 == 24


def test_configs_count_matches_len() -> None:
    space = SearchSpace(
        embedding_model=("m1", "m2"),
        metric=("cosine", "L2"),
        text_field=("a", "b"),
        k_neighbors=(5, 10, 20),
    )
    configs = list(space.configs())
    assert len(configs) == len(space)


def test_configs_default_is_the_k_sweep() -> None:
    configs = list(SearchSpace().configs())
    # Only k varies; everything else is the single default value.
    assert [c["k_neighbors"] for c in configs] == [5, 10, 20]
    assert {c["blocker"] for c in configs} == {"vector"}
    assert {c["embedding_model"] for c in configs} == {"all-MiniLM-L6-v2"}
    assert {c["metric"] for c in configs} == {"cosine"}
    assert {c["text_field"] for c in configs} == {"name"}


def test_config_dict_has_the_expected_keys() -> None:
    config = next(SearchSpace().configs())
    assert set(config) == {"blocker", "embedding_model", "metric", "text_field", "k_neighbors"}


def test_k_neighbors_is_the_innermost_dimension() -> None:
    """The load-bearing contract: k varies fastest, outer keys held fixed per k-block.

    Consecutive configs must share ``(blocker, embedding_model, metric,
    text_field)`` while ``k_neighbors`` cycles through its full range, so the
    downstream loop can build one index per outer tuple and reuse it across k.
    """
    ks = (5, 10, 20)
    space = SearchSpace(
        blocker=("vector",),
        embedding_model=("m1", "m2"),
        metric=("cosine", "L2"),
        text_field=("a", "b"),
        k_neighbors=ks,
    )
    configs = list(space.configs())
    outer_keys = ("blocker", "embedding_model", "metric", "text_field")

    for i, config in enumerate(configs):
        # k cycles through the full range in order, fastest-varying.
        assert config["k_neighbors"] == ks[i % len(ks)]

    # Within each block of len(ks) configs the outer tuple is constant; it only
    # advances at block boundaries.
    for start in range(0, len(configs), len(ks)):
        block = configs[start : start + len(ks)]
        outer_tuples = {tuple(c[k] for k in outer_keys) for c in block}
        assert len(outer_tuples) == 1, f"outer keys changed within a k-block at {start}"

    # Every outer tuple is distinct across blocks (no block repeats).
    block_starts = range(0, len(configs), len(ks))
    outer_per_block = [tuple(configs[s][k] for k in outer_keys) for s in block_starts]
    assert len(set(outer_per_block)) == len(outer_per_block)


def test_consecutive_configs_within_a_block_differ_only_in_k() -> None:
    ks = (5, 10, 20)
    space = SearchSpace(embedding_model=("m1", "m2"), k_neighbors=ks)
    configs = list(space.configs())
    for i in range(len(configs) - 1):
        if (i + 1) % len(ks) != 0:  # not a block boundary
            a, b = configs[i], configs[i + 1]
            assert {k: v for k, v in a.items() if k != "k_neighbors"} == {
                k: v for k, v in b.items() if k != "k_neighbors"
            }


def test_all_pairs_in_blocker_axis_still_products() -> None:
    """The product spans all axes; ``all_pairs`` configs carry the same keys.

    (The factory ignores the vector-only keys for the all_pairs path.)
    """
    space = SearchSpace(blocker=("vector", "all_pairs"), k_neighbors=(5,))
    kinds = [c["blocker"] for c in space.configs()]
    assert kinds == ["vector", "all_pairs"]
    assert len(space) == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"blocker": ()},
        {"embedding_model": ()},
        {"metric": ()},
        {"text_field": ()},
        {"k_neighbors": ()},
    ],
)
def test_empty_axis_raises(kwargs: dict[str, tuple[object, ...]]) -> None:
    with pytest.raises(ValueError, match="must be a non-empty tuple"):
        SearchSpace(**kwargs)  # type: ignore[arg-type]


def test_import_is_light() -> None:
    """Importing search_space must not pull faiss/torch/sentence_transformers.

    Run in a fresh interpreter so no other test's heavy imports contaminate
    ``sys.modules``. This module lives on the public API surface, so it must
    stay as import-light as a bare ``import langres`` (see test_import_budget.py).
    """
    code = (
        "import sys\n"
        "import langres.core.autoresearch.search_space  # noqa: F401\n"
        "heavy = sorted(m for m in ('faiss', 'torch', 'sentence_transformers') "
        "if m in sys.modules)\n"
        "assert not heavy, f'unexpectedly imported: {heavy}'\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
