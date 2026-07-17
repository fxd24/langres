"""Shared, reusable loader-contract checks for DeepMatcher-style benchmarks (Wave B).

Every dataset built with :func:`~langres.data._deepmatcher_loader.make_deepmatcher_benchmark`
must satisfy the same structural contract. Rather than re-writing those assertions
in each dataset's test, Wave C loader tests import :func:`assert_loader_contract`
and run it against their ``<X>Benchmark`` instance (optionally pinning the exact
corpus / gold-pair counts as evidence).

The module name is ``_``-prefixed so pytest does not collect it as a test file;
it holds no tests of its own, only the reusable checker. Exercised here on the
tiny fixture (``tests/data/test_tiny_fixture.py``).
"""

import re
from typing import Any

from langres.data.benchmark import Benchmark, gold_pairs_from_clusters

#: A split-safe corpus id: a single alpha char + integer (the split parses int(id[1:])).
_SPLIT_SAFE_ID = re.compile(r"^[A-Za-z]\d+$")


def assert_loader_contract(
    benchmark: Any,
    *,
    expected_corpus_size: int | None = None,
    expected_gold_pairs: int | None = None,
    seed: int = 0,
) -> None:
    """Assert a benchmark honors the DeepMatcher loader contract.

    Checks, against ``benchmark.load()`` and ``benchmark.split(...)``:

    1. **Protocol conformance** — the instance is a runtime ``Benchmark`` and
       exposes the ``BlockingBenchmark`` shape the method registry needs
       (``schema`` / ``blocking_k`` / callable ``build_blocker``).
    2. **Id scheme** — every corpus id is ``<char><int>`` and ids are unique.
    3. **Closed-world partition** — ``gold_clusters`` partition the corpus exactly.
    4. **Gold-pair consistency** — ``gold_pairs`` equals the within-cluster pairs,
       and every gold pair's two ids are in the corpus.
    5. **Leakage-free split** — train/test record ids are disjoint and cover the
       corpus, and no gold cluster straddles the split.

    The ``BlockingBenchmark`` checks are structural (``hasattr`` / ``callable``) so
    the contract stays dependency-light: ``build_blocker`` lazy-imports the
    ``[semantic]`` stack, so it is only *called* as an optional smoke, guarded so a
    core-only environment (no faiss) skips it rather than failing.

    Args:
        benchmark: A loaded benchmark instance (class already constructed).
        expected_corpus_size: If given, assert the corpus has exactly this many
            records (pin the vendored count as evidence).
        expected_gold_pairs: If given, assert exactly this many gold match pairs.
        seed: Split seed to exercise.
    """
    assert isinstance(benchmark, Benchmark), "benchmark does not satisfy the Benchmark protocol"

    # BlockingBenchmark shape (methods.py's registry needs it). Structural only —
    # methods.BlockingBenchmark is a plain Protocol, never isinstance-checked.
    assert hasattr(benchmark, "schema"), "benchmark is missing 'schema' (BlockingBenchmark)"
    assert hasattr(benchmark, "blocking_k"), "benchmark is missing 'blocking_k' (BlockingBenchmark)"
    assert callable(benchmark.build_blocker), "benchmark.build_blocker is not callable"
    # Optional runtime smoke — build_blocker lazy-imports the [semantic] stack, so
    # guard it: a core-only env (no faiss) skips this rather than failing.
    try:
        blocker = benchmark.build_blocker(benchmark.blocking_k)
        assert type(blocker).__name__ == "VectorBlocker"
    except ImportError:
        pass  # [semantic] extra absent — the structural checks above are the contract.

    corpus, gold_clusters, gold_pairs = benchmark.load()

    # 2. Id scheme.
    ids = [record.id for record in corpus]
    assert ids, "corpus is empty"
    for rid in ids:
        assert _SPLIT_SAFE_ID.match(rid), f"id {rid!r} is not <char><int> (split-unsafe)"
    all_ids = set(ids)
    assert len(all_ids) == len(ids), "duplicate corpus ids"

    # 3. Closed-world partition: every id in exactly one cluster.
    clustered = [rid for cluster in gold_clusters for rid in cluster]
    assert len(clustered) == len(all_ids), "gold_clusters do not partition the corpus (id count)"
    assert set(clustered) == all_ids, "gold_clusters do not cover exactly the corpus ids"

    # 4. Gold-pair consistency.
    assert gold_pairs == gold_pairs_from_clusters(gold_clusters), (
        "gold_pairs must equal the within-cluster pairs of gold_clusters"
    )
    for pair in gold_pairs:
        assert len(pair) == 2, f"gold pair {pair!r} is not a 2-element frozenset"
        assert set(pair) <= all_ids, f"gold pair {pair!r} references ids absent from the corpus"

    # 5. Leakage-free split.
    train_records, test_records, train_clusters, test_clusters = benchmark.split(
        corpus, gold_clusters, seed=seed
    )
    train_ids = {record.id for record in train_records}
    test_ids = {record.id for record in test_records}
    assert train_ids.isdisjoint(test_ids), "train/test record leakage"
    assert train_ids | test_ids == all_ids, "split drops or invents corpus ids"
    for cluster in train_clusters:
        assert cluster <= train_ids, "a train gold cluster straddles the split (leakage)"
    for cluster in test_clusters:
        assert cluster <= test_ids, "a test gold cluster straddles the split (leakage)"
    assert {rid for c in train_clusters for rid in c} == train_ids, "train clusters != train ids"
    assert {rid for c in test_clusters for rid in c} == test_ids, "test clusters != test ids"

    # 6. Optional pinned counts (evidence).
    if expected_corpus_size is not None:
        assert len(corpus) == expected_corpus_size, (
            f"expected {expected_corpus_size} records, got {len(corpus)}"
        )
    if expected_gold_pairs is not None:
        assert len(gold_pairs) == expected_gold_pairs, (
            f"expected {expected_gold_pairs} gold pairs, got {len(gold_pairs)}"
        )
