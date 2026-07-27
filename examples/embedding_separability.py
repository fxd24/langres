"""Measure whether an embedding resource separates known matches from non-matches.

Run it:

    uv run --env-file .env python examples/embedding_separability.py
    uv run --env-file .env python examples/embedding_separability.py \\
        --model BAAI/bge-small-en-v1.5 --instruction "Find the duplicate record for: "

**Why this example loads a real model.** It used to embed five hand-written
strings with ``FakeEmbedder(dimension=32)`` — hash-derived pseudo-random vectors
— and print a "separability margin". That number could not distinguish a good
embedder from a broken one, because no embedder was involved: the demo measured
its own hash function and reported a healthy-looking float either way. An
example whose output does not depend on the thing it claims to measure is worse
than no example.

So this runs the real path: a real checkpoint over a real labelled benchmark
(``fodors_zagat``, the small one that ships in the wheel), and reports ROC-AUC
over gold pairs versus a seeded sample of non-gold pairs. The margin is kept as
a secondary number because it is intuitive, but AUC is the one to read — a
margin is not scale-free across models, and AUC is.

For the model-by-model sweep this example is the single-cell version of, see
``examples/research/embedder_ladder.py`` and
``docs/research/20260727_embedder_ladder.md``.
"""

from __future__ import annotations

import argparse
import random
from collections.abc import Sequence

import numpy as np

from langres.resources import Embedder, SentenceTransformer, SentenceTransformerRuntimeConfig

BENCHMARK = "fodors_zagat"
NEGATIVE_SAMPLE = 5_000
SEED = 0


def _unit(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize rows so a dot product really is a cosine.

    Not an assumption to make about the input: whether the vectors arrive
    normalized depends on the resource's ``normalize_embeddings`` runtime config,
    and calling an unnormalized dot product "cosine" would report a number whose
    scale silently depends on that flag.
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.asarray(vectors / np.where(norms == 0.0, 1.0, norms))


def cosine_scores(
    query_vectors: np.ndarray,
    document_vectors: np.ndarray,
    pairs: Sequence[tuple[int, int]],
) -> np.ndarray:
    """Return one cosine score per declared pair, scoring both directions.

    ``query_vectors`` supplies the query side and ``document_vectors`` the
    document side, so an instruction applied to queries only is reflected exactly
    as it is at search time. **The maximum over both directions is taken**, which
    is what a blocker effectively does — every record is a query once, so a pair
    is retrieved if either direction ranks it. That also keeps the two label
    classes comparable: gold pairs come out of a ``sorted()`` id tuple, which on a
    source-prefixed linkage benchmark makes every positive an A-query→B-document
    pair while the sampled negatives are mixed. See
    ``examples/research/embedder_ladder.py:separability_auc`` for the long form.
    """
    queries = _unit(query_vectors)
    documents = _unit(document_vectors)
    first = [i for i, _ in pairs]
    second = [j for _, j in pairs]
    forward = np.einsum("ij,ij->i", queries[first], documents[second])
    backward = np.einsum("ij,ij->i", queries[second], documents[first])
    return np.asarray(np.maximum(forward, backward))


def labelled_pairs(
    ids: Sequence[str],
    gold_clusters: Sequence[set[str]],
    *,
    seed: int = SEED,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Gold pairs and a seeded sample of non-gold pairs, as index pairs.

    Non-gold pairs are sampled rather than enumerated: the full set is quadratic
    and overwhelmingly negative, so an exhaustive scan would be slow and would
    not change the ranking statistic.
    """
    from itertools import combinations

    position = {record_id: index for index, record_id in enumerate(ids)}
    gold = {
        frozenset(pair) for cluster in gold_clusters for pair in combinations(sorted(cluster), 2)
    }
    matches = [
        (position[a], position[b])
        for a, b in (tuple(sorted(pair)) for pair in gold)
        if a in position and b in position
    ]

    rng = random.Random(seed)
    non_matches: list[tuple[int, int]] = []
    # Sampled WITHOUT replacement: a repeated pair is the same observation counted
    # twice, which narrows the negative class's apparent spread without adding
    # information. On a small corpus (fodors_zagat is ~860 records) collisions are
    # not rare, so this is a real effect, not a formality.
    seen: set[frozenset[int]] = set()
    # Bounded: a tiny or fully-connected gold set has few (or no) non-gold pairs
    # to find, and an unbounded rejection sampler would spin forever on it.
    for _ in range(NEGATIVE_SAMPLE * 20):
        if len(non_matches) >= NEGATIVE_SAMPLE:
            break
        i, j = rng.randrange(len(ids)), rng.randrange(len(ids))
        if i == j or frozenset({i, j}) in seen:
            continue
        if frozenset({ids[i], ids[j]}) in gold:
            continue
        seen.add(frozenset({i, j}))
        non_matches.append((i, j))
    return matches, non_matches


def separability(
    embedder: Embedder,
    texts: Sequence[str],
    ids: Sequence[str],
    gold_clusters: Sequence[set[str]],
    *,
    instruction: str | None = None,
) -> tuple[float, float]:
    """Return ``(roc_auc, mean-match-minus-mean-non-match cosine)``.

    Args:
        embedder: Any ``langres.resources`` embedding resource.
        texts: Blocking text per record, aligned with ``ids``.
        ids: Record ids, aligned with ``texts``.
        gold_clusters: The closed-world gold partition.
        instruction: Optional query-side instruction. Documents stay generic,
            which is the asymmetric shape instructional checkpoints document.
    """
    from langres.metrics.metrics import roc_auc_score

    documents = embedder.embed(list(texts)).vectors
    if instruction is None:
        queries = documents
    else:
        queries = embedder.embed([instruction + text for text in texts]).vectors

    matches, non_matches = labelled_pairs(ids, gold_clusters)
    match_scores = cosine_scores(queries, documents, matches)
    non_match_scores = cosine_scores(queries, documents, non_matches)

    labels = [True] * len(match_scores) + [False] * len(non_match_scores)
    scores = np.concatenate([match_scores, non_match_scores])
    auc = float(roc_auc_score(labels, scores.tolist()))
    margin = float(np.mean(match_scores) - np.mean(non_match_scores))
    return auc, margin


def main(argv: Sequence[str] | None = None) -> None:
    """Measure a real embedder on a real labelled benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--benchmark", default=BENCHMARK)
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--device", default=None, help="torch device, e.g. mps / cpu")
    args = parser.parse_args(argv)

    from langres.core.blockers.vector import concat_comparable_fields
    from langres.data.registry import get_benchmark

    corpus, gold_clusters, _ = get_benchmark(args.benchmark).load()
    texts = [concat_comparable_fields(record) for record in corpus]
    ids = [str(record.id) for record in corpus]

    embedder = SentenceTransformer(
        args.model,
        runtime_config=SentenceTransformerRuntimeConfig(device=args.device, batch_size=64),
    )
    auc, margin = separability(embedder, texts, ids, gold_clusters, instruction=args.instruction)

    print(f"model:      {args.model}")
    print(f"benchmark:  {args.benchmark} ({len(corpus)} records)")
    print(f"instruction: {args.instruction!r}")
    print(f"ROC-AUC (gold vs sampled non-gold): {auc:.4f}")
    print(f"mean match-minus-non-match cosine:  {margin:.4f}")
    print("Read the AUC: a cosine margin is not comparable across models, AUC is.")


if __name__ == "__main__":
    main()
