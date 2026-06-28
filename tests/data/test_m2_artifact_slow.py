"""M2 EXIT — fresh-process artifact identity proof (the brainsquad contract).

Proves the second half of the M2 exit criterion: a saved Resolver artifact runs
a brainsquad-style ``.resolve()`` call end-to-end **in a fresh process** and
yields *identical* clusters to the in-process run. ``resolve()`` is the ONLY M2
consumption call — ``.link()`` / ``.stream_against()`` are M5 stubs and are never
touched here.

The proof, deterministically and with ZERO spend (real MiniLM embeddings for
blocking; the WeightedAverageJudge is a zero-cost feature-bag scorer):

1. load -> split (seed=0) -> tune threshold on TRAIN -> build the resolver ->
   ``resolve`` the held-out TEST dicts IN-PROCESS -> clusters_A.
2. ``resolver.save(<dir>)`` writes the artifact directory (manifest + FAISS
   sidecar built over the test corpus).
3. A FRESH subprocess that first ``import langres.data.er_benchmarks`` (so
   ``RestaurantSchema`` is in the registry before ``Resolver.load``) reads the
   exact same test dicts from a JSON file, calls
   ``Resolver.load(<dir>).resolve(<dicts>)``, and writes its clusters as JSON.
4. clusters_A is asserted equal to clusters_B **canonically** — each clustering
   reduced to a sorted tuple of sorted-id tuples, so the check is independent of
   cluster order and within-cluster id order.

Marked ``slow`` (it embeds the corpus) but RUNS in CI.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from langres.data.er_benchmarks import (
    build_restaurant_resolver,
    load_fodors_zagat,
    split_restaurant_corpus,
    tune_threshold_on_train,
)

# Canonical, order-independent form of a clustering: a sorted tuple of
# sorted-id tuples. Used on both sides of the in-process vs fresh-process check.
Canonical = tuple[tuple[str, ...], ...]


def _canonical(clusters: list[set[str]]) -> Canonical:
    return tuple(sorted(tuple(sorted(c)) for c in clusters))


# Run in a fresh interpreter: import the package (registers RestaurantSchema),
# load the artifact, resolve the SAME test dicts, write canonical clusters out.
_SUBPROCESS_SCRIPT = """
import json
import sys

import langres.data.er_benchmarks  # noqa: F401 — registers RestaurantSchema
from langres.core.resolver import Resolver

artifact_dir, records_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
records = json.loads(open(records_path).read())
clusters = Resolver.load(artifact_dir).resolve(records)
canonical = sorted(sorted(c) for c in clusters)
open(out_path, "w").write(json.dumps(canonical))
"""


@pytest.mark.slow
def test_m2_artifact_resolve_identity_fresh_process(tmp_path: Path) -> None:
    # 1. Build the M2 resolver exactly as the skeleton does (tune on TRAIN only).
    corpus, gold_clusters = load_fodors_zagat()
    train_records, test_records, train_clusters, _ = split_restaurant_corpus(
        corpus, gold_clusters, test_size=0.3, seed=0
    )
    best_threshold = tune_threshold_on_train(train_records, train_clusters)
    resolver = build_restaurant_resolver(best_threshold)

    # In-process resolve over the held-out test dicts (the brainsquad call shape).
    test_dicts = [r.model_dump() for r in test_records]
    clusters_a = resolver.resolve(test_dicts)
    assert clusters_a, "expected the baseline to find at least one multi-record cluster"

    # 2. Persist the artifact directory and the exact dicts the subprocess reuses.
    artifact_dir = tmp_path / "artifact"
    resolver.save(artifact_dir)
    assert (artifact_dir / "resolver.json").exists()

    records_path = tmp_path / "test_records.json"
    records_path.write_text(json.dumps(test_dicts))

    # 3. Fresh subprocess: import the package, load the artifact, resolve.
    out_path = tmp_path / "clusters_b.json"
    subprocess.run(
        [
            sys.executable,
            "-c",
            _SUBPROCESS_SCRIPT,
            str(artifact_dir),
            str(records_path),
            str(out_path),
        ],
        check=True,
    )
    clusters_b = [set(c) for c in json.loads(out_path.read_text())]

    # 4. Identical clusters, compared canonically (order-independent).
    assert _canonical(clusters_a) == _canonical(clusters_b)
