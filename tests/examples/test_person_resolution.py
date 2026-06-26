"""Deterministic proof that the Person embeddings + LLM strong path runs end-to-end.

Drives the importable core of ``examples/person_resolution.py`` (``build_resolver``
+ ``run_demo``) with its deterministic fake LLM client — no network in the
default suite. Asserts the M1-critical blocking signal (Pair-Completeness over
the known duplicates), that the pipeline recovers the known duplicate clusters,
that a save/reload round-trip is identical, and that honest cost is a real float.

The live path (one real OpenRouter call) is gated behind ``OPENROUTER_API_KEY``.
"""

from __future__ import annotations

import os

import pytest

from examples.person_resolution import (
    KNOWN_DUPLICATE_GROUPS,
    KNOWN_DUPLICATE_PAIRS,
    PERSON_RECORDS,
    FakeLLMClient,
    PersonSchema,
    build_resolver,
    run_demo,
)
from langres.core import Resolver
from langres.core.blockers.vector import VectorBlocker
from langres.core.metrics import evaluate_blocking


def _canonical(clusters: list[set[str]]) -> frozenset[frozenset[str]]:
    return frozenset(frozenset(c) for c in clusters)


def test_fake_client_mirrors_llm_judgement() -> None:
    """The fake matcher catches the realistic variations and rejects look-alikes."""
    from examples.person_resolution import _names_match

    # Duplicates: accents, abbreviation, name-order, order-swap.
    assert _names_match("Joséphine Goube", "Josephine Goube")
    assert _names_match("J. Goube", "Joséphine Goube")
    assert _names_match("Goube, Josephine", "Joséphine Goube")
    assert _names_match("Liang Wei", "Wei Liang")
    # Non-duplicates: shared first name, near-name.
    assert not _names_match("Maria Garcia", "Maria Silva")
    assert not _names_match("John Smith", "Jonathan Smith")


def test_run_demo_strong_path_is_deterministic_and_correct() -> None:
    """End-to-end: blocking recall, cluster recovery, save/reload, honest cost."""
    results = run_demo()

    # 1. Blocking Pair-Completeness — the duplicates must reach the judge.
    assert results["blocking_recall"] >= 0.95
    assert results["blocking_stats"].missed_matches_count == 0

    # 2. The pipeline recovers exactly the known duplicate clusters.
    assert _canonical(results["clusters"]) == _canonical(KNOWN_DUPLICATE_GROUPS)
    assert results["bcubed"]["f1"] >= 0.95
    assert results["pairwise"]["f1"] >= 0.95

    # 3. save -> load -> re-resolve is identical (pickle-free manifest).
    assert results["identical"] is True
    assert _canonical(results["reloaded_clusters"]) == _canonical(KNOWN_DUPLICATE_GROUPS)

    # 4. Honest cost is a real, non-negative float (0.0 with the fake client).
    assert isinstance(results["total_cost"], float)
    assert results["total_cost"] >= 0.0
    assert results["num_judgements"] > 0


def test_predicted_pairs_cover_known_duplicate_pairs() -> None:
    """Every known duplicate pair ends up co-clustered (pairwise recall == 1.0)."""
    results = run_demo()
    predicted_pairs = {
        frozenset({a, b})
        for cluster in results["clusters"]
        for a in cluster
        for b in cluster
        if a < b
    }
    assert KNOWN_DUPLICATE_PAIRS <= predicted_pairs


def test_run_demo_is_repeatable() -> None:
    """Two independent runs produce byte-identical clusters and cost."""
    first = run_demo()
    second = run_demo()
    assert _canonical(first["clusters"]) == _canonical(second["clusters"])
    assert first["total_cost"] == second["total_cost"]


def test_manifest_is_pickle_free_and_secret_free() -> None:
    """The persisted resolver.json carries the model ref but no client/secret."""
    manifest = run_demo()["manifest"]
    assert "openrouter/openai/gpt-4o-mini" in manifest  # model ref persisted
    assert "client" not in manifest  # no live client
    assert "api_key" not in manifest.lower()
    assert "pickle" not in manifest.lower()


def test_serialized_blocker_recovers_blocking(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A reloaded Resolver's blocker still captures the duplicate pairs."""
    resolver = build_resolver(FakeLLMClient())
    blocker = resolver.blocker
    assert isinstance(blocker, VectorBlocker)  # narrows Blocker[Any] -> VectorBlocker
    texts = [str(r["name"]) for r in PERSON_RECORDS]
    blocker.vector_index.create_index(texts)

    resolver.save(tmp_path / "person_v0")
    reloaded = Resolver.load(tmp_path / "person_v0")
    reloaded.module.client = FakeLLMClient()  # type: ignore[attr-defined]

    candidates = list(reloaded.blocker.stream(PERSON_RECORDS))
    stats = evaluate_blocking(candidates, KNOWN_DUPLICATE_GROUPS)
    assert stats.candidate_recall >= 0.95


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"),
    reason="requires OPENROUTER_API_KEY for a real LLM call",
)
def test_live_smoke_single_pair() -> None:
    """One real OpenRouter pair returns a usable judgement (gated on a key)."""
    from langres.core.models import ERCandidate
    from langres.core.modules.llm_judge import LLMJudge

    judge: LLMJudge[PersonSchema] = LLMJudge.from_env(
        model="openrouter/openai/gpt-4o-mini", temperature=0.0, entity_noun="person"
    )
    pair: ERCandidate[PersonSchema] = ERCandidate(
        left=PersonSchema(**PERSON_RECORDS[0]),
        right=PersonSchema(**PERSON_RECORDS[1]),
        blocker_name="manual",
    )
    judgement = next(iter(judge.forward(iter([pair]))))
    assert 0.0 <= judgement.score <= 1.0
    assert isinstance(judgement.provenance["cost_usd"], float)
