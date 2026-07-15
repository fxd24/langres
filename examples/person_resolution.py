"""Person resolution: the embeddings + LLM "strong path", end-to-end and deterministic.

This is the runnable proof that langres's target architecture — semantic
blocking (real sentence-transformer embeddings + FAISS ANN) feeding an LLM judge,
clustered into entities, and round-tripped through ``save``/``load`` — actually
runs on Person-shaped records. It is wired **manually** from ``langres.core``
primitives via ``Resolver(...)`` (the declarative ``from_schema(blocker=, matcher=)``
builder is intentionally out of scope for M0.5).

It stays deterministic and free by injecting a tiny **fake** LLM client whose
``completion(...)`` returns canned ``MATCH/NO_MATCH`` text derived from a
name-normalization rule that mirrors how an LLM would judge these records
(accent-stripped, lower-cased, order-insensitive, initial-aware). Swap that fake
for ``LLMMatcher.from_env(...)`` and the exact same pipeline calls a real model.

Run it::

    OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=1 uv run python examples/person_resolution.py

What it shows:

1. Build the strong-path ``Resolver`` manually: ``VectorBlocker`` (declarative
   ``schema=`` + ``text_field=``, cosine FAISS) -> ``comparator=None`` (the judge
   reads raw entities) -> ``LLMMatcher`` -> ``Clusterer``.
2. Resolve a small Person fixture with known duplicates into entity clusters.
3. Report **blocking Pair-Completeness** (the M1-critical "is blocking even
   catching the duplicates" signal) and **honest cost** (0.0 with the fake).
4. ``save`` the pipeline to a human-readable, pickle-free ``resolver.json``,
   reload it, re-attach the fake client, and prove the clustering is identical.
5. Optionally run one real pair through ``LLMMatcher.from_env`` when
   ``OPENROUTER_API_KEY`` is set (skipped cleanly otherwise).

``print`` is allowed in examples (this is demonstration, not library code).
"""

from __future__ import annotations

import re
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from langres.core import Resolver
from langres.core.blockers.vector import VectorBlocker
from langres.core.clusterer import Clusterer
from langres.core.embeddings import SentenceTransformerEmbedder
from langres.core.indexes.vector_index import FAISSIndex
from langres.core.metrics import (
    calculate_bcubed_metrics,
    calculate_pairwise_metrics,
    evaluate_blocking,
)
from langres.core.models import PairwiseJudgement
from langres.core.matchers.llm_judge import LLMMatcher

# ----------------------------------------------------------------------------
# Schema + fixture: a handful of people with KNOWN duplicates.
# ----------------------------------------------------------------------------


class PersonSchema(BaseModel):
    """A minimal Person entity: an id, a name, and a few optional attributes."""

    id: str = Field(description="Unique entity identifier")
    name: str = Field(description="Person's full name (the field we embed)")
    role: str | None = Field(default=None, description="Job title / role")
    org: str | None = Field(default=None, description="Organization / employer")
    linkedin_url: str | None = Field(default=None, description="LinkedIn profile URL")


# 10 records. Two true duplicate clusters with realistic name variation
# (accents vs stripped, abbreviated initial, name-order swap) plus look-alike
# non-duplicates that must stay separate (shared first name; near-name).
PERSON_RECORDS: list[dict[str, Any]] = [
    # Cluster A — Joséphine Goube (4 surface forms of the same person).
    {"id": "p1", "name": "Joséphine Goube", "role": "CEO", "org": "Tech4Good"},
    {"id": "p2", "name": "Josephine Goube", "role": "Chief Executive", "org": "Tech4Good"},
    {"id": "p3", "name": "J. Goube", "org": "Tech4Good"},
    {"id": "p4", "name": "Goube, Josephine", "linkedin_url": "https://linkedin.com/in/jgoube"},
    # Cluster B — Liang Wei (name-order variation, common for Chinese names).
    {"id": "p5", "name": "Liang Wei", "role": "Researcher", "org": "DeepMind"},
    {"id": "p6", "name": "Wei Liang", "role": "Research Scientist", "org": "DeepMind"},
    # Non-duplicates (must NOT merge).
    {"id": "p7", "name": "Maria Garcia", "role": "Designer"},
    {"id": "p8", "name": "Maria Silva", "role": "Designer"},
    {"id": "p9", "name": "John Smith", "org": "Acme"},
    {"id": "p10", "name": "Jonathan Smith", "org": "Acme"},
]

# Ground-truth entity clusters (multi-record only) and the duplicate pairs they
# imply — the blocking Pair-Completeness target.
KNOWN_DUPLICATE_GROUPS: list[set[str]] = [
    {"p1", "p2", "p3", "p4"},
    {"p5", "p6"},
]
KNOWN_DUPLICATE_PAIRS: set[frozenset[str]] = {
    frozenset({a, b}) for group in KNOWN_DUPLICATE_GROUPS for a in group for b in group if a < b
}


# ----------------------------------------------------------------------------
# Deterministic fake LLM client.
# ----------------------------------------------------------------------------


def _normalize_tokens(name: str) -> list[str]:
    """Lower-case, strip accents, drop punctuation -> sorted alnum tokens.

    ``"Joséphine Goube"`` and ``"Goube, Josephine"`` both normalize to
    ``["goube", "josephine"]`` — accent- and order-insensitive.
    """
    folded = unicodedata.normalize("NFKD", name)
    ascii_only = folded.encode("ascii", "ignore").decode("ascii").lower()
    tokens = [t for t in re.split(r"[^a-z0-9]+", ascii_only) if t]
    return sorted(tokens)


def _names_match(a: str, b: str) -> bool:
    """True iff two names plausibly denote the same person.

    Mirrors an LLM's judgement on these records with a transparent rule: same
    number of normalized tokens, each aligned pair either equal or one being a
    single-letter initial of the other (so ``"J. Goube"`` matches
    ``"Josephine Goube"`` but ``"John Smith"`` does not match ``"Jonathan Smith"``).
    """
    ta, tb = _normalize_tokens(a), _normalize_tokens(b)
    if len(ta) != len(tb):
        return False
    for x, y in zip(ta, tb, strict=True):
        if x == y:
            continue
        if len(x) == 1 and y.startswith(x):
            continue
        if len(y) == 1 and x.startswith(y):
            continue
        return False
    return True


@dataclass
class _FakeUsage:
    prompt_tokens: int
    completion_tokens: int


@dataclass
class _FakeMessage:
    content: str


@dataclass
class _FakeChoice:
    message: _FakeMessage


@dataclass
class _FakeResponse:
    choices: list[_FakeChoice]
    usage: _FakeUsage


class FakeLLMClient:
    """A canned, deterministic stand-in for a real LLM client.

    Implements only the ``.completion(...)`` surface ``LLMMatcher.forward`` calls.
    It extracts the two record names from the rendered prompt and answers
    ``MATCH``/``NO_MATCH`` via :func:`_names_match`, so the whole pipeline runs
    offline, for free, and identically on every invocation.
    """

    # The judge dumps each entity as JSON, so the prompt contains exactly two
    # ``"name": "..."`` fields — Record A then Record B.
    _NAME_RE = re.compile(r'"name":\s*"([^"]*)"')

    def completion(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        **kwargs: Any,
    ) -> _FakeResponse:
        prompt = messages[0]["content"]
        names = self._NAME_RE.findall(prompt)
        if len(names) != 2:  # pragma: no cover - prompt shape is fixed
            raise ValueError(f"Expected two names in prompt, found {len(names)}: {names!r}")

        is_match = _names_match(names[0], names[1])
        verdict = "MATCH" if is_match else "NO_MATCH"
        score = 0.95 if is_match else 0.05
        reasoning = (
            f"Names '{names[0]}' and '{names[1]}' normalize to the same person."
            if is_match
            else f"Names '{names[0]}' and '{names[1]}' denote different people."
        )
        content = f"{verdict}\nScore: {score}\nReasoning: {reasoning}"
        return _FakeResponse(
            choices=[_FakeChoice(message=_FakeMessage(content=content))],
            usage=_FakeUsage(prompt_tokens=len(prompt) // 4, completion_tokens=16),
        )


# ----------------------------------------------------------------------------
# Pipeline construction + run.
# ----------------------------------------------------------------------------


def build_resolver(client: Any, *, k_neighbors: int = 5, threshold: float = 0.5) -> Resolver:
    """Wire the embeddings + LLM strong path manually into a Resolver.

    Args:
        client: The LLM client injected into the ``LLMMatcher`` (the deterministic
            :class:`FakeLLMClient` here; ``LLMMatcher.from_env`` in production).
        k_neighbors: Nearest neighbours per record for the vector blocker.
        threshold: Clusterer match threshold over judge scores.

    Returns:
        A runnable, serializable Resolver (declarative blocker -> ``None``
        comparator -> LLM judge -> clusterer).
    """
    blocker: VectorBlocker[PersonSchema] = VectorBlocker(
        vector_index=FAISSIndex(
            embedder=SentenceTransformerEmbedder("all-MiniLM-L6-v2"),
            metric="cosine",
        ),
        schema=PersonSchema,
        text_field="name",
        k_neighbors=k_neighbors,
    )
    judge: LLMMatcher[PersonSchema] = LLMMatcher(
        client=client,
        model="openrouter/openai/gpt-4o-mini",
        temperature=0.0,
        entity_noun="person",
    )
    return Resolver(
        blocker=blocker,
        comparator=None,  # the LLM judge reads raw entities; no comparison vector
        matcher=judge,
        clusterer=Clusterer(threshold=threshold),
    )


def _canonical(clusters: list[set[str]]) -> frozenset[frozenset[str]]:
    """Order-independent cluster identity for equality checks."""
    return frozenset(frozenset(c) for c in clusters)


def run_demo() -> dict[str, Any]:
    """Run the deterministic strong path end-to-end and return its key results.

    Returns a dict with the predicted clusters, blocking recall, total cost,
    BCubed/pairwise F1, and whether a save/reload round-trip reproduced the
    clustering identically — everything the test asserts on.
    """
    client = FakeLLMClient()
    resolver = build_resolver(client)

    # Build the blocker's index explicitly so we can measure blocking quality
    # on the candidates before the LLM ever runs. The Resolver reuses this same
    # built index (it never re-embeds an identical corpus).
    blocker = resolver.blocker
    assert isinstance(blocker, VectorBlocker)  # narrows Blocker[Any] -> VectorBlocker
    texts = [str(r["name"]) for r in PERSON_RECORDS]
    blocker.vector_index.create_index(texts)
    candidates = list(blocker.stream(PERSON_RECORDS))
    blocking_stats = evaluate_blocking(candidates, KNOWN_DUPLICATE_GROUPS)

    # Score (LLM judge) then cluster. predict() returns the judgements so we can
    # sum honest cost without paying for a second pass.
    judgements: list[PairwiseJudgement] = resolver.predict(PERSON_RECORDS)
    clusters = resolver.clusterer.cluster(judgements)
    total_cost = sum(float(j.provenance["cost_usd"]) for j in judgements)

    bcubed = calculate_bcubed_metrics(clusters, KNOWN_DUPLICATE_GROUPS)
    pairwise = calculate_pairwise_metrics(clusters, KNOWN_DUPLICATE_GROUPS)

    # save -> load -> re-attach the fake client -> re-resolve identically.
    with tempfile.TemporaryDirectory() as tmp:
        artifact_dir = Path(tmp) / "person_v0"
        resolver.save(artifact_dir)
        manifest = (artifact_dir / "resolver.json").read_text()
        reloaded = Resolver.load(artifact_dir)
        # from_config rebuilds the judge with a lazy env client; re-inject the
        # SAME fake so the reloaded run is deterministic and free.
        reloaded.module.client = client  # type: ignore[attr-defined]
        reloaded_clusters = reloaded.clusterer.cluster(reloaded.predict(PERSON_RECORDS))

    identical = _canonical(clusters) == _canonical(reloaded_clusters)

    return {
        "clusters": clusters,
        "candidates": candidates,
        "blocking_recall": blocking_stats.candidate_recall,
        "blocking_stats": blocking_stats,
        "num_judgements": len(judgements),
        "total_cost": total_cost,
        "bcubed": bcubed,
        "pairwise": pairwise,
        "identical": identical,
        "reloaded_clusters": reloaded_clusters,
        "manifest": manifest,
    }


def maybe_live_smoke() -> None:
    """Run ONE real pair through ``LLMMatcher.from_env`` if a key is present.

    Gated on ``OPENROUTER_API_KEY`` so the default run is offline and free.
    Cost is trivial (a single pair). Prints the real ``completion_cost``.
    """
    import os

    if not os.getenv("OPENROUTER_API_KEY"):
        print("\n[live smoke] OPENROUTER_API_KEY not set — skipping real LLM call.")
        return

    from langres.core.models import ERCandidate

    judge: LLMMatcher[PersonSchema] = LLMMatcher.from_env(
        model="openrouter/openai/gpt-4o-mini", temperature=0.0, entity_noun="person"
    )
    pair: ERCandidate[PersonSchema] = ERCandidate(
        left=PersonSchema(**PERSON_RECORDS[0]),
        right=PersonSchema(**PERSON_RECORDS[1]),
        blocker_name="manual",
    )
    judgement = next(iter(judge.forward(iter([pair]))))
    print("\n[live smoke] real LLM judgement on (p1, p2):")
    print(f"  score={judgement.score:.3f}  cost=${judgement.provenance['cost_usd']:.6f}")
    print(f"  reasoning: {judgement.reasoning}")


def _print_clusters(title: str, clusters: list[set[str]]) -> None:
    print(f"\n{title}")
    for cluster in sorted(sorted(c) for c in clusters):
        print(f"  {cluster}")


def main() -> None:
    # Quiet langres' info/warning chatter (e.g. the honest "completion_cost
    # unavailable -> 0.0" note the fake response triggers) for a clean demo.
    import logging

    logging.getLogger("langres").setLevel(logging.ERROR)

    print("=" * 78)
    print("Person resolution — embeddings + LLM strong path (deterministic)")
    print("=" * 78)

    results = run_demo()

    _print_clusters("Predicted clusters:", results["clusters"])

    recall = results["blocking_recall"]
    print(
        f"\nBlocking Pair-Completeness (recall over known duplicate pairs): {recall:.3f}"
        f"  [target >= 0.95]  {'PASS' if recall >= 0.95 else 'LOW'}"
    )
    stats = results["blocking_stats"]
    print(
        f"  candidates={stats.total_candidates}  "
        f"missed={stats.missed_matches_count}  "
        f"avg_per_entity={stats.avg_candidates_per_entity:.1f}"
    )

    print(
        f"\nEnd-to-end accuracy vs known duplicates: "
        f"BCubed F1={results['bcubed']['f1']:.3f}  "
        f"Pairwise F1={results['pairwise']['f1']:.3f}"
    )

    print(
        f"\nHonest cost: ${results['total_cost']:.6f} over "
        f"{results['num_judgements']} LLM judgements (0.0 with the fake client)."
    )

    print("\nresolver.json is human-readable, pickle-free, and carries NO client/secret.")
    print("Manifest head (note the judge's model ref but no API key):")
    for line in results["manifest"].splitlines()[:14]:
        print(f"  {line}")

    _print_clusters("Clusters after save -> load -> re-resolve:", results["reloaded_clusters"])
    assert results["identical"], "Reloaded clustering differs from the original!"
    print("\nSave/reload round-trip: clustering is IDENTICAL ✓")

    maybe_live_smoke()

    print("\n" + "=" * 78)
    print("Strong path ran end-to-end. ✓")
    print("=" * 78)


if __name__ == "__main__":
    main()
