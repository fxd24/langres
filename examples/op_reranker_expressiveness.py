"""Expressiveness proof for the Op algebra (epic #193, W2 de-risk).

**The claim.** The old ``langres.core`` had exactly ONE legal in-pipeline shape:
``blocker -> comparator -> matcher -> clusterer`` — four fixed slots, the matcher
pinned to a single position before the clusterer. The new algebra
(:mod:`langres.core.op`) says *position is not a type*: every in-pipeline stage is
one :class:`~langres.core.op.Op` (``Pairs -> Pairs``) in one of two roles —
:class:`~langres.core.op.Score` ("same rows, new scores") or
:class:`~langres.core.op.Select` ("same scores, fewer rows"). Because both ends of
an ``Op`` are the same :class:`~langres.core.pairs.Pairs` carrier, ``Op``\\ s
*compose freely*: you can put a ``Score`` **after** a ``Select``.

That single freedom is a **reranker**, and it is inexpressible in four fixed
slots. This script builds two pipelines out of the additive Op adapters
(:mod:`langres.core.op_adapters`) and runs both **end-to-end at $0** (rapidfuzz
string similarity only — no LLM, no embeddings):

1. ``baseline`` — the classic four-slot shape, expressed as Ops:
   ``BlockerSource -> MatcherScore -> Select(THRESHOLD) -> ClustererStage``.

2. ``reranker`` — a topology the four-slot core could NOT express, a ``Score``
   AFTER a ``Select``:
   ``BlockerSource -> Score(cheap name sim) -> Select(TOPK) ->
   Score(sharper name+address sim) -> Select(THRESHOLD) -> ClustererStage``.
   The second ``Score`` rescores the survivors of the first selection
   (retrieve-then-rerank).

Both pipelines use the **same** threshold and the **same** clusterer, so the only
difference is the inserted TOPK-prune + rescoring ``Score``. On the fixed dataset
below the reranker recovers the true duplicate structure, while the name-only
baseline over-merges same-name / different-address branches.

**Selection is now first-class (W3-b).** The adapters in ``op_adapters.py`` bridge
Source / Score / ClusterStage / Finalize, and W3-b added the two concrete
``Select`` ops — :class:`~langres.core.op.ThresholdSelect` (keep the rows that
clear the price) and :class:`~langres.core.op.TopKSelect` (keep each anchor's k
best) — to :mod:`langres.core.op`. So selection is a real pipeline stage now, not
a knob folded into the clusterer's own threshold; this script imports both from
src rather than defining its own.

Run it:  ``uv run python examples/op_reranker_expressiveness.py``
"""

from __future__ import annotations

from langres.core.blockers.all_pairs import AllPairsBlocker
from langres.core.clusterer import Clusterer
from langres.core.matchers.rapidfuzz import RapidfuzzMatcher
from langres.core.models import CompanySchema
from langres.core.op import Records, Sequential, ThresholdSelect, TopKSelect
from langres.core.op_adapters import BlockerSource, ClustererStage, MatcherScore


# --------------------------------------------------------------------------------------
# Fixed in-memory dataset (a dozen records; no benchmark download).
#
# True duplicate structure:
#   * {a1, a2}  Acme       -- same name AND same address (true dup)
#   * {g1, g2}  Globex     -- same name AND same address (true dup)
#   * i1, i2    Initech    -- SAME name, DIFFERENT address (two branches, NOT a dup)
#   * u1, u2    Umbrella    -- SAME name, DIFFERENT address (two branches, NOT a dup)
#   * s1, h1, w1, p1        -- singletons
# The Initech/Umbrella branches are the trap: a name-only matcher (sim = 1.0)
# fuses them; only a matcher that also weighs the address keeps them apart.
# --------------------------------------------------------------------------------------

RECORDS: Records = [
    {"id": "a1", "name": "Acme Inc", "address": "1 Main St"},
    {"id": "a2", "name": "Acme Inc", "address": "1 Main Street"},
    {"id": "g1", "name": "Globex Corporation", "address": "500 Oak Ave"},
    {"id": "g2", "name": "Globex Corporation", "address": "500 Oak Avenue"},
    {"id": "i1", "name": "Initech", "address": "5 North Ave"},
    {"id": "i2", "name": "Initech", "address": "820 South Blvd"},
    {"id": "u1", "name": "Umbrella LLC", "address": "3 River Rd"},
    {"id": "u2", "name": "Umbrella LLC", "address": "77 Hill St"},
    {"id": "s1", "name": "Soylent Foods", "address": "7 Green Way"},
    {"id": "h1", "name": "Hooli", "address": "1 Infinite Loop"},
    {"id": "w1", "name": "Wayne Enterprises", "address": "1007 Mountain Dr"},
    {"id": "p1", "name": "Stark Industries", "address": "10880 Malibu Point"},
]

# The single knobs shared by BOTH pipelines, so the only variable is the topology.
THRESHOLD = 0.85
TOPK = 3


def _cheap_scorer() -> MatcherScore[CompanySchema]:
    """A cheap, coarse ``Score``: name-only rapidfuzz similarity ($0)."""
    matcher = RapidfuzzMatcher[CompanySchema](field_extractors={"name": (lambda e: e.name, 1.0)})
    return MatcherScore(matcher, out_space="heuristic")


def _sharp_scorer() -> MatcherScore[CompanySchema]:
    """A sharper rescoring ``Score`` σ: name AND address, so branches separate ($0)."""
    matcher = RapidfuzzMatcher[CompanySchema](
        field_extractors={
            "name": (lambda e: e.name, 0.5),
            "address": (lambda e: e.address or "", 0.5),
        }
    )
    return MatcherScore(matcher, out_space="heuristic")


def _source() -> BlockerSource[CompanySchema]:
    """The pipeline entry: all-pairs blocking over the fixed dataset ($0)."""
    return BlockerSource(AllPairsBlocker(schema=CompanySchema))


def _clusterer_stage() -> ClustererStage[CompanySchema]:
    """Pure transitive closure over whatever the final ``Select`` kept.

    Threshold 0.0 makes the clusterer treat every surviving (already-selected) row
    as an edge, so the explicit ``Select`` — not the clusterer — is the match gate.
    """
    return ClustererStage(Clusterer(threshold=0.0))


def _merge_groups(clusters: list[set[str]]) -> list[list[str]]:
    """Order-independent view of the non-singleton clusters (the actual merges)."""
    return sorted(sorted(cluster) for cluster in clusters if len(cluster) > 1)


def run_baseline() -> list[set[str]]:
    """The classic four-slot shape, expressed as Ops and run by manual ``forward`` chaining.

    ``BlockerSource -> MatcherScore(name only) -> Select(THRESHOLD) -> ClustererStage``.
    A ``Sequential`` validates the wiring at construction (it has no ``forward`` —
    topology is code); execution is the manual ``.forward()`` chain below.
    """
    source, score, select, cluster = (
        _source(),
        _cheap_scorer(),
        ThresholdSelect[CompanySchema](THRESHOLD),
        _clusterer_stage(),
    )
    Sequential([source, score, select, cluster])  # wiring check runs here

    pairs = source.forward(RECORDS)
    pairs = score.forward(pairs)
    pairs = select.forward(pairs)
    return cluster.forward(pairs)


def run_reranker() -> list[set[str]]:
    """A ``Score`` AFTER a ``Select`` — the topology four fixed slots cannot express.

    ``BlockerSource -> Score(cheap name sim) -> Select(TOPK) ->
    Score(sharper name+address sim) -> Select(THRESHOLD) -> ClustererStage``.

    The first ``Select(TOPK)`` keeps each record's ``k`` best candidates by the
    cheap score; the second ``Score`` rescores *only those survivors* with a
    sharper matcher; the final ``Select(THRESHOLD)`` gates on the sharper score.
    Same threshold and clusterer as the baseline — the rescoring is the only added
    freedom.
    """
    source = _source()
    cheap = _cheap_scorer()
    topk = TopKSelect[CompanySchema](TOPK)
    sharp = _sharp_scorer()
    threshold = ThresholdSelect[CompanySchema](THRESHOLD)
    cluster = _clusterer_stage()

    # A Score (sharp) sits AFTER a Select (topk): the wiring guard accepts it,
    # because every Op is Pairs -> Pairs and composes freely.
    Sequential([source, cheap, topk, sharp, threshold, cluster])  # wiring check runs here

    pairs = source.forward(RECORDS)
    pairs = cheap.forward(pairs)
    pairs = topk.forward(pairs)  # keep each record's TOPK candidates
    pairs = sharp.forward(pairs)  # rescore ONLY the survivors (rerank)
    pairs = threshold.forward(pairs)
    return cluster.forward(pairs)


def main() -> None:
    """Run both pipelines at $0 and print the contrast."""
    truth = [["a1", "a2"], ["g1", "g2"]]  # the only true duplicate pairs
    baseline = _merge_groups(run_baseline())
    reranker = _merge_groups(run_reranker())

    print("langres Op algebra — expressiveness proof (#193), all $0 (rapidfuzz only)\n")
    print(f"records: {len(RECORDS)}   shared threshold: {THRESHOLD}   topk: {TOPK}\n")

    print("True duplicate merges:")
    print(f"  {truth}\n")

    print("BASELINE   Source -> Score(name) -> Select(THRESHOLD) -> ClustererStage")
    print("           (the classic four-slot shape, as Ops)")
    print(f"  merges:  {baseline}")
    print("  -> over-merges the same-name / different-address branches (Initech, Umbrella)\n")

    print("RERANKER   Source -> Score(name) -> Select(TOPK) -> Score(name+address)")
    print("                  -> Select(THRESHOLD) -> ClustererStage")
    print("           (a Score AFTER a Select — INEXPRESSIBLE in four fixed slots)")
    print(f"  merges:  {reranker}")
    print("  -> the sharper rescore of the survivors separates the branches\n")

    assert reranker == truth, f"reranker should recover the true merges, got {reranker}"
    assert baseline != reranker, "the reranker topology should change the outcome"
    tightened = [group for group in baseline if group not in reranker]
    print(
        f"Result: the reranker is tighter — it dropped {len(tightened)} wrong merge(s): {tightened}"
    )
    print("The four-slot core could not place that second Score; the Op algebra composes it.")


if __name__ == "__main__":
    main()
