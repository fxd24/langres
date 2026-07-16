"""Import-tangle ratchet: the cyclic core of the import graph must not grow.

The architecture refactor's central risk is that a wave couples two more modules
together and no gate notices. That is not hypothetical -- **PR #169** (the
``core/__init__`` fragment split) grew the module SCC from **29 to 43** and the
runtime SCC from **5 to 12**, and CI stayed green the whole way. This file is the
gate that would have caught it.

Why SCC membership, and not "mutual pairs"
------------------------------------------
The obvious metric -- count the pairs of modules that import each other -- is
**worse than useless here**, and PR #169 proves it. Measured across that merge
(``tools/import_graph.py`` run against ``858d605^1`` and ``858d605``):

===========================  ==========  ===========
metric                       pre #169    post #169
===========================  ==========  ===========
modules in a cycle (all)     34          48
largest SCC (all edges)      29          43
largest SCC (runtime)        5           12
**mutual pairs**             **6**       **6**
===========================  ==========  ===========

The mutual-pair count did not merely go down -- it did not move *at all*. The
same six pairs, before and after, while the runtime tangle more than doubled. The
knot ``langres <-> langres.core`` stopped being a mutual pair only because
``langres/__init__`` now reaches core via ``langres._exports``: a 2-cycle became a
3-cycle. Same loop, one more relay on it. (At the *package* level the metric is
actively perverse -- it improves, 4 -> 3.) A mutual-pair ratchet would have waved
PR #169 through without a flicker.

So this gate counts **membership of a strongly connected component**: the set of
modules you cannot untangle from each other, however many hops the loop takes.
Adding a relay module to a cycle makes that set *bigger*, never smaller. That is
the property that makes the ratchet honest, and it is why no metric here counts
edges or pairs.

The two views, and why both are pinned
--------------------------------------
``tools/import_graph.py`` classifies each import as toplevel / function-local /
TYPE_CHECKING, which lets us gate two genuinely different graphs:

* **runtime** (toplevel edges only) -- what actually executes on ``import
  langres``. This is the graph ``tests/test_import_budget.py`` protects.
* **all-edges** -- what grimp/import-linter would see. They cannot distinguish a
  lazy function-body import from an eager one (grimp's ``DirectImport`` carries no
  scope field), so every deliberate lazy-extras seam counts as a real edge here.

The two numbers are far apart (12 vs 43) precisely *because* langres uses lazy
imports on purpose. Pinning only one would hide half the tangle.

Per view we pin two complementary facts, and both are load-bearing:

* ``tangled`` -- every module in *any* cycle. Catches a module joining a cycle,
  and catches a brand-new cycle appearing somewhere else entirely (which a
  largest-SCC-only gate would miss).
* ``largest_scc`` -- the biggest single component. Catches two existing cycles
  *merging* (membership unchanged, structure worse), which ``tangled`` misses.

Ratchet down, never up
----------------------
The assertion is **exact equality**, not ``<=``. That is deliberate and it is the
asymmetry that makes this a ratchet:

* **Lowering a baseline is a normal PR.** Decouple something, the test fails
  telling you the new number, you commit it. Two lines in a diff.
* **Raising a baseline is a deliberate act** that must be argued for in review,
  with a comment saying why the coupling is worth it.

``<=`` would quietly rot: an improvement to 40 that nobody records leaves the gate
still accepting 43, so a later regression back to 43 sails through green. Exact
equality keeps the committed number equal to the truth, always.

Cost: pure AST parsing, no imports executed, no network, ~1s. It runs in the
per-PR ``test`` job (``.github/workflows/test.yml``), not just ``test-full``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

TOOLS = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(TOOLS))

from import_graph import ImportKind, build_graph  # noqa: E402


@dataclass(frozen=True, slots=True)
class TangleBaseline:
    """The committed, measured shape of one view of the import graph.

    `kinds` is the edge filter handed to ``ImportGraph.sccs`` -- ``None`` means
    every edge (the grimp/import-linter view).
    """

    view: str
    kinds: tuple[ImportKind, ...] | None
    largest_scc: int
    tangled: frozenset[str]


# ---------------------------------------------------------------------------
# THE BASELINES. Measured on ba4b1b7 (PR #171 merge) with:
#     uv run python tools/import_graph.py kinds
# Lower these freely when you decouple something -- that is the ratchet working.
# RAISING either number means your change coupled the codebase further: say why,
# in a comment, right here, and expect review to push back.
# ---------------------------------------------------------------------------

# Runtime view: 12 modules, one cycle. This is the export-fragment knot --
# `langres/__init__` -> `_exports/*` -> `core` -> `core/_exports/*` -> `resolver`
# -> `presets` and back up to `verbs`. Every one of these executes on a bare
# `import langres`.
RUNTIME = TangleBaseline(
    view="runtime (toplevel edges only -- what `import langres` executes)",
    kinds=(ImportKind.TOPLEVEL,),
    largest_scc=12,
    tangled=frozenset(
        {
            "langres",
            "langres._exports",
            "langres._exports._core",
            "langres._exports._flywheel",
            "langres._exports._training",
            "langres._exports._verbs",
            "langres.core",
            "langres.core._exports",
            "langres.core._exports._resolver",
            "langres.core.presets",
            "langres.core.resolver",
            "langres.verbs",
        }
    ),
)

# All-edges view: 48 modules across three cycles, the largest being 43. The other
# two are small and independent: {core.analysis, core.reports, plotting.blockers}
# and {core.runs, core.trackers}. `tangled` covers all three; `largest_scc` pins
# the 43 so those cycles cannot silently merge into it.
ALL_EDGES = TangleBaseline(
    view="all-edges (incl. lazy/TYPE_CHECKING -- what grimp/import-linter sees)",
    kinds=None,
    largest_scc=43,
    tangled=frozenset(
        {
            "langres",
            "langres._exports",
            "langres._exports._core",
            "langres._exports._data",
            "langres._exports._flywheel",
            "langres._exports._optimize",
            "langres._exports._training",
            "langres._exports._verbs",
            "langres.clients.openrouter",
            "langres.core",
            "langres.core._exports",
            "langres.core._exports._clustering",
            "langres.core._exports._eval",
            "langres.core._exports._flywheel",
            "langres.core._exports._matchers",
            "langres.core._exports._methods",
            "langres.core._exports._resolver",
            "langres.core.analysis",
            "langres.core.anchor_store",
            "langres.core.benchmark",
            "langres.core.eval_report",
            "langres.core.finetune",
            "langres.core.judgement_log",
            "langres.core.matchers.cascade",
            "langres.core.matchers.cascade_judge",
            "langres.core.matchers.llm_judge",
            "langres.core.method_registry",
            "langres.core.presets",
            "langres.core.reports",
            "langres.core.resolver",
            "langres.core.runs",
            "langres.core.trackers",
            "langres.data.data_profile",
            "langres.data.data_profile.base",
            "langres.data.data_profile.builders",
            "langres.data.data_profile.corpus_field",
            "langres.data.data_profile.embedding_section",
            "langres.data.data_profile.embedding_source",
            "langres.data.data_profile.failure_mode",
            "langres.data.data_profile.hero",
            "langres.data.data_profile.label_structure",
            "langres.data.data_profile.mining_readiness",
            "langres.data.data_profile.separability",
            "langres.data.registry",
            "langres.methods",
            "langres.optimize",
            "langres.plotting.blockers",
            "langres.verbs",
        }
    ),
)


def _bullet(modules: list[str]) -> str:
    return "\n".join(f"      {m}" for m in modules)


def _report(base: TangleBaseline, sccs: list[list[str]]) -> str | None:
    """The failure report for `base`, or None if the graph still matches it.

    Says which modules moved and in which direction, because "43 != 44" tells the
    person who broke it nothing about what they broke.
    """
    tangled = frozenset(m for scc in sccs for m in scc)
    largest = max((len(s) for s in sccs), default=0)
    entered = sorted(tangled - base.tangled)
    left = sorted(base.tangled - tangled)
    if not entered and not left and largest == base.largest_scc:
        return None

    worse = len(tangled) > len(base.tangled) or largest > base.largest_scc
    headline = "IMPORT TANGLE GREW" if worse else "IMPORT TANGLE CHANGED"
    lines = [
        f"{headline} -- view: {base.view}",
        "",
        f"  modules in a cycle : {len(base.tangled)} (baseline) -> {len(tangled)} (now)",
        f"  largest SCC        : {base.largest_scc} (baseline) -> {largest} (now)",
        f"  cycles now         : {sorted((len(s) for s in sccs), reverse=True)}",
    ]
    if entered:
        lines += [
            "",
            f"  ENTERED the tangle ({len(entered)}) -- your change coupled these into a cycle:",
            _bullet(entered),
        ]
    if left:
        lines += [
            "",
            f"  LEFT the tangle ({len(left)}) -- these are now decoupled:",
            _bullet(left),
        ]
    lines += [
        "",
        "  Trace any of them with:",
        "      uv run python tools/import_graph.py cycles --toplevel-only"
        if base.kinds
        else "      uv run python tools/import_graph.py cycles",
        f"      uv run python tools/import_graph.py importers {(entered or left or ['<module>'])[0]}",
        "",
    ]
    if worse:
        lines += [
            "  This is the regression PR #169 shipped unnoticed. Prefer breaking the",
            "  coupling over raising the baseline. If the coupling really is worth it,",
            "  raise the number in tests/test_import_tangle.py AND say why in a comment.",
        ]
    else:
        lines += [
            "  You decoupled something -- nice. Ratchet it DOWN: update largest_scc /",
            "  tangled in tests/test_import_tangle.py to the numbers above, so the gate",
            "  holds the new ground.",
        ]
    return "\n".join(lines)


def _check(base: TangleBaseline) -> None:
    sccs = build_graph().sccs(base.kinds)
    report = _report(base, sccs)
    assert report is None, report


def test_runtime_import_tangle_does_not_grow() -> None:
    """The toplevel-only SCC -- the cycle a bare ``import langres`` really executes."""
    _check(RUNTIME)


def test_all_edges_import_tangle_does_not_grow() -> None:
    """The SCC incl. lazy/TYPE_CHECKING edges -- what grimp/import-linter would see."""
    _check(ALL_EDGES)


def test_runtime_tangle_is_a_subset_of_the_all_edges_tangle() -> None:
    """Sanity-check the two baselines against each other.

    Every toplevel edge is also an all-edges edge, so a module tangled at runtime
    is necessarily tangled under the wider view. If this ever fails, a baseline
    above was hand-edited to something the tool never measured.
    """
    assert RUNTIME.tangled <= ALL_EDGES.tangled
    assert RUNTIME.largest_scc <= ALL_EDGES.largest_scc
