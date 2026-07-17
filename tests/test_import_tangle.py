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

The two numbers are far apart (0 vs 10) precisely *because* langres uses lazy
imports on purpose. Pinning only one would hide half the tangle -- and the runtime
view being *empty* is exactly why: pinning only the all-edges 10 would let a new
eager cycle grow back with the gate still green.

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

Why not import-linter (yet)
---------------------------
Measured against import-linter 2.13 / grimp 3.15, not assumed:

* **grimp cannot see the runtime view at all.** ``DirectImport``
  (``grimp/domain/valueobjects.py``) carries only importer/imported/line -- no
  scope field. ``exclude_type_checking_imports`` exists (top-level
  ``[tool.importlinter]`` only), but there is **no equivalent for function-local
  imports**: ``def build(): import faiss`` is recorded as a plain edge. langres
  has 63 function-local edges and 102 edges that exist *only* lazily -- the
  deliberate extras seam that ``tests/test_import_budget.py`` mandates. So a
  ``forbidden`` contract on the heavy deps would fire on the architecture working
  as designed, and import-linter could only ever gate the 43 view, never the 12.
* **A ``layers`` contract is not writable today.** ``layers.py`` raises
  ``ValueError: Missing layer ... does not exist`` -- a hard crash, not a report
  -- and the target packages don't exist yet. Their order isn't decided either.
* **The contracts that *are* true today are not useful.** With the runtime cycle
  gone, 13 of the 14 direct children of ``langres`` sit outside the all-edges
  tangle (only ``langres.methods`` is still in it); an ``independence`` contract
  over them would assert a regression nobody could plausibly commit. Everything
  worth gating is already gated, and better:
  cycles by this file (both views), eager-heavy-imports by
  ``test_import_budget.py`` (which executes the import and reads ``sys.modules``
  -- ground truth, zero false positives on lazy seams).

import-linter earns its place when the refactor's target packages exist and their
order is decided: a ``layers`` contract then asserts *direction*, which this gate
deliberately does not. Until then ``tools/import_graph.py counterfactual
--mapping ...`` answers "what would this split do to the cycles?" with no new
dependency.

Cost: pure AST parsing, no imports executed, no network, ~1s. It runs in the
per-PR ``test`` job (``.github/workflows/test.yml``), not just ``test-full``.
"""

from __future__ import annotations

from dataclasses import dataclass

# `tools` is on the path via [tool.pytest.ini_options] pythonpath in pyproject.toml.
from import_graph import ImportKind, build_graph


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

    #: ``sccs()`` only ever returns components with >1 member, so the fully
    #: decoupled state is expressible exactly and without a sentinel: an empty
    #: ``tangled`` and ``largest_scc = 0`` mean "no module is in any cycle" --
    #: the same pair ``_report``'s ``max(..., default=0)`` computes for it. 0 is
    #: a measurement here, never "unmeasured".


# ---------------------------------------------------------------------------
# THE BASELINES. Re-measured on `refactor/kill-runtime-cycle` (the runtime cycle
# is gone), stacked on the facade-emptying wave before it, with:
#     uv run python tools/import_graph.py kinds
# Lower these freely when you decouple something -- that is the ratchet working.
# RAISING either number means your change coupled the codebase further: say why,
# in a comment, right here, and expect review to push back.
#
# Two waves ratcheted this down, in order:
#
#  1. The facade-emptying wave (`langres.core` stopped re-exporting
#     implementations): all-edges 43 -> 39, tangled 48 -> 44. The four that left
#     were `core/_exports/_clustering`, `_matchers` and `_eval` (deleted or
#     reduced to contracts) plus `core.matchers.cascade_judge`, which only
#     reached the cycle through the `_matchers` re-export. The runtime view did
#     NOT move (12) -- that wave's measured finding was that emptying the facade
#     *cannot* shrink it, because the runtime cycle was never closed by the
#     component re-exports. It was closed by `core/resolver.py`'s toplevel
#     `import langres`, present solely to read `langres.__version__` when
#     stamping an artifact: the floor importing the ceiling for a version string.
#
#  2. Killing exactly that edge (this wave). The version string moved to
#     `langres/_version.py`, a stdlib-only leaf that imports no langres, so
#     `resolver` and `cli` read it without depending on the root. That wave's
#     prediction held exactly: runtime 12 -> 0, and all-edges 39 -> 10 with it.
#
# Two more measured facts for whoever plans the next wave (both from
# `tools/import_graph.py counterfactual --mapping tools/refactor_target_packages.json`):
#
#  * Package knots went 18 -> 16 -> 15 -> **14** (all edges) and 11 -> 10 -> 9 ->
#    **8** (toplevel). The facade wave killed `benchmarks <-> core` and
#    `core <-> optimize`; the runtime-cycle wave killed `architectures <-> root`
#    in both views (`core/resolver.py` maps to `architectures`, and its
#    `import langres` was that knot's only edge); W1's comparator split killed
#    `components <-> core`, the biggest one (see below). `langres._version` maps
#    to `core` for this reason -- see the mapping's own `_comment`.
#  * The biggest knot, `components <-> core`, went from (65 + 16) to (65 + 1) to
#    **GONE** (W1). The 65 always ran components -> core (implementations
#    importing contracts -- the direction the layering wants); the ONE survivor
#    was `core/_exports/_blocking -> core.comparator`, which existed only because
#    the `Comparator` ABC shared `core/comparator.py` with `StringComparator`, so
#    the target mapping sent that whole module to `components`. W1 split it at
#    symbol granularity (ABC -> `core.comparator`, `StringComparator` ->
#    `core.comparators`), and `components -> core` is now a clean ONE-WAY
#    dependency -- it is no longer a mutual pair at all, so it has dropped off
#    the counterfactual's list. Mutual pairs: 15 -> 14 (all edges), 9 -> 8
#    (toplevel).
#
#    Measured caveat for whoever plans the next split: the file-sharing was NOT
#    the only contract -> impl link. `Comparator.from_schema` -- an ABC
#    classmethod whose body constructed `StringComparator` -- would have
#    re-created the same backwards edge from the new module, so splitting the
#    file alone RELOCATES the knot rather than removing it. It had to go (callers
#    now use `StringComparator.from_schema`, the identical factory the ABC merely
#    delegated to). Expect the same shape in the reports.py / benchmark.py splits
#    the mapping's `_comment` still flags: grep the ABC for references to its own
#    concrete subclasses before predicting an edge count.
# ---------------------------------------------------------------------------

# Runtime view: EMPTY. A bare `import langres` executes no import cycle at all.
#
# It was 12 modules -- the export-fragment knot, `langres/__init__` ->
# `_exports/*` -> `core` -> `core/_exports/*` -> `resolver` -> `presets` and back
# up to `verbs`. That loop had exactly ONE edge closing it: `core/resolver.py`
# did a toplevel `import langres` to read `langres.__version__` for the artifact
# manifest -- the floor importing the ceiling for a version string. The string
# now lives in `langres/_version.py`, a stdlib-only leaf that imports no langres,
# so `resolver` and `cli` read it without depending on the root and the loop has
# no edge left to close. (Verified by dropping that single edge from the graph
# and recomputing Tarjan: 12 -> 0. Nothing else was needed.)
#
# This zero is the ratchet's whole point: it is now IMPOSSIBLE to add an eager
# cycle to langres without this test failing. Do not raise it back.
RUNTIME = TangleBaseline(
    view="runtime (toplevel edges only -- what `import langres` executes)",
    kinds=(ImportKind.TOPLEVEL,),
    largest_scc=0,
    tangled=frozenset(),
)

# All-edges view: 18 modules across five cycles, the largest being 9. Killing the
# runtime cycle collapsed this view (39 -> 10, tangled 44 -> 24): the root, both
# `_exports` trees, `verbs` and `optimize` were only ever tangled here *via* that
# same `resolver -> langres` edge, so they left with it. What remains are five
# genuinely lazy knots, none of them touching the root:
#    9  the `data.data_profile.*` section graph
#    3  {core.reports, metrics.analysis, plotting.blockers}
#    2  {curation.anchor_store, core.resolver}
#    2  {benchmarks.runner, langres.methods}
#    2  {tracking.runs, tracking.trackers}
# `tangled` covers all five; `largest_scc` pins the biggest so they cannot
# silently merge into one.
#
# The benchmark-split wave (this PR) RELOCATED one knot member without changing
# the shape: `core/benchmark.py` split into the import-light spec
# (`langres.data.benchmark`) and the harness package (`langres.benchmarks.*`),
# leaving a re-export shim at the old path. The `{core.benchmark, methods}` 2-cycle
# is now `{benchmarks.runner, methods}` -- `methods.py` imports `_cost_track` from
# `benchmarks.runner` (toplevel) and `run_methods` in `benchmarks.runner` imports
# `make_resolver_factory` from `methods` (function-local), the same litellm-seam
# lazy cycle as before, one module renamed. The shim `core.benchmark` left the
# tangle: no module imports it at toplevel anymore, so it has only outgoing edges.
# Predicted by recomputing Tarjan before the move, measured identical after:
# [9, 3, 2, 2, 2] / 18, `largest_scc` still 9.
#
# W4 (the ERModel/architectures wave) ratcheted this DOWN: 10 -> 9, tangled
# 24 -> 23. `langres.core.presets` left, because W4 deleted it outright. It was
# the verbs' machinery -- judge="auto" key-sniffing, build_resolver,
# build_judge -- and once `verbs.py` went, its only remaining importer was
# `core/benchmark.py` reaching for `_effective_budget`, an alias it now imports
# from the `core.spend_cap` leaf that always owned it. Nothing was moved to buy
# this: the module is gone.
#
# The `CostTrack` wave ratcheted it down again: tangled 23 -> 18, and the OTHER
# 9 -- "the matcher/resolver + methods/benchmark knot (litellm seam)" this
# comment used to list first -- did not shrink, it **disintegrated**, into
# {anchor_store, resolver} and {benchmark, methods}. Five modules left in one
# move: `clients.openrouter`, `core.finetune`, `core.matchers.cascade`,
# `core.matchers.llm_judge`, `core.method_registry`.
#
# ONE edge held all nine together: `clients/openrouter.py` -- an HTTP client --
# imported `CostTrack` from `core/benchmark.py`, a 1.7k-line benchmark harness,
# to build one in `make_token_cost_track`. The floor importing the ceiling, and
# the same shape as PR #176's `resolver -> langres.__version__`. Both statements
# were lazy (one TYPE_CHECKING, one function-local), which is exactly why the
# runtime view was already 0 while this view carried a 9: the edge was invisible
# to `import langres` and fully visible to grimp. `CostTrack` (and its
# `CostBasis` alias) now live in the `core.usage` leaf beside the `LLMUsage`
# that `CostTrack.usage` holds -- tokens are the fact, dollars are derived, so
# the two models belong together and `core.usage` still imports nothing from
# langres. The aggregator `benchmark._cost_track` deliberately did NOT move: it
# reads the harness's `PairwiseJudgement.provenance` conventions, so it would
# have dragged `core.models` in and cost `core.usage` the leaf property that
# made it a safe home. Verified by dropping the edge and recomputing Tarjan
# BEFORE writing the move: predicted [9, 3, 2, 2, 2] / 18, measured the same.
#
# `largest_scc` stays 9 and that is a measurement, not an oversight: the two 9s
# were always different components that happened to be the same size (the note
# this comment used to carry about `largest_scc` being unable to tell them
# apart). The litellm-seam 9 is gone; the `data.data_profile.*` 9 -- a
# self-contained knot in a package this wave never touched -- is what the number
# now pins. Whoever unpicks that one gets to lower it.
#
# Measured fact for whoever plans the next wave: the NEW `langres.architectures`
# package did **not** join the tangle, in either view. Its modules import
# downward only (`core.resolver`, `core.registry`, the component packages) and
# keep every heavy dep ([semantic]/[llm]) inside `_topology`'s function body. An
# architecture that grows a toplevel import of a module which imports it back
# would show up here as an ENTERED line.
#
# The metrics-package wave (`core/{metrics,analysis,debugging,diagnostics}.py` ->
# the `langres.metrics` package, back-compat shims left at the old paths) did NOT
# change the shape -- it RELOCATED one knot member. The `{core.analysis,
# core.reports, plotting.blockers}` 3-cycle is now `{metrics.analysis,
# core.reports, plotting.blockers}`: `analysis` moved out of core, and the ONE
# edge that would otherwise have GROWN this cycle to 4 (`core.reports:960`
# reaching `analysis`) was repointed at `metrics.analysis` so `core.analysis` --
# now a pure re-export shim -- stays a leaf outside every cycle. Predicted by
# recomputing Tarjan before the move, measured identical after: [9, 3, 2, 2, 2] /
# 18, `largest_scc` still 9. `metrics.analysis` is the ONLY member below that
# swapped names.
ALL_EDGES = TangleBaseline(
    view="all-edges (incl. lazy/TYPE_CHECKING -- what grimp/import-linter sees)",
    kinds=None,
    largest_scc=9,
    tangled=frozenset(
        {
            "langres.curation.anchor_store",
            "langres.benchmarks.runner",
            "langres.core.reports",
            "langres.core.resolver",
            "langres.tracking.runs",
            "langres.tracking.trackers",
            "langres.metrics.analysis",
            "langres.data.data_profile.base",
            "langres.data.data_profile.builders",
            "langres.data.data_profile.corpus_field",
            "langres.data.data_profile.embedding_section",
            "langres.data.data_profile.failure_mode",
            "langres.data.data_profile.hero",
            "langres.data.data_profile.label_structure",
            "langres.data.data_profile.mining_readiness",
            "langres.data.data_profile.separability",
            "langres.methods",
            "langres.plotting.blockers",
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
