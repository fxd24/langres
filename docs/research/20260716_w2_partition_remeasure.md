# W2 partition re-measure: does the 12-package split still pay for itself?

**Date:** 2026-07-16
**Branch:** `docs/w2-remeasure` (based on `origin/feat/arch-refactor` @ `f7aeff0`)
**Status:** read-only measurement. No `src/` change, no PR, no merge.

> **Headline.** No. **W2 as specified does not reproduce its own justification and
> should not run as written.** The split's deliverable is a layered,
> contract-enforceable architecture. Measured on the current graph, the target
> mapping produces **one strongly-connected component containing 13 of the 15
> packages** — an `import-linter` `layers` contract would fail on essentially the
> whole architecture. Meanwhile the *largest single knot* (`architectures ↔ core`,
> 30 + 2 edges) is **not a code problem at all**: both its back-edges are
> `langres.core`'s public facade re-exporting `Resolver` and the method registry —
> which this repo's `CLAUDE.md` **mandates**. The plan conflates `langres.core`
> (a public facade namespace, by documented and test-enforced policy) with `core`
> (a proposed bottom contracts layer). Those cannot be the same package, and no
> amount of file-moving reconciles them.
>
> Underneath that sits a structural fact the plan never states: **a file move cannot
> delete a module-level import edge.** Every current module cycle survives the split
> verbatim. The split only decides whether a cycle is *contained* in one package
> (invisible to a layers contract) or *spread across several* (a contract failure).
> The `[10]` model-layer SCC — W4's footprint — is spread across **6 packages** by
> this mapping, making it strictly *worse* for contract enforcement than the
> intra-`core` tangle it is today.
>
> Separately and independently of the graph: **Stream B is not a safe file move.**
> I built the wheel both ways. Renaming `data/` → `datasets/` silently grows the
> wheel from **2.2 MB to 16.1 MB**, newly shipping **13.9 MB (87% of the wheel) of
> third-party CSVs langres has no licence to redistribute** — with a **successful
> build and zero warnings**.

---

## 0. What I measured, and how to reproduce it

Everything below comes from `tools/import_graph.py`, which is gated by
`test_edges_match_grimp_exactly` (set equality against grimp — the engine
`import-linter` actually runs). **No import greps, no hand-rolled AST scans.**

```bash
uv sync --all-extras

# Baseline gates (the task named these two)
uv run pytest tests/test_import_tangle.py tests/test_import_budget.py -q
#   -> 39 passed, 1 skipped
#   (the skip is "scikit-learn is installed ([trained] extra present)" -- unrelated)

# The TOOL's OWN gate lives in a THIRD file (see finding F0 below)
uv run pytest tests/test_import_graph.py -q          # -> 46 passed

# Current graph
uv run python tools/import_graph.py cycles
uv run python tools/import_graph.py kinds

# The counterfactual, both views + every cross-package edge
uv run python tools/import_graph.py counterfactual \
    --mapping tools/refactor_target_packages.json
uv run python tools/import_graph.py counterfactual \
    --mapping tools/refactor_target_packages.json --show-edges 200
```

Four questions the CLI does not expose are answered by driving the tool's **own
API** (`build_graph`, `PackageMapping`, `counterfactual`) — still the gated
instrument, never a grep. `tmp/` is gitignored, so those scripts cannot be
committed and would die with the worktree; **they are reproduced inline in
Appendix A** so this report is self-contained and re-runnable.

The B6 wheel experiment (§5) is reproduced inline in **Appendix B**.

**Every number is reported in both views**, because conflating them is the
mechanism of a previous error on this plan:

- **all-edges** — what grimp/`import-linter` see: includes function-local and
  `TYPE_CHECKING` imports.
- **toplevel** — what the runtime actually executes on `import langres`.

---

## F0. Findings that contradict the framing — read these first

The brief asked me to report anything that contradicts it. Four things do.

### F0.1 The named gate is not in the files I was told to run — **process gap**

The brief says the tool "is gated by `test_edges_match_grimp_exactly`" and then
instructs: run `tests/test_import_tangle.py tests/test_import_budget.py` to confirm
the baseline. **That gate is in neither file.** It lives in
`tests/test_import_graph.py:253`. Running only the two named files validates the
*ratchet* but leaves the *measuring instrument itself* unverified.

I ran it separately: **46 passed**, grimp agreement holds. The measurements below
stand — but the prescribed baseline command does not, by itself, establish that
they can.

### F0.2 `architectures ↔ core` is **30 + 2**, not 26 + 2 — the framing's number is stale

| | framing | measured (all-edges) | measured (toplevel) |
|---|---|---|---|
| `architectures ↔ core` | 26 + 2 | **30 + 2** | **28 + 2** |

Four of the delta are B1's `spend_cap`, which did not exist when 26 was counted:

```
langres.core.spend_cap:52 [toplevel]      -> langres.core.inspection
langres.core.spend_cap:53 [toplevel]      -> langres.core.matcher
langres.core.spend_cap:59 [type-checking] -> langres.core.models
langres.core.spend_cap:54 [toplevel]      -> langres.core.spend
```

This is the same failure mode as the plan's error #9 (a fan-out number gone stale
under its own change). It does not change the conclusion — it *reinforces* it: the
number moves every time anyone touches the tree, which is an argument against
planning a 148-file move around a number measured once.

### F0.3 The mapping file is **honest and current** — the framing's premise is wrong

The brief says the mapping "was written against the pre-W1 tree" and suspects
homeless files. **It was not, and there are none.** The file has been actively
maintained: it documents the W1 comparator symbol-split, the W-1 `_exports`
fragments (#169/#170), `_version` → `core` (#176), and B1's `spend`/`spend_cap`.

Measured (Appendix A.1), **both** directions:

| check | result |
|---|---|
| real modules discovered | **152** |
| unmapped real modules (homeless files) | **0** |
| stale `exact` entries (map a module that no longer exists) | **0** |
| stale `prefix` entries (match no real module) | **0** |

`_version.py` is already mapped — to `core`, deliberately, with a *measured*
justification in the file's comment (mapping it to `root` re-creates an
`architectures ↔ root` knot).

**One real gap, though:** the gate
`test_shipped_refactor_mapping_covers_every_module` only enforces direction 1
(every module has a home). **Direction 2 — an entry naming a module that no longer
exists — is ungated.** It is clean today; nothing keeps it clean. That is a
one-line test if W2 ever proceeds.

### F0.4 "14" now *coincidentally* matches — do not read this as vindication

W0 established that the plan's "14 cross-package cycles" matched no definition of
cycle. On today's graph the all-edges **mutual-knot count is exactly 14**. This is
a coincidence: the graph has changed substantially (W1/#175/#176/#177/B1) and the
plan's "14" was never a knot count. The matching integer is numerology, not
evidence. The *cycle* count under the same mapping is **1** (an SCC of 13
packages).

---

## 1. Q1 — Re-derive the stream partition on the current graph

Plan's streams: **A** = metrics+report+profile+benchmarks · **B** =
datasets+training+curation · **C** = tracking+optimize.

Measured by condensing modules straight onto streams (Appendix A.4), with all
other packages collapsed to `rest`:

| edge | all-edges | toplevel |
|---|---|---|
| B → A | **13** | 11 |
| A → B | **4** | 1 |
| C → A | 3 | 1 |
| C → B | 1 | 0 |
| A → C, B → C | 0 | 0 |

**The plan's A↔B claim reproduces.** A↔B is still a **mutual pair** in both views
(plan said B→A 12 / A→B 3; now 13 / 4). "Sequence B after A" remains incoherent —
this part of the plan's reasoning survived W1 intact.

The four A→B edges (the minority direction, i.e. the ones to break):

```
langres.core.fit_report:31                [toplevel]      -> langres.core.harvest
langres.data.data_profile.builders:150    [function-local]-> langres.data.registry
langres.data.data_profile.embedding_source:379 [function-local] -> langres.core.anchor_store
langres.eval:49                           [type-checking] -> langres.data.registry
```

**C is a pure consumer**: it imports A and B, nothing imports it. C has no mutual
pair with either. So C *could* be sequenced independently — **except**:

**The decisive result: all three streams plus `rest` form a single SCC**
(`[['A','B','C','rest']]`, in *both* views). No stream is independent, because each
is tangled with `rest` (= `core`/`components`/`architectures`). **The stream
partition does not decompose the work.** Re-deriving it on the real shape does not
rescue it; it confirms there is nothing to rescue. The independence the plan needed
does not exist at stream granularity.

---

## 2. Q2 — Counterfactual the split exactly

```
uv run python tools/import_graph.py counterfactual --mapping tools/refactor_target_packages.json
```

| | ALL edges | TOPLEVEL |
|---|---|---|
| cross-package cycles (SCCs of the package graph) | **1** | **1** |
| largest SCC | **13 packages** | **9 packages** |
| mutual package pairs | **14** | **8** |

The all-edges SCC is:
`architectures ↔ benchmarks ↔ clients ↔ components ↔ core ↔ curation ↔ datasets ↔ metrics ↔ plotting ↔ profile ↔ report ↔ tracking ↔ training`

**That is 13 of the 15 packages in one cycle.** The 12-package split, applied
exactly as mapped, does not produce a layered architecture. It produces one tangle
with more names.

### 2.1 The structural fact the plan never states

**A pure file-move cannot delete a module-level import edge.** It only relabels
which package each module belongs to. Therefore *every* current module-level SCC
survives the split byte-for-byte. The only thing the mapping decides is whether an
SCC lands **inside one package** (invisible to a layers contract) or **spans
several** (a contract failure).

Measured (Appendix A.2):

| current SCC | fate under the mapping |
|---|---|
| **[10]** model layer | **SPANS 6 PACKAGES** — architectures(4), components(2), clients(1), curation(1), benchmarks(1), training(1) |
| **[9]** `data.data_profile.*` | **CONTAINED** in `profile` |
| **[3]** analysis/reports/plotting | **SPANS 3** — metrics(1), report(1), plotting(1) |
| **[2]** runs/trackers | **CONTAINED** in `tracking` |

So the split **converts two contained tangles into cross-package cycles** and
contains the other two. That is the whole graph-theoretic content of W2.

### 2.2 A second-order finding: the runtime cycles are *created by the split*

The toplevel module graph has **no SCCs at all** (`SCC sizes, TOPLEVEL only: []` —
the runtime is a DAG, thanks to #176). Yet the counterfactual reports a **9-package
toplevel cycle**.

Condensing a DAG onto packages *manufactures* cycles that do not exist in the code:
`a1 → b1` and `b2 → a2` is acyclic at module level but a cycle at package level.
**Every runtime cross-package cycle in this counterfactual is therefore
self-inflicted by the partition choice, not present in the code.** Since the
toplevel module graph is a DAG, a partition with zero toplevel cross-package cycles
provably exists. The proposed one is simply not it.

### 2.3 Every knot by name, with edge counts and verdict

**Verdict key** — *file-level*: a move (or a mapping fix) genuinely drops the edge ·
*code-level*: a move only **relocates** it; a code change is required.

| # | knot (all-edges) | edges | toplevel? | verdict |
|---|---|---|---|---|
| 1 | `architectures ↔ benchmarks` | 1 + 5 | ✅ 1+2 | **code-level** — `methods` ↔ `core.benchmark` mutual symbol dep |
| 2 | `architectures ↔ core` | **30 + 2** | ✅ 28+2 | **mapping artifact** — both back-edges are `core/_exports/*` |
| 3 | `architectures ↔ curation` | 3 + 3 | — | **code-level** — `resolver` ↔ `anchor_store` |
| 4 | `benchmarks ↔ datasets` | 1 + 7 | — | back-edge is 1 `TYPE_CHECKING` import; **not a runtime knot** |
| 5 | `benchmarks ↔ profile` | 1 + 1 | — | both `TYPE_CHECKING`; **not a runtime knot** |
| 6 | `benchmarks ↔ report` | 2 + 1 | ✅ 1+1 | **code-level** — `eval_report` → `benchmark`, `testing` → `reports` |
| 7 | `clients ↔ tracking` | 1 + 4 | — | **mapping artifact** — carving `clients.tracking` out of its parent `__init__` |
| 8 | `core ↔ curation` | 3 + 9 | ✅ 3+8 | **mapping artifact** — all 3 back-edges are `core/_exports/*` |
| 9 | `core ↔ metrics` | 2 + 3 | — | **code-level** — ABC method *bodies* |
| 10 | `core ↔ report` | 4 + 2 | ✅ 4+1 | **mixed** — 1 facade + **3 code-level (return types)** |
| 11 | `core ↔ tracking` | 4 + 4 | ✅ 3+4 | **mapping artifact** — all back-edges are `core/_exports/*` |
| 12 | `core ↔ training` | 3 + 6 | ✅ 2+5 | **mapping artifact** — all back-edges are `core/_exports/_training` |
| 13 | `metrics ↔ report` | 1 + 6 | ✅ 1+3 | **code-level** — `analysis` ↔ `reports` |
| 14 | `plotting ↔ report` | 1 + 4 | — | **code-level** — `reports` body → `plotting.blockers` |

**Tally: 5 of 14 knots (2, 7, 8, 11, 12) are packaging artifacts, not code
problems. 8 are code-level — a move relocates them. 1 is mixed.**

### 2.4 The code-level knots I verified by reading the code

The brief warned this distinction is decisive and non-obvious, and that more cases
exist. They do. I read every one rather than trusting the edge kind.

**(a) `core → report` — held by return types (the W1 finding reproduces).** Not the
`abstractmethod`; the *signatures*:

```
core/blocker.py:192      ) -> CandidateInspectionReport:
core/blocker.py:227      ) -> BlockerEvaluationReport:
core/clusterer.py:107    ) -> ClusterInspectionReport:
core/inspection.py:40    ) -> ScoreInspectionReport:
```

And worse than annotations — `clusterer.py:135` and `:216` **construct**
`ClusterInspectionReport(...)` in the method body. Moving `reports.py` into a
`report` package cannot drop this edge; it makes it cross-package. Removing it
requires deleting `evaluate()`/`inspect()` from the contract ABCs — a public API
break, not a move.

**(b) `core → metrics` — held by ABC method bodies.** Function-local imports inside
concrete methods on the contract classes:

```python
# core/blocker.py:274
from langres.core.analysis import evaluate_blocker_detailed
return evaluate_blocker_detailed(candidates, gold_clusters, k_values)
# core/clusterer.py:341
from langres.core.metrics import evaluate_clustering
return evaluate_clustering(predicted_clusters, gold_clusters)
```

Exactly the shape of the W1 `Comparator.from_schema` trap: the dependency is in the
**body**, so splitting the file changes nothing.

**(c) `architectures ↔ benchmarks` — a knowingly-managed code cycle.** The source
already documents it (`core/benchmark.py:430-435`):

> *"The two cannot be a single runtime import here without closing the
> `core.benchmark -> methods -> core.benchmark` cycle, so the combined contract is
> expressed for the type checker only."*

`methods.py:64` imports `CostTrack, _cost_track` **from** `core.benchmark`, while
`core.benchmark:59-60` imports `presets`/`resolver` and `:871` imports
`make_resolver_factory`. Mutual, symbol-level, and it survives any move. The fix is
the `_version.py` precedent: extract `CostTrack`/`_cost_track` to a leaf.

---

## 3. Q3 — Is `core` contracts-only after the split, or does the tangle just move?

**Both, and the interesting half is neither of the options as posed.**

**The `[10]` component does not dissolve — and W4 lifting `presets`/`resolver`/
`method_registry` into `architectures/` is not what would dissolve it.** No member
of `[10]` maps to `core`, so the model-layer tangle *does* leave `core`. But it does
not go away: it lands **spread across 6 packages** (architectures, components,
clients, curation, benchmarks, training) and becomes a cross-package cycle — a
`layers` violation where today it is an invisible intra-`core` SCC. A file move
cannot dissolve it because a file move cannot delete an edge. **Only a code change
can, and that code change is W4.**

**And it does *not* survive as `architectures ↔ core`.** That knot — the largest in
the whole counterfactual at 30 + 2 — has exactly two backward edges, and both are
facade fragments:

```
langres.core._exports._methods:6   [toplevel] -> langres.core.method_registry
langres.core._exports._resolver:14 [toplevel] -> langres.core.resolver
```

Nothing in `core`'s *contracts* depends on `architectures`. The dependency is the
**public facade** re-exporting `Resolver` and the method registry.

### 3.1 The real architectural blocker the plan does not name

`langres.core` is simultaneously:

1. a **public facade** that re-exports `Resolver`, the method registry, `harvest`,
   `review`, `runs`, `trackers`, `calibration`, …, and
2. the proposed **contracts-only bottom layer**.

**These are incompatible**, and — importantly — **(1) is not an accident anyone can
quietly revoke.** It is documented policy in this branch's `CLAUDE.md`:

> *"**`langres.core` re-exports contracts, not implementations.** It carries the
> data models, the `Blocker`/`Comparator`/`Matcher`/`Clusterer` base types, the
> opt-in capability Protocols …, **the `Resolver` + registry, the method registry
> and the training/tracking primitives** — the things a pipeline is *written
> against*."*

Verified in the source: `core/_exports/_resolver.py` eagerly imports `Resolver`
(plus the registry + serialization seam) and `core/_exports/_methods.py` eagerly
imports the six `method_registry` symbols. Those two files **are** the 2 back-edges
of the 30 + 2 knot. Nothing in `core`'s *contracts* depends on `architectures` — the
**facade** does, exactly as instructed.

**This is already a half-solved problem, which is the useful part.** A
"facade-emptying wave" has run: `tests/test_import_budget.py::TestCoreLazyGetattr::
test_implementations_are_not_re_exported` pins 17 symbols that must **not** resolve
on `langres.core` (`AllPairsBlocker`, `LLMMatcher`, `StringComparator`,
`AnchorStore`, `FAISSIndex`, …), with the stated rationale that re-exporting an
implementation *"puts `langres.core` back above the components it sits beneath and
re-knots the import graph."* That is precisely this report's argument — already
accepted, already ratcheted.

But the wave stopped at *implementations*. `Resolver`, the method registry, and the
training/tracking primitives stayed on the facade **by design**, because CLAUDE.md
classifies them as things pipelines are written against. So the residual knots are
not a bug in the mapping — **the mapping is faithfully encoding shipped policy.**

**The plan's error is a naming conflation, not a layout problem:** "`langres.core`
the public namespace" and "`core` the bottom layer" are two different objects that
the plan treats as one. No file move can reconcile them. The decision the owner
actually faces is:

- **(i)** `langres.core` stays a public facade (CLAUDE.md unchanged) and the
  *contracts* become the bottom layer under a different package identity — then the
  facade is a **top** layer and the knots vanish; or
- **(ii)** `langres.core` becomes contracts-only — which means **dropping `Resolver`
  and the method registry from `langres.core`**, a public API break and a CLAUDE.md
  rewrite that the plan never scoped.

That decision — not a 148-file move — is what unblocks the layering. It currently
has no owner and is being answered by a comment in a JSON file.

### 3.2 Measured: the facade decision, not the code, is the dominant term

I re-ran the counterfactual with the facade re-homed to a top layer — i.e. option
**(i)** above, modelled as a mapping change (Appendix A.3). Note this is *not* a
free lunch: it is a decision about what `langres.core` **is**, and it implies the
contracts layer gets a different package identity. It costs no source edits; it
costs an architecture ruling.

| variant | ALL: SCCs / largest / knots | TOPLEVEL: SCCs / largest / knots |
|---|---|---|
| **V0** shipped mapping | 1 / **13** / **14** | 1 / **9** / **8** |
| **V1** `core/__init__` + `core/_exports/*` → `facade` (top layer) | 1 / 13 / **10** | 1 / 9 / **4** |
| **V2** = V1 + `core.reports` → `core` | **2** / **10** / **8** | **1** / **4** / **1** |

**V1 deletes 4 knots — including `architectures ↔ core` entirely — without touching
a line of code.** V2 takes the runtime view to a single remaining knot
(`architectures ↔ benchmarks`, i.e. the `CostTrack` cycle) and a 4-package SCC.

**But read V2 honestly — it is a trade, not a win.** Putting `reports` into `core`
is precisely *not* "core is contracts-only", and the tangle **relocates rather than
drops**: `metrics ↔ report` and `plotting ↔ report` simply become `core ↔ metrics`
(5 + 4) and `core ↔ plotting` (4 + 1), and the `[3]` SCC persists as
`core ↔ metrics ↔ plotting`. This is the *same relocation illusion* the brief warned
about, reproduced under my own experiment. I report it as a measurement of where the
tangle lives, **not as a recommendation**.

Note also V2's residual all-edges SCC is **[10]** — exactly the model-layer
component. It is invariant under every mapping I tried, which is the point: **it is
code, and only W4 touches it.**

---

## 4. Q4 — Is the mapping file still honest?

**Yes — fully, in both directions.** See **F0.3**. 152 modules, 0 unmapped, 0 stale
`exact`, 0 stale `prefix`. `_version.py` is mapped, deliberately and with a measured
rationale.

The only defect is a **missing gate**: staleness in the reverse direction (an entry
naming a dead module) is unenforced. Cheap to add; only worth adding if W2 lives.

---

## 5. Q5 — B6, the licensing landmine: **confirmed, and worse than described**

**Verified, empirically, by building the wheel both ways — not by reading the
globs.**

- **The excludes exist**: `pyproject.toml:144-159`, **14** patterns.
- **They are path-literal**: every one is rooted at the literal prefix
  `src/langres/data/datasets/…`.
- **They are hand-maintained and unowned by tooling**: the table's own first line
  says *"Hand-edited: uv does not manage `[tool.hatch.*]` tables."* Nothing
  rewrites them when a directory is renamed.

Stream B moves `data/` → `datasets/` (`"langres.data": "datasets"` in the mapping).
The literal paths then match nothing.

**What the wheel would ship post-move — measured, not predicted:**

| | control (today) | treatment (`data/` → `datasets/`, excludes untouched) |
|---|---|---|
| build result | success | **success — zero warnings** |
| files | 181 | 211 |
| **uncompressed size** | **2.2 MB** | **16.1 MB** |
| **CSV files shipped** | **13** | **43** |
| third-party CSVs newly shipped | 0 | **13.9 MB = 87% of the wheel** |

Single largest newly-shipped file: `datasets/datasets/dblp_scholar/tableB.csv`,
**8.06 MB**. The control ships only the 13 licensed/synthetic CSVs (tiny_fixture,
febrl_person BSD-3, fodors_zagat, and the `peeters_sampled_test.csv` id/label
triples) — exactly as the comment claims. So the excludes are load-bearing *today*
and they are **the only thing** standing between the project and a repeat of the
91%-third-party-data wheel it already shipped once.

**The failure is silent.** Hatchling does not warn on an exclude pattern that
matches nothing. The build succeeds, the wheel is publishable, and the only visible
symptom is a size jump nobody is gating on. This is a licence-compliance
regression — reputationally and legally the most expensive item in this report, and
it is **entirely independent of whether the graph argument holds**.

**Mitigation is cheap and should land regardless of the W2 decision:** a test that
asserts the built wheel contains no CSV outside the licensed allowlist. That gate
makes the rename safe *and* protects the current tree, which today relies on 14
hand-edited literals staying in sync with a directory name by luck.

---

## 6. Recommendation

The measurements do not support running W2 as written. They point at a code-first
sequence. Four live options, with real trade-offs from this graph.

### Option A — Execute W2 as specified (12 packages) — **not recommended**

- **Delivers:** the target directory layout.
- **Measured cost:** 152 modules relabelled, **148 test files** touched.
- **Measured benefit:** *negative* for the stated goal. You end with **1 SCC of 13
  packages / 14 knots** (all-edges) and a **9-package toplevel cycle** that
  **does not exist in the code today** (§2.2). Two contained tangles (`[10]`,
  `[3]`) become cross-package. The largest knot is a facade artifact (§3.2), and
  Stream B silently ships 13.9 MB of unlicensed data (§5).
- **Reversal cost:** very high — a 148-file move is not casually undone, and every
  in-flight branch conflicts.

### Option B — W4-only: fix the code, leave the layout alone — **recommended**

- **Rationale:** the `[10]` model-layer SCC is the actual architecture problem, it
  is **invariant under every mapping tested**, and **only a code change dissolves
  it**. It is W4's footprint already.
- **Blast radius:** ~10 modules (`presets`, `resolver`, `method_registry`,
  `benchmark`, `methods`, `llm_judge`, `cascade`, `finetune`, `anchor_store`,
  `clients.openrouter`) — vs 152 relabelled + 148 test files for Option A.
- **Delivers:** the architecture win the split was supposed to buy, measured on the
  existing ratchet (`tests/test_import_tangle.py`), with no test-file churn.
- **Precedent that this works:** `_version.py` (#176) took a 12-module runtime cycle
  to **0** by extracting one leaf. The `CostTrack`/`_cost_track` extraction (§2.4c)
  is the identical move and would clear the last toplevel knot in V2.
- **Reversal cost:** low — small, reviewable, per-module commits.
- **Cost:** does not deliver the directory layout. If the layout is a goal in its
  own right, this defers it — but on these numbers the layout was never what
  delivered the architecture.

### Option C — Subset: split only what the tool shows a move genuinely fixes

- **Honest candidates are few.** `profile` (the `[9]` SCC is **contained**) and
  `tracking` (`[2]` **contained**) can be carved out without spreading a cycle.
- **Measured caveat:** carving `tracking` out immediately creates the
  `clients ↔ tracking` artifact (knot #7) via `clients/__init__`'s
  `TYPE_CHECKING` reference to `clients.tracking`. So even the "safe" subset is
  not free.
- **Explicitly excluded from any subset: Stream B** until the wheel gate exists (§5).
- **Value:** modest and mostly cosmetic — neither package is where the pain is.
- **Reversal cost:** low-ish (2 packages), but it spends churn on the tangles that
  were *already* fine.

### Option D — Defer W2, keep the ratchet, record the measurement

- Keep `tests/test_import_tangle.py` as the ratchet (it is working — it is why the
  runtime is at 0).
- Land the **wheel-contents licence gate** (§5) now, independently — it protects the
  current tree regardless.
- Add the reverse-direction mapping-staleness gate (§4) only if W2 revives.
- **Cost:** the plan's target layout stays unbuilt, and the mapping file needs
  periodic re-measurement to stay honest (it has been maintained well so far).

### My recommendation

**B, then re-measure; with D's wheel gate landed immediately and independently.**

The reasoning is one measured sentence: **the split cannot fix a single code-level
cycle** (a move cannot delete an edge), **8 of 14 knots are code-level**, **5 more
are the unresolved "is `langres.core` a facade or a layer?" question wearing a
disguise**, and the one component that matters — `[10]` — is W4's footprint and is
invariant under every partition I tested. Do W4, take the `[10]` SCC down in code,
then re-run this exact counterfactual. The partition question gets *easier* after
W4, and it may well dissolve: with `[10]` gone and the facade question settled, the
residual toplevel tangle measured here is a **single** knot.

Note that Option B is also the option the repo is *already* executing. The
facade-emptying wave (§3.1) and `_version.py` (#176) were both code-first,
symbol-level fixes, and both worked — #176 took a 12-module runtime cycle to zero.
W2 is the one wave that proposes to buy layering with moves instead. On this graph,
moves do not buy layering.

The two decisions that are **not** contingent on any of this, and should land on
their own merits:

1. **The wheel licence gate** (§5) — a live, silent, 13.9 MB compliance regression
   sitting behind 14 hand-edited path literals.
2. **Ownership of the facade question** (§3.1) — "is `langres.core` a public facade
   or a contracts-only layer?" is an architecture decision with a public-API blast
   radius, not a transcription detail. Today CLAUDE.md says *facade* and the plan
   assumes *layer*; the contradiction is currently being resolved by a comment in a
   JSON file. It drives the largest knot in the entire report, and it is answerable
   in an afternoon — **before** anyone moves 148 test files.

---

## Appendix A — the analysis scripts (inline; `tmp/` is gitignored)

Each drives `tools/import_graph.py`'s gated API. Run from the repo root with
`uv run python -` or save under `tmp/`.

### A.1 Mapping staleness, both directions (§4, F0.3)

```python
import sys
from pathlib import Path
from collections import Counter
sys.path.insert(0, str(Path("tools").resolve()))
from import_graph import DEFAULT_PACKAGE_ROOT, PackageMapping, build_graph

graph = build_graph(DEFAULT_PACKAGE_ROOT)
mapping = PackageMapping.from_json(Path("tools/refactor_target_packages.json"))
modules = set(graph.modules)
print(f"real modules: {len(modules)}")
print("unmapped (homeless):", mapping.unmapped(modules) or "(none)")
print("stale exact:", sorted(k for k in mapping.exact if k not in modules) or "(none)")
print("stale prefix:", sorted(
    p for p in mapping.prefix
    if not any(m == p or m.startswith(f"{p}.") for m in modules)) or "(none)")
print(Counter(mapping.target(m) for m in modules))
```

### A.2 Where each current module SCC lands under the mapping (§2.1)

```python
import sys
from pathlib import Path
from collections import Counter
sys.path.insert(0, str(Path("tools").resolve()))
from import_graph import DEFAULT_PACKAGE_ROOT, ImportKind, PackageMapping, build_graph

graph = build_graph(DEFAULT_PACKAGE_ROOT)
mapping = PackageMapping.from_json(Path("tools/refactor_target_packages.json"))
for label, kinds in (("ALL edges", None), ("TOPLEVEL only", (ImportKind.TOPLEVEL,))):
    print(f"\n=== {label} ===")
    for scc in graph.sccs(kinds):
        pkgs = Counter(mapping.target(m) for m in scc)
        print(f"  SCC[{len(scc)}] ->",
              "CONTAINED" if len(pkgs) == 1 else f"SPANS {len(pkgs)} PACKAGES", dict(pkgs))
```

### A.3 Mapping variants: how much of the tangle is the facade decision (§3.2)

```python
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path("tools").resolve()))
from import_graph import (DEFAULT_PACKAGE_ROOT, ImportKind, PackageMapping,
                          build_graph, counterfactual)

graph = build_graph(DEFAULT_PACKAGE_ROOT)
raw = json.loads(Path("tools/refactor_target_packages.json").read_text())
e = {k: v for k, v in raw["exact"].items() if not k.startswith("_")}
p = {k: v for k, v in raw["prefix"].items() if not k.startswith("_")}

def report(name, exact, prefix):
    m = PackageMapping(exact=exact, prefix=prefix)
    assert m.unmapped(graph.modules) == []
    print(f"\n=== {name} ===")
    for label, kinds in (("ALL edges", None), ("TOPLEVEL", (ImportKind.TOPLEVEL,))):
        r = counterfactual(graph, m, kinds)
        print(f"  {label:9} | SCCs {len(r.cycles)} | largest "
              f"{max((len(c) for c in r.cycles), default=0)} | knots {len(r.mutual_pairs)}")
        for a, b in r.mutual_pairs:
            print(f"      {a} <-> {b} ({len(r.cross_edges[(a,b)])} + {len(r.cross_edges[(b,a)])})")

report("V0 shipped mapping", dict(e), dict(p))
v1e, v1p = dict(e), dict(p)
v1p["langres.core._exports"] = "facade"; v1e["langres.core"] = "facade"
report("V1 core facade -> top layer", v1e, v1p)
v2e, v2p = dict(v1e), dict(v1p)
v2e["langres.core.reports"] = "core"
report("V2 = V1 + core.reports -> core", v2e, v2p)
```

### A.4 Stream partition (§1)

```python
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path("tools").resolve()))
from import_graph import (DEFAULT_PACKAGE_ROOT, ImportKind, PackageMapping,
                          build_graph, counterfactual)

graph = build_graph(DEFAULT_PACKAGE_ROOT)
raw = json.loads(Path("tools/refactor_target_packages.json").read_text())
pkg = PackageMapping(
    exact={k: v for k, v in raw["exact"].items() if not k.startswith("_")},
    prefix={k: v for k, v in raw["prefix"].items() if not k.startswith("_")})
STREAM = {"metrics": "A", "report": "A", "profile": "A", "benchmarks": "A",
          "datasets": "B", "training": "B", "curation": "B",
          "tracking": "C", "optimize": "C"}
m = PackageMapping(exact={mod: STREAM.get(pkg.target(mod), "rest") for mod in graph.modules},
                   prefix={})
for label, kinds in (("ALL edges", None), ("TOPLEVEL", (ImportKind.TOPLEVEL,))):
    r = counterfactual(graph, m, kinds)
    print(f"\n=== {label} ===  SCCs: {r.cycles}")
    for (a, b), edges in sorted(r.cross_edges.items()):
        if a in "ABC" and b in "ABC":
            print(f"   {a} -> {b}: {len(edges)}")
```

---

## Appendix B — the B6 wheel experiment (§5)

Control vs treatment, run from the repo root. The treatment is the *only* change
Stream B makes to this file's world: rename the directory, touch nothing else.

```bash
# CONTROL -- today's tree
mkdir -p tmp/b6/control
rsync -a --exclude='.git' --exclude='.venv' --exclude='tmp' \
         --exclude='htmlcov' --exclude='dist*' ./ tmp/b6/control/
(cd tmp/b6/control && uv build --wheel --out-dir dist_ctl)

# TREATMENT -- Stream B's move: data/ -> datasets/, pyproject.toml untouched
mkdir -p tmp/b6/treat
rsync -a --exclude='.git' --exclude='.venv' --exclude='tmp' \
         --exclude='htmlcov' --exclude='dist*' --exclude='__pycache__' ./ tmp/b6/treat/
mv tmp/b6/treat/src/langres/data tmp/b6/treat/src/langres/datasets
(cd tmp/b6/treat && uv build --wheel --out-dir dist_tr)   # succeeds, ZERO warnings
```

```python
# inspect either wheel
import zipfile, glob
z = zipfile.ZipFile(glob.glob('dist_*/*.whl')[0])
csvs = [n for n in z.namelist() if n.endswith('.csv')]
tot = sum(z.getinfo(n).file_size for n in z.namelist())
csv = sum(z.getinfo(n).file_size for n in csvs)
print(f"{len(z.namelist())} files, {tot/1e6:.1f} MB uncompressed")
print(f"{len(csvs)} CSVs, {csv/1e6:.1f} MB ({100*csv/tot:.0f}% of wheel)")
```

Observed:

| | control | treatment |
|---|---|---|
| build | success | **success, zero warnings** |
| files / size | 181 / **2.2 MB** | 211 / **16.1 MB** |
| CSVs | **13** (all licensed/synthetic) | **43** |
| CSV bytes | 0.2 MB | **14.1 MB (88%)** |

Largest newly-shipped: `langres/datasets/datasets/dblp_scholar/tableB.csv` — 8.06 MB.

**Suggested gate (independent of the W2 decision):** assert the built wheel's CSV
members are a subset of the licensed allowlist (`tiny_fixture/*`, `febrl_person/*`,
`fodors_zagat/*`, `*/peeters_sampled_test.csv`). That is a property of the shipped
artifact rather than of 14 hand-edited path literals, so it survives any rename.
