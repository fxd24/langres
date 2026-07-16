# Import-Graph Baseline — re-deriving the architecture refactor's numbers

> **Status:** Results / audit (2026-07-16). Measured on `main` @ `4e5b736` with
> `tools/import_graph.py`, committed in the same change. Every number below carries the
> exact command that reproduces it.
>
> **Why this exists:** the refactor plan's wave *order* rests on one quantitative claim —
> *"naive file-move split → 14 cross-package cycles; contract-first → 6"* — produced by an
> uncommitted script. Four evidence errors had already been found in the plan's chain, each
> a text-search inference presented as a measured fact. This doc makes the measurement
> reproducible and re-derives the claims. **The headline result is negative**, which is the
> point: it is cheaper to learn now than after sequencing waves on a fiction.
>
> **Reads with:** `tools/import_graph.py` (the instrument), `tests/test_import_graph.py`
> (its gate, incl. the grimp cross-validation), `tests/test_import_budget.py` (the
> lazy-import contract §2 turns out to constrain).

---

## 0. Verdict table

| # | Claim | Verdict |
|---|---|---|
| 1 | `core.models` 50 in / 2 out is the floor; `core.matcher` 20/3; `core.presets` 3/16; `core.resolver` 8/33 is the ceiling | **CORRECTED** — all four wrong; `core.registry` (23/**0**) is the real floor, `langres.core` (43 out) the real ceiling. `resolver.py:44 import langres` confirmed |
| 2 | `toplevel=386, function-local=63, type-checking=56`; 99 lazy-only edges; SCCs `[5]` / `[29,3,2]` | **CORRECTED (numbers) / CONFIRMED (conclusion)** — true counts are 391/63/59, 102 lazy-only. **SCCs are byte-identical.** The load-bearing lazy-import finding stands |
| 3 | `core/benchmark.py` has 14 importers across four streams | **CORRECTED** — 15. Every named line number is exact; the 15th (`core/__init__.py:115`) was *hidden by a bug in the measuring script* |
| 4 | Stream B→A = 12, A→B = 3 → bidirectional | **CONFIRMED (conclusion) / CORRECTED (count)** — B→A = 13, A→B = 3. All three named A→B edges exact. **Ordering B after A cannot fix it** |
| 5 | naive file-move → **14** cross-package cycles; contract-first → **6** | **NOT REPRODUCIBLE** — no definition of "cycle" yields 14 (see §5). The "6" has **no reproducible definition at all**: the contract-first split was never specified at symbol granularity |

**A fifth evidence error, in the instrument itself.** The rescued `tmp/w0-graph/graph.py`
mis-resolves `from PKG import SUBMODULE`, crediting the edge to the package rather than the
submodule. It **invents 15 phantom edges and misses 23 real ones**. Verified against grimp
(§1.1). Claim 3's "14" is a direct casualty. Claim 1's numbers match *neither* the buggy
script nor the corrected tool, so they came from neither.

---

## 1. The instrument

```bash
uv run python tools/import_graph.py --help
```

`tools/import_graph.py` parses source with `ast` — it never executes an import, so heavy
extras (torch, litellm, faiss) are never loaded. Subcommands: `fan`, `kinds`, `importers`,
`cycles`, `counterfactual`.

### 1.1 Why not just use grimp? And why grimp is nevertheless ground truth

`grimp` is the engine behind `import-linter`. It resolves imports authoritatively — but it
reports **edges, not the *kind* of each import**: it cannot say whether an import is
module-level, function-local, or `TYPE_CHECKING`-only. That distinction is the entire
subject of §2, so the AST tool has to exist.

What the AST tool must *not* do is disagree with grimp about the edges themselves. So that
is a test, not a hope:

```
tests/test_import_graph.py::test_edges_match_grimp_exactly
```

```
my edges   : 493
grimp edges: 493
EXACT MATCH: True
```

The rescued scripts fail this. grimp, asked directly:

```
from langres.core import _report_html   # data_profile/hero.py:25
  grimp edge -> langres.core._report_html : True
  grimp edge -> langres.core             : False
```

`graph.py` appends only `node.module` for `ast.ImportFrom` and never inspects `node.names`,
so it books that edge against `langres.core` — the package `__init__` — instead of the
submodule. Measured consequence:

| | ad-hoc `graph.py` | corrected tool |
|---|---|---|
| unique edges | 485 | **493** |
| phantom edges invented | **15** | 0 |
| real edges missed | **23** | 0 |

The 15 phantoms are all *into package `__init__` files* (`data_profile/* → langres.core` ×8,
`data/* → langres.data` ×6, `eval_report → langres.core`). The 23 misses are the real
submodule edges (`data_profile/* → core._report_html|_svg`, `data/* → data._benchmark_utils`,
`langres.core → core.benchmark|metrics|optimizers`).

> **A hypothesis that did not survive.** The expectation — mine and the orchestrator's — was
> that routing edges through `langres.core`'s `__init__` (a member of both SCCs) would
> **manufacture** cycles, so the corrected tangle should be *smaller*. **That is false.** The
> phantom and missing edges compensate, and the SCCs are byte-identical (§2). The bug is
> real and had to be fixed for the tool to be admissible, but it changed **no** SCC
> conclusion. Recorded here because "the fix will make it better" was itself an unverified
> hypothesis, and the discipline that catches four evidence errors has to catch this one too.

---

## 2. Kinds and the two SCC views — the load-bearing finding

```bash
uv run python tools/import_graph.py kinds
```

```
import STATEMENTS by kind (one statement can carry several edges):
        toplevel = 393
  function-local = 78
   type-checking = 59

unique edges by kind:
        toplevel = 391
  function-local = 63
   type-checking = 59

edges that exist ONLY via function-local/TYPE_CHECKING: 102
  (invisible to `import langres` at runtime -- but VISIBLE to grimp/import-linter)

SCC sizes, ALL edges (what import-linter sees) : [29, 3, 2]
SCC sizes, TOPLEVEL only (what runtime sees)  : [5]

=== runtime (toplevel-only) SCCs ===
  ['langres', 'langres.core', 'langres.core.presets', 'langres.core.resolver', 'langres.verbs']
```

| | previously circulated | corrected | |
|---|---|---|---|
| toplevel unique edges | 386 | **391** | resolver fix |
| function-local | 63 | 63 | — |
| type-checking | 56 | **59** | resolver fix |
| lazy-only edges | 99 | **102** | resolver fix |
| SCC, all edges | `[29,3,2]` | `[29,3,2]` | **identical** |
| SCC, toplevel only | `[5]` | `[5]` | **identical** |

SCC *membership* is byte-identical too, not just the sizes (checked by set comparison, not
by eye — the failure mode that produced evidence error #4 was `grep -c` similarity read as
byte-identity). The three SCCs import-linter sees:

- **[29]** `langres`, `clients.openrouter`, `core`, `core.anchor_store`, `core.benchmark`, `core.eval_report`, `core.finetune`, `core.judgement_log`, `core.matchers.{cascade,cascade_judge,llm_judge}`, `core.method_registry`, `core.presets`, `core.resolver`, `data.data_profile{,.base,.builders,.corpus_field,.embedding_section,.embedding_source,.failure_mode,.hero,.label_structure,.mining_readiness,.separability}`, `data.registry`, `methods`, `optimize`, `verbs`
- **[3]** `core.analysis`, `core.reports`, `plotting.blockers`
- **[2]** `core.runs`, `core.trackers`

### 2.1 What this decides about W0's import-linter contract

**102 of 493 edges (21%) exist only via function-local or `TYPE_CHECKING` imports.** They do
not run on `import langres`. import-linter counts them anyway:

- `exclude_type_checking_imports` defaults to **False** — and even set True, it only covers
  the 59 `TYPE_CHECKING` edges.
- There is **no option to exclude function-local imports**. grimp has no notion of them.

Meanwhile `tests/test_import_budget.py` *mandates* those lazy imports: they are how
`import langres` stays free of torch/litellm/faiss. The two views differ starkly — a
**5**-module runtime tangle vs. a **29**-module tangle import-linter sees.

> **A "zero waivers" import-linter goal fights the lazy-import architecture head-on.** The
> 24 extra modules in the [29] SCC are not a runtime problem to fix; they are the
> lazy-import contract working as designed. W0 must either scope its contracts to the
> runtime view, or budget waivers for ~102 deliberate edges. This is a configuration
> decision, and it is forced.

---

## 3. Fan-in / fan-out

```bash
uv run python tools/import_graph.py fan --top 12
```

```
fan-in fan-out  module
    52       2  langres.core.models
    23       0  langres.core.registry
    22       4  langres.core.reports
    21       3  langres.core.matcher
    15       2  langres.core.metrics
    15       7  langres.core.benchmark
    14       1  langres.core.embeddings
    14       3  langres.core.indexes.vector_index
    13       4  langres.core.blockers.all_pairs
    13       8  langres.core.blockers.vector
    11       1  langres.core.runs
    10       2  langres.core.comparator
```

| module | plan claims | actual | verdict |
|---|---|---|---|
| `core.models` | 50 / 2 | **52** / 2 | CORRECTED |
| `core.matcher` | 20 / 3 | **21** / 3 | CORRECTED |
| `core.presets` | 3 / 16 | 3 / **13** | CORRECTED |
| `core.resolver` | 8 / 33 | **7** / **28** | CORRECTED |

**These came from neither script.** The ad-hoc `graph.py` and the corrected tool agree
*exactly* on all four (52/2, 21/3, 3/13, 7/28), so the resolver bug does not explain the
gap. The plan's fan table is independently wrong.

Two structural corrections that matter more than the digits:

- **`core.models` is not the floor. `core.registry` is** — 23 in / **0** out. It is the only
  module in the package with meaningful fan-in and *zero* first-party dependencies, i.e. the
  one module already sitting where "contracts only" wants everything. `core.models` (fan-out
  2) is second.
- **`core.resolver` is not the ceiling. `langres.core` is** — 43 out vs. resolver's 28. The
  fattest dependency in the package is `core/__init__.py`, the re-export facade. §5 shows
  this is not cosmetic.
- `resolver.py:44` really is `import langres`, toplevel — **CONFIRMED**. It is the single
  edge closing the runtime [5] SCC (`architectures → root` in §5).

---

## 4. `core/benchmark.py`'s importers — 15, not 14

```bash
uv run python tools/import_graph.py importers langres.core.benchmark
```

```
--- langres.core.benchmark: fan-in=15 fan-out=7
      langres:70 [type-checking]
      langres.bootstrap.labelers:23 [toplevel]
      langres.clients.openrouter:30 [type-checking], 271 [function-local]
      langres.core:115 [type-checking]
      langres.core.eval_report:38 [toplevel]
      langres.data._deepmatcher_loader:47 [toplevel]
      langres.data.abt_buy:35 [toplevel]
      langres.data.amazon_google:43 [toplevel]
      langres.data.data_profile.builders:63 [type-checking]
      langres.data.er_benchmarks:19 [toplevel], 24 [toplevel]
      langres.data.febrl_person:39 [toplevel]
      langres.data.registry:40 [type-checking]
      langres.eval:30 [type-checking], 152 [function-local]
      langres.methods:64 [toplevel]
      langres.optimize:36 [type-checking]
```

**Every line number the plan names is exact.** Stream A ×2 (`eval_report.py:38`,
`data_profile/builders.py:63`), Stream B ×7 (`labelers.py:23`, `_deepmatcher_loader.py:47`,
`abt_buy.py:35`, `amazon_google.py:43`, `er_benchmarks.py:19,24`, `febrl_person.py:39`,
`registry.py:40`), Stream C ×1 (`optimize.py:36`), W4 ×1 (`methods.py:64`). Credit where due:
this claim was carefully done.

The count is still wrong, and the reason is instructive:

```
ad-hoc (buggy) importer count : 14   <- the plan claims 14
corrected importer count      : 15
importers the bug HID: ['langres.core']
```

The plan's 14 **is** the buggy script's output, reproduced exactly. The hidden 15th importer
is `core/__init__.py:115` — `from langres.core import benchmark` — i.e. **the very package
the refactor wants to reduce to "contracts only" is itself an importer of the 1773-line
harness it wants to evict.** The measurement error concealed precisely the edge that matters
most to the plan's thesis.

Also unlisted: `clients.openrouter:30,271`. So the true spread is **five** call sites, not
four: Stream A ×2, Stream B ×7, Stream C ×1, W4 ×1, root ×3 (`langres:70`, `eval:30,152`,
`core:115`), **plus `clients` ×1**.

---

## 5. Stream A ↔ B — bidirectional, confirmed

Stream assignment per the task's definition (A = profile + metrics/report + benchmark;
B = datasets/curation/training). Reproduce with the snippet in §7.

```
Stream A --> Stream B: 3 edges
    langres.core.fit_report:31 [toplevel] -> langres.core.harvest
    langres.data.data_profile.builders:150 [function-local] -> langres.data.registry
    langres.data.data_profile.embedding_source:379 [function-local] -> langres.core.anchor_store

Stream B --> Stream A: 13 edges
    langres.bootstrap.labelers:23 [toplevel] -> langres.core.benchmark
    langres.bootstrap.report:33 [toplevel] -> langres.core.metrics
    langres.core.harvest:503 [function-local] -> langres.core.metrics
    langres.data._benchmark_utils:28 [toplevel] -> langres.core.metrics
    langres.data._deepmatcher_loader:47 [toplevel] -> langres.core.benchmark
    langres.data.abt_buy:35 [toplevel] -> langres.core.benchmark
    langres.data.amazon_google:43 [toplevel] -> langres.core.benchmark
    langres.data.er_benchmarks:19 [toplevel] -> langres.core.benchmark
    langres.data.er_benchmarks:24 [toplevel] -> langres.core.benchmark
    langres.data.er_benchmarks:35 [toplevel] -> langres.core.metrics
    langres.data.febrl_person:39 [toplevel] -> langres.core.benchmark
    langres.data.fixed_split_pair_benchmark:48 [toplevel] -> langres.core.metrics
    langres.data.registry:40 [type-checking] -> langres.core.benchmark
```

**A→B = 3, exactly the three edges the plan names, at exactly those line numbers —
CONFIRMED.** B→A = **13**, not 12 (this count is sensitive to the stream definition, which
the plan does not fix precisely; the three A→B edges are not).

**The conclusion holds and is the point: the dependency is bidirectional, so "sequence B
after A" does not resolve it.** Whichever stream lands first, the other's edges are already
in the tree. Note the *asymmetry* the raw counts hide: 2 of the 3 A→B edges are
**function-local**, and the third (`fit_report:31 → harvest`) is a single toplevel import.
The A→B direction is thin and lazy; the B→A direction is thick and toplevel. If a wave order
is wanted, that asymmetry — not the 13-vs-3 headline — is the lever, and it points at
breaking three named edges rather than at sequencing.

---

## 6. The 14-vs-6 counterfactual — not reproducible

```bash
uv run python tools/import_graph.py counterfactual --mapping tools/refactor_target_packages.json
```

The mapping (`tools/refactor_target_packages.json`) transcribes the plan's target layout and
is **total** — every one of the 128 modules is assigned, gated by
`test_shipped_refactor_mapping_covers_every_module` ("no wave may discover a homeless file").

```
=== ALL edges (import-linter) ===
  cross-package cycles (SCCs of the package graph): 1
      [15] architectures <-> benchmarks <-> clients <-> components <-> core <-> curation <->
           datasets <-> metrics <-> optimize <-> plotting <-> profile <-> report <-> root <->
           tracking <-> training
  mutual package pairs (one fixable knot each): 18

=== TOPLEVEL only (runtime) ===
  cross-package cycles (SCCs of the package graph): 1
      [11] architectures <-> benchmarks <-> components <-> core <-> curation <-> metrics <->
           optimize <-> report <-> root <-> tracking <-> training
  mutual package pairs (one fixable knot each): 11
```

### 6.1 No definition of "cycle" yields 14

"14 cycles" needs a unit. Every plausible one was tried:

| metric | ALL edges | TOPLEVEL only |
|---|---|---|
| number of SCCs (>1) | 1 | 1 |
| **size** of the largest SCC | **15** | **11** |
| mutual package pairs (2-cycles) | **18** | **11** |
| elementary circuits (`nx.simple_cycles`) | 26206 | 395 |
| cross-package edges | 379 | 269 |
| ordered package pairs with ≥1 edge | 81 | 57 |

**None is 14.** The nearest are 15 (largest SCC, all edges) and 11 (both toplevel metrics).
26206 elementary circuits is why the unit must be stated: the naive split does not produce
"14 cycles you fix one by one", it produces **one giant 15-package tangle**.

### 6.2 The "6" has no reproducible definition

There is no per-symbol split spec — confirmed with the plan's owner. The entire W1
specification is four lines of prose ("split `reports.py` → contracts + metric models",
"`benchmark.py` → Protocol + harness", …) with no mapping from the ~25 top-level defs in
`benchmark.py` (1773 lines) or the mixed contents of `reports.py` (1164 lines) to
destinations. **The "6" therefore cannot be reproduced, confirmed, or refuted** — it is not
a measurement. Reverse-engineering a split that makes the number come out at 6 would repeat
the original sin one level up, so it was not attempted.

### 6.3 The 18 knots, by name, with the fix for each

The minority direction is the one to break. `[RUNTIME]` = present in toplevel-only edges;
`[LAZY]` = grimp/import-linter sees it, runtime does not.

| # | knot | break | fix |
|---|---|---|---|
| 1 | `components ↔ core` | `core → components` (**16**) | `core/__init__.py:27–58` re-exports every component. Facade problem (§6.4) |
| 2 | `architectures ↔ core` | `core → architectures` (2) | `core/__init__.py:60,89` → `method_registry`, `resolver`. Facade |
| 3 | `core ↔ curation` | `core → curation` (4) | `core/__init__.py:28,33,48,90` → `anchor_store`, `canonicalizer`, `harvest`, `review`. Facade |
| 4 | `core ↔ training` | `core → training` (3) | `core/__init__.py:69,70,132` → `methods_calibrate`, `methods_prompt`, `calibration`. Facade |
| 5 | `benchmarks ↔ core` `[LAZY]` | `core → benchmarks` (1) | `core/__init__.py:115` → `core.benchmark`. Facade. **The edge the ad-hoc bug hid (§4)** |
| 6 | `core ↔ optimize` `[LAZY]` | `core → optimize` (1) | `core/__init__.py:115` → `core.optimizers`. Facade |
| 7 | `core ↔ metrics` | `metrics → core` (3) | `analysis:17`, `debugging:37`, `metrics:24` → `core.models`. **Not a cycle if `core` holds only contracts** — this is the split working as intended; the back-edge is `core/__init__` re-exporting metrics |
| 8 | `core ↔ report` | `report → core` (2) | `eval_report:50`, `reports:31` → `core.models`. Same shape as #7 |
| 9 | `core ↔ tracking` | `tracking → core` (2) | `judgement_log:49,50` → `core.models`, `core.matcher`. Same shape as #7 |
| 10 | `metrics ↔ report` | `metrics → report` (1) | `analysis:18` → `core.reports`. Real: `analysis` mixes metric computation with report models. Dissolves **only if** `reports.py` is actually split — the unspecified split (§6.2) |
| 11 | `architectures ↔ benchmarks` | `architectures → benchmarks` (1) | `methods.py:64` → `core.benchmark`, against `benchmark.py:435,871` → `methods`. **A live 2-cycle today**, spanning Stream A and W4 |
| 12 | `benchmarks ↔ report` | `report → benchmarks` (1) | `eval_report.py:38` → `core.benchmark`. Needs the `Benchmark` Protocol extracted to `core/` — the unspecified split |
| 13 | `architectures ↔ root` | `architectures → root` (1) | **`resolver.py:44` `import langres`** — the single edge closing the runtime [5] SCC. One line; highest value/effort ratio in the table |
| 14 | `architectures ↔ curation` `[LAZY]` | `architectures → curation` (3) | `resolver.py:56,81,1354` → `harvest`, `anchor_store` ×2 |
| 15 | `benchmarks ↔ datasets` `[LAZY]` | `benchmarks → datasets` (1) | `eval.py:49` → `data.registry`, TYPE_CHECKING only |
| 16 | `benchmarks ↔ profile` `[LAZY]` | `benchmarks → profile` (1) | `eval.py:36` → `data.data_profile`, TYPE_CHECKING only |
| 17 | `clients ↔ tracking` `[LAZY]` | `clients → tracking` (1) | `clients/__init__.py:31` → `clients.tracking`, TYPE_CHECKING only |
| 18 | `plotting ↔ report` `[LAZY]` | `plotting → report` (1) | `plotting/blockers.py:9` → `core.reports`, TYPE_CHECKING only |

Seven of the 18 (#5, #6, #14–#18) are **lazy-only**: no runtime cycle exists. They are
import-linter artifacts, and §2.1 decides whether they need waivers or a scoped contract.

### 6.4 The facade is the biggest single lever — and it is not enough

Knots #1–#6 — **6 of 18** — have a minority direction consisting **100%** of edges from one
file, `core/__init__.py`. It re-exports the library (fan-out 43, the package ceiling), so
`core` structurally depends on nearly every future package.

Scoring the facade as its own package (a one-line mapping change, no code moved):

```
mutual pairs  : 18 -> 11   (toplevel: 11 -> 5)
largest SCC   : 15 -> 16   (toplevel: 11 -> 12)
```

**The knots drop by 7 — and the giant SCC does not dissolve. It grows.** The facade is
*inside* the tangle; relabelling it moves the boundary without cutting anything. The facade
must be **emptied** (lazy re-export), not reclassified. Recorded because "the facade is the
whole story" is exactly the tidy conclusion this doc exists to distrust.

---

## 7. Reproducing §5

```python
# uv run python - <<'EOF'
import sys; sys.path.insert(0, "tools")
from import_graph import build_graph
from collections import defaultdict

def stream(m: str) -> str | None:
    if m.startswith("langres.data.data_profile"): return "A"
    if m in ("langres.core.metrics", "langres.core.analysis", "langres.core.diagnostics",
             "langres.core.debugging", "langres.core.eval_report", "langres.core.fit_report",
             "langres.core._svg", "langres.core._report_html", "langres.core.reports",
             "langres.core.benchmark"): return "A"
    if m.startswith("langres.data") or m.startswith("langres.bootstrap"): return "B"
    if m in ("langres.core.finetune", "langres.core.calibration", "langres.core.methods_prompt",
             "langres.core.methods_calibrate", "langres.core.review", "langres.core.harvest",
             "langres.core.anchor_store", "langres.core.canonicalizer"): return "B"
    return None

cross = defaultdict(list)
for e in build_graph().edges:
    a, b = stream(e.importer), stream(e.imported)
    if a and b and a != b: cross[(a, b)].append(e)
for k in sorted(cross):
    print(f"Stream {k[0]} --> Stream {k[1]}: {len(cross[k])} edges")
    for e in sorted(cross[k], key=lambda e: (e.importer, e.lineno)):
        print(f"    {e.importer}:{e.lineno} [{e.kind}] -> {e.imported}")
# EOF
```

---

## 8. What this means for W0

1. **The wave order cannot rest on 14-vs-6.** Neither number is reproducible: 14 matches no
   definition, and 6 has no definition. The naive split yields **one 15-package tangle**, not
   a list of 14 separable cycles — a qualitatively different problem from the one the plan
   is sequenced against.
2. **The import-linter contract is a forced decision, not a default** (§2.1). 21% of edges
   are deliberately lazy; grimp counts them; there is no flag to exclude function-local
   imports. Scope the contract to the runtime view or budget ~102 waivers.
3. **Three named edges beat any sequencing** (§5): A→B is 3 edges, two of them
   function-local. `resolver.py:44` (§6.3 #13) alone closes the runtime [5] SCC.
4. **`core/__init__.py` is the fattest single dependency in the package** (fan-out 43) and
   solely causes 6 of 18 knots — but emptying it, not relabelling it, is what pays (§6.4).
5. **The measurement must stay in-repo.** Four evidence errors reached a plan; a fifth lived
   in the instrument and silently hid the one edge (§4) that most undercuts the plan's
   thesis. `tools/import_graph.py` is committed and gated by `test_edges_match_grimp_exactly`
   so the next claim can be re-run instead of re-argued.
