# DX audit: the examples we wish we had

*Written against `main` @ `4c428eb` (2026-07-10), immediately after the Peeters
replication landed (#102). Every claim below was verified by running the code or
reading the named file — none is inferred from a name.*

> **Status (updated when the PR-1 "eval honesty" branch landed).** This is a
> point-in-time audit; the gaps below were real when written. Since then:
>
> | Gap | Status |
> |---|---|
> | **G1** — `evaluate()` reports dollars, no tokens, no spend cap | **Addressed.** `CostTrack.usage` carries the summed token vector; `evaluate()` is spend-capped by default (`budget_usd`, `DEFAULT_BUDGET_USD = $1.00`) and reports `budget_exceeded` when a single in-flight call overruns the cap. |
> | **G2** — no public seam from a benchmark to candidates | **Addressed.** `Resolver.candidates()` and `langres.eval.candidates_for()`. |
> | **G3** — the eval path always sweeps, and selects the threshold on the test labels | **Addressed.** The sweep stays the default (a fixed cut serves no judge family), but it now warns that `best_threshold` is an argmax fitted to the reported gold, and `threshold=` grades honestly at a fixed cut. |
> | **G4** — langres ships no test double | **Addressed.** `langres.testing.ScriptedJudge`. |
> | **G5** — the verbs cannot express "use this prompt" | **Open** — tracked as issue #103. |
> | **G6** — the docs describe modules that do not exist | **Addressed.** `langres.tasks`/`flows`/`ui`, `core.Optimizer`, `core.Evaluator`, `EmbedBlocker`, `EmbedSim`, `SyntheticGenerator` removed from `docs/TECHNICAL_OVERVIEW.md`. |
>
> Leaving the original text unedited: the audit's value is the evidence trail for
> *why* those seams exist, and rewriting it in place would erase that.

---

## 0. The finding, in one number

`examples/research/peeters_llm_em_replication.py` is **1,658 lines**. Exactly
**12 of them touch langres.**

The other 1,646 are paper-fidelity plumbing: regenerating the pair set, fetching
and diffing the authors' archived answers, the cost ledger, the resume store, the
CLI, the dry-run estimator. And the twelve that *are* langres reach for
`LLMJudge` and `classify_pairs` **directly** — never `link()`, `dedupe()`,
`Resolver`, or `evaluate()`.

> When we did real ER research with our own framework, we bypassed our own
> framework.

That is the whole audit. Everything below is the itemised *why*.

**What this is not.** It is not a complaint about that file. It was written to be
*trustworthy* — resumable, cost-honest, archive-verified — not to be *beautiful*,
and it succeeded at what it was for. The bug is that it sits in `examples/`, the
shop window, with nothing telling a reader it is a research harness rather than a
tutorial.

---

## 1. The shop window today

`examples/` holds **50 files / 18,681 lines**. The distribution is the problem:

| | files | lines | note |
|---|---|---|---|
| `examples/research/*` | 38 | 15,798 | harnesses. Longest: 1,658 / 1,176 / 976 |
| `examples/*` (top level) | 10 | 2,491 | mixed. `flywheel_closed_loop.py` alone is 791 |
| `examples/data/*` | 2 | 392 | fixture generators |
| genuinely tutorial-shaped | **2** | **96** | `quickstart_verbs.py` (58), `judgement_log_demo.py` (38) |

`examples/quickstart_verbs.py` is **good** — 58 lines, runs offline, and is
honest about `judge="auto"` failing loudly instead of silently falling back to
fuzzy matching. It is one file in fifty. Nobody tells the reader which fifty.

`docs/GETTING_STARTED.md` is also good, and points at a "runnable twin" —
`examples/flywheel_closed_loop.py`, **791 lines**. A runnable twin should not be
791 lines.

---

## 2. The four gaps, with evidence

### G1 — `evaluate()` reports dollars, but no tokens and no spend cap

> **Correction (verified by running it).** An earlier draft of this note claimed
> `evaluate()` "structurally cannot report cost." **That was wrong.** It returns a
> `JudgePairEval` whose `.cost` is a real `CostTrack`, aggregated from each
> judgement's `provenance["cost_usd"]` by the default `_cost_track`
> (`benchmark.py:516`). `report.cost.usd_total` works today. The docstring's
> "drops the paid-judge `runner` / cost knobs" refers to the *inputs*, not the
> output. Verified with a scripted judge: `cost.usd_total == $0.0030`.

The real gaps are narrower and sharper:

1. **The `LLMUsage` fact layer never reaches the eval report.** `CostTrack` carries
   `usd_total, usd_per_1k_pairs, est_usd_per_100k, escalation_rate,
   llm_calls_per_candidate` — **no tokens, no `cost_is_real`**. So the OTel token
   vector that #102 landed on `PairwiseJudgement.provenance`, and that #101's design
   makes the *fact* from which dollars are derived, dies at the judgement boundary.
   You cannot re-price an `evaluate()` result, and you cannot tell whether its
   dollars were provider-billed or estimated.
2. **`evaluate()` has no spend cap.** It drops the `runner`
   (`BudgetedModuleRunner`), which is the only thing that bounds spend. The
   friendly one-liner, handed a paid judge and 1,206 pairs, will spend **unbounded**.
   That is a worse defect than the one I originally alleged.

### G2 — no public seam from a benchmark to candidates

`get_benchmark("abt_buy")` returns an object exposing
`load, build_blocker, schema, split, blocking_k, threshold_grid`. There is **no**
`candidates()` / `gold_pairs()`.

The consequence is visible in our own flagship example —
`examples/research/portfolio_race.py:208`:

```python
candidates = list(resolver._candidates([r.model_dump() for r in test_records]))
```

**A private method.** Plus `gold_pairs_from_clusters` lives in
`langres.core.benchmark` and is exported from neither `langres.core.__all__` nor
`langres.eval`. When your own examples reach through the wall, the wall is in the
wrong place.

### G3 — the eval path always sweeps, and selects the threshold on the test labels

**What `threshold` means.** A judge emits a `score` per pair. `classify_pairs`
(`metrics.py:303`) predicts *match* iff `score >= threshold`. It is the decision
cutoff that turns a score into a yes/no.

`evaluate()` never takes one. It sweeps `DEFAULT_PAIR_GRID` — 19 points from 0.05
to 0.95 — and reports P/R/F1 **at the threshold that maximises F1 on the very gold
set it is scoring against**. Two consequences:

1. **The headline F1 is optimistically biased.** It is a max over 19 thresholds
   fitted to the labels being reported on. That is threshold selection on the test
   set. (`run_method` does this honestly — `benchmark.py:585` picks the argmax on a
   *train* curve. `evaluate()` does not.)
2. **For a binary judge the sweep is meaningless.** Peeters' protocol — and every
   Yes/No paper prompt — emits scores in `{0.0, 1.0}`, so all 19 thresholds produce
   an identical partition and identical F1. Verified: the reported
   `best_threshold` comes back as **`0.05`**, the first grid point, an artifact of
   the argmax tie-break rather than a property of the matcher.

There is no `threshold=0.5` fixed-decision option, which is why the replication
computed its metrics through `classify_pairs` by hand.

### G4 — langres ships no test double

`docs/TESTING_AT_ZERO_COST.md` documents the `judge=<Module instance>` escape
hatch, but the fake it recommends is **DSPy's `DummyLM`** — so trying langres for
free requires the `[llm]` extra and an eager `import dspy` (disk cache, sqlite).

Meanwhile the repo's own suite hand-rolls the same double four ways:
`DummyModule` (×4), `ScriptedJudge`, `FakeJudge`, `DummyBlocker`.

There is no `langres.testing`. "Try the library with everything mocked — no LLM
calls, no embeddings, no infra" is currently something the user must build.

### G5 (already filed) — the verbs cannot express "use this prompt"

`judge="auto"` and `judge="zero_shot_llm"` both build a `DSPyJudge`. `LLMJudge` —
the only class with a prompt seam — is unreachable by name from `link()`,
`dedupe()`, or `Resolver.from_schema()`. Tracked as **#103**.

### G6 — the docs describe modules that do not exist *and should not*

Verified by import — none of these exist, and no `DeduplicationTask`,
`EntityLinkingTask` or `CompanyFlow` class exists anywhere in `src/` or `tests/`:

| symbol | exists? | documented in |
|---|---|---|
| `langres.tasks` | **no** | `USE_CASES.md`, `POC.md`, `TECHNICAL_OVERVIEW.md` |
| `langres.flows` | **no** | `TECHNICAL_OVERVIEW.md`, `USE_CASES.md` |
| `langres.ui` (Streamlit) | **no** | `TECHNICAL_OVERVIEW.md` |
| `Optimizer.finetune` | **no** | `docs/research/20260701_er_seam_audit.md` |

The interesting question is not whether they exist but whether they *should*.
**They should not — and every intent behind them already shipped under a better
name.** `docs/ROADMAP.md`, the current statement of direction, mentions
`tasks`/`flows`/`ui` **zero times**.

| the 2025 idea | its intent | what actually serves it now |
|---|---|---|
| `langres.tasks` (`DeduplicationTask`) | an out-of-the-box entry point | **the verbs** — `dedupe()` / `link()`. Declarative, schema-optional, spend-capped. |
| `langres.flows` (`CompanyFlow`) | a reusable, portable domain "brain" | **the `Resolver` artifact** — `from_schema` + `save`/`load` via the config-registry. A *serialized* brain beats a hand-written class. |
| `langres.ui` (Streamlit labeler) | close the human-in-the-loop | **the CLI** — `langres review` / `export-csv` / `import-csv`, over `ReviewQueue` + `CorrectionLog`. |

`langres.ui` is the one to reject on principle, not just on redundancy. ROADMAP §1:
*"Engine intelligence in langres; data, persistence, **visibility** in the
consumer."* A shipped Streamlit app **is** visibility. It belongs to brainsquad, on
the same side of the seam as streaming and temporal support. Shipping it would put
langres on both sides of its own architectural boundary.

`TECHNICAL_OVERVIEW.md` is stale in two further ways, both worth fixing in the same
pass: it describes a **two**-layer API (`tasks` / `core`) that has been a
**three**-layer one (verbs → `Resolver` → `core`) since M0; it calls `ReviewQueue`
"a storage backend (e.g. a simple SQLite database)" when it is a truncating JSONL
*snapshot* regenerated from the judgement log (`review.py:115-123`); and its
"Observability & Tracing (TBD)" section predates #99 (trackers) and #101 (the OTel
cost vocabulary).

**Recommendation: delete `tasks`/`flows`/`ui` from the docs rather than build them,
and record that the intents survived while the modules did not.**

---

## 3. The examples we wish we had

Three files. Each is the **spec** for an API, not a description of one. `BLOCKED`
marks a line that cannot be written today; the gap it proves is named inline.

### `examples/01_dedupe_a_csv.py` — zero labels, no key, no downloads

```python
"""Dedupe a list of records with zero labels. Offline, $0."""
from langres import dedupe

records = [
    {"id": "1", "name": "Acme Corporation", "city": "New York"},
    {"id": "2", "name": "Acme Corp",        "city": "New York"},
    {"id": "3", "name": "Unrelated Bakery", "city": "Miami"},
]

for cluster in dedupe(records, judge="string", threshold=0.6):
    print(sorted(cluster))
```

**Status: writable today.** This is `quickstart_verbs.py` minus the env-var
explainer. It needs to become the front door, not file 7 of 50.

### `examples/02_bring_your_own_prompt.py` — the one that matters

```python
"""Score any prompt on a benchmark: how good, and at what price?"""
from langres.core import LLMJudge
from langres.eval import evaluate, get_benchmark

PAPER_PROMPT = (
    "Do the two product descriptions refer to the same real-world product? "
    "Answer with 'Yes' if they do and 'No' if they do not.\n"
    "Entity 1: '{left}'.\nEntity 2: '{right}'."
)

bench = get_benchmark("abt_buy")
candidates, gold = bench.candidates(split="test")   # BLOCKED (G2): no public seam.
                                                    # portfolio_race.py:208 uses resolver._candidates()

judge = LLMJudge(
    model="openrouter/openai/gpt-4o-mini-2024-07-18",
    prompt_template=PAPER_PROMPT,        # ✓ landed in #102
    response_parser="binary_yes_no",     # ✓ landed in #102
    temperature=0.0,                     # ✓ default since #102
)

report = evaluate(judge, candidates, gold, threshold=0.5)   # BLOCKED (G3): always sweeps a
                                                           # grid, picks argmax-F1 on the test labels
                                                           # ALSO: no spend cap (G1.2) — unbounded spend

print(f"F1 {report.pair.f1:.2f}  ·  ${report.cost.usd_total:.4f}")   # ✓ both work today
print(f"   {report.cost.usage.input_tokens} in-tokens, real={report.cost.cost_is_real}")
#          ^^^^^^^^^^^^^^^^^^ BLOCKED (G1.1): CostTrack has no tokens, no cost_is_real
```

**Twelve lines. Three blockers.** This is what the 1,658-line replication should
have looked like, and it is the single example that would have *shown langres
off*: bring a prompt, get accuracy and dollars.

Note the shape of the surviving G1: dollars are already there. What is missing is
the *token vector* those dollars were derived from — the exact fact #101 argues is
primary — and any bound on spend.

Its mocked twin — what CI runs, and what a user runs before spending a cent:

```python
from langres.testing import ScriptedJudge      # BLOCKED (G4): langres.testing does not exist
judge = ScriptedJudge({("a1", "b1"): 1.0, ("a1", "b2"): 0.0})
report = evaluate(judge, candidates, gold, threshold=0.5)
```

### `examples/03_make_it_cheaper.py` — the flywheel, small

```python
"""Cut cost without losing F1: review the margin, train a student, cascade."""
from langres.core import CascadeJudge
from langres.core.review import select_for_review
from langres.core.harvest import harvest_labeled_pairs, derive_threshold_from_pairs

margin = select_for_review(judgements, n=50)        # the uncertain band
pairs  = harvest_labeled_pairs("corrections.jsonl") # what the human fixed
tau    = derive_threshold_from_pairs(pairs)         # kills the magic constant

cheap = CascadeJudge(student=student, teacher=judge, band=(0.2, 0.8))
report = evaluate(cheap, candidates, gold, threshold=tau)
print(f"F1 {report.f1:.2f} at ${report.cost_usd:.4f} — was ${baseline.cost_usd:.4f}")
```

**Status: mostly writable**, inherits G1/G3. This is langres' actual
differentiator and there is no short example of it — only the 791-line
`flywheel_closed_loop.py`.

---

## 4. The CI contract

Curated examples run on **every PR**, and they run **fully mocked**:

- **no real LLM calls** — a `langres.testing` double in the `judge=` slot (G4),
  never a live client, never a key. `env -u OPENROUTER_API_KEY` does **not**
  make a process keyless (litellm's import-time `load_dotenv()` walks up the
  tree); injection is the only real guarantee.
- **no real embeddings** — an injected fake embedding service, no
  sentence-transformers download, no torch.
- **no infra** — no Qdrant, no network, no disk cache.

Rationale beyond hygiene: *if a curated example cannot run mocked, a user cannot
try langres without spending money.* "Can this run mocked?" and "is this pleasant
to try?" are the same question. A doc that lies gets caught by a red build —
which is exactly what would have stopped `langres.tasks` from living in three
docs (G6).

---

## 5. Proposed order

The examples are the acceptance tests. Each gap is done when its `BLOCKED` line
deletes.

| wave | fixes | unblocks |
|---|---|---|
| **A** | `langres.testing` (`ScriptedJudge`, `FakeEmbedder`) — one double, replacing 4 hand-rolled copies | G4 → CI, and the mocked twin of 02 |
| **A** | curate `examples/` vs `examples/research/` (label harnesses "not a tutorial"); **delete** `tasks`/`flows`/`ui` from the docs; fix `TECHNICAL_OVERVIEW`'s two-layer API and `ReviewQueue`-is-SQLite claims | G6 |
| **B** | `CostTrack` carries the `LLMUsage` token vector + `cost_is_real`; `evaluate()` accepts a `runner`/`budget_usd` so the one-liner cannot spend unbounded | G1 → example 02 & 03 |
| **B** | `evaluate()` accepts a fixed `threshold=`; when it sweeps, say so and warn that `best_threshold` is fitted to the labels being reported | G3 |
| **B** | public `bench.candidates(split=…)` seam; re-export `gold_pairs_from_clusters` | G2 → kills `resolver._candidates()` |
| **C** | `judge="prompt_llm"` preset reaching `LLMJudge` | G5 (#103) |

The sharpest single line item is **`evaluate()` has no spend cap**. Everything else
degrades an experiment; that one bills for it.

Wave A is pure addition and unblocks the CI contract. Wave B is the one that
makes example 02 — the one that sells the library — writable at all.
