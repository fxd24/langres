# DX audit: the examples we wish we had

*Written against `main` @ `4c428eb` (2026-07-10), immediately after the Peeters
replication landed (#102). Every claim below was verified by running the code or
reading the named file — none is inferred from a name.*

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

### G1 — `evaluate()` structurally cannot report cost

`langres.eval.evaluate` is the bring-your-own-data one-liner. Its own docstring:

> "A thin wrapper over `evaluate_judge_on_candidates` that **drops** the raw
> judgements (and the paid-judge `runner` / **cost knobs**) …"

So langres' headline eval call cannot answer *"what F1, at what price?"* — the
exact question modern LLM-era ER is a trade-off over, and the one the cost design
note (#101) exists to serve. `LLMUsage` (landed in #102) is the fact layer; nothing
at the eval layer surfaces it.

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

### G3 — the eval path assumes a threshold sweep

`evaluate(..., grid=(0.05, …, 0.95))` always sweeps and reports best-F1. Peeters'
protocol — and every binary Yes/No paper prompt — has **no threshold**. There is
no `threshold=0.5` fixed-decision option. We had to compute metrics through
`classify_pairs` by hand.

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

### G6 — the docs describe modules that do not exist

Verified by import:

| symbol | exists? | documented in |
|---|---|---|
| `langres.tasks` | **no** | `USE_CASES.md`, `POC.md`, `TECHNICAL_OVERVIEW.md` |
| `langres.flows` | **no** | `TECHNICAL_OVERVIEW.md`, `USE_CASES.md` |
| `langres.ui` (Streamlit) | **no** | `TECHNICAL_OVERVIEW.md` |
| `Optimizer.finetune` | **no** | `docs/research/20260701_er_seam_audit.md` |

A newcomer's first read is partly fiction.

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

report = evaluate(judge, candidates, gold, threshold=0.5)   # BLOCKED (G3): always sweeps a grid

print(f"F1 {report.f1:.2f}  ·  ${report.cost_usd:.4f}  ·  {report.usage.input_tokens} in-tokens")
#                                 ^^^^^^^^^^^^^^^^^^ BLOCKED (G1): evaluate() drops cost
```

**Twelve lines. Three blockers.** This is what the 1,658-line replication should
have looked like, and it is the single example that would have *shown langres
off*: bring a prompt, get accuracy and dollars.

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
| **A** | curate `examples/` vs `examples/research/` (label harnesses "not a tutorial"); delete doc fiction | G6 |
| **B** | `evaluate()` returns cost + usage; add fixed `threshold=` alongside `grid=` | G1, G3 → example 02 & 03 |
| **B** | public `bench.candidates(split=…)` seam; re-export `gold_pairs_from_clusters` | G2 → kills `resolver._candidates()` |
| **C** | `judge="prompt_llm"` preset reaching `LLMJudge` | G5 (#103) |

Wave A is pure addition and unblocks the CI contract. Wave B is the one that
makes example 02 — the one that sells the library — writable at all.
