# Getting started: the langres flywheel, end to end

**Start here.** This is the lifecycle at altitude — how a langres project goes
from *no labels* to a *cheap, self-improving matcher* with a human in the loop.
Every step below carries a short runnable snippet **inline**; the links point
*down* to the mechanics (they are for depth, never for the code you need to get
moving).

langres closes a loop most entity-resolution (ER) tools leave open: a frontier
LLM acts as the first **judge** (the component that scores whether two records
match) and bootstraps *silver* labels — machine-generated, not yet
human-verified — with just a prompt; a human reviews only the **uncertain
margin**: the candidate pairs whose scores fall closest to the decision
threshold, the ones the judge is least sure about and the only ones a human
needs to look at. Those labels tune a **cheaper judge** — in production a DSPy
prompt-tuned smaller LLM, in this page's $0 demo a classical *student* model
trained on them — and a **cascade** runs the cheap judge everywhere while
escalating only the still-uncertain pairs back to the frontier.

```
   ┌────────────────────────────────────────────────────────────────────┐
   │                         THE DATA FLYWHEEL                          │
   │                                                                    │
   │   day 1: LLM judge  ──►  log every call  ──►  select the margin    │
   │   (dedupe, capped)       (log="…jsonl")       (select_for_review)  │
   │        ▲                                             │             │
   │        │                                             ▼             │
   │   cascade: cheap    ◄──  tune a cheaper judge ◄──  human review    │
   │   judge everywhere,      (DSPy prompt-tune a       (langres review │
   │   LLM only in band        smaller LLM / .fit)       / CSV export)  │
   │        │                        ▲                                  │
   │        └────────────────────────┘   save/load the whole pipeline   │
   └────────────────────────────────────────────────────────────────────┘
```

**Its runnable twin is [`examples/flywheel_min.py`](https://github.com/fxd24/langres/blob/main/examples/flywheel_min.py)** —
the core of the loop (steps 1–4: log → review the margin → CSV round-trip →
harvest → data-driven threshold → re-run → tearsheet, a one-page HTML quality
report) in ~90 lines, offline at **$0**. Run it while you read:

```bash
uv run python examples/flywheel_min.py
```

The **full** seven steps — including the trained student and the cascade —
run at $0 in [`examples/flywheel_closed_loop.py`](https://github.com/fxd24/langres/blob/main/examples/flywheel_closed_loop.py),
a deeper fixture-driven harness that drives the `langres.core` primitives
directly, bypassing the named architectures' one-call `.dedupe()`/`.compare()`
methods and the CLI.

---

## Two architectures: offline (`FuzzyString`) vs an LLM (`VectorLLMCascade`)

There is no key-sniffing front door — no `matcher="auto"` that picks a model
for you. You construct the architecture you want, and each one is honest about
what it costs before you ever call `.dedupe()`.

### `FuzzyString` — $0, offline

No API key, no network, no model download. rapidfuzz string similarity does the
matching. Good for a first taste and for CI; weaker on messy real data. **This
snippet runs verbatim:**

```python
from langres.architectures import FuzzyString

records = [
    {"id": "1", "name": "Acme Corporation", "city": "New York"},
    {"id": "2", "name": "Acme Corp", "city": "New York"},
    {"id": "3", "name": "Totally Different Co", "city": "Chicago"},
]

result = FuzzyString(threshold=0.6).dedupe(records)
# result       -> [{'1', '2'}]      (singletons like '3' are dropped)
# result.architecture == "FuzzyString", result.score_type == "heuristic"
```

### `VectorLLMCascade` — paid, because you constructed it

`VectorLLMCascade(llm=...)` blocks with a vector index, scores every pair for
free with an embedding student, and escalates only the uncertain band to a
real LLM judge — under a **default $1 spend cap**:

```python
from langres.architectures import VectorLLMCascade

model = VectorLLMCascade(llm="openrouter/openai/gpt-4o-mini")  # needs [llm] + [semantic]
result = model.dedupe(records)          # makes real, spend-capped API calls
# result.architecture == "VectorLLMCascade", result.backbone == "openrouter/openai/gpt-4o-mini"
```

**No key, no silent fallback.** Without `OPENROUTER_API_KEY`/`OPENAI_API_KEY`
set, `VectorLLMCascade(...).dedupe(...)` fails with the provider's own error the
first time it makes a call — there is no heuristic standing by to quietly
degrade to fuzzy matching. A spend-cap breach instead raises `BudgetExceeded`
(root-exported from `langres`) carrying the partial judgements, never a silent
bill:

```python
from langres import BudgetExceeded
from langres.architectures import VectorLLMCascade

try:
    result = VectorLLMCascade(llm="openrouter/openai/gpt-4o-mini", budget_usd=0.01).dedupe(records)
except BudgetExceeded as exc:
    print(exc)  # the cap tripped; exc carries the partial judgements
```

> **You cannot bootstrap ER from nothing.** With zero labels and no LLM, there
> is no honest signal to separate true duplicates from look-alikes.
> `VectorLLMCascade` is the real starting point for a new entity type;
> `FuzzyString` is the deterministic, free path you opt into on purpose.

### Setup — the extras each step needs

`FuzzyString` needs nothing beyond the core install. `VectorLLMCascade` needs
both of its backbones' extras — it blocks with a vector index and judges with a
real LLM:

```bash
uv sync                               # core: FuzzyString, $0
uv sync --extra llm --extra semantic  # VectorLLMCascade's two backbones
uv sync --extra trained               # [trained]: scikit-learn behind RandomForestMatcher + derive_threshold
```

---

## The lifecycle, step by step

The steps below tell one continuous story. `records` is a list of dicts, each
with a **stable `id`** (see [operating notes](#two-operating-notes-for-the-loop)
— this matters). Snippets are minimal excerpts of the real API; the
runnable twins wire them together: [`flywheel_min.py`](https://github.com/fxd24/langres/blob/main/examples/flywheel_min.py)
covers steps 1–4; [`flywheel_closed_loop.py`](https://github.com/fxd24/langres/blob/main/examples/flywheel_closed_loop.py)
covers all seven.

### 1. Day 1 — dedupe with the LLM, under a cap

Start with the teacher. `VectorLLMCascade(llm=...).dedupe(records)` blocks
(pre-selects the record pairs worth comparing), scores every candidate pair —
free with the embedding student, escalating only the uncertain band to the
LLM — and clusters, spend-capped:

```python
from langres.architectures import VectorLLMCascade

model = VectorLLMCascade(llm="openrouter/openai/gpt-4o-mini")
result = model.dedupe(records)              # $1 cap by default
# result.architecture names the class, result.backbone the LLM id;
# override the cap with budget_usd=
```

Depth: the architectures layer and its contract live in
[`../README.md`](https://github.com/fxd24/langres/blob/main/README.md#quickstart-named-architectures-dedupe-and-compare).

### 2. Log every judgement from day 1

The loop is only as good as its signal. Opt into `log=` on the very first run —
it records every judge call (ids, score, verdict, model, cost) to a JSONL file
with **zero overhead when omitted**. Every architecture's `.dedupe()`/`.compare()`
takes it. This is the flywheel *inlet*:

```python
result = model.dedupe(records, log="judgements.jsonl")
# FuzzyString(threshold=0.6).dedupe(records, log="judgements.jsonl") is the $0 version
```

Depth: [`EXPERIMENTS.md` § Signal log](EXPERIMENTS.md) for the record shape and
the `features=True` PII note.

### 3. Review at the margin — `select_for_review` + `langres review`

Don't review everything — review where the judge was *least sure*.
`select_for_review` reads the log, picks the uncertain margin (and mixes in a
small random **audit** slice for unbiased trust measurement), and
`ReviewQueue.write` snapshots it. This cell is copy-paste complete — it writes
`queue.jsonl` and prints the exact next command:

```python
from langres import JudgementLog, select_for_review, ReviewQueue

rows = JudgementLog("judgements.jsonl").read()
items = select_for_review(
    rows, strategy="uncertainty", threshold=0.6, margin=0.15, records=records
)
ReviewQueue("queue.jsonl").write(items)
print(f"{len(items)} pairs to review. Next:  uv run langres review queue.jsonl")
```

Then a human answers the queue. The **primary review path is the CSV
round-trip** — `langres export-csv` writes the queued pairs to a plain `.csv`
file you can open in any spreadsheet; fill the `label` column with `y`/`n`,
and `langres import-csv` reads the answers back:

```bash
uv run langres export-csv queue.jsonl to_label.csv   # fill the 'label' column (y/n)
uv run langres import-csv to_label.csv queue.jsonl   # -> corrections.jsonl
```

`langres review queue.jsonl` is the quick terminal loop (a `y/n/s/q` prompt per
pair) for developers who prefer to stay in the shell — each answer is appended
to `corrections.jsonl` immediately, so quitting never loses work and a re-run
resumes. (`uv run langres --version` reports your build.) Depth:
[`EXPERIMENTS.md` § Signal log](EXPERIMENTS.md).

### 4. Harvest silver + gold into labeled pairs

`harvest_labeled_pairs` merges the logged verdicts (**silver** — weak labels)
with the human corrections (**gold** — overrides), keyed order-independently by
pair. `derive_threshold_from_pairs` then reads a data-driven cut off the result:

```python
from langres.core.harvest import (
    CorrectionLog, harvest_labeled_pairs, derive_threshold_from_pairs,
)

corrections = CorrectionLog("corrections.jsonl").read()
pairs = harvest_labeled_pairs(rows, corrections)   # verdicts=silver, corrections=gold
threshold = derive_threshold_from_pairs(pairs)     # data-driven, not a magic constant
```

> **Circularity caveat.** Calibrating on silver labels *alone* is circular — a
> judge's own verdicts can only recover the cut that produced them, so
> `derive_threshold_from_pairs` **warns** when every label is silver. Overlay
> human corrections before you trust the threshold. (Training a *different*
> model on silver labels is legitimate — that is exactly the next step.)

Depth: [`EXPERIMENTS.md` § Flywheel harvest](EXPERIMENTS.md) and
[`TUTORIAL_YOUR_OWN_CSV.md` § 4](TUTORIAL_YOUR_OWN_CSV.md#4-calibrate-the-threshold-from-a-few-labels).

### 5. Train the cheap student

Now spend the harvested labels on making the judgement cheaper. This demo fits
a `RandomForestMatcher` — a **trainable judge**, the loop's $0 stand-in for the
cheaper model — on the harvested labels, then calibrates *its own* threshold on
*its own* scores:

```python
from langres.core.matchers.random_forest_judge import RandomForestMatcher
from langres.training.calibration import derive_threshold

student = RandomForestMatcher(feature_specs=comparator.feature_specs)
student.fit(iter(train_candidates), train_labels)          # labels from step 4
student_threshold = derive_threshold(student_scores, heldout_labels)
```

> **The classical student is the $0 plumbing demo — the production rung is a
> cheaper LLM.** This step's `RandomForestMatcher` is Magellan-style supervised
> matching, shipped as an honest baseline and free plumbing for the loop. The
> LLM-native pattern is to spend the same harvested labels on **prompt-tuning a
> smaller LLM** (`DSPyMatcher`): a precision-tuned DSPy prompt signature let a cheap
> model beat an uncompiled frontier model at lower cost (see `docs/ROADMAP.md`;
> automatic MIPROv2 *compilation* was measured and cut — the signature is the
> lever). Fine-tuning a small LM on these labels is the roadmap's next rung.
> Whichever student you pick, calibrate it on **its own** scores, never the
> teacher's — `prob_rf` and `prob_llm` are different scales.

Depth: [`EXPERIMENTS.md` § The fit seam](EXPERIMENTS.md) and
[`examples/flywheel_closed_loop.py`](https://github.com/fxd24/langres/blob/main/examples/flywheel_closed_loop.py)
(the deeper harness, which builds `train_candidates` / `student_scores` for you).

### 6. Cascade — cheap everywhere, frontier only at the margin

`CascadeMatcher` runs the student on every pair and escalates **only** pairs whose
student score lands in an uncertainty `band` back to the frontier judge. One
threshold cuts the mixed student/teacher stream (both emit `[0, 1]`
probabilities). A hand-built matcher like this is exactly what the mid-level
`Resolver` is for — the named architectures each fix their own topology, so a
custom composition (this cascade wraps the `RandomForestMatcher` step 5
trained, not `VectorLLMCascade`'s own embedding student) goes here instead:

```python
from langres import Resolver
from langres.core.matchers.cascade_judge import CascadeMatcher

cascade = CascadeMatcher(student=student, escalation=teacher, band=(0.35, 0.65))
resolver = Resolver.from_schema(Contact, matcher=cascade, threshold=student_threshold)
result = resolver.dedupe(records)
```

> **Derive the band from data, don't hard-code it.** A `±0.15` constant is the
> same magic-number mistake step 4 just killed.
> [`examples/flywheel_closed_loop.py`](https://github.com/fxd24/langres/blob/main/examples/flywheel_closed_loop.py)
> widens the band around the student threshold until it captures ~20% of
> calibration-split scores, and prints the derivation. Beyond pairwise cascading, **set-wise judging**
> (`SelectMatcher`) is the direction that judges a whole candidate group at once —
> see [`docs/ADDING_A_METHOD.md`](ADDING_A_METHOD.md).

### 7. Save and load the whole pipeline

Freeze the configured pipeline — schema, blocker, matcher (including a fitted
CascadeMatcher student), threshold — into a reusable artifact. Every `ERModel`
(a named architecture or the `resolver` from step 6) has `.save`/`.load`:

```python
resolver.save("artifacts/contacts_v1")            # resolver.json + per-child sidecars
reloaded = Resolver.load("artifacts/contacts_v1") # fitted student round-trips, no pickle
```

Serialization is config-registry based (human-readable JSON manifest + sidecar
state, no pickle, no code execution). Depth:
[`DX_RESOLVER.md`](DX_RESOLVER.md) and
[`TUTORIAL_YOUR_OWN_CSV.md` § 5](TUTORIAL_YOUR_OWN_CSV.md#5-save-and-load-the-pipeline).

---

## Where langres fits (and where it doesn't)

Be honest about the landscape. **A review queue, a CSV hand-off, and a trained
matcher are not new** — active learning plus clerical review has been table
stakes in entity resolution since [dedupe](https://github.com/dedupeio/dedupe)
(~2014) and [Zingg](https://github.com/zinggAI/zingg), and supervised matching
on labeled pairs goes back to Magellan and, further, to Fellegi–Sunter clerical
review. langres does not claim to have invented any of that.

**What langres actually adds** is the *bootstrap*: an **LLM teacher generates
the silver labels**, so a brand-new entity type with **zero** labeled data has
signal on day one, and the human reviews **only the uncertain margin** — under
honest per-call cost accounting, on a **seam where every judge is swappable**
(string ↔ embedding ↔ LLM ↔ trained ↔ cascade, one interface). That is the
delta, not the loop mechanics.

**vs. Splink — different lane, complement not compete.**
[Splink](https://github.com/moj-analytical-services/splink) does unsupervised
Fellegi–Sunter probabilistic linkage at population scale on a SQL backend.
langres is code-first, judge-agnostic, and aimed at the zero-label bootstrap.

- **Use Splink when** you have hundreds of thousands to millions of records in a
  data warehouse and want scalable, unsupervised probabilistic record linkage.
- **Use langres when** you have no labels and want an LLM to bootstrap them, you
  want to swap matching methods behind one seam without a rewrite, and you want a
  human reviewing only the uncertain margin with honest cost per call.

**Governance & trust.** The `strategy="audit"` slice — a seeded random sample
over *all* judged pairs, not just the uncertain ones — is the mechanism that
measures quality without bias and catches **confident false merges** that margin
sampling never surfaces. It is the trust knob for a matcher you have to defend.

**Privacy posture.** `select_for_review` and the review queue are **ids-only by
default** — record content is copied into the queue only when you explicitly
pass `records=`. For PII datasets, omit `records=`: the reviewer works from ids
(or a separate secured store), and no record content leaves your data.

---

## Two operating notes for the loop

Two things the flywheel depends on that are easy to miss:

1. **The loop needs explicit, stable ids.** A schema-less `.dedupe(records)` call
   with no `"id"` key assigns **positional** ids (`0`, `1`, `2`, …). Positional ids
   are per-run: a re-run's id `3` may be a *different* record, so the judgement
   log, review queue, and corrections from different runs **won't join back to
   the same records**. Give every record a stable `"id"` (or pass
   `schema=<YourModel>`) so every seam keys on the same identity across runs.

2. **One log file per run, or dedupe rows before harvest.**
   `harvest_labeled_pairs` emits **one `LabeledPair` per judgement row**, so
   appending re-runs to the same `judgements.jsonl` **duplicate-weights** the
   training data (the same pair counted twice). Either write a fresh log file
   per run, or dedupe rows by pair before harvesting.

---

## Next

- [`../README.md`](https://github.com/fxd24/langres/blob/main/README.md) — the named architectures, install, and API-stability table.
- [`TUTORIAL_YOUR_OWN_CSV.md`](TUTORIAL_YOUR_OWN_CSV.md) — a messy CSV → clusters
  in 15 minutes, with threshold calibration and save/load.
- [`EXPERIMENTS.md`](EXPERIMENTS.md) — the experimentation DX: racing judges,
  the signal log, the harvest, the budget seam.
- [`../examples/flywheel_min.py`](https://github.com/fxd24/langres/blob/main/examples/flywheel_min.py) —
  the loop's core (steps 1–4: log → review → harvest → threshold → tearsheet),
  runnable at $0.
- [`../examples/flywheel_closed_loop.py`](https://github.com/fxd24/langres/blob/main/examples/flywheel_closed_loop.py) —
  this whole page — all seven steps including student + cascade — at $0
  (core primitives directly; bypasses the named architectures and the CLI).
