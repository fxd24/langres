# Getting started: the langres flywheel, end to end

**Start here.** This is the lifecycle at altitude — how a langres project goes
from *no labels* to a *cheap, self-improving matcher* with a human in the loop.
Every step below carries a short runnable snippet **inline**; the links point
*down* to the mechanics (they are for depth, never for the code you need to get
moving).

langres closes a loop most ER tools leave open: an expensive judge bootstraps
*silver* labels, a human reviews only the uncertain margin, those labels train a
*cheap* student, and a **cascade** runs the student everywhere while escalating
only the still-uncertain pairs back to the expensive judge.

```
   ┌─────────────────────────────────────────────────────────────────────┐
   │                          THE DATA FLYWHEEL                           │
   │                                                                     │
   │   day 1: LLM judge  ──►  log every call  ──►  select the margin     │
   │   (dedupe, capped)       (log="…jsonl")       (select_for_review)   │
   │        ▲                                             │              │
   │        │                                             ▼              │
   │   cascade: cheap  ◄──  train cheap student  ◄──  human review       │
   │   student + escalate    (RandomForestJudge.fit +           (langres review /  │
   │   only in the band       derive_threshold)        CSV round-trip)   │
   │        │                       ▲                                     │
   │        └───────────────────────┘  save / load the whole pipeline    │
   └─────────────────────────────────────────────────────────────────────┘
```

**Its runnable twin is [`examples/flywheel_closed_loop.py`](../examples/flywheel_closed_loop.py)** —
the same eight stages against committed fixtures, at **$0** (a deterministic
local stand-in plays the frontier judge). Run it end to end while you read:

```bash
uv run python examples/flywheel_closed_loop.py
```

---

## Two lanes: keyed (default) vs keyless

langres has **one** front door with two honest lanes. Pick yours before you
start.

### Keyless lane — `judge="string"` ($0, offline)

No API key, no network, no model download. rapidfuzz string similarity does the
matching. Good for a first taste and for CI; weaker on messy real data. **This
snippet runs verbatim:**

```python
from langres import dedupe

records = [
    {"id": "1", "name": "Acme Corporation", "city": "New York"},
    {"id": "2", "name": "Acme Corp", "city": "New York"},
    {"id": "3", "name": "Totally Different Co", "city": "Chicago"},
]

result = dedupe(records, judge="string", threshold=0.6)
# result       -> [{'1', '2'}]      (singletons like '3' are dropped)
# result.judge_used == "string", result.score_type == "heuristic"
```

### Keyed lane — `judge="auto"` (the default; bring an LLM)

`dedupe(records)` defaults to `judge="auto"`, which picks a real LLM judge from
`OPENROUTER_API_KEY` / `OPENAI_API_KEY` and names the model — and that money is
involved — *before* any paid call, under a **default $1 spend cap**:

```python
from langres import dedupe, NoJudgeAvailableError

try:
    result = dedupe(records)                 # judge="auto"; needs a key + [llm]
except NoJudgeAvailableError as exc:
    print(exc)  # no key -> a clean, actionable error, NOT a wrong answer
```

**Fail, don't fall back.** With no key, `"auto"` **raises
`NoJudgeAvailableError`** (root-exported from `langres`) instead of quietly
degrading to fuzzy matching: unsupervised string matching over-merges on
unlabeled data — one silent bad answer is worse than one loud error. Offline
matching is always available, but only as the *explicit* `judge="string"`
opt-in. A spend-cap breach likewise raises `BudgetExceeded` (also root-exported)
carrying the partial judgements — never a silent bill.

> **You cannot bootstrap ER from nothing.** With zero labels and no LLM, there
> is no honest signal to separate true duplicates from look-alikes. The keyed
> lane is the real starting point for a new entity type; the keyless lane is the
> deterministic, free fallback you opt into on purpose.

### Setup — the extras each step needs

The core install is enough for the keyless lane. The flywheel adds two opt-in
extras, one per capability:

```bash
uv sync                     # core: the keyless "string" lane, $0
uv sync --extra llm         # [llm]: the LLM teacher for the keyed lane / bootstrap
uv sync --extra trained     # [trained]: scikit-learn behind RandomForestJudge + derive_threshold
# (need both? `uv sync --extra llm --extra trained`)
```

---

## The lifecycle, step by step

The steps below tell one continuous story. `records` is a list of dicts, each
with a **stable `id`** (see [operating notes](#two-operating-notes-for-the-loop)
— this matters). Snippets are minimal excerpts of the real API; the
[runnable twin](../examples/flywheel_closed_loop.py) wires them all together.

### 1. Day 1 — dedupe with the LLM, under a cap

Start with the teacher. `dedupe(records)` (default `judge="auto"`) blocks,
scores every candidate pair with the LLM, and clusters — spend-capped:

```python
result = dedupe(records)                    # judge="auto", $1 cap by default
# result.judge_used names the model that ran; override the cap with budget_usd=
```

Depth: the verb layer and its contract live in
[`../README.md`](../README.md#quickstart-dedupe-and-link).

### 2. Log every judgement from day 1

The loop is only as good as its signal. Opt into `log=` on the very first run —
it records every judge call (ids, score, verdict, model, cost) to a JSONL file
with **zero overhead when omitted**. This is the flywheel *inlet*:

```python
result = dedupe(records, log="judgements.jsonl")   # or judge="string" to stay $0
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
round-trip** — export the queue to a spreadsheet, label it, import it back:

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

Now imitate the expensive teacher cheaply. Fit a `RandomForestJudge` — a **trainable
judge** — on the harvested labels, then calibrate *its own* threshold on *its
own* scores:

```python
from langres.core.modules.random_forest_judge import RandomForestJudge
from langres.core.calibration import derive_threshold

student = RandomForestJudge(feature_specs=comparator.feature_specs)
student.fit(iter(train_candidates), train_labels)          # labels from step 4
student_threshold = derive_threshold(student_scores, heldout_labels)
```

> **This is Magellan-style supervised matching, not LLM distillation.** langres
> trains a small classical model on the harvested labels — it does **not**
> compile or distill the LLM's prompt (that path was measured and cut; see
> `docs/ROADMAP.md`). Calibrate the student on **student** scores, never the
> teacher's — `prob_rf` and `prob_llm` are different scales.

Depth: [`EXPERIMENTS.md` § The fit seam](EXPERIMENTS.md) and the
[runnable twin](../examples/flywheel_closed_loop.py) (which builds
`train_candidates` / `student_scores` for you).

### 6. Cascade — cheap everywhere, frontier only at the margin

`CascadeJudge` runs the student on every pair and escalates **only** pairs whose
student score lands in an uncertainty `band` back to the frontier judge. One
threshold cuts the mixed student/teacher stream (both emit `[0, 1]`
probabilities):

```python
from langres.core.modules.cascade_judge import CascadeJudge

cascade = CascadeJudge(student=student, escalation=teacher, band=(0.35, 0.65))
result = dedupe(records, judge=cascade, threshold=student_threshold)
```

> **Derive the band from data, don't hard-code it.** A `±0.15` constant is the
> same magic-number mistake step 4 just killed. The
> [runnable twin](../examples/flywheel_closed_loop.py) widens the band around
> the student threshold until it captures ~20% of calibration-split scores, and
> prints the derivation. Beyond pairwise cascading, **set-wise judging**
> (`SelectJudge`) is the direction that judges a whole candidate group at once —
> see [`docs/ADDING_A_METHOD.md`](ADDING_A_METHOD.md).

### 7. Save and load the whole pipeline

Freeze the configured pipeline — schema, blocker, judge (including a fitted
CascadeJudge student), threshold — into a reusable artifact. Drop to the
`Resolver` the verbs sit on:

```python
from langres import Resolver

resolver = Resolver.from_schema(Contact, judge=cascade, threshold=student_threshold)
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

1. **The loop needs explicit, stable ids.** A schema-less `dedupe(records)` with
   no `"id"` key assigns **positional** ids (`0`, `1`, `2`, …). Positional ids
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

- [`../README.md`](../README.md) — the verbs, install, and API-stability table.
- [`TUTORIAL_YOUR_OWN_CSV.md`](TUTORIAL_YOUR_OWN_CSV.md) — a messy CSV → clusters
  in 15 minutes, with threshold calibration and save/load.
- [`EXPERIMENTS.md`](EXPERIMENTS.md) — the experimentation DX: racing judges,
  the signal log, the harvest, the budget seam.
- [`../examples/flywheel_closed_loop.py`](../examples/flywheel_closed_loop.py) —
  this whole page, runnable at $0.
</content>
</invoke>
