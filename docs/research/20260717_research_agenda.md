# Research Agenda — the experiments the framework unlocks

> **Status:** Agenda / living note (2026-07-17). This is a **backlog of questions**, not
> a plan and not a result. It records what is worth *measuring* once the core framework
> is ready, why each question matters to langres, what is already known (with citations),
> what would settle it, and what it would cost.
>
> **Scope boundary.** Framework/build work is **out of scope** here. `docs/ROADMAP.md`
> owns the milestones and `docs/plans/` owns executable build plans. This doc is the
> layer *above* both: what the framework is *for*. An item here becoming buildable is a
> signal to write a plan, not to expand this doc.
>
> **Reads with:**
> - `docs/THEORY.md` — the mathematical foundation. **⚠ Not on `main` as of this
>   writing**: it lives on the unmerged branch `docs/theory-foundation`, and every
>   reference below is pinned to **`7a106ce`** ("docs(theory): ground THEORY.md in the
>   literature; retract four claims") so the citation does not dangle. Items A1–A3 are
>   the experiments its §6.5 explicitly asks for; this doc does **not** re-derive it.
> - `docs/research/20260707_data_prep_hard_case_mining_survey.md` — the mining/selection
>   literature backbone. Items C4/C5 are its retrieval-training thread, made measurable.
> - `docs/research/20260715_data_prep_architecture.md` §6 — the data-prep *backlog*. This
>   doc is the wider agenda; that one owns data-prep method choice.
> - `docs/research/20260701_er_seam_audit.md` — the SOTA seam audit (issue #55).

---

## 0. How to read this

Every item has the same five fields:

| Field | Means |
|---|---|
| **The question** | Stated so it can come out *no*. If it cannot fail, it is not here. |
| **Why it matters to langres** | The decision it changes. An item that changes no decision is deleted. |
| **What's already known** | Cited. Primary sources only. Confidence flagged. |
| **What would settle it** | The concrete measurement — not "investigate". |
| **Cost & prerequisites** | $0 / CPU / GPU / paid-API, and what must land first. |

**Method discipline** (this project has been burned — see `.claude/rules/expert-knowledge.md`):
every number below is transcribed from a primary source **read directly**, or is marked
`[unverified]`. A hypothesis is labelled a hypothesis. Novelty is not a goal — the
owner's standard is *"We don't care how much we are contributing with novelty. We care
that what we have is good and correct."* Several items below exist to find out that we
are **wrong**, which is the cheapest possible outcome.

---

## 1. The board

*(priority/cost table — filled in below as items are drafted)*

---

## 6. Thread E — grounding

Neither item here produces a method. Both exist to stop us being confidently wrong:
E1 checks that *published* numbers hold, E2 checks that we are not re-deriving a solved
problem under a different vocabulary.

### E1 — Reproduction studies: do published ER results hold?

**The question.** Take a published ER result, re-run it, and check it reproduces. Not
"is it a good method" — *is the number real, and does it survive contact with our
harness?*

**Why it matters to langres.** Two distinct payoffs, and the second is the real one:

1. Every method langres claims to compose is a method langres must be able to *reproduce*.
   A seam that cannot reproduce the thing it wraps is a seam in name only.
2. **Reproduction is the cheapest available test of our own instrument.** When a
   replication misses, the prior should not be "the paper is wrong" — it is usually us.
   A miss localizes a bug in our harness that no unit test would catch, because it is a
   *semantic* bug (wrong split, wrong metric, wrong pair set), not a crashing one.

**What's already known.** This is tractable here, and it has already paid off — twice,
in opposite directions:

- **It works.** The Peeters/MatchGPT replication (`docs/research/`, and the merged
  work behind it) hit **99.25% per-pair agreement** on **all 1206 Abt-Buy pairs**, on
  both `gpt-4o-mini` and `gpt-4o`, for **$0.28** — and the authors' archived answers
  reproduce their published F1 exactly. A full replication of a paid-LLM ER result costs
  less than a coffee.
- **It catches things.** `THEORY.md` §12 (erratum 7) records a citation this project
  itself got wrong: a claimed "face clustering collapses to 0.37" was read off a
  **0–100** scale (so: 0.37 **percent** — ≈100× the natural misreading), from a run that
  scored **BCubed F = 66.96**, using **HAC, not transitive closure**. The doc's own
  verdict: *"A citation that fails this badly under checking is worse than none."*
  Erratum 11 records a "33%" NIL rate with **no source at all**.

**What would settle it.** Pick targets where the authors published enough to be checked
(archived predictions ≫ archived weights ≫ a table in a PDF), and report per-pair
agreement, not just the headline metric — agreement is what localizes a harness bug; a
matching F1 can hide two compensating errors. Priority targets, in order of how much
they'd change what we do:

| Target | Why it, specifically | Reproducible from |
|---|---|---|
| **Ditto** (Li et al., VLDB 2020, [2004.00584](https://arxiv.org/abs/2004.00584)) | The number our LLM placement is measured against; project history says we land 0.14–0.21 short of it. Either the gap is real and instructive, or our harness differs. | Code + published splits |
| **AnyMatch** ([2409.04073](https://arxiv.org/abs/2409.04073)) | The data-recipe bet (`20260715_data_prep_architecture.md`) rests on it. A 124M GPT-2 is cheap to re-run. | Code + LODO protocol |
| **Jellyfish** ([2312.01678](https://arxiv.org/abs/2312.01678)) | Caveat already on file: Amazon-Google was **seen** in training, Abt-Buy is zero-shot. A replication that ignores this reproduces a leak. | Weights on HF |

**Cost & prerequisites.** **Cheap.** Peeters-class paid replication ≈ **$0.30–$5**.
Ditto/AnyMatch are **CPU-feasible, GPU-preferred** (single consumer GPU). Prerequisite:
the benchmark registry (`langres/data/registry.py`) already has the loaders — the
binding constraint is *protocol fidelity* (exact split, exact pair set), not compute.
**Blocked by nothing.** This is the item to run first if the goal is confidence rather
than discovery.

---

### E2 — Cross-domain ER: who else has this problem, and under what name?

**The question.** Many fields rediscover entity resolution under their own vocabulary —
record linkage, deduplication, coreference resolution, entity alignment (KG), author
name disambiguation, face clustering, product matching, identity resolution,
bioinformatics sequence matching. Which of their concepts are **genuinely the same
operation**, and which are **superficially similar and transfer badly**? And which of
their techniques transfer *to the others*?

**Why it matters to langres.** langres's central bet — from `docs/ROADMAP.md` §1 and
`THEORY.md` §3 ("one operation, seven names") — is that these are the *same* operation
at different feasible classes. That bet is either langres's reason to exist or its
biggest unforced error, and **it is currently asserted, not surveyed**. Two concrete
consequences:

- If the shared core is real, the feature-bag/architecture seam should accept a
  coreference or KG-alignment workload with **no new abstraction**. That is a falsifiable
  claim about our API, testable without new research.
- If it is not real — if, say, coreference's discourse structure or KG alignment's graph
  structure is load-bearing rather than incidental — then the honest move is to **narrow
  the claim**, not to widen the framework.

**What's already known.** More than the framing above suggests, and `THEORY.md` already
carries the receipts — this item is a survey, and it starts from a non-empty base:

- **The silos are documented, and are closing.** `THEORY.md` §12 erratum 13 retracts its
  own "the literature is siloed" claim as *"True until recently"* — and notes it was
  documented **by** the field itself: **Leone et al., PVLDB 15(8), 2022**. That paper is
  the natural starting point, and its existence means the survey's premise must be
  *"how far did they get"*, not *"nobody has looked"*.
- **The cross-domain concepts are already partly mapped.** §12 erratum 12 records that
  the assignment constraint, dropped by DB/ER toolkits, is **alive in KG entity
  alignment** — where NIL travels under the name **"dangling entities"**. Erratum 10
  distinguishes Fellegi–Sunter abstention (from *uncertainty*) from NIL (asserting
  *non-existence*) — a genuine, non-obvious semantic difference between two things that
  look identical. Erratum 7 is a cross-domain citation (face clustering) that broke on
  contact.
- **The precedent for a generalist exists.** **Unicorn** (Tu et al., SIGMOD 2023,
  OpenReview `388Cge6WPN`) unifies matching tasks in one MoE encoder — but, per §0
  erratum 9, *"Its axis is **data-element type**, not constraint shape"*. So the
  unification question is live and *partly answered by someone else*, on a different
  axis than ours.
- **Prior art per operation is inventoried** in `THEORY.md` §0 (Fellegi–Sunter 1969,
  AJAX, meta-blocking, BFKPT, Swoosh, Dedupalog). That is the *data-integration* column
  of the table this item needs to complete; the other columns (NLP coreference, KG
  alignment, vision clustering, bibliometrics, bioinformatics) are empty.

**What would settle it.** A survey with a **forced comparison table**, one row per field,
whose columns are langres's own operations — carrier, score $\sigma$, selection $\pi$,
feasible class $\mathcal{F}$, merge, and *what plays the role of blocking*. The
discipline that makes it worth doing rather than a reading list:

- Each cell is filled with the field's **own words + citation**, not our paraphrase
  (this is the `THEORY.md` §0 method, and it is what makes §0 survive review).
- Every claimed transfer must name **a technique that moves**, and be marked
  transferred / **not** transferred / untested. "Both do clustering" is not a finding.
- Deliberately hunt **false friends**. The value is concentrated in the *negatives*:
  erratum 10 (FS-abstention ≠ NIL) is worth more than ten confirmed similarities,
  because it is the kind of mistake the framework would otherwise encode in an API.
- Output is a section in this folder plus, where a transfer is real, **one issue per
  transfer** — not a framework change.

**Cost & prerequisites.** **$0** — literature only, no compute. Bounded by reading time,
which is the real cost (this is the largest-effort $0 item on the board). Prerequisite:
none. **Sequencing:** worth doing *before* any framework generalization work, and it is
the natural companion to landing `docs/theory-foundation` — but nothing downstream
blocks on it, so it is a background thread, not a gate.

---

*(items follow)*
