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
>   writing** (it lives on the unmerged branch `docs/theory-foundation`). Items A1–A3
>   below are the experiments its §6.5 explicitly asks for; this doc does **not**
>   re-derive it.
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

*(items follow)*
