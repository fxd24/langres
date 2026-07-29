# Friction Log

This document tracks technical issues encountered during development, their root causes, and remedies. Helps future contributors avoid the same pitfalls.

---

## OpenMP Thread Conflicts (macOS + Python 3.13)

**Problem:** FAISS tests segfault with `OMP: Error #179: Function pthread_mutex_init failed` on macOS with Python 3.13. Root cause: Multiple libraries (torch, scikit-learn, faiss-cpu) each bundle their own OpenMP runtime (`libomp.dylib`), causing thread initialization conflicts.

**Remedy:** Set environment variables in `.env` to force single-threaded OpenMP mode: `OMP_NUM_THREADS=1` and `KMP_DUPLICATE_LIB_OK=1`. The prek pre-push hook automatically loads these via `uv run --env-file .env pytest`. Minimal performance impact at POC scale (<10K entities).

### The same conflict also presents as a silent **deadlock**, not only a segfault (2026-07-28)

**Problem:** `examples/research/prompt_axis.py` hung for 3h16m at 0% CPU with no
error, no traceback and no network activity, immediately after logging
`Loading SentenceTransformer model from …/embeddinggemma-300m`. Three earlier
models (MiniLM, bge, e5) had already completed 208 rows, so it looked like "a
problem with Gemma" rather than an environment problem — which is what made it
expensive to find.

**Measured, not inferred:**

| condition | Gemma load |
|---|---|
| no faiss imported (repo id *or* local snapshot path) | **1.3 s** |
| faiss imported + index built, `KMP_DUPLICATE_LIB_OK=TRUE` only | **never returns** (killed at 180 s) |
| same, plus `OMP_NUM_THREADS=1` | **1.6 s** |

**Two traps worth naming:**

1. **`KMP_DUPLICATE_LIB_OK` alone is not sufficient.** It suppresses the *abort*
   (`OMP: Error #15`), not the *deadlock*. A run can therefore have the variable
   set, produce no error whatsoever, and hang forever. `OMP_NUM_THREADS=1` is the
   one that actually fixes it.
2. **Dropping `--env-file .env` is not a cosmetic deviation.** A fresh git
   worktree has **no `.env`** (it is gitignored), so `uv run --env-file .env`
   fails outright there and the tempting move is to drop the flag and set
   `KMP_DUPLICATE_LIB_OK` by hand — which is exactly the state above. In a
   worktree, set the variables explicitly instead:
   `KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false uv run python …`

---

## MPS memory is not reclaimed between models, so a long sweep OOMs (2026-07-28)

**Problem:** the same `prompt_axis.py` sweep died at row 324 of 400 with a torch
MPS out-of-memory asking for **42.44 GiB** — while loading
`Qwen/Qwen3-Embedding-0.6B`, a 0.6B model in fp16 whose weights are ~1.2 GiB. The
number is absurd on its face, which is the useful clue: it is not one model's
footprint, it is four models' worth of cached allocations plus the corpus vectors
the sweep had accumulated over the preceding three checkpoints.

**Cause:** the MPS allocator caches freed blocks rather than returning them, and
Python objects that outlive their loop iteration keep the model graph alive. One
of these was a real leak — the `blocker` bound in the inner loop pinned the
embedder after the `del` list had already been extended once — but fixing it only
moved the ceiling, it did not remove it. A sweep long enough will still hit it.

**Remedy: one benchmark per process.** The harness resumes at cell granularity
(`--resume` reads the rows file), so a driver loop that invokes it once per
`(model, benchmark)` and lets the process exit is both the fix and free — the
embedding cache is on disk, so nothing is recomputed:

```sh
for b in fodors_zagat abt_buy amazon_google wdc_computers; do
  KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
    uv run python examples/research/prompt_axis.py \
      --models Qwen/Qwen3-Embedding-0.6B --benchmarks "$b" --resume
done
```

**The trap:** it fails **late** and **partially**. Three models and 324 rows had
already been written, so the failure looks like "Qwen3 is too big for this
machine" rather than "this process has been alive too long". Running Qwen3
*first* would have succeeded and hidden the ceiling entirely. Treat "a model that
loads fine alone but OOMs in a sweep" as an allocator-lifetime problem, not a
model-size one.

---

## Wave 3 run-as-user DX numbers (2026-07-03)

A genuine fresh-environment pass measuring what a real newcomer experiences,
each against a target budget. Method: fresh `uv venv` + `uv pip install` into a
temp dir with an **isolated (cold) `UV_CACHE_DIR`** so downloads are real, on a
fast (~1 Gbps) connection. Numbers are network-bound — a newcomer on a slower
link will see proportionally longer install times (download sizes given so the
number is interpretable, not just a wall-clock figure from one machine).

| Metric | Measured | Target budget | Verdict |
|---|---|---|---|
| Cold install — core only (`uv pip install langres`) | **2.3 s** (63 MB) | < 30 s | ✅ PASS |
| Cold install — `[llm]` extra (dspy/litellm/openai) | **2.4 s** (207 MB) | < 60 s | ✅ PASS |
| Cold install — `[semantic]` extra (torch/faiss/sentence-transformers) | **6.8 s** (921 MB) | < 120 s | ✅ PASS |
| `python -c "import langres"` (cold interpreter) | **~0.2 s** pure / **~0.55 s** via `uv run` | < 2 s | ✅ PASS |
| TTHW (fresh venv → first successful dedupe) | **~2.5 s** (2.3 s core install + 0.2 s run) | < 60 s | ✅ PASS |
| LOC-to-first-cluster (`examples/quickstart_models.py`) | **3 statements / ~10 lines** (import + records literal + `dedupe(...)` + print loop) | ≤ 10 lines | ✅ PASS |

**Import time — the W0.4 lazy-import win holds.** `import langres` is ~0.2 s
pure-interpreter (well under the 2 s budget), and a direct check confirms the
heavy stacks stay out of `sys.modules` on a bare import:
`torch`, `litellm`, and `sentence_transformers` are all `False` after
`import langres`. The PEP 562 `__getattr__` lazy resolution (see
`tests/test_import_budget.py`) means a newcomer who only names the offline
`FuzzyString` architecture never pays torch's import cost.

**TTHW is dominated by the (tiny) core install, not by langres itself.**
`quickstart_models.py` runs offline at $0 through `FuzzyString` — the
architecture with no paid model slot, so it cannot spend, needs no API key, no
network, no embedding-model download. It prints `2 cluster(s) found` in ~0.2 s.
From a cold `uv venv` to that first cluster is ~2.5 s end-to-end. The heavier
`[semantic]` / `[llm]` paths are only needed once a newcomer names a model that
needs them — `Resolver.from_schema(matcher="embedding")` or
`VectorLLMCascade(llm=...)`; there is no automatic row-count-based switch, and
the quickstart deliberately names `FuzzyString` so the first-run experience
needs neither.

**No new friction found** in this pass — the packaging/import DX cleaned up in
W0.4 (lazy heavy imports, core/extras split) is holding. The one caveat worth
recording: cold-install wall-clock is network-bound, so the `[semantic]` figure
(921 MB, torch-dominated) is the one a newcomer on a slow link will feel; the
core and `[llm]` paths stay light.

---

*Add new friction items here as they're discovered.*
