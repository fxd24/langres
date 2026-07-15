# Friction Log

This document tracks technical issues encountered during development, their root causes, and remedies. Helps future contributors avoid the same pitfalls.

---

## OpenMP Thread Conflicts (macOS + Python 3.13)

**Problem:** FAISS tests segfault with `OMP: Error #179: Function pthread_mutex_init failed` on macOS with Python 3.13. Root cause: Multiple libraries (torch, scikit-learn, faiss-cpu) each bundle their own OpenMP runtime (`libomp.dylib`), causing thread initialization conflicts.

**Remedy:** Set environment variables in `.env` to force single-threaded OpenMP mode: `OMP_NUM_THREADS=1` and `KMP_DUPLICATE_LIB_OK=1`. Pre-commit hooks automatically load these via `uv run --env-file .env pytest`. Minimal performance impact at POC scale (<10K entities).

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
| LOC-to-first-cluster (`examples/quickstart_verbs.py`) | **3 statements / ~10 lines** (import + records literal + `dedupe(...)` + print loop) | ≤ 10 lines | ✅ PASS |

**Import time — the W0.4 lazy-import win holds.** `import langres` is ~0.2 s
pure-interpreter (well under the 2 s budget), and a direct check confirms the
heavy stacks stay out of `sys.modules` on a bare import:
`torch`, `litellm`, and `sentence_transformers` are all `False` after
`import langres`. The PEP 562 `__getattr__` lazy resolution (see
`tests/test_import_budget.py`) means a newcomer who only wants the string-judge
`dedupe()` path never pays torch's import cost.

**TTHW is dominated by the (tiny) core install, not by langres itself.**
`quickstart_verbs.py` runs offline at $0 through the explicitly pinned
zero-spend `"string"` judge — no API key, no network, no embedding-model
download. (The default `matcher="auto"` requires an API key and raises
`NoMatcherAvailableError` without one — offline fuzzy matching is an explicit
opt-in, not a fallback.) It prints `2 cluster(s) found` in ~0.2 s. From a cold `uv venv` to that first
cluster is ~2.5 s end-to-end. The heavier `[semantic]` / `[llm]` paths are only
needed once a newcomer graduates past the toy dataset (N > ~100 records) or
wants a real LLM judge; the quickstart deliberately pins `matcher="string"` so the
first-run experience needs neither.

**No new friction found** in this pass — the packaging/import DX cleaned up in
W0.4 (lazy heavy imports, core/extras split) is holding. The one caveat worth
recording: cold-install wall-clock is network-bound, so the `[semantic]` figure
(921 MB, torch-dominated) is the one a newcomer on a slow link will feel; the
core and `[llm]` paths stay light.

---

*Add new friction items here as they're discovered.*
