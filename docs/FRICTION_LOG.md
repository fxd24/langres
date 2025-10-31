# Friction Log

This document tracks technical issues encountered during development, their root causes, and remedies. Helps future contributors avoid the same pitfalls.

---

## OpenMP Thread Conflicts (macOS + Python 3.13)

**Problem:** FAISS tests segfault with `OMP: Error #179: Function pthread_mutex_init failed` on macOS with Python 3.13. Root cause: Multiple libraries (torch, scikit-learn, faiss-cpu) each bundle their own OpenMP runtime (`libomp.dylib`), causing thread initialization conflicts.

**Remedy:** Set environment variables in `.env` to force single-threaded OpenMP mode: `OMP_NUM_THREADS=1` and `KMP_DUPLICATE_LIB_OK=1`. Pre-commit hooks automatically load these via `uv run --env-file .env pytest`. Minimal performance impact at POC scale (<10K entities).

---

*Add new friction items here as they're discovered.*
