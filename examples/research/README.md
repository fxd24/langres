# Research examples

Experiment-tier scripts (harness code, not the library contract) that reproduce
langres milestones and probe methods. Run any of them with `uv run python
examples/research/<script>.py`; the ML-heavy ones want the OpenMP remedy
(`OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE`, or `uv run --env-file .env`).

## Autoresearch: the propose→run→eval→keep loop (epic #145)

**`blocking_recall_autoresearch.py`** — the E1 **proof that the loop closes**.

The autoresearch loop is a small hill-climber: **propose** a config, **run** it,
**evaluate** it into metrics, and **keep** it iff an `Objective` says it beats the
incumbent. Because ER F1 saturates near 99%, the loop steers on a *loss-like*
signal instead — here, continuous **recall@budget**:

```
maximize candidate_recall  subject_to  reduction_ratio >= 0.985
```

Point `langres.optimize` at a modest, frozen grid over the **amazon_google**
benchmark (2 metrics × 2 blocking texts × an ascending `k` sweep) and watch it
climb:

- **Recall@budget climbs.** `SearchSpace` sweeps `k_neighbors` innermost, so more
  neighbours surface more true pairs and the *incumbent* candidate_recall ratchets
  up trial by trial. The script prints the progress curve so the climb is visible.
- **The budget is a real gate.** The top-of-sweep configs spend ~2× the
  comparisons for no extra recall, breach the reduction-ratio floor, and are
  **rejected** — the loop keeps the best *feasible* incumbent. That is the
  recall-vs-cost tradeoff, made explicit (not a saturated F1 threshold).
- **Every trial is logged off-git — accepted *and* rejected.** The loop persists
  each trial to a local, owned `RunStore` JSONL under `tmp/` (gitignored); the
  script reads it back to prove nothing was dropped.
- **$0 and offline.** Local `all-MiniLM-L6-v2` embeddings, no LLM; the benchmark
  is vendored. `k` is innermost, so it is only 4 embedding passes over the corpus.

```bash
uv run --env-file .env python examples/research/blocking_recall_autoresearch.py
# needs the [semantic] extra: uv sync --all-extras --no-extra finetune
```

Nothing generated is committed: the `RunStore` JSONL (and the optional
incumbent-recall PNG) land in the gitignored `tmp/autoresearch/`.
