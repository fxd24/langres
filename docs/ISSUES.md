# Known Issues and Gotchas

This document tracks known issues, common pitfalls, and design limitations in langres.

## Blocker Evaluation

### Issue: Fixed k in VectorBlocker Limits Recall Curve Accuracy

**Status**: Known limitation (2025-11-16)

**Problem**:

When evaluating a `VectorBlocker`, the recall curve can only be computed accurately up to the `k_neighbors` value used during candidate generation. This creates a mismatch between:

1. **Candidate generation k**: The `k_neighbors` parameter in `VectorBlocker.__init__()`
2. **Evaluation k_values**: The `k_values` parameter in `evaluate_blocker_detailed()` (default: `[1, 5, 10, 20, 50]`)

If you generate candidates with `k_neighbors=20` but evaluate with `k_values=[1, 5, 10, 20, 50]`, the recall@50 measurement is **not accurate** because you only retrieved 20 neighbors per entity.

**Example of the problem**:

```python
# Generate candidates with k=20
blocker = VectorBlocker(
    schema_factory=...,
    text_field_extractor=...,
    vector_index=index,
    k_neighbors=20,  # ⚠️ Only retrieves 20 neighbors per entity
)
candidates = list(blocker.stream(entities))

# Evaluate with default k_values=[1, 5, 10, 20, 50]
report = blocker.evaluate(candidates, gold_clusters)

# ❌ Problem: recall@50 is computed on only 20 candidates
# The optimal_k calculation will be inaccurate if true optimal k > 20
optimal_k = report.recall_curve.optimal_k(target_recall=0.95)
print(f"Optimal k: {optimal_k}")  # Might say 50, but we never tested k=50!
```

**Impact**:

- **Recall curve**: Values for k > k_neighbors are capped at recall@k_neighbors
- **Optimal k calculation**: Will be wrong if the true optimal k exceeds k_neighbors
- **Model comparison**: May incorrectly rank models if they differ in performance at higher k

**Other metrics are NOT affected**:
- MAP, MRR, NDCG: Computed on all retrieved candidates, still valid
- Separation, score distributions: Computed on all candidates, still valid
- Rank distribution: Computed on all candidates, still valid

**Solution**:

Set `k_neighbors` to be **at least as large** as the maximum k you want to evaluate:

```python
# ✅ Correct approach
MAX_K_TO_EVALUATE = 100  # Want to compute recall@100

blocker = VectorBlocker(
    schema_factory=...,
    text_field_extractor=...,
    vector_index=index,
    k_neighbors=MAX_K_TO_EVALUATE,  # Retrieve enough candidates
)
candidates = list(blocker.stream(entities))

# Now recall curve is accurate up to k=100
report = blocker.evaluate(candidates, gold_clusters, k_values=[1, 5, 10, 20, 50, 100])
```

**Trade-offs**:

- **Higher k_neighbors**:
  - ✅ More accurate recall curve
  - ✅ Better optimal k estimation
  - ❌ Slower candidate generation (more vector search operations)
  - ❌ More memory usage (more candidates stored)

- **Lower k_neighbors**:
  - ✅ Faster candidate generation
  - ✅ Less memory usage
  - ❌ Recall curve capped at k_neighbors
  - ❌ May miss the true optimal k

**Recommendation**:

For **model comparison** (e.g., `examples/compare_embedders_for_funders.py`):
- Use `k_neighbors=100` or higher to get a complete picture
- This ensures optimal_k calculations are accurate
- The extra cost is worth it for proper model selection

For **production deployment**:
- Use the optimal k determined from evaluation
- No need to retrieve extra candidates beyond what you'll use

**Related Files**:
- `examples/compare_embedders_for_funders.py`: Multi-model comparison example
- `src/langres/core/analysis.py`: `_compute_recall_curve()` function (lines 309-394)
- `src/langres/core/blockers/vector.py`: `VectorBlocker` implementation

---

## Embedding Model Names

### Issue: Not All Model Names Are Valid on HuggingFace

**Status**: User error / Documentation needed (2025-11-16)

**Problem**:

Some embedding model names that seem reasonable don't actually exist on HuggingFace. For example:

❌ `intfloat/multilingual-e5-medium` - Does NOT exist
✅ `intfloat/multilingual-e5-base` - Exists (768d)
✅ `intfloat/multilingual-e5-small` - Exists (384d)
✅ `intfloat/multilingual-e5-large` - Exists (1024d)

**Error Message**:

```
OSError: intfloat/multilingual-e5-medium is not a local folder and is not a valid model identifier listed on 'https://huggingface.co/models'
```

**Solution**:

Always verify model names on HuggingFace before using them:
- Visit https://huggingface.co/models
- Search for the model family (e.g., "multilingual-e5")
- Use the exact model identifier from HuggingFace

**Common Model Series**:

**Sentence-Transformers (all-*):**
- `all-MiniLM-L6-v2` (384d, 22M params)
- `all-mpnet-base-v2` (768d, 109M params)
- `paraphrase-MiniLM-L3-v2` (384d, 17M params)

**Multilingual E5 (intfloat):**
- `intfloat/multilingual-e5-small` (384d, 118M params)
- `intfloat/multilingual-e5-base` (768d, 278M params)
- `intfloat/multilingual-e5-large` (1024d, 560M params)

**BGE Series (BAAI):**
- `BAAI/bge-small-en-v1.5` (384d, 33M params)
- `BAAI/bge-base-en-v1.5` (768d, 109M params)
- `BAAI/bge-large-en-v1.5` (1024d, 335M params)
- `BAAI/bge-m3` (1024d, 568M params, multi-lingual)

---

## Future Issues

<!-- Add new issues here as they're discovered -->
