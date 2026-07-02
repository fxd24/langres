---
paths:
  - "src/**"
---

# Architecture & Component Design

**The layered API, the design principles, and what "lightweight & composable"
means in practice.** Read before adding or refactoring a component under `src/`.

## The Layered API (verbs → Resolver → core)

langres exposes three layers, each a thin shell over the one below. (Note: there
is **no** `langres.tasks` or `langres.flows` module — those names were never
built. The real layering is:)

1. **Verbs (`langres.link` / `langres.dedupe`)** — the top DX layer.
   - Target: most users. Schema-optional, zero-label, `judge="auto"` with a
     default spend cap.
   - Returns self-describing results (`LinkVerdict`; a `dedupe` result carrying
     `judge_used` / `score_type` / `fallback_reason`).
   - Philosophy: the one-liner front door. Thin sugar over `Resolver`.

2. **`langres.Resolver`** — the declarative mid-layer.
   - `Resolver.from_schema(schema, judge=...)` builds a default dedup pipeline
     (blocker + comparator + judge + clusterer); `.resolve(records)` runs it;
     `.save`/`.load` serialize it via the config-registry (no pickle).
   - `Resolver.link` / `stream_against` are reserved `NotImplementedError` stubs
     (M5 incremental/cross-source work) — do not document them as working.

3. **Low-Level (`langres.core`)**: Composable primitives for custom pipelines.
   - Target: advanced users building bespoke pipelines.
   - Real components: `Module` (judge), `Blocker` (`AllPairsBlocker`,
     `VectorBlocker`), `Comparator` (`StringComparator`), `Clusterer`, plus
     judges (`LLMJudge`, `EmbeddingScoreJudge`, `WeightedAverageJudge`, …) and
     `core.calibration.derive_threshold`.
   - Philosophy: Like PyTorch's primitives.
   - **Not yet built** (roadmap, don't reference as existing): `Canonicalizer`,
     a general `Optimizer` (only `optimizers.BlockerOptimizer` exists),
     constrained `Clusterer`, set-wise / trained judge families.

## Key Design Principles

- **Pydantic-First**: All data models use Pydantic for validation
- **Full Observability**: Every `PairwiseJudgement` carries provenance and reasoning
- **Composable**: Components should be reusable across different tasks
- **Optimizable**: Support both hyperparameter tuning (Optuna) and prompt optimization (DSPy)
- **Cost-Aware**: Consider API costs, computation costs, and optimization budgets

## Component Design: Lightweight & Composable

**langres is "lightweight and composable" - but what does that mean in practice?**

### Single Responsibility Principle (SRP)
Each component should have **ONE reason to change**. If you need "and" to describe what a class does, it's doing too much.

**Example:**
- ❌ Bad: "VectorBlocker normalizes schema AND extracts text AND generates embeddings AND builds indexes AND searches"
- ✓ Good: "VectorBlocker orchestrates candidate generation by delegating to injected services"

### Lightweight = Single Abstraction Level
A component is lightweight when it:
- Has **≤3 constructor dependencies** (more suggests multiple responsibilities)
- Operates at **single abstraction level** (don't mix high-level orchestration with low-level library calls)
- Is **≤200 lines per class** (not a hard rule, but a warning sign)
- Can be described **without "and"** in a single sentence

**Red flags for over-complex components:**
- Importing from multiple domains (e.g., `faiss` AND `transformers` AND `networkx` in same class)
- Hard to test (must mock concrete libraries like SentenceTransformer)
- Mixed abstractions (Blocker directly calling `faiss.IndexFlatL2()` instead of `VectorIndex.add()`)

### Composition Patterns: Extract Helper Classes
When a component handles distinct technical concerns, extract them:

```python
# Instead of VectorBlocker doing everything:
class VectorBlocker:
    def __init__(self, ..., model_name, ...):
        self.model = SentenceTransformer(model_name)  # ❌ Direct dependency

# Extract services and inject them:
class EmbeddingService:
    """Helper: Only generates embeddings."""
    def encode_batch(self, texts: list[str]) -> np.ndarray: ...

class VectorBlocker:
    def __init__(self, ..., embedding_service: EmbeddingService, ...):
        self.embedding_service = embedding_service  # ✓ Injected dependency
```

**Benefits of extraction:**
- ✓ Single responsibility (EmbeddingService only does embeddings)
- ✓ Testable (mock interface, not concrete library)
- ✓ Reusable (use same EmbeddingService in Module)
- ✓ Swappable (try different embedding models)

### When to Extract Helper Classes
Extract when you see:
1. **Multiple technical libraries**: Same class imports `faiss` AND `transformers`
2. **Hard to test**: Must mock concrete libraries (SentenceTransformer, FAISS)
3. **Reuse potential**: Logic needed in multiple places (embeddings in Blocker AND Module)
4. **Multiple "and"s**: "Does schema normalization AND text extraction AND embedding AND indexing"

**When NOT to extract:**
- Truly trivial (1-2 line lambda)
- No reuse (used once, unlikely to change)
- Already simple (meets lightweight criteria)

**📋 See `.agent/component-design-principles.md` for comprehensive guidance**, including:
- Complete SRP examples with before/after code
- Decision framework for when to extract helper classes
- VectorBlocker case study showing proper decomposition
- Composition patterns (service classes, strategy pattern, factories)
- Common anti-patterns and how to avoid them
- Component design checklist

## When Adding New Components

1. **Blockers**: Must implement candidate generation (`stream`) and schema
   normalization.
2. **Judges (Modules)**: Must yield `PairwiseJudgement` objects from `forward`.
   There is **no single registration seam yet** — a new *public, name-selectable*
   judge must be wired into **all three** dispatch sites, or it will be rejected
   by whichever path doesn't know it:
   - `methods.py:_make_module_builder` — the benchmark / method-registry path.
   - `core/resolver.py:_build_module_for_judge` — what `Resolver.from_schema(judge=...)`
     dispatches on.
   - `core/presets.py:build_judge` — what the verbs (`link` / `dedupe`, incl.
     `"auto"`) dispatch on.
   (A single public method-registration API that collapses these is deferred to
   issue #55; see `TODOS.md`.) A judge you only ever pass as a `Module` instance
   — `dedupe(records, judge=MyJudge(...))` — needs none of this wiring.
3. **Composition happens in `Resolver`**, not a `Task` class: a resolve is
   blocker → (compare) → judge → clusterer. The verbs (`link` / `dedupe`) are
   the user-facing sugar over it.

Add docstrings to all public methods, and include a usage example in `examples/`
for any new user-facing component.

## Common Patterns

### Judge (Module) Implementation

```python
class SomeJudge(Module):
    def forward(self, candidates):
        """Yield a PairwiseJudgement for each candidate pair."""
        for pair in candidates:
            score = self._compute_similarity(pair)
            yield PairwiseJudgement(
                left_id=pair.left.id,
                right_id=pair.right.id,
                score=score,
                score_type="calibrated_prob",  # one of the models.py literals
                decision_step="some_judge",
                provenance={},
            )
```

### Wiring it into a pipeline

```python
# Low-level: build a Resolver from a schema, pick the judge, run it.
resolver = Resolver.from_schema(MySchema, judge="string")
clusters = resolver.resolve(records)   # -> list[set[str]]

# High-level: the verbs do the same thing with schema inference + spend cap.
from langres import dedupe
result = dedupe(records)               # judge="auto" by default
```
