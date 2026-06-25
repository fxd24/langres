---
paths:
  - "src/**"
---

# Architecture & Component Design

**The two-layer API, the design principles, and what "lightweight & composable"
means in practice.** Read before adding or refactoring a component under `src/`.

## The Two-Layer API

1. **High-Level (`langres.tasks`)**: Pre-built task runners for common use cases
   - Target: 80% of users
   - Examples: `DeduplicationTask`, `EntityLinkingTask`
   - Philosophy: Like scikit-learn's Pipeline

2. **Low-Level (`langres.core`)**: Composable primitives for custom pipelines
   - Target: 20% of users (advanced use cases)
   - Components: `Module`, `Blocker`, `Optimizer`, `Clusterer`, `Canonicalizer`
   - Philosophy: Like PyTorch's primitives

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

1. **Blockers**: Must implement candidate generation and schema normalization
2. **Flows (Modules)**: Must yield `PairwiseJudgement` objects
3. **Tasks**: Should compose Blocker + Flow + optional Optimizer
4. **All Components**: Should support both `.run()` and `.compile()` methods where appropriate

Add docstrings to all public methods, and include a usage example in `examples/`
for any new component.

## Common Patterns

### Task Implementation

```python
class SomeTask:
    def __init__(self, flow: Module, blocker: Blocker):
        self.flow = flow
        self.blocker = blocker

    def compile(self, gold_data, metric: str):
        """Optimize hyperparameters on gold data"""
        pass

    def run(self, data):
        """Execute the task on input data"""
        pass
```

### Flow (Module) Implementation

```python
class SomeFlow(Module):
    def forward(self, candidates):
        """Yield PairwiseJudgement for each candidate pair"""
        for pair in candidates:
            score = self._compute_similarity(pair)
            yield PairwiseJudgement(
                left_id=pair.left.id,
                right_id=pair.right.id,
                score=score,
                score_type="calibrated_prob"
            )
```
