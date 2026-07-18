---
paths:
  - "src/**"
---

# Architecture & Component Design

**The layered API, the design principles, and what "lightweight & composable"
means in practice.** Read before adding or refactoring a component under `src/`.

## The Layered API (architectures → ERModel → core)

langres exposes three layers, each a thin shell over the one below. (Note: there
is **no** `langres.tasks` or `langres.flows` module — those names were never
built. There is also no `langres.link` / `langres.dedupe` module-level verb and
no `matcher="auto"` — W4 deleted both outright, along with `core.presets`: naming
a model is the user's job, not a heuristic's. The real layering is:)

1. **Named architectures (`langres.architectures`)** — the top DX layer.
   - Target: most users. A whole ER pipeline is a class you construct —
     `FuzzyString()` (all-pairs blocking + string similarity, the **$0, offline,
     no-key architecture** — it has no paid model slot, so it cannot spend) or
     `VectorLLMCascade(llm=...)` (vector blocking + a cheap embedding student +
     an LLM escalated only at the margin — **paid, because you constructed it**,
     never because a heuristic sniffed an environment variable). `.dedupe(records)`
     and `.compare(a, b)` are methods on the model, not module-level functions.
   - Both accept `budget_usd=` (default `DEFAULT_BUDGET_USD`) and cap real spend;
     `FuzzyString` can never trip the cap since it has no paid slot to spend from.
     A breach raises `BudgetExceeded` (root-exported) carrying the partial
     judgements.
   - Returns self-describing results (`LinkVerdict`; a `DedupeResult` carrying
     `architecture` — the model class that ran — and `backbone` — the LLM id /
     embedder name, or `None` when nothing with weights ran — plus `score_type`).
   - Philosophy: the one-liner front door, made explicit instead of a heuristic
     string switch. Thin sugar over `ERModel`. `compare` is deliberately not
     named `link`: that name is reserved for `ERModel.link` (cross-source, two
     record *sets* — a reserved M5 stub, see below).

2. **`langres.Resolver`** (a plain alias of `ERModel`; `Resolver is ERModel`) — the declarative mid-layer.
   - `Resolver.from_schema(schema, matcher=...)` builds a default dedup pipeline
     (blocker + comparator + matcher + clusterer); `.resolve(records)` runs it;
     `.save`/`.load` serialize it via the config-registry (no pickle).
   - **Spend-capped too** (B1), not just the named architectures above (they
     share the same `ERModel` machinery): `budget_usd=` on the
     constructor and `from_schema`, defaulting to `DEFAULT_BUDGET_USD`. The
     `SpendMonitor` is built ONCE per instance, so N `resolve()` calls share one
     budget instead of getting a fresh one each. `None` = the default, **not**
     uncapped (`spend_cap.UNCAPPED_BUDGET_USD` is the deliberate opt-out).
     The cap wraps at *scoring* time and never sits in the `module` slot, so
     `fit()`'s isinstance checks and `save()` still see the raw matcher.
   - **Scoring through the matcher? Go through `Resolver._scorer()`.** Because
     the slot stays raw, `self.module` is a *public, uncapped* scorer: calling
     `.module.forward(...)` bills past `budget_usd` and reports to no ledger.
     That is not theoretical — `AnchorStore._judge` did exactly this and spent
     uncapped. `_scorer()` is the ONE seam (`resolve` / `predict` / `fit` /
     `AnchorStore.assign` all use it); `tests/core/test_resolver_spend_cap.py`
     AST-bans `<any>.module.forward(...)` in `src/`. Note it returns a *fresh*
     wrapper around the CURRENT `self.module` sharing one long-lived
     `SpendMonitor` — the monitor is the durable thing; caching the wrapper
     would pin a stale matcher (`dedupe` re-wraps the slot after construction).
   - `Resolver.link` / `stream_against` are reserved `NotImplementedError` stubs
     (M5 incremental/cross-source work) — do not document them as working.

3. **Low-Level (`langres.core`)**: Composable primitives for custom pipelines.
   - Target: advanced users building bespoke pipelines.
   - Real components: `Matcher` (judge), `Blocker` (`AllPairsBlocker`,
     `VectorBlocker`), `Comparator` (`StringComparator`), `Clusterer`, plus
     matchers (`LLMMatcher`, `EmbeddingScoreMatcher`, `WeightedAverageMatcher`,
     `CascadeMatcher` — cheap student + escalate-at-the-margin, …) and
     `core.calibration.derive_threshold`. The flywheel/labelling seam lives in
     the **`langres.curation`** package (`review`, `harvest`, `anchor_store`,
     `canonicalizer` + the dissolved `langres.bootstrap` cold-start): pick the
     uncertain margin with `curation.review.select_for_review` / `ReviewQueue`,
     harvest with `curation.harvest` (`harvest_labeled_pairs`,
     `Correction`/`CorrectionLog`, `derive_threshold_from_pairs`). The harvest/
     review symbols stay re-exported on the `langres.core` facade
     (`langres.core.select_for_review`, …); the deep `langres.core.{review,
     harvest,anchor_store,canonicalizer}` module paths are temporary W2-sweep
     back-compat shims re-exporting from curation.
   - **`langres.core` itself re-exports only the *contracts*** — the models, the
     `Blocker`/`Comparator`/`Matcher`/`Clusterer` base types, the opt-in
     capability Protocols (`Inspectable`, the `fit` mixins), the `Resolver` +
     registry, the method registry, training/tracking. The implementations are
     imported from the package that owns them:

     ```python
     from langres.core.blockers    import AllPairsBlocker, VectorBlocker
     from langres.core.comparators import StringComparator
     from langres.core.matchers    import CascadeMatcher, LLMMatcher
     from langres.core.clusterers  import CorrelationClusterer
     from langres.core.embeddings  import SentenceTransformerEmbedder
     from langres.core.indexes     import FAISSIndex
     ```

     The `Comparator` ABC vs `StringComparator` split (W1) is the worked example:
     a contract that imports its own implementation — even indirectly, via a
     factory like the old `Comparator.from_schema` — sits *above* the components
     that depend on it. Build the default via the impl's own factory
     (`StringComparator.from_schema`) instead.

     Re-exporting an implementation from `langres.core` puts the floor above the
     components it sits beneath and re-knots the import graph — `langres.core`
     must stay importable *by* the components, not the reverse. The ratchet in
     `tests/test_import_tangle.py` measures it.
   - Philosophy: Like PyTorch's primitives.
   - **Search is not core.** The autoresearch engine lives in
     `langres.autoresearch` (`objective`/`search_space`/`factory`/`loop` +
     `blocker_optimizer.BlockerOptimizer`) behind the `langres.optimize` facade,
     *outside* `core`: `core` is ER modelling, search is a consumer of it. The
     dependency is one-way and measured — `core → autoresearch` and
     `core → optimize` are both **0** edges in every kind; adding one would put
     search beneath the primitives it drives.
     The engine is a package and the facade is a *module* (`optimize.py`) on
     purpose: `langres.optimize` is a **callable** (`_exports/_optimize.py` binds
     the attribute to the function), so any submodule under that name is
     unreachable by attribute traversal — `import langres.optimize.loop as l`
     raises `ImportError`. Don't put the engine back under it.
   - **Not yet built** (roadmap, don't reference as existing): a general
     `Optimizer` (only `langres.autoresearch.blocker_optimizer.BlockerOptimizer`
     exists).

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
   `forward` is the **only** abstract method on `Matcher` — the contract is
   deliberately thin. Optional capabilities are opt-in, runtime-checkable
   structural Protocols, never abstract methods that every judge must stub out:
   score inspection is `core/inspection.py:Inspectable`
   (`inspect_scores`; the shared body stays in
   `core/reports.py:_inspect_scores_impl`, so opting in is a 2-line
   pass-through) and the trainable roles are the mixins in `core/fit.py`.
   Callers detect either with `isinstance(component, <Protocol>)`. Adding an
   `@abstractmethod` to `Matcher` is a breaking change to every judge in the
   repo *and* every user subclass — make the capability a Protocol instead.
   A *public, name-selectable* judge is registered **once**, in the single
   method registry (`core/method_registry.py` — the v0.3 unification that
   closed issue #55's three-site wiring debt): a `MethodSpec` carries the
   builder plus identity metadata (`score_type`, `default_threshold`,
   `default_model`, `accepted_kinds`, `needs_comparator`, `requires_extra`).
   Both remaining dispatch paths — `core/resolver.py:_build_module_for_judge`
   (`Resolver.from_schema`, the low-level `matcher=` switch) and
   `methods.py:_make_module_builder` (the benchmark harness) — resolve names
   through it, so a name means exactly one construction everywhere. (A third
   path, the verbs' `core/presets.py:build_judge`, existed pre-W4 and is gone —
   `matcher="auto"` sniffed an API key and spent on whatever it found; naming a
   model explicitly replaced it, not a better heuristic.) `from_schema`'s own
   name tuple stays a per-layer *policy*, not registration (no `"auto"`).
   Method ids are bare names; `/` is reserved for future `author/method`
   namespacing. Spec builders must lazy-import heavy deps
   (dspy/litellm/sklearn) inside the build function — the registry is
   eager-imported by `langres.core` (see `tests/test_import_budget.py`).
   A judge you only ever pass as a `Matcher` instance — e.g.
   `Resolver.from_schema(schema, matcher=MyJudge(...)).dedupe(records)`, or
   `ERModel(blocker=..., comparator=..., matcher=MyJudge(...), clusterer=...)`
   directly — needs none of this wiring. For `save`/`load`, the component
   config-registry (`core/registry.py:register`) remains a separate, orthogonal
   namespace.
3. **Composition happens in `ERModel`**, not a `Task` class: a resolve is
   blocker → (compare) → matcher → clusterer. The named architectures in
   `langres.architectures` (`FuzzyString`, `VectorLLMCascade`) are the
   user-facing sugar over it — each fixes its own topology and exposes
   `.dedupe()`/`.compare()` as methods.

## Backbones: `ModelRef` is the ONE model-reference concept

**Architecture = topology** (which components, in what order). **Backbone = what
fills a model slot.** Swapping a backbone must never mint a new architecture, so
a component never invents its own model-reference shape: it takes a
`langres.core.model_ref.ModelRef` (via `normalize_model_ref`, which accepts a
plain string, a dict, or a ref).

`ModelRef` is a stdlib-only leaf (it imports nothing from `langres`), frozen,
validated in `__post_init__`, and **weightless** — reference strings only, so it
round-trips as JSON config via `to_config`. Its fields: `base`, `kind`,
`adapter`, `adapter_revision`, `api_base`, `revision`. Base and adapter Hub
artifacts carry independent revision pins.

**`kind` is the discriminator, and routing reads nothing else:**

| `kind` | `base` names | runs |
|---|---|---|
| `api` | a litellm id (`openai/gpt-4o`, `gpt-5-mini`) | served (litellm) |
| `endpoint` | a model served at `api_base` | served (litellm) |
| `hf` | a Hugging Face Hub id (`org/name`) | in-process |
| `local` | a local directory path | in-process |

Rules that are load-bearing, each for a measured reason:

- **Never route on the filesystem** (B17). `backend_for(kind)` is the whole rule.
  The predecessor probed `os.path.isdir(base)`, so the same saved config resolved
  to a *different backend in a different working directory* (reproduced: a
  `./gpt-5-mini` directory flipped that API id litellm → transformers). A path is
  recognized by **syntax** (`./`, `../`, `/`, `~`) — so a bare relative dir name is
  an API id, not a path. Inference is total over non-empty strings: a multi-slash
  id is never a Hub id (those carry one slash), so it goes to `api` and litellm —
  which owns the real provider list — reports any error.
- **`revision` pins an `hf` ref** (B16). Without it `org/name` drifts as the Hub
  moves and an "identical versioned config" is not identical across time.
- **Don't guess a provider typo.** `org/name` cannot be disambiguated from a
  typo'd provider by syntax: the real org `mistralai` scores 0.875 against the
  `mistral` provider while the typo `opeani`→`openai` scores 0.833, so no difflib
  cutoff exists. A caller who means an API model names `kind="api"`.
- **DSPy-backed slots are litellm-only** (B10). DSPy routes *every* completion
  through litellm (`lm.py:forward` → `litellm_completion`); `lm_local` shells out
  to sglang and points litellm back at localhost. So `require_litellm_routable`
  rejects local dirs and adapters at construction. It deliberately admits `hf`:
  litellm knows 146 providers and the prefix table 26, so 120 real provider ids
  infer as `hf` and rejecting them would break working code.

A method declares which backbones it can run via `MethodSpec.accepted_kinds`, and
`check_backbone` enforces it on every dispatch path — an empty set means "no model
slot", so `model=` raises rather than being silently dropped. See
`docs/ADDING_A_METHOD.md`.

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
                score=score,                    # a *ranker*: the caller's threshold decides
                score_type="calibrated_prob",   # one of the models.py literals (a required
                                                # family tag even when score is None)
                decision_step="some_judge",
                provenance={},
            )
```

A judge is a *ranker* (emits `score`, threshold decides) **or** a *decider* (a
binary Yes/No judge: emit `decision=True/False` and leave `score=None` — a
fabricated `0.0`/`1.0` would lie). Setting neither is an **abstention**
(`is_abstain`). Never test `score >= threshold` yourself — ask
`langres.core.predicted_match(judgement, threshold)`, the one place that answers
"is this a match" (decision wins over score; an abstention returns `None`, and is
excluded from the predicted set — never graded a confident "no").

### Wiring it into a pipeline

```python
# Low-level: build a Resolver from a schema, pick the matcher, run it.
resolver = Resolver.from_schema(MySchema, matcher="string")
clusters = resolver.resolve(records)   # -> list[set[str]]

# High-level: a named architecture does the same thing with schema inference +
# spend cap -- no "auto" string to opt out of, because there is no key-sniffing
# default: FuzzyString cannot spend at all (it has no paid model slot).
from langres.architectures import FuzzyString
result = FuzzyString().dedupe(records)   # $0, offline, no key
```
