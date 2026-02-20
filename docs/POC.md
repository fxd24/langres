# langres: Proof of Concept
## TODO Add as well the idea of training a classifier (classical) after the blocker to decide whether is a match or not.

## Purpose

Before committing to the full langres architecture, we must validate the core hypothesis: **that a composable, low-level API built on Pydantic data contracts can effectively support entity resolution strategies ranging from classical string matching to LLM-based judgment, with full observability and optimization-readiness**.

This POC focuses exclusively on the `langres.core` layer—the foundational primitives that will underpin all higher-level abstractions.

## The Hypothesis

The langres architecture proposes three key innovations:

1. **Separation of Concerns**: Blocker (candidate generation + schema normalization) and Module (comparison logic) as distinct, composable primitives
2. **Universal Data Contracts**: `ERCandidate` and `PairwiseJudgement` as the type-safe interfaces between all components
3. **Provenance-First Design**: Every decision carries full metadata for observability and future optimization

**What we need to prove**: These abstractions work in practice, not just in theory.

## Three Approaches to Validate

We will implement three progressively sophisticated entity resolution strategies on a canonical task: **company deduplication**.

### Approach 1: Classical String Matching
**Pure rapidfuzz with tunable thresholds**

This baseline validates:
- The `Module.forward()` abstraction feels natural for rule-based logic
- `PairwiseJudgement` provenance captures decision metadata
- The architecture doesn't add unnecessary complexity for simple cases

**Why this matters**: If our framework makes simple things harder, it fails the "worse is better" test.

### Approach 2: Semantic Vector Search
**Embedding-based blocking with cosine similarity matching**

This intermediate approach validates:
- The Blocker/Module separation clarifies responsibilities
- ANN-based candidate generation integrates cleanly
- Schema normalization in Blocker keeps Module logic portable

**Why this matters**: This is the modern ER baseline—embeddings + approximate nearest neighbor search. Our framework must make this pattern trivial.

### Approach 3: Hybrid Blocking + LLM Judge ⭐
**Vector blocking for recall, LLM judgment for precision**

This target architecture validates:
- Cascade patterns (cheap checks → expensive LLM) fit the Module abstraction
- Cost and reasoning tracking work via provenance
- The system can achieve competitive accuracy (≥0.85 BCubed F1)

**Why this matters**: This is the architecture langres is designed to enable—composing classical, semantic, and LLM-based techniques with full observability and cost control.

## What Success Looks Like

### Functional Requirements
- All three approaches run end-to-end on the same test data
- Swapping Module implementations requires minimal code changes
- Every `PairwiseJudgement` contains actionable provenance

### Quality Requirements
- **Approach 1**: Baseline performance (BCubed F1 ≥ 0.70)
- **Approach 2**: Semantic improvement (BCubed F1 ≥ 0.75)
- **Approach 3**: Competitive with SOTA (BCubed F1 ≥ 0.85)

### Architectural Requirements
- Clear extension points for future Optimizer integration
- Data contracts are ergonomic (not fighting Pydantic)
- Blocker recall ≥0.95 (ANN search doesn't miss true matches)

## Core Components to Build

### Data Contracts (Foundation)
- `ERCandidate[SchemaT]`: The normalized pair passed to Modules
- `PairwiseJudgement`: The rich decision output with provenance
- `CompanySchema`: Domain model for test data

### Core Primitives
- `Module`: Abstract base for all comparison logic
- `Blocker`: Abstract base for candidate generation + normalization
- `Clusterer`: Graph-based entity formation (transitive closure)

### Concrete Implementations
- **Approach 1**: `RapidfuzzModule`
- **Approach 2**: `VectorBlocker`, `EmbeddingModule`
- **Approach 3**: `LLMJudgeModule`, `CascadeModule`

### Evaluation Infrastructure
- BCubed F1 metric (cluster-level quality)
- Synthetic company dataset (500 pairs, challenging cases)
- Cost tracking (API calls, USD per decision)

## Development Approach: Test-Driven

We will follow strict TDD discipline:

1. **Write tests first**: Define expected behavior via tests
2. **Watch them fail**: Confirm tests catch the missing implementation
3. **Implement minimally**: Write just enough code to pass
4. **Refactor**: Clean up while staying green

**Rationale**: This POC is not throwaway code. These components become the production library. TDD ensures we build a solid foundation with 100% test coverage.

## Key Questions This POC Answers

1. **Composability**: Can we swap Modules without touching Blocker or Clusterer?
2. **Observability**: Does `PairwiseJudgement` provide sufficient provenance for debugging and optimization?
3. **Performance**: Does the cascade pattern in Approach 3 reduce cost while maintaining quality?
4. **Extensibility**: Are the optimization hooks (tunable parameters, prompt templates) clear from the implementation?
5. **Ergonomics**: Do the Pydantic data contracts help or hinder development?

## What This POC Is Not

- ❌ **Not a prototype to be discarded**: This is production code with TDD rigor
- ❌ **Not the full framework**: No `langres.tasks`, no `Optimizer`, no `SyntheticGenerator`
- ❌ **Not a benchmark competition**: We validate feasibility, not SOTA optimization
- ❌ **Not feature-complete**: No HITL, no canonicalization, no record linkage

## Go/No-Go Decision

**After implementing all three approaches:**

### Go → Build Full Framework
If:
- Approach 3 achieves ≥0.85 BCubed F1
- Component swapping is trivial (high composability)
- Provenance enables post-hoc analysis and optimization
- TDD code quality is production-ready

### No-Go → Iterate on Architecture
If:
- Quality targets not met (framework overhead kills performance)
- Abstractions feel unnatural (fighting the framework)
- Data contracts are too rigid or too loose
- Optimization hooks are unclear

## Timeline

**4 weeks, focused sprints:**

- **Week 1**: Data contracts + Approach 1 (classical baseline)
- **Week 2**: Blocking primitives + Approach 2 (semantic)
- **Week 3**: LLM integration + Approach 3 (hybrid)
- **Week 4**: Evaluation, benchmarking, documentation

**Deliverable**: `docs/POC_RESULTS.md` with findings and recommendation

## Why This Matters

Entity resolution is a mature field with established tools (Dedupe.io, recordlinkage) and commercial systems (Tamr, AWS Entity Resolution). langres only justifies its existence if:

1. **The two-layer API actually reduces complexity** (vs. monolithic config-driven systems)
2. **Composability enables rapid experimentation** (swap Modules, not rewrite pipelines)
3. **Full observability enables optimization** (Optuna/DSPy can interrogate provenance)
4. **The framework grows with user sophistication** (tasks → core as needs evolve)

**This POC validates these assumptions before we build the full system.**

## Conclusion

This is not a speculative prototype. This is the foundation layer of langres, built with production-quality TDD from day one.

The three approaches represent the spectrum of modern entity resolution: classical (rapidfuzz), semantic (embeddings), and hybrid (LLM). If our core abstractions serve all three elegantly, we've validated the architecture.

**Success means**: Proceed to `langres.tasks`, `Optimizer`, and the full two-layer API with confidence.

**Failure means**: Iterate on `langres.core` abstractions before scaling up.

The POC is the critical validation phase—invest 4 weeks now to avoid 6 months of architectural regret.
