# Future Requirements and Enhancements

This document tracks features and enhancements that are **out of scope for the POC** but should be considered for post-POC development based on user feedback and real-world usage.

---

## Interactive Example Exploration (Post-POC)

### Motivation

During POC development, we identified the need for **interactive exploration of diagnostic examples** beyond simple lists. Users want to:
- Rank failure examples by severity/importance
- Filter examples by various criteria (score range, text length, cluster size)
- Group examples by patterns (all long text, all abbreviations, etc.)
- Export examples to pandas DataFrame for custom analysis
- Visualize score distributions with examples overlaid

**Current State (POC)**: Simple `.diagnose()` method returns lists of examples
```python
diagnostics = report.diagnose(candidates, entities)
for ex in diagnostics.missed_matches[:10]:
    print(f"{ex.left_text} <-> {ex.right_text}")
```

**Desired Future State**: Rich exploration API
```python
explorer = report.explore(candidates, entities)

# Top-K ranking by severity
worst = explorer.missed_matches.top_k(10, by="baseline_similarity")
for ex in worst:
    print(f"{ex.left_text} <-> {ex.right_text} (severity: {ex.severity_score:.2f})")

# Filtering by criteria
long_text = explorer.missed_matches.filter(lambda ex: len(ex.left_text) > 100)
overlap_zone = explorer.separation_failures.in_range(score_min=0.4, score_max=0.6)

# Grouping by patterns
by_cluster = explorer.missed_matches.group_by("cluster_id")
for cluster_id, examples in by_cluster.items():
    print(f"Cluster {cluster_id}: {len(examples)} missed matches")

# Export for custom analysis
df = explorer.to_dataframe()
df.sort_values("severity_score", ascending=False).head(20)

# Visualization (optional, requires matplotlib)
explorer.plot_score_distribution(highlight_failures=True)
```

### Requirements

#### 1. **ExampleExplorer API**
Interactive object for exploring diagnostic examples with fluent API.

**Components**:
- `RankedErrorCollection`: Collection with `.top_k()`, `.filter()`, `.in_range()`, `.group_by()`
- `ExampleExplorer`: Entry point with category accessors (`.missed_matches`, `.separation_failures`, etc.)
- `SeverityCalculator`: Compute multidimensional severity scores

**Key Features**:
- Lazy evaluation (only compute when accessed)
- Chainable operations (`.filter().top_k()`)
- Pandas export for advanced analysis
- Optional visualization (matplotlib integration)

#### 2. **Severity Metrics**
Multidimensional scoring to rank "worst performers" by actionable criteria.

**Severity Dimensions by Error Type**:

**Missed Matches**:
- `baseline_similarity`: High similarity by other metrics (cosine, edit distance) → critical to fix
- `cluster_size`: Larger clusters → more impactful miss
- `text_length`: Longer text → may indicate embedder max_length issue
- `field_coverage`: Number of populated fields → data quality signal

**High-Scoring Non-Matches**:
- `score_distance`: Distance from decision threshold → how confident was wrong prediction
- `baseline_similarity`: Low similarity by other metrics → model bias indicator
- `domain_conflict`: Cross-domain pairs (e.g., "Apple Inc." vs "Apple Fruit") → semantic confusion

**Separation Failures** (overlap zone between true/false distributions):
- `distance_from_median`: How far from true/false medians → hardest cases
- `score_variance`: High variance → unstable predictions
- `baseline_agreement`: Agreement across multiple metrics → systematic vs random error

**Example Usage**:
```python
# Find the 10 most critical missed matches (high baseline similarity)
critical = explorer.missed_matches.top_k(10, by="baseline_similarity")

# These are pairs that SHOULD have matched (high string similarity)
# but the embedder missed them → indicates embedder limitation
```

#### 3. **Pattern Detection (Optional, Advanced)**
Auto-detect patterns in failures to generate insights.

**Pattern Examples**:
- "Missed matches are significantly longer than average (>100 chars)" → Try model with longer context
- "Separation failures contain abbreviations (Corp/Corporation, Ltd/Limited)" → Add abbreviation normalization
- "High-scoring non-matches are all cross-domain (e.g., locations vs people)" → Add domain classifier
- "False positives cluster in specific industry (e.g., all tech companies)" → Domain-specific tuning needed

**Implementation Approach**:
- Statistical tests (length distribution, field coverage)
- Text analysis (regex patterns, entity types)
- Clustering (group failures by embedding similarity)

**Caution**: Pattern detection requires heuristic maintenance and may produce false insights. Should be opt-in and clearly marked as "suggestions" not "facts."

#### 4. **Jupyter Integration**
Rich display in Jupyter notebooks for interactive exploration.

**Features**:
- `_repr_html_()` for pretty-printed examples with syntax highlighting
- Interactive widgets for filtering/sorting (ipywidgets)
- Inline visualizations (matplotlib/plotly)
- Progress bars for expensive operations (tqdm)

**Example**:
```python
# In Jupyter
explorer = report.explore(candidates, entities)
explorer  # Shows rich HTML table with top-10 failures, filter controls, charts
```

### Implementation Phases

**Phase 1: Core ExampleExplorer** (1-2 weeks)
- Implement `RankedErrorCollection` with `.top_k()`, `.filter()`
- Implement `ExampleExplorer` with category accessors
- Add `.explore()` method to `BlockerEvaluationReport`
- Add severity score fields to example models
- Tests (100% coverage)

**Phase 2: Severity Calculation** (1 week)
- Implement `SeverityCalculator` with multidimensional scoring
- Add baseline similarity computation (rapidfuzz, cosine)
- Add cluster size, text length metrics
- Integrate into example extraction
- Tests

**Phase 3: Advanced Features** (1-2 weeks, based on user demand)
- Pandas export (`.to_dataframe()`)
- Matplotlib visualization (`.plot_score_distribution()`)
- Pattern detection (opt-in)
- Jupyter rich display

**Phase 4: Documentation** (3-5 days)
- Create `docs/DIAGNOSTICS.md` with examples
- Add to `examples/compare_embedders_for_funders.py`
- Video tutorial showing interactive exploration

### Success Criteria

**Must Have**:
- Users can rank failures by severity in <5 lines of code
- Top-K API is discoverable (IDE autocomplete)
- Performance: <100ms to rank 1000 examples
- Memory: <50 MB for 10K examples with full metadata

**Nice to Have**:
- Pandas export for advanced users
- Jupyter rich display for exploratory analysis
- Pattern detection suggestions (with clear "experimental" label)

### User Feedback Questions

Before implementing, validate with users:
1. How often do you want to explore failures beyond top-10?
2. What sorting criteria matter most? (severity, similarity, cluster size, text length)
3. Do you use Jupyter? Would interactive widgets help?
4. What patterns would be most useful to auto-detect?
5. Would you export to pandas for custom analysis?

### Related Research

Comprehensive research on this topic was completed on 2025-11-16:
- **Direction 1**: Unified Report Architecture (`.with_diagnostics()` pattern)
- **Direction 2**: Component Methods (`.diagnose()` vs `.explore()`)
- **Direction 3**: Top-K Analysis (severity metrics, ranking algorithms)
- **Direction 4**: Consistency (inspect → evaluate → diagnose progression)
- **Direction 5**: Memory/Performance (candidates storage, generator patterns)

**Documents** (archived in `.agent/` for future reference):
- `20251116_unified_report_architecture_analysis.md` - Pattern B (Report Builder)
- `TOP_K_FAILURE_ANALYSIS_ARCHITECTURE.md` - Severity metrics taxonomy
- `TOP_K_IMPLEMENTATION_GUIDE.md` - Ready-to-implement code (1,500 LOC)
- `MEMORY_AND_PERFORMANCE_OPTIMIZATION.md` - Performance benchmarks
- Full research: ~10,000 lines across 15 documents

**Key Findings**:
- Memory is not a constraint (examples: 27 KB per 100)
- Pattern B (`.with_diagnostics()`) is Pydantic-compatible
- Multidimensional severity is superior to single score
- Generator patterns for examples (memory-efficient)

**Recommendation**: When implementing, start with research documents as reference architecture.

---

## Auto-Generated Insights (Future, Maybe)

### Motivation

Users may benefit from automatic pattern detection in failures, similar to how AutoML tools suggest hyperparameters.

**Example Insights**:
- "Missed matches have 2.5x longer text than average → Try model with max_length > 512"
- "80% of separation failures contain abbreviations → Add abbreviation expansion"
- "High-scoring non-matches are all from cluster X → Check for entity type confusion"

### Concerns

**Maintenance Burden**:
- Heuristics become stale as models evolve
- False positives erode trust
- Hard to test objectively ("is this a good insight?")

**Recommendation**: Only implement if user research shows strong demand. If implemented:
- Mark as "experimental suggestions" not "facts"
- Allow users to disable
- Provide confidence scores
- Log when suggestions are followed/ignored (feedback loop)

---

## Advanced Visualization (Future, Optional)

### Motivation

Visual exploration of score distributions, example clusters, and failure patterns.

### Features

**Score Distribution Plots**:
```python
explorer.plot_score_distribution(
    highlight_failures=True,  # Overlay missed matches
    show_threshold=0.5,       # Mark decision boundary
)
```

**Example Cluster Visualization**:
```python
# Project examples into 2D space (UMAP/t-SNE)
explorer.plot_example_clusters(
    method="umap",
    color_by="error_type",  # missed_match, false_positive, etc.
)
```

**Rank Distribution**:
```python
# Show where true matches rank in top-K
explorer.plot_rank_distribution()
```

### Implementation

- Use matplotlib for static plots (no dependencies)
- Optional plotly for interactive (requires extra install)
- Integrate with Jupyter `_repr_html_()` for auto-display

### Recommendation

Wait for user feedback - many users prefer exporting to pandas and using their own viz tools.

---

## Multi-Component Diagnostics (Future, V2)

### Motivation

Users may want unified diagnostics across Blocker + Module + Clusterer.

**Example**:
```python
# Current (POC): Component-level diagnostics
blocker_diag = blocker_report.diagnose(...)
module_diag = module_report.diagnose(...)  # When Module.evaluate() exists
cluster_diag = clusterer_report.diagnose(...)

# Future: Pipeline-level diagnostics
pipeline_diag = pipeline.diagnose(
    candidates=candidates,
    judgements=judgements,
    clusters=clusters,
    gold_clusters=gold_clusters,
)

# Trace failures through pipeline
pipeline_diag.trace_failure(entity_id="E123")
# Shows:
# 1. Blocker: Found 20 candidates (ranked #5, score 0.85)
# 2. Module: Scored 0.65 (below 0.7 threshold)
# 3. Clusterer: Not merged (threshold 0.7)
```

### Recommendation

**Defer until post-POC**. Current `PipelineDebugger` serves this purpose. Re-evaluate once:
- Module.evaluate() exists
- Clusterer returns Pydantic reports
- Users report needing pipeline-level tracing

---

## Integration with Optimization (V2+)

### Motivation

Diagnostics could inform optimization by prioritizing fixes for high-impact failures.

**Example**:
```python
# Use diagnostics to weight optimization
optimizer = Optimizer(
    metric="bcubed_f1",
    focus_on=diagnostics.worst_failures(n=100),  # Optimize for hardest cases
)
```

### Recommendation

**Far future** - requires Optimizer implementation first (not in POC scope).

---

## Summary

**Priority 1 (Post-POC, High Confidence)**:
- Interactive Example Exploration (`ExampleExplorer` with `.top_k()`, `.filter()`)
- Severity Metrics (multidimensional ranking)
- Pandas export

**Priority 2 (Post-POC, Medium Confidence)**:
- Jupyter rich display
- Basic visualizations (score distribution)

**Priority 3 (V2+, Low Confidence - Needs User Validation)**:
- Auto-generated insights
- Advanced visualization (cluster plots, UMAP)
- Multi-component pipeline diagnostics
- Integration with Optimizer

**Decision Gates**:
1. Complete POC successfully (BCubed F1 ≥ 0.85)
2. Gather user feedback on `.diagnose()` method
3. Validate demand for interactive exploration
4. Implement Priority 1 features
5. Re-evaluate Priority 2-3 based on usage data

---

## Related Documentation

- `docs/POC.md` - Current POC scope and success criteria
- `docs/ISSUES.md` - Known issues and gotchas
- `docs/TECHNICAL_OVERVIEW.md` - Component APIs
- `.agent/` - Comprehensive research on diagnostic architectures (2025-11-16)
