# Changelog

## [Unreleased] - POC Phase

- Designed two-layer API architecture and POC validation plan (3 approaches: classical, semantic, LLM hybrid)
- Implemented core primitives (`Module`, `Blocker`, `Clusterer`) with Pydantic data contracts and 100% test coverage
- Completed Approach 1 (classical baseline): `AllPairsBlocker` + `RapidfuzzModule` end-to-end pipeline

### M1: Cold-start gold-set bootstrapping (LLM-teacher)

Reusable, entity-type-agnostic `langres.bootstrap` package that mines hard-negative
candidate pairs from a blocker, labels them with a budget-capped LLM teacher, and
emits a versioned gold set + coverage/calibration report. Validated on the
**Fodors-Zagat** restaurant benchmark (864 records / 112 cross-source matches).

- **Data contract + adapter**: `GoldPair`/`GoldSet` (versioned Pydantic, JSON
  save/load), `RestaurantSchema` (computed `embed_text`), `load_fodors_zagat`,
  and a blocking k-sweep that pins `DEFAULT_BLOCKING_K=5` (Pair-Completeness 0.9911).
- **Mining + labeling**: `HardNegativeMiner` (three-stratum similarity sampling),
  `TeacherLabeler` (hard $20 budget cap via pre-flight pair cap + per-pair token
  tally + blind-cost abort, `enable_langfuse=False` client), plus `GroundTruth`/`Fake`
  labelers for deterministic, zero-spend CI runs.
- **Metrics + report**: added `cohens_kappa`, `matthews_corrcoef`, `brier_score`,
  `expected_calibration_error` (equal-mass bins), `reliability_bins` to `core.metrics`;
  `BootstrapReport` covers Pair-Completeness, teacher-vs-truth agreement (F1/kappa/MCC),
  calibration (Brier/ECE of P(match) vs is-match), and an agreement-convergence curve.
- **`Bootstrapper`** orchestrator wires blocker → cross-source filter → miner → labeler
  → gold set + report; deterministic real-embedding example + slow CI test.
- **EXIT (real GLM-5.2 teacher run, $1.28)**: 1382-pair gold set committed at
  `data/gold_sets/fodors_zagat/`; Pair-Completeness 0.9911. Teacher-vs-truth over
  the **full** cross-source band (n=1382, closed-world truth): F1 0.446,
  precision 0.288, recall 0.991, kappa 0.368, MCC 0.472; calibration Brier 0.194 /
  ECE 0.195. **Finding**: the raw GLM teacher is high-recall / low-precision and
  overconfident on this band (its 0.999-confidence bin is only ~27% true matches),
  so the report does its job — surfacing that raw teacher labels need a
  precision-raising step (threshold/secondary review) before use as final gold.
  (An earlier draft scored only the 213 pairs whose records both appeared in a
  match cluster, which hid the teacher's false positives and inflated F1 to 0.873;
  the loader now returns the complete closed-world partition so every cross-source
  pair is evaluated.)

### Component Inspection Methods (Progressive Pipeline Building)

**Added exploratory analysis capabilities to core components** - enables parameter tuning WITHOUT ground truth labels:

- **Report Models** (`langres.core.reports`):
  - `CandidateInspectionReport`: Statistics and examples for blocker output
  - `ScoreInspectionReport`: Score distribution analysis for module output
  - `ClusterInspectionReport`: Cluster size distribution and singleton analysis
  - All reports support `.to_markdown()`, `.to_dict()`, and `.stats` property

- **Inspection Methods**:
  - `Blocker.inspect_candidates(candidates, entities, sample_size)`: Explore candidate generation without labels
    - Implemented in `VectorBlocker` with k_neighbors tuning recommendations
  - `Module.inspect_scores(judgements, sample_size)`: Analyze score distributions without labels
    - Implemented in `LLMJudgeModule`, `RapidfuzzModule`, and `CascadeModule`
    - Includes threshold recommendations based on distribution
  - `Clusterer.inspect_clusters(clusters, entities, sample_size)`: Review clustering results without labels
    - Singleton rate analysis and threshold tuning recommendations

- **Example**: exploratory inspect → tune → re-inspect → iterate workflow
  - Demonstrates parameter calibration without expensive labeling
  - All three inspection methods (the standalone `progressive_pipeline_building.py`
    example was removed in M0, superseded by `examples/resolver_company_dedup.py`)

**Key Benefits**:
- **Progressive discovery**: Build pipelines incrementally with feedback at each stage
- **Label-free exploration**: Understand pipeline behavior before expensive labeling
- **Actionable recommendations**: Rule-based parameter tuning suggestions
- **Human + AI readable**: Markdown reports for humans, JSON for agents
- **Type-safe**: Full mypy strict mode compliance with generic SchemaT support
