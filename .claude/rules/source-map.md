---
paths:
  - "src/**"
---

# langres Source Map

> **Path-scoped:** loads only when you read/edit a file under `src/`. The root
> `CLAUDE.md` carries the always-on guidance and a condensed top-level map; this
> file carries the per-module detail — what each file is *for*, and the traps
> the paths alone don't tell you.
>
> **Keep this in sync with the code.** When a change moves a module, changes what
> a package exports, or retires a shim, update the entry here in the **same**
> change — not a follow-up.

```
langres/
├── src/langres/
│   ├── architectures/  # Named ER pipelines: FuzzyString ($0/offline), VectorLLMCascade (paid) — construct one, call .dedupe()/.compare()
│   ├── optimize.py     # langres.optimize / score_blocking: the import-light facade (stdlib-only module top; every engine import is lazy). A MODULE, not a package — `langres.optimize` is a CALLABLE, so a submodule under that name is unreachable by attribute traversal (`import langres.optimize.loop as l` → ImportError). The engine lives next door:
│   ├── autoresearch/   # the autoresearch ENGINE — blocking search, NOT ER modelling, so it sits outside core (depends on core one-way; core imports nothing from here)
│   │   ├── __init__.py     # docstring only — exports NOTHING, which is what keeps it import-light (factory/blocker_optimizer are heavy)
│   │   ├── objective.py    # the immutable keep-if-better scorer (Pareto + log_loss)
│   │   ├── search_space.py # the declarative config grid the loop enumerates
│   │   ├── factory.py      # config -> runnable blocker. HEAVY ([semantic] at module top) — lazy-import only
│   │   ├── loop.py         # propose → run → evaluate → keep, over tracking.runs persistence
│   │   └── blocker_optimizer.py  # BlockerOptimizer (Optuna study; optuna is dev-only — lazy-import only)
│   ├── eval.py         # Curated evaluation facade (lazy): evaluate, list_benchmarks/get_benchmark, ER metrics
│   ├── benchmarks/     # ER benchmark HARNESS — race methods into a table / score a judge on a fixed candidate set. Internal plumbing: users reach it ONLY via a dataset from `langres.data.get_benchmark(...)` + `langres.eval.evaluate(...)` / `candidates_for(...)`; __init__ exports nothing (import-light, like autoresearch/)
│   │   ├── runner.py       # run_method / run_methods -> BenchmarkTable; tracks, resolver-bcubed, tune_threshold, cost helpers
│   │   └── judge_eval.py   # JudgePairEval, evaluate / evaluate_judge_on_candidates (scorer-isolating, spend-capped), BudgetedModuleRunner
│   ├── cli.py          # langres CLI: review / export-csv / import-csv (labeling loop)
│   ├── _exports/       # per-domain fragments composing the ROOT __all__ + lazy maps (add a root export HERE, not in __init__.py)
│   ├── core/           # Low-level primitives + the Resolver
│   │   ├── _exports/       # same, for langres.core (add a core export HERE, not in core/__init__.py)
│   │   ├── resolver.py     # ERModel (aliased Resolver): the class + from_schema / fit / the anchor surface; no matcher="auto"
│   │   ├── _model_state.py     # ERModel layer: what a model IS — slots, identity, the 3 construction doors, schema binding
│   │   ├── _model_run.py       # ERModel layer: how it RUNS — block → (compare) → score → cluster; dedupe / compare
│   │   ├── _model_persist.py   # ERModel layer: how it PERSISTS — the resolver.json manifest + per-slot sidecars (no pickle)
│   │   ├── _artifacts.py       # component ⇄ ComponentSpec adapters (the leaf _model_persist serializes each slot with)
│   │   ├── inputs.py       # normalize_records: raw dicts -> (schema, normalized records); schema inference for a schema-less dedupe()/compare()
│   │   ├── results.py      # LinkVerdict / DedupeResult — architecture + backbone + score_type + threshold
│   │   ├── spend.py, spend_cap.py  # SpendMonitor/BudgetExceeded ledger + SpendCappedMatcher (the ONE enforcer) + DEFAULT_BUDGET_USD; core leaf, so ERModel/every architecture can cap
│   │   ├── method_registry.py  # ONE MethodSpec registry: judge/method name -> builder + identity (all three dispatch paths resolve here)
│   │   ├── registry.py     # component config-registry (type_name -> class) for save/load
│   │   ├── blocker.py, blockers/   # AllPairsBlocker, VectorBlocker
│   │   ├── comparator.py, comparators/  # Comparator ABC (contract) + StringComparator (impl)
│   │   ├── module.py, modules/, judges/  # Module (judge) ABC + LLMJudge, CascadeJudge, etc.
│   │   ├── clusterer.py            # Clusterer (transitive closure)
│   │   ├── runs.py, judgement_log.py, trackers/  # → back-compat SHIMS; observability moved to langres.tracking (below). `# TEMPORARY: deleted by the W2 sweep`
│   │   ├── calibration.py, finetune.py, fit_report.py, methods_prompt.py, methods_calibrate.py  # W2 back-compat shims → langres.training.* (real modules moved; marked `# TEMPORARY: deleted by the W2 sweep`)
│   │   └── reports.py              # inspection/evaluation report models (ScoreInspectionReport, BlockerEvaluationReport, ...)
│   ├── curation/       # human-in-the-loop labelling + gold-set cold-start (the dissolved langres.bootstrap). core/{review,harvest,anchor_store,canonicalizer}.py are TEMPORARY W2-sweep back-compat shims re-exporting from here
│   │   ├── review.py       # select_for_review + ReviewQueue (pick the uncertain margin)
│   │   ├── harvest.py      # Correction/CorrectionLog, harvest_labeled_pairs, derive_threshold_from_pairs, align_pairs
│   │   ├── anchor_store.py         # AnchorStore / ClusterDelta (hold the anchors; assign incoming records)
│   │   ├── canonicalizer.py        # Canonicalizer (survivorship: fold a cluster into one golden record)
│   │   └── base.py, miners.py, models.py, labelers.py, bootstrapper.py, report.py, _pairs.py  # gold-set cold-start: Miner/Labeler, HardNegativeMiner, GoldPair/GoldSet, Bootstrapper, BootstrapReport
│   ├── methods.py      # method registry / _make_module_builder (benchmark path)
│   ├── clients/        # OpenRouter client, SpendMonitor, pricing
│   ├── tracking/       # observability, NOT ER modelling — so it sits beside core (depends on core one-way; the langres.core facade re-exports these names for back-compat)
│   │   ├── runs.py         # RunContext/RunRecord/RunStore (JSONL, content-addressed recipe_id) + capture_run/git_sha/dataset_fingerprint
│   │   ├── judgement_log.py    # JudgementLog + LoggingMatcher (logs every judge call: ids, score, verdict, model, cost)
│   │   ├── factories.py    # create_wandb_tracker / create_trackio_tracker (also lazily resolved via langres.clients)
│   │   └── trackers/       # ExperimentTracker Protocol + NoOpTracker/MultiTracker + resolve_tracker; lazy MlflowTracker/WandbTracker/TrackioTracker (each pulls its heavy backend only when wired)
│   ├── metrics/        # ER metrics + diagnostics — they SCORE a resolution, not the modelling contract, so beside core (public via langres.eval; back-compat shims at core.metrics/.analysis/.debugging/.diagnostics)
│   │   ├── metrics.py      # BCubed / pairwise / ranking ER metrics (ranx lazy for [eval] MRR/NDCG/MAP)
│   │   ├── analysis.py     # evaluate_blocker_detailed, extract_false_positives/missed_matches
│   │   ├── debugging.py    # PipelineDebugger + Candidate/Score/Cluster stats
│   │   └── diagnostics.py  # FalsePositiveExample / MissedMatchExample / DiagnosticExamples
│   ├── report/         # the shared $0 rendering seam (presentation, NOT modelling — so it sits beside core, not in it)
│   │   ├── _svg.py         # pure-stdlib inline-SVG chart primitives (line_chart/bar_chart); imports nothing from langres
│   │   ├── _report_html.py # shared HTML scaffold: document()/section()/_num/_histogram/safe_auc
│   │   └── eval_report.py  # EvalReport, the $0 tearsheet (public home: langres.eval / root langres, both lazy)
│   ├── training/       # fitting/calibrating a matcher (what PRODUCES a tuned model, NOT ER modelling — so it sits beside core, like report/). Imports core one-way; core → training is non-zero by design (resolver.fit + the _exports/_training surface), see tests/test_import_tangle.py
│   │   ├── finetune.py         # QLoRA/LoRA training (run_finetune, QLoRA, LabeledCandidate); peft/trl/bitsandbytes/torch stay lazy inside QLoRATrainer.train ([finetune])
│   │   ├── calibration.py      # derive_threshold + the Platt/isotonic Calibrator (sklearn at module scope, [trained])
│   │   ├── fit_report.py       # the FitReport fit digest (held on ERModel.fit_report_)
│   │   ├── methods_prompt.py   # Bootstrap / MIPRO / GEPA — the prompt-optimize Method objects (dspy lazy)
│   │   └── methods_calibrate.py # Platt / Isotonic — the score-calibrate Method objects
│   └── data/           # benchmark dataset loaders (FZ, Amazon-Google, ...)
│       ├── benchmark.py # the benchmark SPEC (a dataset IS a benchmark): the Benchmark protocol + PairTrack / gold_pairs_from_clusters / DEFAULT_PAIR_GRID a dataset carries. Import-light; the langres.benchmarks harness depends on it ONE-WAY (never data -> benchmarks)
│       └── registry.py # name→benchmark manifest: list_benchmarks() / get_benchmark()
├── tests/              # Test suite
├── examples/           # Usage examples (quickstart_models.py is the offline quickstart)
└── docs/               # Documentation
```

**Not built yet** (roadmap — do not reference as existing): `tasks`/`flows`
modules, a general `Optimizer`, a synthetic data generator.
