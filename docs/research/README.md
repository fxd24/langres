# `docs/research/` — research notes, surveys, and the reference spine

This folder is langres's **research memory**: literature surveys, design notes, experiment
result write-ups, and strategic audits — the sources and reasoning the framework is built
on. It is the answer to *"where do we write down everything we find in research?"*

**Conventions**
- Filenames are dated: `YYYYMMDD_topic.md`. Newest work is not a rewrite of older work —
  each doc is a durable record; supersessions are noted in-doc, not by deletion.
- **Research vs. plans:** *this* folder holds the *why* (literature, design rationale,
  results). Executable, wave-structured build plans live in **`docs/plans/`**. A survey
  here typically has a sibling plan there (e.g. the #86 survey → the data-layer plan).
- Every doc **owns its own full reference list**. The consolidated per-theme spine at the
  bottom of this file is a *curated index of load-bearing anchors*, not a mirror of every
  citation — go to the doc for depth.
- The verify-before-asserting rule applies: docs flag confidence and mark unverified
  claims. Reference entries below without a link are cited by venue/year (no URL was
  fabricated).

---

## Index

| Doc | Kind | What it is |
|---|---|---|
| [`20260701_er_seam_audit.md`](20260701_er_seam_audit.md) | audit | ER research landscape as a **seam audit** — 17 method deep-dives + 3 cross-cutting lenses against langres's "composable seam" claim; where it holds and breaks vs. SOTA (feeds issue #55). |
| [`20260702_w1_trained_family_results.md`](20260702_w1_trained_family_results.md) | results | W1.2 trained-family replication — `FellegiSunterMatcher` + `RandomForestMatcher`, $0 CPU-only. |
| [`20260703_w2_person_benchmark_results.md`](20260703_w2_person_benchmark_results.md) | results | W2.1 person resolution on **FEBRL4**, config-only, $0 (five free local methods). |
| [`20260703_w3_paid_smoke_results.md`](20260703_w3_paid_smoke_results.md) | results | W3 paid smoke — set-wise `SelectMatcher` vs. pairwise, measured ($4.65, spend-capped). |
| [`20260707_data_prep_hard_case_mining_survey.md`](20260707_data_prep_hard_case_mining_survey.md) ⭐ | survey | **Data prep & hard-case mining for EM** — the S1–S6 mining taxonomy, the label-noise trap, augmentation, LLM-era selection, the DSPy consumer, and the blocking-vs-matching / decoder-embedding architecture (§11–12). The literature backbone (#86). |
| [`20260709_cost_accounting_design.md`](20260709_cost_accounting_design.md) | design | Cost accounting — tokens are the fact, dollars derived; what to store vs. derive (issue #100). |
| [`20260710_dx_audit_examples_we_wish_we_had.md`](20260710_dx_audit_examples_we_wish_we_had.md) | audit | DX audit: the examples we wish we had (every claim run/verified against `main`). |
| [`20260710_logprob_credence_probe.md`](20260710_logprob_credence_probe.md) | results | The logprob-credence probe — does an LLM judge know when it's wrong? A **gate** run before adding `confidence` to `PairwiseJudgement`. |
| [`20260713_model_identity_and_hub.md`](20260713_model_identity_and_hub.md) | design | Model identity + a publishing seam — the "HF-transformers moment": tie the verbs to which model is used. |
| [`20260714_training_surface_design.md`](20260714_training_surface_design.md) | design | The training surface — judge→`Matcher` rename, one-verb `fit`, the **miner contract** (plain functions → `LabeledPair`), the `finetune()`/serve path, `ColVal`. |
| [`20260715_data_prep_architecture.md`](20260715_data_prep_architecture.md) ⭐ | design | **Data-preparation architecture** — the method-*shape* taxonomy, the failure-mode-driven curation loop (profile → diagnose → fix → train), the existing-code reuse map, AnyMatch-first build order, and the experiment backlog. The *design* over the #86 survey. |
| [`20260716_import_graph_baseline.md`](20260716_import_graph_baseline.md) | results/audit | **Import-graph baseline** for the architecture refactor (frozen @ `4e5b736`, + a post-W-1 delta @ `858d605`) — re-derives the fan table, the kinds/SCC split, and the cross-package cycle counterfactual with the committed `tools/import_graph.py`. Mostly a **negative result**: the "14 vs 6 cycles" claim does not reproduce, 21% of edges are deliberately-lazy ones import-linter cannot ignore, and the W-1 fragment split grew the tangle rather than shrinking it — **relabelling is not emptying**. |

| [`20260717_research_agenda.md`](20260717_research_agenda.md) ⭐ | agenda | **The research agenda** — the experiments the framework *unlocks*, once it is ready (framework work itself is out of scope). 13 items across five threads: the staging result (measure $\varphi$; the Zamani follow-up), **$0 diagnostics** we can run today, the retrieval/matching ladder (the embedder-as-matcher; RocketQA denoising; ADORE), prompt-tuning's ceiling and the fine-tune handoff, and grounding (reproduction; cross-domain ER). Each item is question → why it matters → what's known (cited) → what would settle it → cost & prerequisites, with every item priced ($0 / CPU / GPU / paid) and sequenced. Two findings shape it: the cheapest items (**B1** closure diagnostic, **B2** benchmark profile) are **$0 and one script away**, and several "proposals" are **already built** (`CorrelationClusterer`, `QdrantHybridRerankingIndex`, `VectorBlocker(query_prompt=…)`) — so they are measurements, not builds. |
| [`20260727_embedder_ladder.md`](20260727_embedder_ladder.md) | results | **Which embedding model at which measured parameter count**, and does a query-side instruction change the answer — agenda item C3, measured. Generated by `examples/research/embedder_ladder.py` from a tracked JSONL (`20260727_embedder_ladder_rows.jsonl`) plus the reference per-record recall the paired CIs are built from (`20260727_embedder_ladder_reference_recall.json`); re-running reproduces the table instead of appending to it. Parameter counts are `sum(p.numel())` from the loaded model — no size table, no small/base/large label. Read the caveats before quoting a number: the metric is candidate recall + separability AUC (never F1 at a threshold), recall is capped by a *reachable ceiling* the cross-source filter imposes, model-vs-model deltas carry cluster-resampled 95% CIs against the shipped default, saturation is imported from a separate stream, and every benchmark here is linkage — the ladder says nothing about dedup. The axis only became measurable at all once `query_prompt` stopped being a silent no-op on the blocking path. **Carries two corrections to its own first release** (and to the #239 PR body): the headline delta was measured in the `none` arm — *neither* side prompted, i.e. langres's default configuration — not "with only the query side driven"; and the asymmetric document+query recipe **is** expressible through the shipped API (the document prefix rides on the embedder's `prompt_name`, not through `create_index`), it was merely undocumented and unchecked. The report now also splits its recommendation by **licence**, because langres is Apache-2.0 and the best-measured checkpoint is not. |

| [`20260728_prompt_axis_findings.md`](20260728_prompt_axis_findings.md) ⭐ | results | **Do better instructions make instruction-following embedders better at blocking, and can we pick good ones rather than guessing?** Five checkpoints × four benchmarks × up to seven prompt recipes, reported **per model × per benchmark and never averaged**, on candidate recall at fixed `k` (never F1 at a threshold). Every documented prompt format is quoted from the checkpoint's *own* `config_sentence_transformers.json` or model card at a **pinned revision** — none inferred from another model's convention, since they have nothing in common. **The answer is three tiers, not two:** the checkpoint's own *retrieval* prompt is a reliable win (significantly positive on ≥1 benchmark for **three of the four** instruction-trained models, never significantly negative anywhere); substituting your own task text into its template shape is a **coin flip** (helped Qwen3 +0.0388, cost Gemma −0.1589) and never beat the model's own default on recall; a raw English sentence outside any template hurt **4 of 5** checkpoints — EmbeddingGemma is the exception, and is kept in the table rather than dropped. **"Documented" is not sufficient, which is the trap:** Gemma's `clustering` template is exactly as official as its retrieval one and costs **−0.3183** Δ per-record recall. Two model-card predictions were tested: Qwen3's "1% to 5%" held, and e5's — `query:` on both sides for symmetric tasks, which ER blocking looks exactly like — bought **no measurable improvement**. Every interval is **corrected for multiplicity** (Holm, one `(model, arm)` per family), which is what removes e5 from tier 1: both its documented families fail at Holm's first step, so no e5 recipe is distinguishable from no prompt at all, and the three withdrawn cells are the study's entire retraction list. Carries its **own** corrections: an interval that closes on zero is reported as a boundary case, not a win; the model-specificity claim was rebuilt on `er_symmetric` (the one genuinely cross-model-identical arm) after the arm originally cited turned out to differ per checkpoint by design; two cited snapshot hashes were wrong because the harness loaded by mutable repo name; and an arm-vs-arm argument resting on non-overlapping intervals was withdrawn as never having been a test. Generated by `examples/research/prompt_axis.py`; rows carry the checkpoint revision and a prompt fingerprint, so a re-run cannot silently mix studies. |
| [`20260727_closure_diagnostic.md`](20260727_closure_diagnostic.md) | results | **B1 answered** — does an output cluster contain a pair we judged and *rejected*? Rebuilt from the `JudgementLog`'s `verdict` (never `score < threshold`; the v4 schema persists no threshold, and `predicted_match` gives `decision` precedence), both traps counted rather than assumed, and re-clustered with `CorrelationClusterer` over the identical judgements. |
| [`20260728_external_reproduction.md`](20260728_external_reproduction.md) | results | **E1 applied to blocking — the first time langres re-derives someone else's published number with its own components.** Re-runs DeepBlocker's and UniBlocker's candidate-set protocol (`C = union over a in A of topK(a -> B)`, `|C| = K*|A|`, PC over the *raw* positive list) through `SentenceTransformerEmbedder` + `FAISSIndex`, and compares against numbers transcribed by hand from the primary PDFs into a companion reference JSON (`20260728_external_reproduction_reference.json`), each tagged with the table it came from. Generated by `examples/research/external_reproduction.py` from a tracked JSONL. **The verdict splits into two answers and they must not be run together.** *Unconditional* — the harness computes the literature's quantity: the candidate set is theirs to the pair (`|C| = 150 x 2,616 = 392,400` against DeepBlocker's printed `392.4k` on `dblp_scholar`), their `PQ = hits/(k*|A|)` reproduces to 2 dp on five of six UniBlocker rows (and is sharp enough to expose a typo in their Table 2), and the internal `score_blocking` crosscheck reproduces the embedder ladder digit-for-digit. *Conditional* — every recall comparison, and they do not all agree equally well: two land within 0.7–1.5 pp (`dblp_scholar` 98.82 vs 98.1; `dblp_acm` 96.76/82.11 vs 95.28/81.00), `fodors_zagat` is 5.4 pp off in our favour and unexplained, and the three product benchmarks are 4–7 pp apart as points and only *overlap* once the gold-set interval is applied. **All of it rests on our gold pairs corresponding to theirs, which is argued from provenance and verified nowhere** — neither paper publishes a pair list, and an equal pair *count* is not an identical pair *set*. **Read Section C first**: on `abt_buy`/`amazon_google`/`walmart_amazon` our gold set is smaller than the papers' (the DeepMatcher labelled splits, whose positives already survived someone's blocker), so a higher PC there is expected and is *not* evidence of anything — Section B2 gives a conditional interval, not a bound. `fodors_zagat` is the outlier the report cannot explain: its gold count is identical yet we sit 5.4 pp *above* UniBlocker's STransformer row, so it evidences the protocol arithmetic rather than the model. Carries three honest negatives: UniBlocker's **mAP column does not reproduce** from the formula their §4.2 prints (their PC/PQ do), one of their PQ cells is internally inconsistent with their own PC, and **no published blocking baseline exists** for `wdc_computers` or `febrl_person` as we ship them. These numbers are **not** `score_blocking` output and must not be compared to the embedder ladder's — Section F attributes that gap to its two causes (directional vs symmetric kNN; raw positives vs transitive closure). |
| [`20260727_portfolio_annotation.md`](20260727_portfolio_annotation.md) | results | **B2 answered** — the per-benchmark saturated / unsaturated / structurally-caveated annotation with the number behind each verdict, the rules fixed before looking, and the portfolio-level gaps (no registered dedup benchmark; six of nine loadable sets are git-checkout-only). Adds the one missing measurement, `VocabularyOverlapSection`. |
| [`20260728_threshold_default.md`](20260728_threshold_default.md) | results | **Should `fit(derive_threshold=True)` be the default?** — the derived cut vs. the front door's hard-coded `0.5`, per benchmark, per $0 method, at three seeds, every number graded on a **corpus-disjoint** held-out split. Three things come out of it: whether the derived cut earns the seat; that "flip the default" is not a one-character change but three verified raising paths; and **the split trap** — `align_pairs`' entity-disjoint split degenerates on a dense (k-NN-candidate) label set, so `FitReport`'s own held-out numbers are not trustworthy at that scale. Tables regenerate from the tracked JSON via `--render`. |

**Related, elsewhere:** `docs/plans/20260715_data_layer_plan.md` (the buildable profiler +
mining seam), `docs/plans/20260713_training_loop_plan.md`, `docs/plans/20260708_tracking_observability_plan.md`.

---

## Framework foundations — key references by theme

The load-bearing anchors, grouped. Confidence and "no arXiv" flags carried from the source
docs. Depth (mechanism, langres mapping, caveats) lives in the doc that cites each.

### Entity matching / data integration
- **DeepMatcher / Magellan** — Mudgal et al., SIGMOD 2018 (the "benchmarks are trivially
  solvable" motivation; blocking-derived hard negatives).
- **Ditto** — Li et al., VLDB 2020, [2004.00584](https://arxiv.org/abs/2004.00584) (COL/VAL
  serialization; augmentation operators).
- **AnyMatch** — Zhang et al. 2024, [2409.04073](https://arxiv.org/abs/2409.04073)
  (GPT-2-124M, LODO zero-shot; hard-positive mining + augmentation — the reproduction target).
- **Fine-tuning LLMs for EM** — Steiner, Peeters & Bizer 2024, [2409.08185](https://arxiv.org/abs/2409.08185)
  (LLM-generated training data for EM; mixed results — read before a fine-tune-selection design).
- **Can Foundation Models Wrangle Your Data?** — Narayan et al., VLDB 2022, [2205.09911](https://arxiv.org/abs/2205.09911).
- **Jellyfish** — [2312.01678](https://arxiv.org/abs/2312.01678) (Llama-2-13B for data-prep tasks).
- **Unicorn** — Tu et al., SIGMOD/PACMMOD 2023 *(no arXiv; OpenReview `388Cge6WPN`)* — one
  MoE encoder across matching tasks, explicit zero-shot; the generalist-EM precedent.
- **ER surveys** — Christophides et al., CSUR 2020, [1905.06397](https://arxiv.org/abs/1905.06397);
  Papadakis et al., CSUR 2020, [1905.06167](https://arxiv.org/abs/1905.06167) (the four-stage
  blocking→processing→matching→clustering pipeline).

### Hard-example / hard-negative mining
- **OHEM** — Shrivastava et al., CVPR 2016, [1604.03540](https://arxiv.org/abs/1604.03540).
- **Focal Loss** — Lin et al., ICCV 2017.
- **DPR** — Karpukhin et al., EMNLP 2020 (BM25 lexical hard negatives).
- **ANCE** — Xiong et al., ICLR 2021, [2007.00808](https://arxiv.org/abs/2007.00808)
  (index-refreshed model-mined negatives).
- **RocketQA** — Qu et al., NAACL 2021, [2010.08191](https://arxiv.org/abs/2010.08191)
  (cross-encoder denoised hard negatives — the EM safety rail).
- **NV-Retriever** — Moreira et al. 2024, [2407.15831](https://arxiv.org/abs/2407.15831)
  (positive-aware false-negative removal; TopK-PercPos).

### Difficulty / coreset / label-noise / curriculum
- **EL2N / GraNd** — Paul et al., NeurIPS 2021, [2107.07075](https://arxiv.org/abs/2107.07075)
  (EL2N ≈ `|1 − p(match)|`, no extra training).
- **Forgetting events** — Toneva et al., ICLR 2019, [1812.05159](https://arxiv.org/abs/1812.05159).
- **Dataset Cartography** — Swayamdipta et al., EMNLP 2020 (confidence × variability; the
  "ambiguous" region = cleanest informative-pair notion).
- **Confident Learning / Cleanlab** — Northcutt et al., JAIR 2021, [1911.00068](https://arxiv.org/abs/1911.00068)
  (the label-noise disambiguator — run before trusting any difficulty score).
- **Core-Set / k-Center** — Sener & Savarese, ICLR 2018, [1708.00489](https://arxiv.org/abs/1708.00489).
- **Moderate Coreset** — Xia et al., ICLR 2023; **Herding** — Welling 2009.
- **SemDeDup** — Abbas et al. 2023, [2303.09540](https://arxiv.org/abs/2303.09540).
- **Beyond neural scaling laws** (prototypicality pruning) — Sorscher et al., NeurIPS 2022,
  [2206.14486](https://arxiv.org/abs/2206.14486) (**pruning beats power-law scaling**).
- **D4** — Tirumala et al., NeurIPS 2023, [2308.12284](https://arxiv.org/abs/2308.12284)
  (SemDeDup + prototypicality-diversify).
- **Curriculum learning** — Bengio et al., ICML 2009 *(no arXiv)*; **Competence-based
  curriculum** — Platanios et al., NAACL 2019, [1903.09848](https://arxiv.org/abs/1903.09848).

### LLM-era data selection
- **LIMA** — Zhou et al. 2023, [2305.11206](https://arxiv.org/abs/2305.11206) (cap and curate).
- **AlpaGasus** — Chen et al. 2023, [2307.08701](https://arxiv.org/abs/2307.08701) (LLM scores each example).
- **DEITA** — Liu et al. 2024, [2312.15685](https://arxiv.org/abs/2312.15685) (complexity × quality × diversity).
- **IFD / Superfiltering** — Li et al., NAACL 2024 [2308.12032](https://arxiv.org/abs/2308.12032) / ACL 2024 [2402.00530](https://arxiv.org/abs/2402.00530) (GPT-2-size proxy transfers the ranking; no paid LLM).
- **LESS** — Xia et al., ICML 2024, [2402.04333](https://arxiv.org/abs/2402.04333) (gradient-matched selection for a target task).

### Embedding / retrieval training (the blocker target)
- **SimCSE** — Gao et al. 2021, [2104.08821](https://arxiv.org/abs/2104.08821).
- **GTR** — Ni et al. 2021, [2112.07899](https://arxiv.org/abs/2112.07899) (scale/batch as the lever).
- **Margin-MSE** — Hofstätter et al. 2020, [2010.02666](https://arxiv.org/abs/2010.02666)
  (cross-encoder teacher margins ARE the bi-encoder's labels — the matcher→blocker data engine).
- **TAS-B** — Hofstätter et al. 2021, [2104.06967](https://arxiv.org/abs/2104.06967)
  (dual-teacher + topic-aware batch sampling).
- **GISTEmbed** — Solatorio 2024, [2402.16829](https://arxiv.org/abs/2402.16829)
  (guide model masks false in-batch negatives).
- **GradCache** — Gao et al. 2021, [2101.06983](https://arxiv.org/abs/2101.06983)
  (huge in-batch-negative batches under fixed memory).
- **E5 / CCPairs** — Wang et al. 2022, [2212.03533](https://arxiv.org/abs/2212.03533);
  **E5-mistral** — [2401.00368](https://arxiv.org/abs/2401.00368) (LLM-synthesized training data).
- **BGE-M3** — [2402.03216](https://arxiv.org/abs/2402.03216); **Arctic-Embed** —
  [2405.05374](https://arxiv.org/abs/2405.05374); **Gecko/FRet** — [2403.20327](https://arxiv.org/abs/2403.20327)
  (LLM-relabelled positives + hard negatives).
- **Synthetic-query generation** — Doc2Query [1904.08375](https://arxiv.org/abs/1904.08375);
  InPars [2202.05144](https://arxiv.org/abs/2202.05144) / InPars-v2 [2301.01820](https://arxiv.org/abs/2301.01820);
  Promptagator [2209.11755](https://arxiv.org/abs/2209.11755) (generate-then-filter; the ER
  analog is synthetic entity-variant anchors filtered by the matcher).

### Data mixture / domain reweighting (the generalist question)
- **DoReMi** — Xie et al., NeurIPS 2023, [2305.10429](https://arxiv.org/abs/2305.10429)
  (proxy-model group-DRO learns per-domain weights).
- **DoGE** — Fan et al., ICML 2024, [2310.15393](https://arxiv.org/abs/2310.15393)
  (gradient-alignment source scoring — closest to LODO source-selection).
- **Data Mixing Laws** — Ye et al. 2024, [2403.16952](https://arxiv.org/abs/2403.16952).

### Weak supervision / active learning
- **Data programming** — Ratner et al., NeurIPS 2016, [1605.07723](https://arxiv.org/abs/1605.07723);
  **Snorkel** — Ratner et al., VLDB 2018, [1711.10160](https://arxiv.org/abs/1711.10160)
  (label model over noisy labeling functions).
- **FlyingSquid** — Fu et al., ICML 2020, [2002.11955](https://arxiv.org/abs/2002.11955)
  (closed-form label model).
- **ALMSER-GB** — Primpeli & Bizer, ISWC 2021 *(no arXiv; `wbsg-uni-mannheim/ALMSER-GB`)* —
  EM-native graph-boosted active learning for multi-source ER.
- **BADGE** — Ash et al., ICLR 2020, [1906.03671](https://arxiv.org/abs/1906.03671)
  (uncertainty × diversity in one acquisition — gradient-embedding k-means++).
- **Cluster-Margin** — Citovsky et al., NeurIPS 2021, [2107.14263](https://arxiv.org/abs/2107.14263);
  **BAIT** — Ash et al., NeurIPS 2021, [2106.09675](https://arxiv.org/abs/2106.09675).
- **DIAL** — Jain et al., VLDB 2022, [2104.03986](https://arxiv.org/abs/2104.03986)
  (separate objectives for blocker recall vs. matcher precision).
- **DTAL** — Kasai et al., ACL 2019, [1906.08042](https://arxiv.org/abs/1906.08042) (transfer + active EM).

### Augmentation
- **Ditto** — [2004.00584](https://arxiv.org/abs/2004.00584) (operators; MixDA).
- **Rotom** — Miao et al., SIGMOD 2021 (meta-learned operator policy).
- **Sudowoodo** — Wang et al., ICDE 2023, [2207.04122](https://arxiv.org/abs/2207.04122)
  (contrastive self-supervision, label-free).
- **SyNeg** — [2412.17250](https://arxiv.org/abs/2412.17250); **Syntriever** —
  [2502.03824](https://arxiv.org/abs/2502.03824) (LLM-synthesized hard negatives).

### Prompt optimization (the DSPy lineage)
- **Breunig 2025** — *"Let the Model Write the Prompt"*, [dbreunig.com](https://www.dbreunig.com/2025/06/10/let-the-model-write-the-prompt.html)
  (the DSPy-in-ER inspiration; Qwen3-0.6B 60.7%→82% via MIPROv2; "your eval data is your
  most valuable AI asset").
- **DSPy / MIPROv2** — [2406.11695](https://arxiv.org/abs/2406.11695) (how demo/instruction
  selection works; the pre-filtering-hard-demos hypothesis in survey §7).

### Decoder-LLM embeddings & the recall/precision split
See the #86 survey **§11–§12** for the consolidated treatment (GritLM joint training,
Qwen3-Embedding, E5-mistral instructions, the UniBlocker counter-signal, and how SOTA
divides blocking recall from matching precision).
