# Model identity and a publishing seam — a design note for the "transformers moment"

*Design note for the maintainer's direction question — "think of HF transformers:
anybody can publish their model, and you can use it easily; today `link`/`dedupe`
don't tell you which model is used, and the verbs should be tied to the type of
model one uses" — and for [issue #103](https://github.com/fxd24/langres/issues/103)
(LLMJudge unreachable by name from the verbs). Scope: what "model identity" and
"publishing" should mean in langres, mapped honestly onto the seams that exist,
with staged options. Documentation only; no code changes. Written against `main`
@ `53a08ed` (2026-07-13).*

> **Provenance of claims.** Every `file:line` and every statement about the
> langres codebase below was read directly from `main` @ `53a08ed` while writing
> this note. Statements about Hugging Face transformers/Hub mechanics (§2) are
> **background knowledge, not re-verified against HF docs for this note** — they
> describe the well-known shape of that ecosystem (namespaced ids, revision
> pinning, model cards, `from_pretrained`, `trust_remote_code`), and any
> implementation against `huggingface_hub` must re-check the live API first.
> Everything else is argument.

---

## 0. Executive summary

1. **The maintainer's complaint is verifiably true, twice over.** (a) A
   `dedupe()` result reports only `judge_used` + `score_type`
   (`verbs.py:104-127`) — the resolved LLM model id (`openrouter/openai/gpt-4o-mini`
   vs `openai/gpt-5-mini`, chosen *by which API key happens to be set*,
   `presets.py:197-225`) is computed in `resolve_judge` and then **dropped**
   before the result is built (`presets.py:531-598` — `ResolvedJudge` carries no
   `model` field). The embedding judge is worse: its model
   (`all-MiniLM-L6-v2`, `presets.py:105`) is a hardcoded constant that appears
   in no result, notice, or docstring a verb user sees. (b) A *specific* model
   from a specific paper cannot be named at all: the only prompt-seam judge,
   `LLMJudge`, has no preset name (issue #103), so a published method must be
   hand-wired from `langres.core` and then reports itself as
   `judge_used="custom"` (`presets.py:404-406`) — identity erased at exactly the
   moment it matters.

2. **langres has three name registries and none of them is a model identity.**
   The verb judge names (4, `presets.py:83`), the benchmark method names (7+2,
   `_method_names.py` / `methods.py:185-265`), and the component
   `type_name` config-registry (20 registrations, `core/registry.py`) are three
   unaligned namespaces. None carries an author dimension, none names the
   underlying model, and a new judge must be wired into **three** dispatch
   sites by hand (`.claude/rules/component-design.md`; deferred to #55).

3. **The transformers analogy maps cleanly onto existing seams — until the
   payload.** `pipeline(task, model=...)` ≈ the verbs; `from_pretrained` ≈
   `Resolver.load` (a no-code-execution config-registry rebuild,
   `resolver.py:757-832`); the `evaluate` library ≈ `langres.eval` + the
   benchmark registry (10 datasets). What breaks: an ER "model" is a
   **config + prompt + threshold bundle** whose threshold is not portable across
   score families (`presets.py:89-93`), whose schema coupling is a Python class
   resolved by name (`all_pairs.py:161-166`), and whose most-customized seams
   (`response_parser`, `record_serializer`) are callables that deliberately do
   **not** serialize (`llm_judge.py:485-488`). Trained weights — the thing that
   makes a hub worth visiting — are today the exception (FAISS index, compiled
   DSPy program, fitted RF sidecars), pending the training-loop work (§5).

4. **Recommendation (§6): O3, staged — but sequenced so each stage is useful
   alone.** v0.3 ships O4's fixes *expressed in O1's vocabulary*: one method
   registry that collapses the three dispatch sites (closing #55's debt and
   #103 in the same move), `LLMJudge` reachable by name with named parsers, and
   the resolved model id stamped on every result. v0.4 opens the namespace to
   third parties via a `langres.methods` entry-points group
   (`pip install langres-ditto` → `dedupe(records, judge="ditto")`) plus the
   `evaluate_method(name, benchmark)` one-liner that makes the leaderboard
   story real. `Resolver.from_pretrained` + Hub hosting (O2) is deferred until
   the training loop produces weight-bearing artifacts worth downloading — and
   until the artifact-compat hard-fail (`resolver.py:861-866`) has a policy.

---

## 1. Inventory — the seams that exist today (verified)

### 1.1 The verb judge names and what the verbs report

`JudgeName = Literal["auto", "zero_shot_llm", "embedding", "string"]`
(`presets.py:83`). `judge="auto"` resolves to `("zero_shot_llm", <model>)` from
whichever API key is set — `OPENROUTER_API_KEY` → `openrouter/openai/gpt-4o-mini`
(`clients/openrouter.py:88`), else `OPENAI_API_KEY` → `openai/gpt-5-mini`
(`presets.py:112`) — failing fast with `NoJudgeAvailableError` when neither is
(`presets.py:197-208`). The caller can override with `model=`, but that kwarg
is documented as "Model id override for `zero_shot_llm`" and is **silently
ignored for every other judge** (`presets.py:387-389`, `verbs.py:298`).

What comes back:

- `LinkVerdict(match, score, reasoning, judge_used, score_type, judgement)`
  (`verbs.py:81-101`). No `model` field. The model id *is* reachable via
  `verdict.judgement.provenance["model"]` — but that is a per-judge provenance
  convention (`dspy_judge.py:289,311`; `llm_judge.py:1077`), not a typed
  contract; nothing guarantees a third-party `Module` stamps it.
- `DedupeResult(list[set[str]], judge_used, score_type)` (`verbs.py:104-127`).
  No `model`, and no judgements at all — the model id of a paid run is
  reported once, in a `warnings.warn` notice before scoring
  (`presets.py:220-224`), and is then unrecoverable from the result object.
- `judge="embedding"` never surfaces `all-MiniLM-L6-v2`
  (`_VECTOR_MODEL_NAME`, `presets.py:105`) anywhere a verb user looks. The
  same constant silently governs the blocker that kicks in above 100 records
  for *every* judge (`_ALL_PAIRS_MAX_N`, `presets.py:103`) — so even a
  `judge="string"` run's candidate set depends on an invisible embedding model.
- Passing a `Module` instance reports `judge_used="custom"`
  (`presets.py:404-406`): the escape hatch works, but erases identity.

### 1.2 The method registry (`methods.py`) — names exist, but only for the race

`ALL_METHODS` = `rapidfuzz`, `weighted_average`, `embedding_cosine`,
`llm_judge`, `cascade`, `dspy_judge`, `select_judge` (`_method_names.py`), plus
`fellegi_sunter` / `random_forest` dispatchable-but-not-raced
(`methods.py:33-48`). `_make_module_builder` (`methods.py:185-265`) is a
hand-rolled `if/elif` chain mapping name → module builder;
`make_resolver_factory` (`methods.py:268-326`) requires a `BlockingBenchmark`
(schema + pinned blocker), so **these names are unusable from the verbs or from
`Resolver.from_schema`** — they exist only for the benchmark harness. Note the
registry's `llm_judge` method *does* build `LLMJudge` (`methods.py:209`), so
"the prompt-seam judge by name" already exists on exactly one of the three
dispatch paths — the one end users never call.

### 1.3 The component config-registry and the `Resolver` artifact

`@register(type_name)` → `_COMPONENT_REGISTRY` with a lazy import map for
optional-dependency components and a did-you-mean error
(`core/registry.py:61-132`); a parallel `register_schema` namespace for Pydantic
schemas (`core/registry.py:135-167`). 20 component registrations exist today.
`Resolver.save(path)` writes `resolver.json` — an `ArtifactManifest` of
`ComponentSpec(type_name, slot, config)` entries plus `artifact_version` and
`langres_version` — and per-slot sidecar state dirs for components owning
out-of-band state (FAISS index bytes, compiled DSPy program, fitted model)
(`resolver.py:722-755`). `Resolver.load` rebuilds every slot from the registry
by `type_name`: **"no code and no pickle"** (`resolver.py:15-16`) is an
explicit, load-time-safe invariant worth protecting in any hub design.

Two hard facts that shape §3:

- **Schema coupling.** `AllPairsBlocker.config` is
  `{"schema_type_name": <registered name>}` (`all_pairs.py:161-166`): the
  artifact references its schema *by name*, and loading in a fresh process
  requires that Python class to be registered (`verbs.py:41-45` documents the
  `SchemaNotRegistered` failure for inferred schemas). A published artifact
  cannot carry its schema as data today.
- **Version brittleness.** `_check_versions` **hard-fails in both directions** —
  an artifact older than the current layout raises "no longer readable;
  re-save with this langres build" (`resolver.py:861-866`). Fine for local
  artifacts; fatal for artifacts published to a hub, where the publisher and
  consumer upgrade on different schedules.

### 1.4 `Resolver.from_schema` — the third dispatch site

`_build_module_for_judge` (`resolver.py:82-137`) accepts
`Literal["string", "embedding", "zero_shot_llm"]` or a `Module` — deliberately
no `"auto"` (layering) and **no spend cap** (`resolver.py:380-385`). Its
`zero_shot_llm` branch, like the verbs', builds a `DSPyJudge` — not `LLMJudge`.
The three-site duplication is documented as deliberate layering-preserving debt
(`resolver.py:93-97`, `.claude/rules/component-design.md`), with the single
registration API deferred to #55.

### 1.5 Issue #103 — the prompt seam is unreachable by name

Verified against the issue and the code: `LLMJudge` is the only judge exposing
`system_prompt` / `prompt_template` / `response_parser` / `record_serializer`
(`llm_judge.py:344-352`), it is registered and serializable
(`type_name="llm_judge"`, config carries model/temperature/prompt/system-prompt/
on_parse_error — `llm_judge.py:478-501`), and **no verb or `from_schema` name
reaches it**. The parser/serializer callables are explicitly not serialized —
on `Resolver.load` they revert to defaults (`llm_judge.py:485-488`) — which
matters because a binary yes/no parser (`parse_binary_yes_no`,
`llm_judge.py:122`) is exactly what the Peeters replication (#102) customized.

### 1.6 The evaluation surface

`langres.eval` (lazy facade, `eval.py`): `evaluate(module, candidates,
gold_pairs, *, grid, threshold, budget_usd, ...)` (`benchmark.py:1341-1351`),
`candidates_for(bench, split=...)` — which pins the benchmark's own blocking
config and leakage-free split (`eval.py:102-170`) — plus
`list_benchmarks()`/`get_benchmark(name)` over a static import-light manifest of
9 loadable datasets + external-only OpenSanctions (`data/registry.py:206-320`),
and the `EvalReport` self-contained HTML tearsheet (PR #107). Identity plumbing
that already exists underneath: `Resolver.config_dict()` (hash-safe config
snapshot, `resolver.py:687-720`) feeding `compute_recipe_id`
(`runs.py:223`) for idempotent replay — a *content* identity, but keyed to
in-process config, not to a shareable name.

---

## 2. The transformers analogy, honestly mapped

| HF transformers concept | langres today | Gap |
|---|---|---|
| **Model id `author/name`** (globally unique, namespaced) | Three flat namespaces: 4 verb judge names, 9 method names, 20 `type_name`s. No author dimension; uniqueness only within one install (`registry.py:104-105` raises on collision). | No identity primitive. The same string means different things per layer (`llm_judge` the *method* builds `LLMJudge`; `zero_shot_llm` the *judge name* builds `DSPyJudge`). |
| **Revision pinning** (commit hash, `revision=`) | `artifact_version` (layout version) + `langres_version` (logged, non-fatal) in `resolver.json`; `recipe_id` hashes the config for replay (`runs.py:223`). | No content-addressed revision on a *named* thing; `recipe_id` is per-run infra, not a distribution handle. |
| **Model card** (README + metadata: intended use, eval results, cost) | Nothing. Closest: `EvalReport` (per-run tearsheet) and the §8 open question in `docs/research/20260709_cost_accounting_design.md` — "an artifact that says what it *is* but not what it *achieves and costs* isn't choosable". | The card is the missing half of "evaluate it for themselves": a published method needs pinned claims (dataset, protocol, F1, $/1k pairs) next to its config. |
| **`pipeline(task, model=...)`** | `link`/`dedupe(judge=..., model=...)`. | `judge` names a *family*, not a model; `model=` only affects the LLM family and is silently ignored otherwise (`presets.py:387-389`); result drops the resolved model (§1.1). |
| **`AutoModel.from_pretrained("author/name")`** | `Resolver.load(local_dir)` — registry rebuild, no code execution. | No remote fetch, no namespace, hard-fail on layout drift (`resolver.py:861-866`). |
| **The Hub** (hosting, discovery, versioning) | Nothing. Artifacts are plain directories — which is exactly what `huggingface_hub` hosts well (a repo *is* a directory of files). | Hosting is the easy part; §3-O2 argues the payload and compat policy are the hard parts. |
| **`evaluate` / leaderboards** | `langres.eval.evaluate` + `candidates_for` + 10-dataset registry + `EvalReport`. | No name×benchmark one-liner; no cross-run aggregation (§4). |
| **`trust_remote_code`** | Two trust models already separated: pip packages (arbitrary code, pip's existing trust) vs. artifacts (no code execution by design, `resolver.py:15-16`). | Keep them separated. A hub artifact must never gain code execution; third-party *code* ships as a package (§3-O1). |

### 2.1 Where the analogy breaks

1. **An ER "model" is a bundle, and the threshold is part of it.** The same
   0.7 cut means different things on `"heuristic"` vs `"sim_cos"` vs
   `"prob_llm"` scales — that is why per-judge defaults exist at all
   (`presets.py:89-93`, E12). A published method that omits its calibrated
   threshold is not reproducible; transformers has no equivalent knob.
2. **Input coupling is a Python class, not a tokenizer.** A transformers model
   ships its tokenizer as data; a langres artifact references
   `schema_type_name` and needs the class importable (§1.3). Cross-user
   artifact portability therefore *depends on* code distribution (a schema
   package) or a schema-as-data extension — the artifact alone is not enough.
3. **The most-customized seams are callables that don't serialize.**
   `response_parser`/`record_serializer` revert to defaults on load
   (`llm_judge.py:485-488`). Any publishing story must either name-register
   parsers (config-expressible) or accept that some methods are
   package-only.
4. **Weights are (today) the exception.** Sidecar state exists — FAISS index,
   compiled DSPy program (`dspy_judge.py:216-220`), fitted RF — but the
   flagship publishable artifact class (trained students, fine-tuned small
   LMs) is exactly what the parallel training-loop plan is scoping (§5). A hub
   launched before that exists would host prompt-and-threshold JSON files:
   legitimate, but a thin announcement.

---

## 3. Options

All snippets are against real current signatures; "after" snippets are
illustrative API sketches, not commitments.

### O1 — one method registry + qualified ids + entry-points publishing

Make the *method name* the identity primitive: a single `MethodSpec` registry
that all three dispatch sites resolve through, seeded with the built-ins, and
opened to third parties via a `langres.methods` entry-points group (the
standard Python plugin seam — pytest/flake8/keyring all use it; the repo
already hand-maintains `[project.scripts]`, `pyproject.toml:49-52`, so the
packaging precedent exists in-repo).

**Before (today)** — running a published paper's method:

```python
# Hand-wire core, lose identity:
from langres.core.modules.llm_judge import LLMJudge, parse_binary_yes_no
judge = LLMJudge.from_env(
    model="openrouter/openai/gpt-4o-mini",
    prompt_template=PEETERS_PROMPT,
    response_parser=parse_binary_yes_no,
)
verdict = langres.link(a, b, judge=judge)
verdict.judge_used   # "custom"  <- identity erased (presets.py:404-406)
```

**After (O1):**

```python
# In a third-party package, langres-peeters2023/pyproject.toml:
#   [project.entry-points."langres.methods"]
#   peeters2023 = "langres_peeters2023:METHOD"

verdict = langres.link(a, b, judge="peeters2023",
                       model="openrouter/openai/gpt-4o-mini")
verdict.judge_used   # "peeters2023"
verdict.model        # "openrouter/openai/gpt-4o-mini"
```

with a registry entry shaped roughly like:

```python
class MethodSpec(BaseModel):
    name: str                        # "zero_shot_llm", "peeters2023"
    build: Callable[..., Module[Any]]  # (schema, *, model, **params) -> Module
    default_threshold: float         # absorbs _DEFAULT_THRESHOLDS (presets.py:89)
    score_type: str                  # absorbs _SCORE_TYPE_BY_JUDGE (presets.py:118)
    requires_extra: str | None       # "llm" -> actionable ImportError, like
                                     # data/registry.py:186-194 already does
```

**Id grammar — a decision to make deliberately, once.** The task sketch
`judge="zero_shot_llm/gpt-4o-mini"` collides with reality: model ids
themselves contain slashes (`openrouter/openai/gpt-4o-mini`). First-slash
splitting would parse, but it burns `/` on the model axis. Recommendation
inside O1: keep `model=` an orthogonal kwarg (it already exists on both
verbs), and reserve `/` for HF-style **author namespacing** of third-party
methods (`judge="jdoe/ditto"`), with bare names reserved for built-ins.
Cheap to reserve now; expensive to retrofit after names ship.

- **Benefits.** Collapses the three dispatch sites into one — this *is* the
  #55 debt, and #103 falls out of it (register a `prompt_llm` spec backed by
  `LLMJudge`). Makes `methods.py`'s names (currently harness-only, §1.2)
  usable from the verbs. The announcement story is the strongest of the four:
  "`pip install langres-ditto` → `dedupe(records, judge='ditto')`" is
  concrete, demoable, and mirrors how the Python ecosystem already trusts
  code.
- **Drawbacks.** Code distribution only: no pinned prompt/threshold/eval
  claims travel with the name (a package can change behavior under the same
  name — pip's trust model, for better and worse). Entry-points scanning must
  respect the import-light discipline (`tests/test_import_budget.py`):
  `importlib.metadata.entry_points(group=...)` reads metadata without
  importing, so lazy load-on-first-use preserves the budget — but this must be
  tested, not assumed. Name collisions across packages need a defined rule
  (error loudly, like `registry.py:104-105`).
- **Risks.** The `MethodSpec` refactor touches the verbs' happy path; the
  fail-fast `"auto"` semantics, spend-cap wrapping, and per-judge thresholds
  (`presets.py:364-423`) must survive byte-for-byte. Mitigation: built-ins
  move first behind the existing test suite; the entry-points group lands
  separately.
- **Reversal cost.** Low-to-medium. The registry is additive (the four
  built-in names keep working); the entry-points group can be abandoned
  without breaking anything shipped in core. Public third-party names, once
  adopted, are forever-ish — hence getting the grammar right first.

### O2 — artifact-first: `Resolver.from_pretrained` + Hub hosting

The shareable unit is the serialized `Resolver` (or judge) — `resolver.json`
+ sidecars + a model card — hosted anywhere, with HF Hub as the default
because an artifact is already a directory of files.

**Before (today):**

```python
resolver = Resolver.load("artifacts/company_v0")   # local dir only
```

**After (O2):**

```python
resolver = Resolver.from_pretrained("jdoe/abtbuy-gpt4omini-calibrated")
# -> huggingface_hub.snapshot_download(...) -> Resolver.load(local_dir)
# card: README.md with pinned eval claims + the EvalReport tearsheet embedded
```

- **Benefits.** Carries the *exact* bundle §2.1 says a method is: config +
  prompt (`LLMJudge.config` already serializes `prompt_template` and
  `system_prompt`, `llm_judge.py:478-501`) + calibrated threshold (the
  Clusterer slot) + sidecar weights (compiled DSPy program, fitted models).
  No code execution on load — the invariant holds. The card slot directly
  answers the cost-accounting note's §8 open question (what an artifact
  *achieves and costs*). `from_pretrained('user/ditto-abtbuy')` is the
  headline HF users already know how to read.
- **Drawbacks (all verified against the code).** (a) Schema coupling: the
  artifact names its schema; the consumer needs the class registered (§1.3) —
  so cross-user artifacts *depend on* a code-distribution channel anyway.
  (b) Parser/serializer callables don't travel (`llm_judge.py:485-488`) —
  the Peeters method is not fully expressible as an artifact today.
  (c) A third-party *component* in an artifact still needs its package
  installed, or `get_component` raises `UnknownComponentType`
  (`registry.py:124-131`). (d) **Layout hard-fail:** an artifact saved under
  an older `artifact_version` refuses to load (`resolver.py:861-866`).
  Publishing public artifacts at 0.x under that policy manufactures breakage;
  a compat/migration policy must precede the hub, not follow it.
- **Risks.** Publishing creates compat *obligations* — the reversal cost is
  reputational, not technical. And until trained students exist, most
  artifacts are prompt+threshold JSON: a hub that launches thin can define the
  project as thin.
- **Reversal cost.** Medium-high once public artifacts exist; near-zero while
  it stays a design.

### O3 — layered O1 + O2 (code via pip, weights/config via artifacts)

Exactly transformers' own split: library + third-party *code* through the
package channel; *configs, prompts, thresholds, weights* through artifacts. A
method package can ship its artifact(s), and an artifact can name components
that a method package registers. O1 resolves O2's dependency problems ((a) and
(c) above: schemas and components arrive via the package); O2 gives O1's names
pinned, reproducible payloads and a card.

- **Benefits.** The only option that makes *both* maintainer sentences true:
  "call a specific model from a specific paper" (O1 name) *and* "evaluate it
  for themselves" against pinned claims (O2 card + §4).
- **Drawbacks.** Two workstreams; sequencing is the whole game. Built at once,
  it is over-building for a 0.2.0 library with (today) zero external
  publishers.
- **Reversal cost.** Inherits each layer's; the layering itself adds none.

### O4 — minimal near-term: fix #103 + surface identity, defer everything else

**After (O4):**

```python
clusters = langres.dedupe(records)         # judge="auto"
clusters.judge_used   # "zero_shot_llm"
clusters.model        # "openrouter/openai/gpt-4o-mini"   <- new; today: gone

verdict = langres.link(a, b, judge="prompt_llm",           # new preset -> LLMJudge
                       prompt_template=PEETERS_PROMPT,
                       response_parser="binary_yes_no")    # named, serializable
verdict.model         # "openrouter/openai/gpt-4o-mini"
```

Three concrete edits: (1) a `prompt_llm` preset resolving to `LLMJudge`, wired
into all three dispatch sites, with `response_parser` accepting a *registered
name* so it serializes (fixing `llm_judge.py:485-488`'s round-trip gap for the
built-in parsers); (2) `model` on `ResolvedJudge`/`DedupeResult`/`LinkVerdict`
(and the embedder model surfaced when a `VectorBlocker` is in play); (3) a
decision recorded on #103's open question — new name vs. re-backing
`zero_shot_llm` (recommend **new name**: `zero_shot_llm`→`DSPyJudge` is load-
bearing for compiled-judge work, and a silent backing-class swap changes
behavior for existing callers).

- **Benefits.** Days, not weeks; directly discharges the visibility complaint
  and #103; zero new public surface beyond two result fields and one preset
  name.
- **Drawbacks.** No publishing seam; the three dispatch sites *stay* triplicated
  (the `prompt_llm` wiring adds a fourth branch to each — the debt compounds).
  Names minted ad hoc now (`prompt_llm`, the parser-name convention) become
  the legacy any later registry must honor.
- **Risks / reversal cost.** Minimal; everything is additive. The real cost is
  path-dependence — which is why §6 recommends doing O4 *inside* O1's
  vocabulary rather than before it.

### The announcement story, compared

| Option | The release-post sentence | Honest caveat |
|---|---|---|
| O1 | "`pip install langres-ditto` → `dedupe(records, judge='ditto')` — anyone can publish an ER method as a package." | Claims about quality live in the package README, unpinned. |
| O2 | "`Resolver.from_pretrained('user/ditto-abtbuy')` — download the exact matcher, prompt, threshold, and card." | Thin until trained artifacts exist; schema/component deps still need pip. |
| O3 | Both of the above. | Ships last. |
| O4 | "Every result now tells you exactly which model matched your records — and you can run any paper's prompt by name." | No third-party story yet. |

---

## 4. The evaluation tie-in: named method × registered benchmark

The pieces for "any method, any benchmark, one line, with a cost column"
mostly exist; what is missing is the *join* on identity:

```python
# Today (works, but the method must be hand-built and identity is manual):
from langres import eval as lev
bench = lev.get_benchmark("abt_buy")
candidates, gold = lev.candidates_for(bench, split="test")
result = lev.evaluate(my_judge_module, candidates, gold, budget_usd=2.0)

# The leaderboard-shaped missing one-liner (O1 provides the name -> module join):
result = lev.evaluate_method("peeters2023", "abt_buy",
                             model="openrouter/openai/gpt-4o-mini",
                             budget_usd=2.0)
```

Gaps, in dependency order:

1. **A name→judge builder usable outside the race harness.** `build_judge`
   needs a schema (`presets.py:228`), `make_resolver_factory` needs a
   `BlockingBenchmark` (`methods.py:268`) — nothing today turns
   ("method name", "benchmark name") into a scored table. The O1 registry is
   exactly this function.
2. **Identity on results** (§1.1 / O4), so a row can read
   "peeters2023 · gpt-4o-mini · abt_buy · F1 0.82 · $0.31/1k pairs" — the
   cost column already exists (`CostTrack`, `EvalReport`).
3. **Cross-run aggregation.** `EvalReport` is per-run; `RunRecord`/`recipe_id`
   (`runs.py`) persist runs but nothing renders "methods × benchmarks". The $0
   v0.4 version is a repo-hosted results file + a generated table (the
   `portfolio_race` example is the seed) — a *community* leaderboard (external
   submissions) additionally needs protocol pinning and is out of scope here.
4. **Protocol pinning for comparability.** `candidates_for` already pins the
   benchmark's blocker, `blocking_k`, split, and seed (`eval.py:102-170`) —
   the leaderboard must record those (plus grid/threshold policy: `evaluate`'s
   default grades at best-F1 threshold, `benchmark.py:1360-1361`) so rows are
   apples-to-apples. `recipe_id` covers this within one install; a published
   row needs the protocol fields spelled out.

---

## 5. Adjacent work: the training loop (dependency runs both ways)

A parallel plan is scoping the training loop —
`docs/plans/20260713_training_loop_plan.md` (PR #109, in flight alongside this
note; its §3.6 "Framework deltas" references this doc as the design that
decides where trained checkpoints live and how they are named/versioned). Two
dependencies, both ways:

- **Training → publishing.** Trained students are the artifacts that make
  O2's hub worth visiting — prompt+threshold bundles alone are thin (§2.1.4).
  The training plan's concrete outputs are this design's **first real
  artifact customers**: fine-tuned judge checkpoints (CrossEncoderJudge
  safetensors in a `state_dir` sidecar; Qwen3 QLoRA adapters). The publishing
  seam should not front-run this payload.
- **Publishing → training.** The training loop should *target the existing
  artifact contract* (config + `SerializableState` sidecars,
  `resolver.py:722-755`) so its outputs are publishable without rework — the
  compiled-DSPy path already does this (`dspy_judge.py:216-220`). And the O1
  method identity should be stamped into `JudgementLog` records and
  `RunRecord` at birth, so training-data lineage can say *which named method*
  produced each silver label. Deciding the id grammar in v0.3 (§3-O1) is what
  makes that stamp stable.

The trained-artifact case also sharpens what O2's model card must capture,
beyond §2's config+prompt+threshold bundle (requirements per the training
plan; carried here as the card's target schema, not re-derived):

1. **Base-model identity** (which pretrained weights the checkpoint adapts).
2. **Data recipe including silver-teacher identity** — which LLM harvested
   the labels, at what threshold: exactly the `JudgementLog` lineage stamp
   above, closing the loop.
3. **The full license chain** — base weights *and* training data (e.g. a
   CC-BY-4.0 instruct set is shippable; teacher outputs need provenance).
4. **Eval provenance** — dataset/split and the honest-protocol threshold
   (§4.4's protocol-pinning fields).

One load-path constraint follows: training-only dependencies (unsloth / trl /
peft) stay dev-group-only, so the artifact format must not require them at
load time — inference-side loading rides the existing `[semantic]` extra
(sentence-transformers/torch), consistent with the lazy-extras discipline the
registry already enforces (`registry.py:61-75`).

---

## 6. Recommendation and staged path

> **Recommendation (marked as such): O3, sequenced — v0.3 ships O4's fixes
> expressed in O1's vocabulary; v0.4 ships the open namespace and the
> evaluation join; O2's hub waits for the training loop's artifacts.**
> The alternatives remain live: pure O4 if the next two releases are
> bandwidth-bound (everything in it survives into O1 unchanged *if the naming
> is chosen registry-shaped*); O2-first only if an announcement moment
> matters more than a publisher ecosystem — and then only after the
> artifact-compat policy exists.

**v0.3 (identity):**
1. One `MethodSpec` judge registry replacing the three dispatch switches
   (`presets.py:build_judge`, `resolver.py:_build_module_for_judge`,
   `methods.py:_make_module_builder`) — closing #55's debt; built-ins only;
   `"auto"`'s fail-fast, spend-cap, and per-judge-threshold semantics
   preserved under the existing test suite.
2. Close #103 through it: `judge="prompt_llm"` → `LLMJudge`, with
   `response_parser`/`record_serializer` accepting registered names so the
   config round-trips (`llm_judge.py:485-488`); `zero_shot_llm` keeps backing
   `DSPyJudge` (no silent behavior change).
3. Identity on every result: `model` on `LinkVerdict`/`DedupeResult` (and the
   embedding/blocker model surfaced), stamped from `ResolvedModule.model`
   which already exists (`presets.py:364-370`); same fields into
   `JudgementLog`.
4. Reserve the id grammar: bare names = built-ins; `/` reserved for
   `author/method` namespacing; `model=` stays an orthogonal kwarg.

**v0.4 (publishing + evaluation join):**
5. `langres.methods` entry-points group, loaded lazily (import-budget test
   extended to prove metadata scanning stays light); collision = loud error.
6. `evaluate_method(name, benchmark_name, *, model, budget_usd, ...)` in
   `langres.eval`; a `langres-method-template` example package; the
   repo-hosted methods×benchmarks results table with cost columns.

**v0.5+ (gated on the training loop producing weight-bearing artifacts):**
7. Artifact compat policy (replace the both-directions hard-fail,
   `resolver.py:861-866`, with a documented migration story), the model-card
   format (config + pinned eval claims + `EvalReport`, answering the
   cost-note §8 question), then `Resolver.from_pretrained` over optional
   `huggingface_hub`.

## 7. What we are NOT proposing

- **No hosted langres hub service.** HF Hub (or any file host) suffices;
  langres ships a fetch-and-load path at most.
- **No code execution from artifacts** — no `trust_remote_code` equivalent.
  Third-party *code* travels as a pip package (O1); artifacts stay pure data
  rebuilt through the registry (`resolver.py:15-16`).
- **No renaming of the four existing judge names in 0.3.** `zero_shot_llm`,
  `embedding`, `string`, `auto` keep their exact semantics; identity is added
  around them, not under them.
- **No community-submission leaderboard yet.** §4's v0.4 table is repo-hosted
  and maintainer-curated; open submissions need protocol governance this note
  only sketches.
- **No schema-as-data mechanism yet.** §2.1.2's coupling is real, but the
  package channel covers it for v0.4; a declarative schema format is its own
  design note when artifact portability becomes the bottleneck.

---

## Appendix — key file:line index (verified against `main` @ `53a08ed`)

| Claim | Location |
|---|---|
| `JudgeName` literal (4 names) | `src/langres/core/presets.py:83` |
| Per-judge default thresholds (E12) | `src/langres/core/presets.py:89-93` |
| `_VECTOR_MODEL_NAME = "all-MiniLM-L6-v2"` (invisible embedder) | `src/langres/core/presets.py:105` |
| Auto-judge model pick by API key; gpt-4o-mini / gpt-5-mini | `src/langres/core/presets.py:107-112,197-225`; `src/langres/clients/openrouter.py:88` |
| `resolve_judge` returns the model; verbs drop it | `src/langres/core/presets.py:364-423,531-598` |
| `judge=Module` → `judge_used="custom"` | `src/langres/core/presets.py:404-406` |
| `LinkVerdict` / `DedupeResult` fields (no `model`) | `src/langres/verbs.py:81-101,104-127` |
| `model=` ignored for non-LLM judges | `src/langres/core/presets.py:387-389`; `src/langres/verbs.py:298` |
| Provenance `"model"` is a convention, not a contract | `src/langres/core/modules/dspy_judge.py:289,311`; `llm_judge.py:1077`; `src/langres/core/models.py` (`PairwiseJudgement` has no model field) |
| Method names; harness-only factory | `src/langres/_method_names.py`; `src/langres/methods.py:185-265,268-326` |
| Component registry, lazy map, collision error | `src/langres/core/registry.py:61-132` |
| Schema registry; artifact references schema by name | `src/langres/core/registry.py:135-167`; `src/langres/core/blockers/all_pairs.py:161-166` |
| Artifact manifest; "no code and no pickle" | `src/langres/core/resolver.py:15-16,656-755` |
| Artifact layout hard-fail (both directions) | `src/langres/core/resolver.py:848-866` |
| `from_schema` judge switch (3rd dispatch site, uncapped) | `src/langres/core/resolver.py:82-137,380-385` |
| `LLMJudge` seams; parser/serializer not serialized | `src/langres/core/modules/llm_judge.py:339-352,478-501,485-488,122` |
| Compiled DSPy program persists via sidecar | `src/langres/core/modules/dspy_judge.py:207-220` |
| `evaluate` signature; best-F1-threshold default | `src/langres/core/benchmark.py:1341-1361` |
| `candidates_for` pins blocker/split/seed | `src/langres/eval.py:102-170` |
| Benchmark manifest (9 loadable + external) | `src/langres/data/registry.py:206-320` |
| `config_dict` / `compute_recipe_id` | `src/langres/core/resolver.py:687-720`; `src/langres/core/runs.py:223` |
| Three-dispatch-site rule; #55 deferral | `.claude/rules/component-design.md` ("When Adding New Components") |
| Hand-edited packaging precedent | `pyproject.toml:49-52` |
| Artifact cost/metric provenance open question | `docs/research/20260709_cost_accounting_design.md` §8 |
