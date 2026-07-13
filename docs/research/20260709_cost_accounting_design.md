# Cost accounting in langres — a design note for the tracking layer

*Design note for [issue #100](https://github.com/fxd24/langres/issues/100)
("Cost tracking: comprehensive design"). Scope: how langres should account for
the cost of an experiment run coherently — what it stores, what it derives, where
each concern lives, and how `RunRecord` evolves. Documentation only; no code
changes. Written against `main` @ `33dd0db` (post-PR #99/#98).*

> **Provenance of claims.** Every `file:line` and every statement about the
> langres codebase below was read directly from `main` while writing this note.
> The external prior-art facts in §5 (OpenTelemetry, W&B Weave, Langfuse,
> Phoenix, MLflow, litellm price-table counts, provider cache-billing rules) were
> **supplied to this note as already-verified research** and are cited but **not
> independently re-verified here** — treat them as inputs, and re-check the live
> URLs before implementing against a specific API. Everything else is argument.

---

## 0. Executive summary

1. **The `{llm, embedding, infra}` decomposition #100 proposes is the wrong
   axis.** It mixes one thing that is langres's job (metering a paid model
   invocation) with two that are not. "Embedding cost" is not a cost *category* —
   a paid embedding API is just another metered model call; a local embedder is
   compute, which the ROADMAP brain/body seam explicitly delegates to the
   consumer (like streaming and temporal support). "Infra cost" is consumer-side
   by the same rule. The honest axes are **per-call** (a metered model
   invocation) and **per-stage** (blocking / judging / compiling, already
   expressible via `parent_run_id`).

2. **A cost axis is constitutive of the benchmark seam, not decoration.** langres
   exists to *compose, benchmark, and tune* (ROADMAP §1). "Which method wins at
   what dollar cost" is the second column of every benchmark table langres will
   ever produce. Under-building this is under-building the product.

3. **Tokens are the stored fact; dollars are derived.** Three independent forces
   make a stored dollar scalar a lossy record: ~10× price variance for the *same
   model* across serving providers, prices that move over time (the
   `PRICES_PER_1M` comment already records GLM's price moving —
   `openrouter.py:45`), and the fact that **no price table anywhere carries an
   effective date** (§5). Store the usage vector; derive dollars on read.

4. **Four separated concerns**, today tangled into one scalar and one hardcoded
   table: **metering** (facts — core), **pricing** (policy — tracking layer,
   effective-dated), **budgeting** (the one runtime guardrail that legitimately
   needs a live conservative price — argues for an injected `PriceBook`, not the
   hardcoded module-global table), and **monitoring** (a query over the stored
   records; production monitoring is consumer-side per ROADMAP §5).

5. **Schema**: adopt OpenTelemetry GenAI's token vocabulary flattened to
   snake_case, with OTel's *subset* semantics documented explicitly and
   normalized at the boundary (Anthropic's disjoint counts must be summed);
   adopt litellm's price-key names for the price book; adopt Weave's
   effective-dated derive-on-read for re-pricing; and **preserve `cost_is_real` +
   the provider-reported cost as a distinct authoritative fact** — the one thing
   no other system stores.

6. **`RunRecord` grows additively `v:1 → v:2`.** Honest caveat: `RunStore.read`
   never branches on `v` and there is no migration test (verified —
   `runs.py:414-445`). Store the usage vector + provider-reported cost as facts,
   **plus a denormalized derived total stamped with `price_book_version`**,
   because our store is append-only JSONL (not Weave's SQL) — a reader without
   the price book must still see a number and know which prices produced it.

7. **One open question this note surfaces but does not resolve**: should the
   `Resolver` artifact carry cost + metric provenance, closing the ROADMAP §5 gap
   ("thresholds + metric provenance" — the provenance half was never built)? It
   matters (an artifact that says what it *is* but not what it *achieves and
   costs* is not choosable) and it is a strictly larger change than #100.

8. **Staged plan**: (a) additive usage vector — settles nothing controversial;
   (b) auto-wire `record_cost` + `capture_run` into the benchmark path (the
   deferred "Stream C"), with the DSPy-compile `$0` bug as a sub-task; (c)
   `PriceBook` seam + effective dates; (d) unify the three budget mechanisms.

**Verdict on #100 as written:** its instinct ("design before code") is right and
its list of *sub-tasks* is mostly right, but its central *cost model* (§ question
1, the `{llm, embedding, infra}` split) should be replaced before any of the
sub-tasks land, or every subsequent schema decision inherits the wrong axis.

---

## 1. Current state (verified against `main` @ `33dd0db`)

The tracking spine (PR #99) is real and honest as far as it goes; the gap is that
cost is a **single scalar that almost nothing writes**.

**What exists and is honest:**

- Per-call cost facts already exist, in memory, on every `PairwiseJudgement`.
  `LLMJudge.forward` stamps `provenance` with `model`, `cost_usd`,
  **`cost_is_real`** (bool: OpenRouter-billed vs. pinned estimate), `provider`
  (the upstream serving provider), `prompt_tokens`, `completion_tokens`
  (`llm_judge.py:429-438`; async path identical at `:588-598`).
- `cost_is_real` is set by `_billing` (`llm_judge.py:326-337`), which prefers
  OpenRouter's actual billed cost via usage accounting
  (`parse_openrouter_billing`, `openrouter.py:330-347`, reading the
  `llm_provider-x-litellm-response-cost` hidden header, `openrouter.py:283`) and
  falls back to the pinned-table estimate (`_calculate_cost`,
  `llm_judge.py:711-729`) only when no real cost is present.
- `SpendMonitor` (`openrouter.py:399-455`) is a **passive ledger** (`add` /
  `check`), not a meter — it accumulates whatever honest per-call cost the caller
  feeds it and warns/raises on the total. Real per-call cost is computed upstream
  in `LLMJudge._billing`.

**What is missing or broken:**

- **`RunRecord.spend_usd: float = 0.0` is a single scalar** (`runs.py:188`), with
  `budget_exceeded: bool` beside it (`runs.py:189`). No decomposition, no tokens,
  no provider, no price-version.
- **Almost nothing writes it.** `_RunHandle.record_cost` (`runs.py:515-518`) is
  the only setter, and the *only* callers repo-wide are `tests/test_runs.py:599`
  (synthetic `0.42`) and `examples/research/experiment_tracking_demo.py:137`
  (literal `0.0`). A real `run_methods` run persists **no** spend.
- **`record_cost` does not forward to the tracker.** `log_metrics` /
  `log_artifact` both forward to the `ExperimentTracker` *and* stash for the
  record (`runs.py:504`, `:509`); `record_cost` only mutates the record
  (`runs.py:515-518`). So cost never reaches MLflow / W&B even when a tracker is
  attached.
- **The benchmark path never opens a run.** `benchmark.py`
  (`run_method`/`run_methods`) has zero `capture_run` calls. `capture_run` is
  wired into exactly one production seam: `DSPyJudge.compile`
  (`dspy_judge.py:368`).
- **DSPy compile records `$0`.** That one wired seam carries an in-source NOTE
  deferring the fix to #100: "this compile run records $0 spend … deferred to
  issue #100" (`dspy_judge.py:366-367`). A paid `mipro` compile silently
  under-reports optimization spend.
- **The only per-call disk sink is `JudgementLog`**, opt-in (`log=None` default on
  the verbs); token counts only reach disk with `features=True`.
- **Pricing is a hand-maintained 10-model table.** `PRICES_PER_1M`
  (`openrouter.py:40-65`); an unknown model silently prices to `$0` via the
  `litellm.completion_cost` fallback (`llm_judge.py:725-729`). The table's own
  comment notes a price already moved (`openrouter.py:45`).
- **Three budget mechanisms, three defaults, two failure semantics:**
  | Mechanism | Where | Default | On breach |
  |---|---|---|---|
  | `_SpendCappedModule` | presets / verbs (`presets.py:290`) | `DEFAULT_BUDGET_USD = 1.0` (`presets.py:97`) | **raises** `BudgetExceeded` (carries `partial_judgements`) |
  | `BudgetedModuleRunner` | benchmark (`benchmark.py:924`) | `budget_usd=20.0`, soft `15.0` (`benchmark.py:971-972`) | **silently truncates** the input, returns what was scored |
  | `TeacherLabeler` cap | bootstrap (`labelers.py:201-204`) | its own | pre-flight truncate (mirrors the runner) |
- **The `Resolver` artifact carries components only.** `ArtifactManifest`
  (`serialization.py:72-88`) has `artifact_version`, `langres_version`,
  `components`, `checksums` — no metrics, no cost. ROADMAP §5 promises the
  artifact records "thresholds + **metric provenance**" (`ROADMAP.md:190-192`);
  the provenance half was never built.
- **No model-size field anywhere.** `RunContext.llm_model` (`runs.py:146`) records
  the model id on the recipe, but the stored `metrics` dict (a `MethodResult`
  dump) records the *method* name, not a model id or size class.

---

## 2. Scope, derived from the ROADMAP brain/body seam

The ROADMAP draws one line that settles most of #100's scope questions before
they are asked:

> *Engine intelligence in langres; data, persistence, visibility in the
> consumer.* (`ROADMAP.md:25-27`)

Streaming, temporal, and the cluster store are all delegated to the consumer
(brainsquad) on exactly this basis (ROADMAP §2.3, §5). **Cost falls on the same
fault line**, and it splits into three, not three-of-a-kind:

- **A paid model invocation is engine-side and is langres's job to meter.** An
  LLM judge call and a *paid* embedding-API call are the *same kind of event*: a
  metered model invocation with tokens in, tokens out, and a provider-reported
  cost. langres already meters the first (`_billing`); it should meter the second
  the identical way, through the same seam — not a parallel "embedding cost"
  category.
- **Local compute is not a langres cost category.** A local
  sentence-transformers / fastembed embedder spends GPU-seconds, not dollars.
  Attributing an imputed dollar rate to local compute requires an instance price
  the consumer owns, not langres. `duration_seconds` (`runs.py:179`) is already
  recorded and is the honest, unit-free proxy; that is the right amount of local-
  compute accounting for langres to do.
- **Infrastructure cost is consumer-side by the §5 rule.** `$/hr × hours`,
  GPU-hours, and instance rates live where the infra lives — the orchestration
  layer. langres records the *duration*; the consumer multiplies by *its* rate.

**Consequence: replace the axis.** #100's design-question 1 offers "one scalar, a
structured `{llm, embedding, infra}`, or per-component provenance rolled up." The
right answer is the third framing, but keyed on the two axes the architecture
already has:

- **Per-call** — the atomic metered fact: one model invocation's usage vector +
  provider-reported cost + `cost_is_real` + `model` + `provider`. This is exactly
  what `PairwiseJudgement.provenance` already holds; the work is to *aggregate and
  persist* it, not to invent it.
- **Per-stage** — blocking / judging / compiling, expressed through the
  `parent_run_id` lineage that already exists (`runs.py:130-131`; ROADMAP notes a
  sweep parents its per-seed children and a compile parents its eval runs,
  `EXPERIMENTS.md:277-278`). A stage's cost is the roll-up of its calls; a
  parent's cost is the roll-up of its children. No `{llm, embedding, infra}`
  enum is needed — the *kind* of a call is carried by its `model`/`decision_step`,
  and "embedding vs judge" is a stage/step distinction, not a top-level cost
  bucket.

This reframing is not cosmetic: it removes two fields that would otherwise be
permanent, mostly-`0.0`/`None` columns on `RunRecord` inviting confused readers,
and it aligns cost with the lineage model the tracking layer already commits to.

---

## 3. Four separated concerns

The current design fuses four concerns into `spend_usd` + `PRICES_PER_1M`.
Separating them is the core architectural move.

1. **Metering — facts, in `core`.** Count what happened: tokens by class,
   provider, model, and the provider-reported cost when the provider gives one.
   Facts never encode a price *decision*. This lives where the call is made
   (`LLMJudge`, and later any embedding client) and already half-exists in
   `provenance`.
2. **Pricing — policy, in the tracking layer, effective-dated.** Turn tokens into
   dollars. Prices are *policy* (negotiated rates, provider choice, they move over
   time), so they do not belong hardcoded in a `clients` module-global. A
   `PriceBook` maps `(model, provider, effective_date) → per-token rates`. This is
   read-time and swappable.
3. **Budgeting — the one runtime guardrail that needs a live price.** A pre-flight
   cap must *estimate cost before spending*, so it is the single place core
   legitimately needs a conservative live price (today: `per_token_worst_price`,
   `openrouter.py:184-195`, over `PRICES_PER_1M`). This argues for **injecting a
   `PriceBook`** into the budget mechanisms rather than reaching for the hardcoded
   table — same seam as pricing, conservative mode.
4. **Monitoring — a query over the stored records.** "How much did this sweep
   cost?", "cost per F1 point across methods?" are *reads* over `RunStore`, not a
   new subsystem. Per ROADMAP §5, *production* cost monitoring (live dashboards on
   deployed inference) is consumer-side; langres owns *experiment* cost
   accounting.

The failure mode of the current design is that all four are collapsed: the
hardcoded table is simultaneously the meter's fallback, the only price policy, and
the budget estimator, so you cannot change a price without touching a `clients`
module, cannot re-price history at all, and cannot budget against a negotiated
rate without editing source.

---

## 4. Tokens are the stored fact; dollars are derived

Three verified forces make a stored dollar scalar lossy:

- **~10× same-model provider variance.** OpenRouter serves
  `meta-llama/llama-3.3-70b-instruct` at $0.10/M (DeepInfra) and $1.04/M
  (Together) *for the same model* [supplied]. langres already captures the
  serving `provider` in provenance precisely because it matters — but a bare
  `spend_usd` throws that dimension away.
- **Prices move.** `PRICES_PER_1M`'s own comment records GLM's list price moving
  on a refresh (`openrouter.py:45`). A dollar figure computed last month against
  a price that no longer exists is unreconstructable from a token count if the
  price is gone — *unless the dollars were stored*.
- **No price table carries an effective date.** Across litellm's 2,941 price
  entries, **zero** carry an `effective_date` or price version [supplied]; the
  only price history is the git history of the JSON file. So a stored dollar
  figure is, in practice, *the only surviving record of the price that produced
  it*.

The resolution is the one Weave already ships (§5): **store tokens as the durable
fact and derive dollars on read against an effective-dated price book** — *and*,
because our store is append-only JSONL with no join engine, **also denormalize
the derived total** so a reader without the price book still sees a number
(§6, §7). This is not a contradiction: tokens + provider-reported cost are the
*authoritative facts*; the denormalized derived total is a *convenience stamp*
that records which price version produced it and can always be recomputed.

---

## 5. Prior art — what six tracking systems actually do

*(All facts in this section were supplied to the note as verified research; cite,
don't re-derive. URLs are for the implementer to re-check.)*

| System | Token vocab | Cost storage | Re-pricing | Reasoning tokens | Cache tokens |
|---|---|---|---|---|---|
| **OTel GenAI** | `gen_ai.usage.input_tokens`, `output_tokens`, `cache_read.input_tokens`, `cache_creation.input_tokens`, `reasoning.output_tokens` (all `development`) | **defines no cost attribute at all** — usage-only by design | n/a | yes | yes |
| **W&B Weave** | stored tokens | **cost never stored**; derived on read | **yes** — dated `add_cost`, join at `effective_date <= started_at` | no | `cache_read`/`cache_creation` price cols |
| **Langfuse** | `usage_details: map<str,int>` (open) | `cost_details: map<str,double>` (open); ingested wins over inferred | no (newest `start_date`, not obs-timestamped; issue #12184 wontfix) | — | one-key-only (see trap) |
| **Phoenix / OpenInference** | full token tree | **full cost tree mirroring the token tree** (`llm.cost.prompt_details.cache_read`, …) | "cost is not retroactive" | — | yes |
| **MLflow** | `mlflow.chat.tokenUsage` (`cache_read_input_tokens`, `cache_creation_input_tokens`) | `mlflow.llm.cost`, computed at trace time via litellm | no native re-pricing | no | yes |
| **litellm** | price entries | `input_cost_per_token` (2468), `output_cost_per_token` (2465), `cache_read_input_token_cost` (669), `cache_creation_input_token_cost` (225), `output_cost_per_reasoning_token` (45) | **no effective_date on any of 2,941 entries** | 45 entries | yes |

Sources (re-check before implementing): OTel GenAI semantic conventions
(`open-telemetry/semantic-conventions-genai`, moved 2026-05; token PRs #3163
merged 2026-01-27, #3383 merged 2026-04-27; open PR #197 proposes a further
per-modality restructure, not merged); W&B Weave `llm_token_prices` /
`client.add_cost`; Langfuse `usage_details`/`cost_details` + issue #12184;
Phoenix/OpenInference cost tree; MLflow `mlflow.chat.tokenUsage` / `mlflow.llm.cost`;
litellm `model_prices_and_context_window.json` + `litellm.register_model`.

**Three load-bearing lessons:**

- **Weave is the reference implementation for re-pricing.** Cost is *never*
  stored; it is derived on read by joining stored tokens against a dated
  `llm_token_prices` table (`cache_read_input_token_cost`,
  `cache_creation_input_token_cost`, `effective_date`, `pricing_level`).
  `client.add_cost(llm_id, prompt_token_cost, completion_token_cost,
  effective_date)` inserts a dated row and the *next* query re-derives every
  historical call at the price where `effective_date <= call.started_at`. This is
  exactly the effective-dated derive-on-read langres should adopt — modulo the
  storage caveat in §6/§7 (Weave has SQL; we have JSONL). Weave captures no
  reasoning tokens.

- **THE TRAP: token subset semantics are not universal, and getting it wrong
  double-counts or under-counts silently.** OTel's convention is that
  `cache_read`/`cache_creation` are **included in** `input_tokens` and
  `reasoning` is **included in** `output_tokens` (subsets). Langfuse mandates the
  **opposite**: each token counted in *exactly one* key; `input` must *exclude*
  tokens already counted in another `input_*` key. Raw providers disagree with
  each other too: OpenAI's `prompt_tokens` **includes** `cached_tokens`, while
  Anthropic's `input_tokens` **excludes** cache tokens (so an Anthropic total is
  `input + cache_read + cache_creation`). **Any schema must pick one convention,
  document it, and normalize at the boundary** — or a cross-provider cost
  comparison is quietly wrong. This note picks OTel subset semantics (§6).

- **Cache economics are model- and provider-specific, and this is where langres's
  headline cost wins live.** OpenAI does not bill cache *writes*; its cache-*read*
  discount is 0.5×/0.25×/0.1× **depending on model**. Anthropic *does* bill writes
  (1.25× at 5m TTL, 2.0× at 1h) and reads at 0.1× [supplied]. Minimum cacheable
  prefix on OpenAI is 1024 tokens — so a ~66-token pairwise ER prompt can **never**
  cache, but **DSPy-compiled prompts** (bootstrapped demos + long instructions)
  and **`SelectJudge`'s shared-anchor set-wise prompts** routinely clear 1024
  tokens of stable prefix. Those are exactly the paths where langres's cost story
  lives (cheap distilled student, one-call-per-group set-wise), and today langres
  **cannot see** whether it is re-paying full price for the same prefix thousands
  of times. A usage vector with cache classes is what makes that visible.

---

## 6. Concrete schema proposal

### 6.1 The per-call usage vector (metering, `core`)

Flatten OTel's GenAI token vocabulary to snake_case, adopt OTel's **subset**
semantics, and **normalize at the boundary**:

```
usage = {
    "input_tokens":            int,   # OTel-subset: INCLUDES cache_read + cache_creation
    "output_tokens":           int,   # OTel-subset: INCLUDES reasoning
    "cache_read_input_tokens":  int,  # subset of input_tokens
    "cache_creation_input_tokens": int,  # subset of input_tokens
    "reasoning_tokens":         int,  # subset of output_tokens
}
```

Documented invariant (the anti-trap): `cache_read + cache_creation <=
input_tokens` and `reasoning <= output_tokens`. **Normalization is mandatory at
ingestion**: OpenAI usage maps almost directly (subset already); **Anthropic must
be summed** into `input_tokens` because its provider counts are disjoint. This is
a small, well-specified adapter per provider, and it is the difference between a
correct and a silently-wrong cross-provider table.

This is a strict superset of what `provenance` already carries
(`prompt_tokens`/`completion_tokens`, `llm_judge.py:434-437`): `input_tokens` /
`output_tokens` are the renamed OTel-canonical versions of those two, and the
three cache/reasoning fields are additive and default to `0`.

### 6.2 The price book (pricing, tracking layer, effective-dated)

Adopt litellm's **price-key names** so the book is drop-in compatible with
litellm's own table and its `register_model(dict|url)` runtime override:

```
price_row = {
    "model": str, "provider": str | None, "effective_date": date,
    "input_cost_per_token": float, "output_cost_per_token": float,
    "cache_read_input_token_cost": float | None,
    "cache_creation_input_token_cost": float | None,
    "output_cost_per_reasoning_token": float | None,
    "price_book_version": str,
}
```

Keyed on `(model, provider, effective_date)` — the provider dimension is what
captures the ~10× variance §4 documents; the effective date is the one thing
litellm's 2,941 entries lack. Derivation is Weave's rule: for a call, pick the
row with the greatest `effective_date <= call.started_at`. `PRICES_PER_1M`
becomes the seed data of the *default* price book, not a hardcoded fallback in a
client module.

### 6.3 Preserve the provider-reported cost as a distinct authoritative fact

This is langres's differentiator and must not be lost in the move to
derive-on-read. `_billing` already distinguishes OpenRouter's *actual billed
cost* from a table estimate via `cost_is_real` (`llm_judge.py:326-337`). **No
other system in §5 stores a provider-authoritative billed cost** — Weave/Phoenix
always *derive*, Langfuse takes an ingested cost but records **no flag** for
whether you got the ingested or the inferred number. langres already has the
flag. Keep, per call:

```
provider_reported_cost_usd: float | None   # from usage accounting; None if absent
cost_is_real: bool                          # the flag no one else keeps
```

So a stored call has both a *fact* (what the provider actually billed, when it
told us) and a *derivation* (what the price book computes from tokens). When both
exist they cross-check; when they diverge, `cost_is_real` tells you which to trust.

---

## 7. `RunRecord`: additive `v:1 → v:2`

**Honest caveat, verified:** `RunStore.read` never branches on `v`
(`runs.py:414-445` deserializes every line straight into `RunRecord` with no
version dispatch), and there is no migration test. So a `v` bump today is
*documentation*, not *behavior* — new optional fields are what actually keeps old
lines readable (Pydantic defaults fill them), and the `v` field only earns its
keep once a reader branches on it. Recommend adding both the fields and a
minimal `v`-dispatch + round-trip test in the same change, so `v:2` means
something.

Proposed additive fields on `RunRecord` (all optional, old lines stay valid):

```
# -- Cost (v:2, additive) --
usage_by_key: list[UsageBucket] | None = None     # §6.1 rolled up PER price-book key
provider_reported_cost_usd: float | None = None   # sum of §6.3 facts, when all real
cost_is_real: bool = False                    # True iff every call had a real cost
derived_cost_usd: float | None = None         # §6.2 derive-on-read, DENORMALIZED
price_book_version: str | None = None         # which prices produced derived_cost_usd
# spend_usd / budget_exceeded retained: spend_usd = the run's authoritative total
#   (provider-reported when cost_is_real, else derived), for back-compat readers.

# UsageBucket = the price-book key + the LLMUsage vector accumulated under it:
#   {model, provider, usage: {input_tokens, output_tokens, cache_read_input_tokens,
#                             cache_creation_input_tokens, reasoning_tokens}}
```

**Why `usage_by_key`, not a single flat `usage` vector?** Pricing is keyed on
`(model, provider, effective_date)` (§6.2). A run that mixes models — a
`CascadeJudge` is *defined* by cheap-student-then-expensive-teacher — or that
OpenRouter routes to two serving providers of the same model (~10× apart, §5)
would collapse into one anonymous token total, and re-pricing could then only
apply a single price row to the aggregate. That silently corrupts spend, which is
the exact failure this design exists to prevent. Bucketing by the price-book key
keeps the roll-up re-priceable.

`effective_date` needs no bucket: `RunRecord.started_at` already dates the run.
A run that straddles a price change is priced at its start — the per-call rows in
`JudgementLog` (§6.1) remain the **authoritative, per-call-dated fact**, and the
run roll-up is a denormalized convenience derived from them. When a run is long
enough for that to matter, re-derive from the call log, not the roll-up.

**Why denormalize `derived_cost_usd` when Weave stores nothing?** Weave derives on
read because it has a SQL store that can `JOIN` tokens against a live price table
at query time. Our store is **append-only JSONL** (`RunStore`, `runs.py:367-445`)
with no join engine and, deliberately, no dependency beyond stdlib+pydantic
(`runs.py:4-11`). A reader that only has the JSONL — an agent doing the
idempotent-replay two-liner (`EXPERIMENTS.md:271-275`), a diff across sessions —
must still see a dollar number without loading a price book. So we store the
tokens (re-derivable forever) *and* stamp the derived total with its
`price_book_version` (so a reader knows exactly which prices made it and can
re-derive if it disagrees). Best of both: durable facts + a convenience number
that is never anonymous.

**`spend_usd` stays** and becomes "the run's authoritative total" — provider-
reported when every call was real, else derived — so every existing reader
(`sum(r.spend_usd …)`, `EXPERIMENTS.md:274`) keeps working unchanged.

---

## 8. Open question to surface (not resolve): the artifact's cost + metric provenance

The `Resolver` artifact today is components-only (`ArtifactManifest`,
`serialization.py:72-88`). ROADMAP §5 defines the artifact as "feature extractors
+ blocking funnel config + judge … + thresholds + **metric provenance**"
(`ROADMAP.md:190-192`) — the metric-provenance half was never built.

**Why it matters:** the artifact is the brainsquad integration contract
(ROADMAP §5). An artifact that declares what it *is* (its components) but not what
it *achieves and costs* (held-out F1, `$/1k pairs`, the model + price it was
measured at) is **not choosable** — a consumer picking `person_v1` vs `person_v2`
has no basis in the artifact itself. Cost accounting done well (§6) produces
exactly the numbers that provenance slot wants: a run's `usage`, `derived_cost`,
`price_book_version`, and headline metric are a ready-made "this artifact scored
X at $Y/1k pairs on model M" stamp.

**Why it is out of scope for #100:** it changes the *artifact* contract and the
`save`/`load` surface (`resolver.py`), touches versioned serialization
(`ARTIFACT_VERSION`), and needs a decision on public/private provenance stripping
(consumer-side per §5). That is a strictly larger blast radius than adding fields
to a `RunRecord`. Recommend a follow-up issue that *depends on* #100's schema, so
the artifact's metric-provenance block reuses the same cost vocabulary rather than
inventing a second one.

---

## 9. Answering #100's six design questions

| # | Question | Disposition | Answer |
|---|---|---|---|
| 1 | Cost model / decomposition; back-compat with `v:1` | **settled (reframed)** | Reject `{llm, embedding, infra}` (§2). Store a per-call **usage vector + provider-reported cost** (§6), roll up per-run and per-stage via `parent_run_id`. Back-compat = additive optional fields, `v:1→v:2`, with a real `v`-dispatch test (§7). |
| 2 | Where cost is captured | **settled** | Metering in `core` at the call site (already in `provenance`); auto-wire `record_cost` from the aggregated calls in the `capture_run`/benchmark wrap (Stream C); make `record_cost` **also forward to the tracker** (today it doesn't — `runs.py:515-518`). DSPy compile needs `dspy.settings`/`track_usage` (§10). |
| 3 | Embedding cost | **settled** | Not a category. A **paid** embedding API is a metered model call through the same seam as the judge. **Local** embedding is compute, not dollars — out of scope for $-metering, covered by `duration_seconds`. |
| 4 | Infrastructure cost | **settled (non-goal)** | Consumer-side per ROADMAP §5. langres records `duration_seconds`; `$/hr`, GPU-hours, and instance rates live in the orchestration layer. Explicit non-goal. |
| 5 | Attribution & granularity | **settled** | Per-call is the atom; per-stage and per-run are roll-ups over the existing `parent_run_id` lineage (`runs.py:130`). No new enum. |
| 6 | Budget semantics | **partly open** | Metering/pricing settled; **unifying the three mechanisms** (`presets.py:290` raises, `benchmark.py:924` truncates, `labelers.py:201` truncates — §1) with one default and one documented failure semantic is real design work, staged as (d) in §10. Whether a cap spans categories is moot once "infra/embedding-local" aren't $-categories: the cap is over metered model spend, which is the only dollars langres controls. `status="budget_exceeded"` (`runs.py:80`) already generalizes. |

---

## 10. Staged implementation plan

Ordered so each stage is independently landable and the controversial parts come
last.

**(a) Additive usage vector — settles nothing controversial.**
Add the §6.1 usage fields to `PairwiseJudgement.provenance` (rename
`prompt_tokens`/`completion_tokens` to the OTel-canonical `input_tokens`/
`output_tokens`, keep the old keys as aliases for one release; add the three
cache/reasoning fields defaulting to `0`) and the §6.3
`provider_reported_cost_usd`/`cost_is_real` are already present. Write the
per-provider normalization adapter (OpenAI subset pass-through; **Anthropic sum**).
Purely additive; no pricing, no budgeting touched. Ships the cache-visibility
win of §5 on its own.

**(b) Auto-wire `record_cost` + `capture_run` into the benchmark path (Stream C).**
Open a `capture_run` in `run_method`/`run_methods` (zero today), aggregate the
per-call usage + `SpendMonitor.spent` into `record_cost`, and **make `record_cost`
forward to the tracker** (it doesn't — `runs.py:515-518`). This is the deferred
Stream C the issue names. **Sub-task: the DSPy-compile `$0` bug**
(`dspy_judge.py:366-368`). Note a real constraint: DSPy's `compile` bypasses
`forward()`, so there is **no per-pair `provenance` to sum** — the compile's LM
calls happen inside the optimizer. It needs DSPy's own usage tracking
(`dspy.settings` / `track_usage`; `dspy_judge.py:239` already runs `forward` under
`dspy.context(track_usage=True)`, but `compile` at `:369` uses
`dspy.context(lm=...)` **without** `track_usage`). So the fix is: enable
`track_usage` on the compile context and read DSPy's aggregate usage into
`record_cost`.

**(c) `PriceBook` seam + effective dates.**
Introduce the §6.2 price book (litellm key names, `(model, provider,
effective_date)` key, Weave-style derive-on-read), seed it from `PRICES_PER_1M`,
and compute `derived_cost_usd` + `price_book_version` at record-finalize time
(§7). This is where "prices move" and "provider variance" get their home.

**(d) Unify the three budget mechanisms.**
One budget abstraction with one default and one *documented* failure semantic,
taking an **injected `PriceBook`** (conservative mode) instead of reaching for the
hardcoded table. Reconcile the raise-vs-truncate split (`presets.py` raises,
`benchmark.py`/`labelers.py` truncate) — most cleanly by keeping both behaviors
but making them *one code path with a policy flag*, so the default and the price
basis can't drift across three files.

---

## 11. What we are NOT doing, and why

- **Not adding `{llm, embedding, infra}` cost buckets.** Wrong axis (§2); they
  collapse a metered-call distinction with two consumer-side concerns and leave
  permanent mostly-empty columns on `RunRecord`.
- **Not $-metering local embeddings.** Local compute is GPU-seconds, not dollars;
  the honest unit is `duration_seconds`, already recorded. Imputing a dollar rate
  needs an instance price the consumer owns.
- **Not building an infra-cost model.** Consumer-side per ROADMAP §5. langres
  records duration; the orchestration layer multiplies by its rate.
- **Not adopting Langfuse's one-key-only token convention.** It contradicts OTel's
  subset semantics and would force a second normalization for OTel-shaped
  providers; we pick OTel subsets and document the invariant (§6.1).
- **Not deriving-only like Weave (no stored dollars).** Our JSONL store has no
  join engine; a price-book-less reader must still see a number, so we denormalize
  a version-stamped derived total alongside the durable tokens (§7).
- **Not building live production cost monitoring.** That is consumer-side
  (ROADMAP §5). langres owns *experiment* cost accounting — a query over
  `RunStore`, not a dashboard subsystem.
- **Not changing the `Resolver` artifact in this issue.** The metric/cost
  provenance gap (ROADMAP §5, §8 here) is real but a larger, separate blast
  radius; sequence it as a follow-up that depends on #100's vocabulary.
- **Not a hard `v`-migration engine.** `RunStore.read` doesn't branch on `v` today
  and additive optional fields don't need one; we add fields + a round-trip test,
  not a migration framework (§7).

---

## Appendix — key file:line index (verified against `main` @ `33dd0db`)

- `spend_usd` scalar + `budget_exceeded`: `src/langres/core/runs.py:188-189`
- `record_cost` (only setter; no tracker-forward): `runs.py:515-518` — cf.
  `log_metrics` forwards at `:504`, `log_artifact` at `:509`
- `record_cost` callers repo-wide: `tests/test_runs.py:599`,
  `examples/research/experiment_tracking_demo.py:137`
- `RunStore.read` (no `v` branch): `runs.py:414-445`
- `parent_run_id` lineage: `runs.py:130-131`; `RunStatus`: `runs.py:80`
- per-call provenance (tokens, `cost_usd`, `cost_is_real`, `provider`):
  `src/langres/core/modules/llm_judge.py:429-438` (sync), `:588-598` (async)
- `_billing` (real vs estimate + `cost_is_real`): `llm_judge.py:326-337`;
  `_calculate_cost` fallback: `:711-729`
- OpenRouter real-cost parse: `src/langres/clients/openrouter.py:283-347`
- `PRICES_PER_1M` (10-model table; GLM-moved comment): `openrouter.py:40-65`
  (comment `:45`)
- `SpendMonitor` (passive ledger): `openrouter.py:399-455`;
  `BudgetExceeded` (+`partial_judgements`): `:383-396`
- Three budget mechanisms: `_SpendCappedModule` `src/langres/core/presets.py:290`
  (`DEFAULT_BUDGET_USD=1.0` `:97`); `BudgetedModuleRunner`
  `src/langres/core/benchmark.py:924` (`20.0`/`15.0` `:971-972`);
  `TeacherLabeler` `src/langres/bootstrap/labelers.py:201-204`
- DSPy compile `$0` NOTE + `capture_run`: `src/langres/core/modules/dspy_judge.py:366-368`;
  `track_usage` used in `forward` `:239`, absent in `compile` context `:369`
- `ArtifactManifest` (components only): `src/langres/core/serialization.py:72-88`
- ROADMAP brain/body seam `docs/ROADMAP.md:25-27`; artifact §5 `:188-196`
