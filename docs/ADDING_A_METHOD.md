# Adding a matching method behind the seam

langres' thesis is that a matching method — string similarity, embeddings, an
LLM judge — is implemented **once** as a judge (a `Matcher`) and stays
usable, swappable, and tunable by everyone through one seam. This guide walks the
**current in-repo path** for contributing a new method, using the real
**`SelectMatcher`** (the ComEM-style set-wise judge, W1.1) as the worked example.

> **No public plugin API yet.** There is deliberately **no** external/plugin
> registration entry point today. A new *name-selectable* method must be wired
> into the in-repo dispatch sites below. A single public method-registration API
> that collapses them is **deferred to [issue #55](https://github.com/fxd24/langres/issues/55)**
> (see `TODOS.md`). This document is the contract *as it stands* — what a
> contributor editing the repo does now.
>
> **You may not need any of this.** A judge you only ever pass as a `Matcher`
> instance — `dedupe(records, matcher=MySelectMatcher(...))` — needs **zero**
> registry wiring. The steps below are only for making a method selectable *by
> name* (`"select_judge"`) across the benchmark harness and the two build paths.

---

## 1. Write the judge (a `Matcher`)

A judge is a `Matcher[SchemaT]` whose `forward` yields one `PairwiseJudgement` per
candidate pair. Everything downstream — the `Clusterer`, the metrics harness, the
`JudgementLog` — consumes that flat pairwise stream, so **whatever shape your
scoring logic has, the output contract is pairwise**.

`SelectMatcher` scores a whole *group* at once, so it subclasses
`GroupwiseMatcher` (`src/langres/core/matcher.py`) rather than `Matcher` directly.
`GroupwiseMatcher` **IS-A** `Matcher`: its concrete `forward` derives
`ERCandidateGroup`s from the pairwise stream and dispatches to your
`forward_groups`, then flattens the result back to pairwise. **Set-wise in,
pairwise out** — the group structure never leaks past the class boundary, so the
Resolver spine needs no changes. You implement only `forward_groups` and
`inspect_scores`.

### Riding the `ComparisonVector` seam

An `ERCandidate` carries an optional `comparison: ComparisonVector`
(`src/langres/core/models.py`). This is the two-phase pipeline seam:

- **Comparison-aware judges** (e.g. `WeightedAverageMatcher`, `RandomForestMatcher`,
  `FellegiSunterMatcher`) consume a per-feature `ComparisonVector` that a
  `Comparator` stage attaches upstream, and raise if it's `None`. Wire these with
  a `Comparator` (step 3 returns one alongside the module builder).
- **Self-contained judges** (e.g. `LLMMatcher`, `DSPyMatcher`, `SelectMatcher`) read
  the raw `left`/`right` entities directly and **ignore** `comparison` — so it
  stays `None` for them, and they need no `Comparator`.

`SelectMatcher` is self-contained: it renders each entity to JSON and asks the LLM
to select the matching id. So its method-registry entry (step 3) returns
`(builder, None)` — no comparator.

---

## 2. Emit the right `PairwiseJudgement` — `score_type` and the group-call cost

Every judgement declares a **`score_type`** literal so a downstream reader knows
what scale the `score` is on (thresholds are not comparable across scales). The
allowed values live on `PairwiseJudgement` in `src/langres/core/models.py`:

| `score_type` | Meaning | Emitting judge |
|---|---|---|
| `heuristic` | string-similarity score | `WeightedAverageMatcher`, `RapidfuzzMatcher` |
| `sim_cos` | cosine similarity | `EmbeddingScoreMatcher` |
| `prob_llm` | per-pair LLM probability | `DSPyMatcher`, `LLMMatcher` |
| `prob_group_llm` | **set-wise** LLM decision decomposed to a pair | `SelectMatcher` |
| `calibrated_prob` | calibrated probability | cascade / calibrated modules |
| `prob_fs` / `prob_rf` | Fellegi-Sunter / random-forest probability | `FellegiSunterMatcher`, `RandomForestMatcher` |

`SelectMatcher` emits `score_type="prob_group_llm"` — a distinct literal precisely
because its score comes from a *group* decision (1.0 for the single selected id,
0.0 for the rest), not an independent per-pair judgement. **If your method scores
on a genuinely new scale, add a literal here** (and keep the table above in sync).

### The group-call cost convention

A set-wise judge makes **one** LLM call for K pairs. Pricing each of the K
resulting judgements at the full call cost would overcount spend K-fold. The
convention (`stamp_group_cost` in `src/langres/core/matcher.py`) avoids that:

- the **full `cost_usd`** is stamped on the **first** judgement of the group;
- every **sibling** carries **`$0`**;
- **`provenance["group_id"]`** is set on **all** of them (traceable back to the
  one call);
- **`provenance["group_end"] = True`** marks the **last** judgement — a boundary
  marker so a consumer draining a lazy group stream (the verbs' spend cap,
  `_SpendCappedMatcher`) knows where to stop without peeking past it and
  triggering the next paid call.

`SelectMatcher.forward_groups` prices the call from token usage
(`tokens / 1000 * price_per_1k_tokens`) and calls `stamp_group_cost(...)`. The
existing cost aggregators (`benchmark._cost_track`) already read
`provenance["cost_usd"]`, so a group sums to exactly one call's cost with no
changes on their end. (`SelectMatcher` also degrades gracefully: a malformed,
out-of-group, or over-selecting LLM answer maps the *whole group* to `$0`-scored
"no match" judgements carrying `provenance["select_error"]` rather than raising
mid-stream — CEO #12. The call is still billed, so the same cost convention
applies.)

---

## 3. Register it in the method registry (make it selectable by name)

Since the v0.3 model-identity slice there is **one registration seam**:
`langres.core.method_registry`. A `MethodSpec` carries the builder plus the
name's identity metadata, and all three dispatch paths — the verbs
(`presets.build_judge`), `Resolver.from_schema`
(`resolver._build_module_for_judge`), and the benchmark harness
(`methods._make_module_builder`) — resolve names through it, so registering
once makes the name mean the same thing everywhere. Adding a method is **one
builder function + one spec**:

```python
# src/langres/core/method_registry.py
def _build_select_judge(schema, *, model=None, entity_noun="entity",
                        client=None, comparator=None) -> Matcher[Any]:
    # Lazy import: dspy must stay out of sys.modules unless this method is
    # actually chosen (the registry is eager-imported by langres.core).
    from langres.core.matchers.select_judge import SelectMatcher

    resolved_model = model or DEFAULT_OPENROUTER_MODEL
    judge: SelectMatcher[Any] = SelectMatcher(lm=client, model=resolved_model)
    judge.price_per_1k_tokens = dspy_price_per_1k(resolved_model)  # honest-cost seam
    return judge

register_method(MethodSpec(
    name="select_judge",
    build=_build_select_judge,
    score_type="prob_group_llm",       # the family tag its judgements carry
    default_threshold=0.7,             # E12: per-family threshold scales
    default_model=DEFAULT_OPENROUTER_MODEL,  # what results report as `model`
    accepts_model=True,                # honors a caller model= override
    requires_extra="llm",              # actionable ImportError names the extra
))
```

Method ids are **bare names**; `/` is reserved for future `author/method`
namespacing (model ids keep their slashes in the orthogonal `model=` kwarg).
Two per-layer *policies* remain separate from registration: the verbs'
allowlist (`presets._VERB_JUDGE_NAMES` — join it only if the judge is safe
with no injected client and no fit step) and `from_schema`'s name tuple.
`SelectMatcher` joins neither (it's a benchmark/experiment method that needs an
injected DSPy LM).

Then declare its **membership** in the method tuples at the top of `methods.py`
so the harness races it:

```python
LLM_METHODS: tuple[str, ...] = ("llm_judge", "cascade", "dspy_judge", "select_judge")
ALL_METHODS: tuple[str, ...] = ZERO_SPEND_METHODS + LLM_METHODS
```

Membership placement encodes a real contract:

- **`ZERO_SPEND_METHODS`** — deterministic, no API call (`rapidfuzz`,
  `weighted_average`, `embedding_cosine`).
- **`LLM_METHODS`** — takes an *injected client*. Note the client *shape*
  differs: `llm_judge` wants a LiteLLM client (`client.completion(...)`),
  `cascade` an OpenAI-shaped one, and `dspy_judge`/`select_judge` a **DSPy LM**
  (`dspy.LM` / `DummyLM`). Document which your method expects.
- **Neither tuple** — a method the registry recognizes but that needs an
  explicit `fit` step (`fellegi_sunter`, `random_forest`); the `run_methods`
  race can't build+fit it per grid threshold, so it's driven via
  `Resolver.fit(...)` instead. Register the spec but leave it out of the race
  tuples.

The `price_per_1k_tokens` assignment is the **honest-cost seam**: `SelectMatcher`
prices each call as `tokens/1000 * price_per_1k_tokens`, defaulting to `$0`. A
real paid run must pin the price from the OpenRouter table
(`dspy_price_per_1k(model)` — the registry's DSPy builders do this), or cost
would report `$0` and the live budget-stop would never fire.

---

## 4. Serialization — the config-registry contract (no pickle)

A judge that goes into a `Resolver` slot must be **serializable without pickle**.
The Resolver's `save`/`load` writes a human-readable `resolver.json` manifest and
rebuilds every slot from the component registry **by `type_name`** — no code
execution, no pickle. To participate, a judge provides four things (all visible on
`SelectMatcher` in `src/langres/core/matchers/select_judge.py`):

```python
from langres.core.registry import register

@register("select_judge")                       # (1) registry key
class SelectMatcher(GroupwiseMatcher[SchemaT]):
    type_name: ClassVar[str] = "select_judge"    # (2) mirrored as a class attr

    @property
    def config(self) -> dict[str, object]:        # (3) PURE, JSON-able config
        return {                                  #     never the LM or secrets
            "model": self.model,
            "temperature": self.temperature,
            "entity_noun": self.entity_noun,
        }

    @classmethod
    def from_config(cls, config: dict[str, object]) -> "SelectMatcher[SchemaT]":
        return cls(                               # (4) rebuild from config alone
            lm=None,                              #     (LM rebuilt lazily)
            model=str(config["model"]),
            temperature=float(config["temperature"]),
            entity_noun=str(config["entity_noun"]),
        )
```

Rules the contract enforces:

- **`config` is pure and JSON-serializable** — construction parameters only,
  **never** the injected LM, an API key, or any live object. `from_config` alone
  must rebuild an equivalent (uncompiled) judge, and it builds its LM lazily on
  first use.
- **`@register` may be lazy.** `SelectMatcher` is registered via
  `registry._LAZY_COMPONENT_MODULES` so `import langres.core` doesn't import dspy
  (importing dspy opens a disk cache). `get_component("select_judge")` imports and
  registers it on demand.
- **Out-of-band state, if any, is a separate concern.** A judge with tuned state
  (e.g. `DSPyMatcher`'s compiled program) implements `save_state`/`load_state` and
  the Resolver writes a per-slot sidecar. `SelectMatcher` has **none** —
  `from_config` fully rebuilds it, like any stateless judge.

### A fresh-process round-trip is REQUIRED

In-process `save`/`load` is not enough — the whole point of no-pickle
serialization is that a **different process** can reload the artifact. Prove it:

```python
# In a FRESH interpreter (no prior import of your judge's module):
from langres.core.registry import get_component
cls = get_component("select_judge")       # lazily imports + registers
assert cls.type_name == "select_judge"    # rebuildable by name, no pickle
```

`tests/core/modules/test_dspy_judge.py` does the stronger version as **two**
companion tests — mirror both for your judge:

- `test_resolver_with_dspy_judge_saves_and_loads` — **in-process**: saves a
  `Resolver` (judge in the module slot) and asserts the **manifest shape** —
  `module_spec["type_name"] == "dspy_judge"` and **no `lm`** in the config.
- `test_resolver_load_dspy_judge_in_fresh_process` (`@pytest.mark.slow`) — proves
  the reload itself succeeds in a **fresh subprocess** (`Resolver.load` via
  `langres.core` alone), i.e. that lazy registration fires on demand.

---

## 5. Test expectations

- **100% coverage** (a POC requirement). Cover the happy path, every
  `score_type` branch, and every error branch (for `SelectMatcher`: parse failure,
  unknown-id, over-selection — each maps a whole group to `select_error` without
  raising).
- **LLM paths run on `DummyLM` — $0, offline, deterministic.** Never make a real
  call in a test. Build the judge with `lm=DummyLM([...canned answers...])`; the
  answer dict is keyed by the DSPy signature's output field names (for
  `SelectMatcher`: `{"reasoning": ..., "selected_ids": '["b"]'}`). See
  [`docs/TESTING_AT_ZERO_COST.md`](TESTING_AT_ZERO_COST.md) and
  `tests/core/modules/test_select_judge.py` (DummyLM-driven throughout).
- **Verify the cost convention.** Assert the full cost lands on the first
  judgement, `$0` on siblings, and that a group sums to exactly one call's cost
  (`test_forward_groups_group_cost_sums_to_exactly_one_calls_cost`).
- **Guard against live spend.** Run `uv run pytest -m "not integration"` — never a
  bare `pytest`, which can execute paid integration tests.

---

## Checklist

- [ ] Judge is a `Matcher` (or `GroupwiseMatcher`) yielding `PairwiseJudgement`s.
- [ ] Correct `score_type` (add a literal to `models.py` if genuinely new).
- [ ] Set-wise judges apply the group-call cost convention (`stamp_group_cost`).
- [ ] Honest-cost seam wired (`price_per_1k_tokens` pinned for paid runs).
- [ ] Selectable by name? One `MethodSpec` in `core/method_registry.py` **+**
      the right method tuple — and, if it's also a `from_schema`/verb judge,
      the per-layer allowlists (`presets._VERB_JUDGE_NAMES`, `from_schema`'s
      tuple).
- [ ] `@register` + `type_name` + pure `config` + `from_config` (no pickle).
- [ ] Fresh-process (subprocess) reload proven in a test.
- [ ] 100% coverage; all LLM paths on DummyLM at $0.
