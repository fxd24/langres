# Your own CSV in 15 minutes

You have a messy CSV of records — customers, companies, products — and some rows
are the same real-world thing spelled differently. This tutorial takes you from
that file to **entity clusters** at **$0** (no API key, no LLM, no spend), then
shows how to calibrate the match threshold from a handful of labels and persist
the pipeline for reuse.

> **What "runs on core" depends on your file size.** `dedupe` picks its blocker
> by row count. At **≤ 100 rows** it uses an all-pairs blocker and needs only the
> **core** install (`uv sync`: Pydantic, rapidfuzz, networkx, numpy) — fully
> offline, no network, no model download. **Above 100 rows** it auto-switches to
> an embedding blocker (MiniLM + FAISS) to scale O(N·k) instead of O(N²); that
> path needs the **`[semantic]`** extra and does a **one-time MiniLM model
> download** on first run. Either way it stays **$0** — no API key, no LLM call.
> The examples below use a 5-row file, so they run on core alone. (Threshold
> calibration in step 4 adds one more optional extra, `[trained]` for
> scikit-learn.)

This is the middle rung of the docs ladder:

- **Below:** [`examples/quickstart_verbs.py`](../examples/quickstart_verbs.py) —
  dedupe a list of dicts in ~10 lines.
- **Here:** a real CSV, a typed schema, a calibrated threshold, save/load.
- **Above:** [`docs/EXPERIMENTS.md`](EXPERIMENTS.md) — racing judges, paid LLM
  judges, the budget seam.

---

## 1. From CSV to records

langres verbs (`dedupe` / `link`) take **plain dicts**. pandas turns a CSV into
exactly that with `df.to_dict("records")`:

```python
import pandas as pd

# dtype=str keeps ids and zip codes as strings (no "1" -> 1 surprises).
df = pd.read_csv("contacts.csv", dtype=str)
records = df.to_dict("records")
# records[0] -> {"id": "1", "name": "Acme Corporation", "city": "New York", ...}
```

Two conventions the verbs rely on:

- **A unique `id` per record.** If every row has an `"id"` key, langres uses it;
  if none do, it assigns positional ids. A *mix* raises (it can't tell which
  source is authoritative). A duplicate `id` in a `dedupe` batch also raises.
- **Flat, scalar fields.** Inferred schemas coerce each value to `str | None`
  (and turn `NaN`/`None` into `None`, never the string `"nan"`). A nested
  `list`/`dict` value raises with guidance — pass an explicit schema for those.

---

## 2. Author a Pydantic schema

You *can* skip this — `dedupe(records)` will infer an ephemeral all-`str | None`
schema from the record keys. But an inferred schema is **not durable**: a saved
artifact that references it cannot be reloaded in a fresh process (the class was
minted at runtime and isn't importable by name). For anything you intend to
`save`/`load`, declare a real schema.

A langres schema is any Pydantic model with a **string `id`** field. Declare the
fields you want to match on — the string judge compares them field-by-field:

```python
from pydantic import BaseModel
from langres.core.registry import register_schema

@register_schema("Contact")          # register by name -> durable save/load
class Contact(BaseModel):
    id: str
    name: str
    city: str | None = None
    email: str | None = None
```

`@register_schema("Contact")` records the class in langres' component registry
under a stable name, so a saved pipeline referencing it reloads in any process
that has imported the module — the config-registry contract, no pickle. (Omit it
and in-process use still works; only cross-process `load` of an artifact needs
the name.)

> **What gets matched, and where `embed_text` fits.** The string judge scores on
> the schema's declared `str | None` fields (`name`, `city`, `email` here) via
> rapidfuzz — one similarity per field, weighted and combined; `id` is excluded.
> When the embedding blocker is active (> 100 rows, or `judge="embedding"`),
> `dedupe`/`Resolver.from_schema` derive the *blocking* text from those same
> declared string fields automatically. A schema-level
> `@computed_field embed_text` is a separate, **advanced** convention: it only
> takes effect when you hand-build a `VectorBlocker(schema=..., text_field="embed_text")`
> yourself (the declarative path the benchmark loaders use) — the verbs and
> `from_schema` do **not** read it. You don't need it for this tutorial; just
> declare the fields that identify the entity.

---

## 3. Dedupe — no key, no spend

`dedupe` groups the batch into clusters. Pin `judge="string"` to stay **$0** with
no LLM and no API key; the default `judge="auto"` would pick an LLM judge *if* an
API key is set (and fall back to this same string judge, with one notice, if
not). (For this 5-row file the string judge is also fully offline; on a file over
100 rows the blocker step downloads MiniLM once, per the note at the top — still
$0.)

```python
from langres import dedupe

result = dedupe(records, schema=Contact, judge="string", threshold=0.6)

print([sorted(c) for c in result])
# [['1', '2'], ['3', '4']]      # "Acme Corporation"/"Acme Corp", etc.

print(result.judge_used, result.score_type)
# string heuristic
```

The result is a plain `list[set[str]]` of id-clusters, and it is
**self-describing**: `result.judge_used`, `result.score_type`, and
`result.fallback_reason` tell you exactly what ran. **Singletons are dropped** —
a record that matches nothing does not appear in the output.

> **Entity linking instead of dedupe?** Use `link(left, right, schema=Contact,
> judge="string")` to score one pair; it returns a `LinkVerdict` that is truthy
> iff it's a match (`if link(a, b): ...`) and carries `.score`, `.reasoning`,
> and the full `.judgement`.

### On the threshold

`threshold=0.6` is a **magic constant** — a guess. Different judges score on
different scales (`"heuristic"` for string, `"sim_cos"` for embeddings,
`"prob_llm"` for an LLM), so a number that's right for one is meaningless for
another. `threshold=None` (the default) resolves to a sane per-judge default, but
the honest move is to derive it from data.

---

## 4. Calibrate the threshold from a few labels

Once you can label even a handful of pairs as match / non-match, stop guessing.
Score those pairs with `link`, then hand the scores and labels to
`derive_threshold`, which picks the cut that best separates them (Youden's J on
the ROC curve by default):

```python
from langres.core.calibration import derive_threshold

labeled = [
    (records[0], records[1], True),    # Acme Corporation / Acme Corp  -> match
    (records[2], records[3], True),    # Totally Different Co / Company -> match
    (records[0], records[4], False),   # Acme / Unrelated Bakery        -> no
    (records[1], records[4], False),
]

scores, labels = [], []
for left, right, is_match in labeled:
    verdict = link(left, right, schema=Contact, judge="string")
    scores.append(verdict.score)
    labels.append(is_match)

threshold = derive_threshold(scores, labels)   # -> ~0.907, from the data
clusters = dedupe(records, schema=Contact, judge="string", threshold=threshold)
```

`derive_threshold` needs the `[trained]` extra (scikit-learn):
`uv sync --extra trained`. It needs **both** a positive and a negative label
under the default `"youden"` method (it raises on a single class); pass
`method="percentile", percentile=...` for a label-agnostic cut.

> **Where do labels come from at scale?** langres has a whole *flywheel* for
> this: opt into `dedupe(..., log="judgements.jsonl")` to record every judge
> call, collect human corrections, and harvest them into labeled pairs with
> `langres.core.harvest` — which feeds this exact `derive_threshold`. See
> [`examples/flywheel_threshold_harvest.py`](../examples/flywheel_threshold_harvest.py)
> and [`docs/EXPERIMENTS.md`](EXPERIMENTS.md).

---

## 5. Save and load the pipeline

The verbs rebuild their pipeline per call. To freeze a configured pipeline —
schema, blocker, judge, calibrated threshold — into a reusable artifact, drop to
the `Resolver` (the declarative mid-layer the verbs sit on) and use
`save`/`load`:

```python
from langres import Resolver

resolver = Resolver.from_schema(Contact, judge="string", threshold=threshold)
clusters = resolver.resolve(records)          # same clusters as dedupe()

resolver.save("artifacts/contacts_v1")        # writes resolver.json (+ sidecars)
reloaded = Resolver.load("artifacts/contacts_v1")
assert reloaded.resolve(records) == clusters
```

`save` writes a **human-readable `resolver.json` manifest** plus per-component
sidecar state; `load` rebuilds every slot from the component registry by its
`type_name` — **no pickle, no code execution**. This is why step 2's
`@register_schema("Contact")` matters: `load` looks the schema up by that name.
Load the artifact in a fresh process and it reconstructs identically (import the
module that defines `Contact` first so the registration runs).

---

## Where to go next

- **Spend a little on an LLM judge.** Set `OPENROUTER_API_KEY` and drop the
  `judge=` kwarg: `dedupe(records, schema=Contact)` picks an LLM judge under a
  default **$1 spend cap** (`budget_usd=`). See [`docs/EXPERIMENTS.md`](EXPERIMENTS.md).
- **Test your pipeline in CI without spending.** See
  [`docs/TESTING_AT_ZERO_COST.md`](TESTING_AT_ZERO_COST.md) — inject a DummyLM-backed
  judge for deterministic, offline, $0 assertions.
- **Contribute a new matching method.** See
  [`docs/ADDING_A_METHOD.md`](ADDING_A_METHOD.md).
