# ERModel

The declarative base every named architecture subclasses. `Resolver` is a plain
alias of `ERModel` — one class under two names, not a shim.

::: langres.core.resolver

## Results

What `dedupe()` and `compare()` hand back. Both name the `architecture` that
produced them and the `backbone` that filled its model slot, so a result can
always answer "what model was this?".

::: langres.core.results

## Inputs

The shared input adapter: raw dicts → `(schema, normalized records)`. Every
architecture normalizes identically, so schema inference, id resolution and the
NaN/nested-value rules are one contract rather than per-model behaviour.

::: langres.core.inputs

## Spend cap

The one enforcer. `effective_budget` only resolves `None` to a default number —
nothing is capped unless a matcher is wrapped in `SpendCappedMatcher`.

::: langres.core.spend_cap
