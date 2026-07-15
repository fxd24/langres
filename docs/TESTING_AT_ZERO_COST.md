# Testing an ER pipeline at $0 (DummyLM)

An LLM judge costs money and needs a network. Neither belongs in a unit test or
CI run. langres' verbs (`link` / `dedupe`) and its `Resolver` all accept a
**`Matcher` instance** in the `matcher=` slot — the escape hatch that lets you swap
the real LLM for a **`DummyLM`**-backed judge that replays canned answers
offline, deterministically, and for **$0**.

This is the same seam the project's own test suite runs on (see
[`tests/test_verbs.py`](https://github.com/fxd24/langres/blob/main/tests/test_verbs.py) and
[`tests/core/modules/test_dspy_judge.py`](https://github.com/fxd24/langres/blob/main/tests/core/modules/test_dspy_judge.py)).

> **Why this matters — the spend footgun.** `import langres` eagerly imports
> litellm, whose `load_dotenv()` side effect can silently load an
> `OPENROUTER_API_KEY` from a `.env`. A test that reaches a *real* LLM judge then
> spends **real money** — even in CI. Injecting a DummyLM judge means the paid
> code path is never entered: no key is read for scoring, no request leaves the
> process. Verify a suite is offline with `uv run pytest -m "not integration"`
> (never a bare `pytest`, which can run live integration tests).

---

## The pattern

`DummyLM` (from `dspy.utils.dummies`) is a DSPy language model that returns
**canned answers** instead of calling a provider. You build a real judge
(`DSPyMatcher` for pairwise, `SelectMatcher` for set-wise) around it and pass the
judge as `matcher=`:

```python
from dspy.utils.dummies import DummyLM
from pydantic import BaseModel

from langres import dedupe
from langres.core.matchers.dspy_judge import DSPyMatcher


class Company(BaseModel):
    id: str
    name: str


def test_dedupe_merges_matching_records() -> None:
    # DSPyMatcher's signature has three output fields; a DummyLM answer is a dict
    # keyed by those field names. DummyLM replays this same answer for every
    # call -> fully deterministic.
    answer = {"reasoning": "same company", "match": "True", "match_probability": "0.95"}
    judge = DSPyMatcher(lm=DummyLM([answer] * 20), entity_noun="company")

    records = [
        {"id": "a", "name": "Acme Corp"},
        {"id": "b", "name": "Acme Corporation"},
    ]
    result = dedupe(records, schema=Company, matcher=judge, threshold=0.5)

    assert result == [{"a", "b"}]        # the pair merged into one cluster
    assert result.judge_used == "custom"  # an injected Matcher reports as "custom"
```

No key, no network, no spend — and the assertion is exact because DummyLM's
answer is fixed. The judge still runs the *real* `DSPyMatcher.forward` code
(rendering, parsing, scoring, provenance); only the model call is faked, so you
are testing your pipeline wiring, not a mock of it.

The same injection works for `link` and for a `Resolver` (both blocks below reuse
the `answer` dict and `Company` schema from the test above):

```python
from langres import link

verdict = link(
    {"id": "a", "name": "Acme"},
    {"id": "b", "name": "Acme Inc"},
    schema=Company,
    matcher=DSPyMatcher(lm=DummyLM([answer] * 20), entity_noun="company"),
    threshold=0.5,
)
assert verdict.match is True
assert verdict.score == 0.95          # the parsed match_probability
assert verdict.score_type == "prob_llm"
```

```python
from langres.core import AllPairsBlocker, Clusterer, Resolver

resolver = Resolver(
    blocker=AllPairsBlocker(schema=Company),
    comparator=None,
    matcher=DSPyMatcher(lm=DummyLM([answer] * 20), entity_noun="company"),
    clusterer=Clusterer(threshold=0.5),
)
assert resolver.resolve(records) == [{"a", "b"}]
```

---

## Set-wise judges (SelectMatcher) at $0

A `GroupwiseMatcher` like `SelectMatcher` makes **one** LLM call per candidate
*group* ("which of these K candidates matches the anchor?"). DummyLM tests it the
same way — the canned answer carries the signature's `selected_ids` field:

```python
from dspy.utils.dummies import DummyLM
from langres.core.groups import ERCandidateGroup
from langres.core.matchers.select_judge import SelectMatcher

judge = SelectMatcher(
    lm=DummyLM([{"reasoning": "b matches", "selected_ids": '["b"]'}]),
    entity_noun="company",
)
group = ERCandidateGroup(
    anchor=Company(id="a", name="Acme"),
    members=[Company(id="b", name="Acme Corp"), Company(id="c", name="Other")],
    group_id="a",
)
judgements = list(judge.forward_groups(iter([group])))

scores = {j.right_id: j.score for j in judgements}
assert scores == {"b": 1.0, "c": 0.0}                 # only "b" was selected
assert judgements[0].score_type == "prob_group_llm"
assert judgements[0].provenance["cost_usd"] == 0.0    # DummyLM = $0
```

---

## Key facts

- **`matcher=<Matcher>` is the seam.** Any `Matcher` instance passed to `link` /
  `dedupe` is used verbatim and reported as `judge_used="custom"`; passed to a
  `Resolver`, it just fills the `module` slot.
- **DummyLM is deterministic.** It replays its canned answers, so assertions are
  exact — no flakiness, no seeds.
- **Cost is honest and zero.** Under DummyLM, token counts are 0, so the
  `provenance["cost_usd"]` a judge stamps is `$0`; the built-in spend cap on the
  verbs never trips.
- **DSPy is an extra.** `DSPyMatcher` / `SelectMatcher` / `DummyLM` live behind the
  `[llm]` extra: `uv sync --extra llm` (or `--all-extras`). The string /
  embedding judges need no LLM at all — for those, `matcher="string"` is already a
  $0 offline path with nothing to inject.
- **Guard the suite.** Run `uv run pytest -m "not integration"` to keep live,
  paid integration tests out. A bare `pytest` can spend.
