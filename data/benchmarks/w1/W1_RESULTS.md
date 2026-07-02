# W1.1 — SelectJudge (ComEM-style set-wise) call-count benchmark on Amazon-Google

The falsifiable W1.1 question: does a **set-wise judge that makes ONE LLM call per
anchor group** (instead of one call per pair) actually reduce the number of LLM
calls, and is that reduction real (not an artifact of a skewed group-size
distribution)? Measured structurally on the real Amazon-Google test split
(`AmazonGoogleBenchmark`, seed 0, `k_neighbors=50`) at **$0** with DSPy's
`DummyLM` — a content-agnostic "always no match" judge, so this is a call-count
and plumbing proof, **not** a quality claim (see "Deferred: quality replication"
below). Reproduce with `uv run python examples/w1_select_judge_benchmark.py`;
raw output in `data/benchmarks/w1/select_judge_amazon_google.json`.

## Call-count + honest-cost reduction table

One LLM call costs the same fixed amount regardless of whether it is a pairwise
call (`llm_judge` / `dspy_judge`, one call per candidate pair) or a group call
(`select_judge`, one call per anchor group) — so a call-count reduction is a
cost reduction at **any** fixed per-call price. This is the structural lever
`stamp_group_cost` (W1.0, E5) makes honest: the group's one call cost lands on
its first judgement, $0 on the rest, so the totals below are what a real paid
run's `provenance["cost_usd"]` would sum to, at any price `$P`.

| quantity | naive pairwise judge | SelectJudge | reduction |
| --- | --- | --- | --- |
| test records | 1,374 | 1,374 | — |
| candidate pairs in scope | 48,854 | 48,854 | — |
| LLM calls | 48,854 (1 per pair) | 1,374 (1 per group) | **35.56x** |
| cost at $P per call | 48,854 × $P | 1,374 × $P | **35.56x** |
| cost at $0 (DummyLM, this run) | $0.0000 | $0.0000 | n/a (both $0) |

All 1,374 test-set anchors produced a non-empty group (`n_groups_nonempty ==
n_groups_total == 1,374`) — every anchor had at least one candidate within its
`k=50` nearest neighbors, so the reduction is not inflated by empty groups
being silently skipped.

## Group-size distribution (E3 — a skewed distribution would fake the ratio)

A reduction ratio computed only from the *mean* group size can be misleading if
most groups are trivial (size 1, no savings) and a handful of giant groups pull
the average up. The full distribution rules that out here:

| statistic | value |
| --- | --- |
| min group size | 2 |
| max group size | 50 (the `k_neighbors` cap) |
| mean group size | 35.56 |
| groups at size 1 (no reduction) | **0** |
| groups at size ≥ 20 | 1,183 of 1,374 (86%) |

Full histogram (group size → count of anchors with that many candidate
members), from `select_judge_amazon_google.json`:

```
 2: 1     11: 7    20: 21   29: 26   38: 49   47: 44
 3: 2     12: 7    21: 21   30: 26   39: 55   48: 48
 4: 1     13: 10   22: 16   31: 33   40: 38   49: 47
 5: 3     14: 11   23: 23   32: 34   41: 63   50: 63
 6: 3     15: 12   24: 18   33: 37   42: 58
 7: 3     16: 13   25: 34   34: 33   43: 52
 8: 7     17: 13   26: 39   35: 40   44: 64
 9: 5     18: 12   27: 36   36: 36   45: 64
10: 5     19: 12   28: 22   37: 44   46: 63
```

No group is at the trivial size-1 floor; the distribution is weighted toward
the large end (mode at the `k=50` cap), meaning most anchors genuinely have
many blocked candidates — exactly the regime a set-wise judge is for. The
35.56x reduction is a fair summary of this distribution, not an artifact of a
few outlier groups.

## Group contract invariant holds with SelectJudge in the loop

`stream_groups()` pairs ≡ `stream()` pairs (CEO #14) was re-checked on this
exact run with a real `GroupwiseModule` consumer attached (not just the
generic property tests in `tests/core/test_blocker.py` /
`tests/core/blockers/test_vector.py`): flattening `VectorBlocker.stream_groups()`
back to pairs recovers exactly the 48,854 pairs `VectorBlocker.stream()`
produces on the same 1,374 test records — no dupes, no losses — and
`SelectJudge.forward_groups()` in turn yields exactly one `PairwiseJudgement`
per pair (48,854 judgements from 1,374 calls). The set-wise contract is
additive end-to-end.

## Harness plumbing proof

`select_judge` runs through `run_method` exactly like any other registered
method (`langres.methods.LLM_METHODS`), on the real `AmazonGoogleBenchmark`:

| method | dataset | seed | bcubed_f1 | pair_f1 | usd_total |
| --- | --- | --- | --- | --- | --- |
| select_judge (DummyLM, always "no match") | amazon_google | 0 | 0.8545 | 0.0000 | 0.0000 |

`pair_f1 == 0.000` is expected, not a bug: the injected DummyLM always answers
"no candidate matches," so recall is trivially 0 — this run proves the
*plumbing* (Resolver build, blocking, scoring via
`GroupwiseModule.forward()`'s buffered pairwise→group derivation, clustering,
both metric tracks, cost accounting), not judge quality. `bcubed_f1` stays high
because the clusterer's sanity floor (near-all-singletons) is a reasonable
score on a task where nothing merges.

## Deferred: quality replication vs ComEM's published +16 F1

The ComEM paper (Wang et al., "Match, Compare, or Select? An Investigation of
Large Language Models for Entity Matching," COLING 2025) reports its selecting
strategy substantially improving F1 over pairwise matching on several ER
benchmarks. This branch does **not** attempt to replicate that quality claim —
per the M4.5/M5 plan's $3 LLM budget (spent only at the final W3 verification
gate) and U4's resolution (real-model quality smoke deferred to W3), this
result is a **structural/plumbing** proof only: the call-count lever is real
and measurable at $0; whether a real model's single-call selection matches
K real pairwise calls' quality is an open, explicitly deferred question.

## Design note: 0-or-1 selection contract

`SelectJudge`'s DSPy signature asks the LLM to select **at most one** matching
candidate per group (or none), not an arbitrary subset. This mirrors ComEM's
own "selecting" strategy, which the paper describes as choosing "the" (single)
record most likely to match — not a multi-label filter. This is what makes
"the LLM selected multiple candidates" a well-defined `select_error` case
(`CEO #12`) rather than an arbitrary implementation choice: a group scored 1.0
on more than one member is treated as a broken response for that group, mapped
to whole-group "no match" with `provenance["select_error"]` set, never raised.

## Reproduce

```
uv run python examples/w1_select_judge_benchmark.py
```

Writes `data/benchmarks/w1/select_judge_amazon_google.json`. Deterministic and
$0 (DummyLM only) — safe to re-run.
