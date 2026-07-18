# Reproducibility

Reproduction in langres is a chain of evidence, not a single seed.

```text
resource revisions + operation topology
                ↓
          recipe identity
                ↓
evaluation protocol + fixed test identity
                ↓
 attempt records + measurements
                ↓
 local bundle / Trackio view / Hub artifact
```

## The four identities

- `recipe_id` identifies the logical architecture, resource configuration,
  data, split, and seeds.
- `evaluation_id` identifies the statistical question: protocol, metrics,
  fixed test set, and hardware cohort.
- `cache_id` identifies reusable bytes at a declared operation boundary,
  including source, environment, input order, operation specs, and resource
  revisions.
- `attempt_id` identifies one execution. Retries remain attempts; they are not
  independent statistical samples.

Use `EvaluationProtocol` as the declaration and `ExperimentReport` as the
result. Do not compare rows only because their architecture labels match.

## Clean and dirty claims

A clean official claim requires committed source plus the relevant lock,
environment, dataset/test, and model revisions. If Git provenance is
unavailable, langres hashes installed sources and marks the result dirty.
Untracked files are part of dirty state; a clean-looking commit SHA alone is
not sufficient evidence.

Dirty exploratory runs are useful and resumable, but publish them as
exploratory. Do not relabel them after the fact.

## Local-first handoff

Keep these together:

- the `EvaluationProtocol` and `ExperimentReport`;
- the `RunStore` JSONL, which retains terminal and failed attempts;
- any declared stage cache needed for exact resume;
- the source commit/dirty hash and dependency lock;
- resource `ModelRef` values and immutable revisions;
- measured usage and the original `PriceSnapshot`.

Trackio is an optional view, not the authoritative record:

```bash
uv run python examples/research/trackio_reproduction.py
```

The example sets `space_id=None`, so Trackio stays local. Configure an HF Space
only as an explicit publication step.

## Pretrained bundles and the Hub

Local bundle creation and loading need no Hub client:

```bash
uv run python examples/research/hub_lifecycle.py --path artifacts/acme-v1
```

The bundle contains the allowlisted resolver topology, resource references,
compatibility facts, generated model card, and only the aggregate measurement
summary you explicitly pass. It does not implicitly include datasets, records,
generations, caches, credentials, or model weights.

Remote publication is explicit and private in the example:

```bash
uv run python examples/research/hub_lifecycle.py \
  --path artifacts/acme-v1 \
  --push-to your-org/acme-v1
```

`from_pretrained(repo_id, revision=...)` resolves and records the immutable Hub
commit separately from each resource revision. See
[Hugging Face model sharing](HUGGING_FACE.md) for claim levels and the exact
allowlist/security contract.

## Privacy defaults

- `RunStore` and stage caches are local unless you choose a remote tracker.
- Trackio is local unless `space_id` is set.
- Hub upload happens only through `push_to_hub`.
- Prompt-bearing configuration requires `allow_sensitive_config=True`.
- Benchmark datasets and user records are never uploaded implicitly.
- Persisted error messages redact complete authorization values, but logs and
  aggregate summaries still deserve review before publication.

## Performance claims

Quality comparison requires the same `evaluation_id`. Latency, throughput,
memory, and cost comparison additionally requires a compatible hardware/runtime
cohort. Record device, dtype, quantization, batching, workers, warm/cold state,
provider/region, and concurrency when they can affect the number. Repricing
stored tokens changes a derived cost estimate; it does not change the original
observed charge or make two infrastructure cohorts equivalent.
