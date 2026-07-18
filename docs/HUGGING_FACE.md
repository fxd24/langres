# Hugging Face model sharing

langres publishes a complete entity-resolution architecture as a validated
bundle around its existing `resolver.json` artifact. The model topology,
selection thresholds, resource references, and registered operation specs stay
in `resolver.json`; `langres-artifact.json` adds the exact file allowlist,
checksums, compatibility facts, model references, claim level, and optional
measurement summary.

Install the Hub client only when you need remote download or upload:

```bash
pip install "langres[hub]"
```

Local save/load does not require that extra:

```python
model.save_pretrained(
    "artifacts/acme-er",
    measurement_summary={
        "protocol_id": "official-v1",
        "evaluation_id": "amazon-google-v1",
        "dataset_ids": ("amazon-google:test",),
        "quality": {"pair_f1": 0.91},
        "tokens": {"input_tokens": 123_456, "output_tokens": 7_890},
        "cost": {"usd": 2.34},
        "performance": {"p95_latency_ms": 84.2},
        "hardware": {"accelerator": "NVIDIA L4"},
    },
)

reloaded = langres.ERModel.from_pretrained("artifacts/acme-er")
```

A complete zero-network local round-trip, plus an explicitly private
`push_to_hub` path that runs only when `--push-to` is supplied, lives in
`examples/research/hub_lifecycle.py`.

Prompt-bearing configuration is excluded by default. If the model contains a
custom/default serialized prompt, inspect `model.config_dict()` and opt in
explicitly:

```python
model.save_pretrained(
    "artifacts/prompted-er",
    allow_sensitive_config=True,
)
```

The outer manifest and generated card then state that prompt-bearing
configuration is included.

Credentials are never publishable, even with `allow_sensitive_config=True`.
Keep API keys, authorization headers, tokens, and similar secrets out of
serialized LiteLLM `provider` / `extra_body` options; inject them through the
runtime client or environment instead.

Remote operations use the same journey:

```python
result = model.push_to_hub(
    "acme/entity-resolver",
    private=True,
    revision="main",
    commit_message="Publish official-v1 result",
)

reloaded = langres.ERModel.from_pretrained(
    "acme/entity-resolver",
    revision=result.commit_oid,
)
print(reloaded.pretrained_source_.resolved_revision)
```

`from_pretrained` first resolves a branch or tag to an immutable repository
commit, downloads only `langres-artifact.json`, validates its bounded schema
against repository metadata, then downloads the exact allowlist at that commit.
It rejects traversal, symlinks, missing allowlisted files, unknown sizes, checksum
mismatches, unknown component/operation types, and malformed layouts before
`ERModel.load()` constructs a component. Files not named by the manifest are
ignored and never downloaded, which keeps repeated repository updates usable
without weakening the exact download allowlist. Artifacts never supply Python
module paths and never enable remote code.

Bundles are deliberately state-free in this first release: only
`resolver.json`, the generated model card, and an optional aggregate measurement
summary are eligible. `save_pretrained` rejects Resolver sidecars because
existing sidecar formats may contain corpus rows, compiled prompts, NumPy data,
or native FAISS binaries. Publish a fresh configuration-only model; sharing
trusted learned state requires a future explicit per-component safe-publication
contract.

An artifact is accepted only by the same langres minor release that created it
(patch releases remain compatible). In-place directory overwrite is
intentionally unsupported; publish to a new directory and switch the reference
after validation.

## Claim levels

- `reference-only` (default): the bundle contains topology and pinned or
  unpinned resource references, not copied model weights.
- `benchmark-reproducible`: requires named protocol/evaluation/datasets, at
  least one quality metric, plus full `organization/repository` Hub ids and
  immutable 40-character commit SHAs for every Hugging Face base and adapter.
  API, endpoint, local, shorthand, branch, and tag references are not eligible
  for this claim. It still references immutable Hub weights rather than copying
  them.
- `frozen-weights`: deliberately rejected for now. Local trained resource paths
  are not yet copied and rebased into the bundle, so claiming frozen weights
  would be misleading.

The model card and optional `measurement-summary.json` contain only explicitly
selected aggregate facts. Dataset rows, generations, judgement logs, local
caches, and tracker credentials are never included. Prompt-bearing resolver
configuration is included only with `allow_sensitive_config=True` and is
declared in both manifest and card. Tokens are forwarded to the Hub client for
the requested operation and are never written to the bundle or model card.
The outer manifest also records the install extras needed by optional
components, including `[trained]` for calibrators and random-forest matchers.

Hub artifact revision provenance is separate from resource `ModelRef`
revisions: `pretrained_source_.resolved_revision` identifies the bundle commit,
while each resource revision identifies the actual embedder, reranker, or LLM.
