"""Guarded 18-cell paid proof. Planning is free; execution needs two confirmations."""

from __future__ import annotations

import argparse
import hashlib
import re
from collections.abc import Callable, Mapping
from pathlib import Path

from langres.core.clusterer import Clusterer
from langres.core.model_ref import ModelRef
from langres.core.op import Stage, ThresholdSelect, TopKSelect
from langres.core.op_adapters import ClustererStage
from langres.core.resolver import ERModel
from langres.core.spend import SpendMonitor
from langres.data.registry import get_benchmark
from langres.experiments import (
    ArchitectureFactory,
    EvaluationProtocol,
    Experiment,
    expand_official_proof_matrix,
)
from langres.resources import (
    CrossEncoderReranker,
    Generate,
    LiteLLM,
    Parse,
    Rerank,
    Retrieve,
    SentenceTransformer,
)
from langres.tracking.runs import dataset_fingerprint

from _research_foundation import ResearchRecord

BUDGET_USD = 20.0
PAID_CONCURRENCY = 1
CONFIRMATION = "I_ACCEPT_USD_20_MAXIMUM"
BENCHMARKS = ("amazon_google", "abt_buy")
_FULL_COMMIT_SHA = re.compile(r"[0-9a-fA-F]{40}")


def _require_immutable_hf_revision(label: str, ref: ModelRef) -> None:
    """Reject mutable HF branches/tags before any model load or paid work."""
    if ref.kind != "hf":
        return
    if ref.revision is None or _FULL_COMMIT_SHA.fullmatch(ref.revision) is None:
        raise ValueError(
            f"{label} requires an immutable 40-hex Hugging Face commit SHA; "
            f"got revision={ref.revision!r}"
        )


def _dataset_fingerprints() -> dict[str, str]:
    fingerprints: dict[str, str] = {}
    for benchmark_id in BENCHMARKS:
        corpus, clusters, pairs = get_benchmark(benchmark_id).load()
        fingerprints[benchmark_id] = dataset_fingerprint(corpus, [clusters, pairs])
    return fingerprints


def build_protocol(
    fingerprints: Mapping[str, str] | None = None,
) -> EvaluationProtocol:
    """Build the proof policy; execution replaces planning markers with hashes."""
    resolved_fingerprints = dict(fingerprints or {})
    identity_inputs = resolved_fingerprints or {
        benchmark: "resolve-before-paid-execution" for benchmark in BENCHMARKS
    }
    test_id = hashlib.sha256(
        "|".join(f"{key}:{value}" for key, value in sorted(identity_inputs.items())).encode()
    ).hexdigest()
    return EvaluationProtocol.official_proof(
        benchmark_ids=BENCHMARKS,
        dataset_fingerprints=resolved_fingerprints,
        dataset_revisions=(
            {}
            if resolved_fingerprints
            else {benchmark: "resolve-before-paid-execution" for benchmark in BENCHMARKS}
        ),
        fixed_test_set_id=f"sha256:{test_id}",
        split_seed=0,
        budget_usd=BUDGET_USD,
    )


def _resources(
    embedder_ref: ModelRef,
    reranker_ref: ModelRef,
    llm_ref: ModelRef,
) -> tuple[SentenceTransformer, CrossEncoderReranker, LiteLLM]:
    return (
        SentenceTransformer(embedder_ref),
        CrossEncoderReranker(reranker_ref),
        LiteLLM(llm_ref),
    )


def build_factories(
    *,
    embedder_ref: ModelRef,
    reranker_ref: ModelRef,
    llm_ref: ModelRef,
) -> tuple[ArchitectureFactory, ...]:
    """Build the four named operation chains plus one custom topology."""
    for label, ref in (
        ("embedder_ref", embedder_ref),
        ("reranker_ref", reranker_ref),
        ("llm_ref", llm_ref),
    ):
        _require_immutable_hf_revision(label, ref)

    def factory(
        name: str,
        make_ops: Callable[
            [float, SentenceTransformer, CrossEncoderReranker, LiteLLM],
            list[Stage],
        ],
        *,
        stochastic: bool = False,
        replay_boundary: int | None = None,
    ) -> ArchitectureFactory:
        def build(threshold: float, monitor: SpendMonitor) -> ERModel:
            embedder, reranker, llm = _resources(embedder_ref, reranker_ref, llm_ref)
            return ERModel.from_topology(
                ops=make_ops(threshold, embedder, reranker, llm),
                replay_boundary=replay_boundary,
                monitor=monitor,
            )

        return ArchitectureFactory(
            name=name,
            factory=build,
            cache_semantics="stochastic" if stochastic else "deterministic",
        )

    def retrieve(
        threshold: float,
        embedder: SentenceTransformer,
        _reranker: CrossEncoderReranker,
        _llm: LiteLLM,
    ) -> list[Stage]:
        return [
            Retrieve(embedder, schema=ResearchRecord, k=50),
            ThresholdSelect(threshold),
            ClustererStage(Clusterer(threshold=0.0)),
        ]

    def retrieve_rerank(
        threshold: float,
        embedder: SentenceTransformer,
        reranker: CrossEncoderReranker,
        _llm: LiteLLM,
    ) -> list[Stage]:
        return [
            Retrieve(embedder, schema=ResearchRecord, k=50),
            Rerank(reranker),
            ThresholdSelect(threshold),
            ClustererStage(Clusterer(threshold=0.0)),
        ]

    def retrieve_llm(
        _threshold: float,
        embedder: SentenceTransformer,
        _reranker: CrossEncoderReranker,
        llm: LiteLLM,
    ) -> list[Stage]:
        return [
            Retrieve(embedder, schema=ResearchRecord, k=50),
            TopKSelect(10),
            Generate(llm),
            Parse(),
            ThresholdSelect(0.5),
            ClustererStage(Clusterer(threshold=0.0)),
        ]

    def retrieve_rerank_llm(
        _threshold: float,
        embedder: SentenceTransformer,
        reranker: CrossEncoderReranker,
        llm: LiteLLM,
    ) -> list[Stage]:
        return [
            Retrieve(embedder, schema=ResearchRecord, k=50),
            Rerank(reranker),
            TopKSelect(10),
            Generate(llm),
            Parse(),
            ThresholdSelect(0.5),
            ClustererStage(Clusterer(threshold=0.0)),
        ]

    def custom(
        threshold: float,
        embedder: SentenceTransformer,
        reranker: CrossEncoderReranker,
        _llm: LiteLLM,
    ) -> list[Stage]:
        return [
            Retrieve(embedder, schema=ResearchRecord, k=100),
            Rerank(reranker),
            TopKSelect(20),
            ThresholdSelect(threshold),
            ClustererStage(Clusterer(threshold=0.0)),
        ]

    return (
        factory("Retrieve", retrieve, replay_boundary=1),
        factory("RetrieveRerank", retrieve_rerank, replay_boundary=2),
        factory("RetrieveLLM", retrieve_llm, stochastic=True),
        factory("RetrieveRerankLLM", retrieve_rerank_llm, stochastic=True),
        factory("CustomTopology", custom, replay_boundary=3),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-paid", action="store_true")
    parser.add_argument("--confirm", default="")
    parser.add_argument("--concurrency", type=int, choices=(PAID_CONCURRENCY,), default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/official-paid-proof"))
    parser.add_argument("--embedder", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--embedder-revision")
    parser.add_argument("--reranker", default="cross-encoder/ms-marco-MiniLM-L6-v2")
    parser.add_argument("--reranker-revision")
    parser.add_argument("--llm", default="openrouter/openai/gpt-4o-mini")
    args = parser.parse_args()

    protocol = build_protocol()
    cells = expand_official_proof_matrix(protocol)
    print(
        f"preflight: {len(cells)} cells, concurrency={args.concurrency}, "
        f"stopping threshold=USD {BUDGET_USD:.2f}"
    )
    if not args.execute_paid:
        print(
            "plan only: dataset hashes are unresolved; no model load, "
            "network request, or paid call was made"
        )
        print(f"confirmation phrase: {CONFIRMATION}")
        print(
            "next command: uv run python examples/research/official_paid_proof.py "
            f"--execute-paid --confirm {CONFIRMATION} --concurrency {PAID_CONCURRENCY} "
            "--embedder-revision <40-hex-commit-sha> "
            "--reranker-revision <40-hex-commit-sha>"
        )
        return
    if args.confirm != CONFIRMATION:
        raise SystemExit(f"refusing paid execution: pass --confirm {CONFIRMATION}")
    if protocol.budget_usd != BUDGET_USD or args.concurrency != PAID_CONCURRENCY:
        raise SystemExit(
            "refusing paid execution: the USD 20 stopping threshold and concurrency 1 are mandatory"
        )
    embedder_ref = ModelRef(
        base=args.embedder,
        kind="hf",
        revision=args.embedder_revision,
    )
    reranker_ref = ModelRef(
        base=args.reranker,
        kind="hf",
        revision=args.reranker_revision,
    )
    try:
        _require_immutable_hf_revision("embedder_ref", embedder_ref)
        _require_immutable_hf_revision("reranker_ref", reranker_ref)
    except ValueError as exc:
        raise SystemExit(f"refusing official execution: {exc}") from exc

    protocol = build_protocol(_dataset_fingerprints())
    try:
        factories = build_factories(
            embedder_ref=embedder_ref,
            reranker_ref=reranker_ref,
            llm_ref=ModelRef(base=args.llm, kind="api"),
        )
    except ValueError as exc:
        raise SystemExit(f"refusing official execution: {exc}") from exc
    report = Experiment(
        architectures=factories,
        protocol=protocol,
        store=args.output_dir / "runs.jsonl",
        cache_dir=args.output_dir / "stage-cache",
        budget_usd=BUDGET_USD,
        fail_fast=True,
    ).run()
    print(report.to_markdown())


if __name__ == "__main__":
    main()
