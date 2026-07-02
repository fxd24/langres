"""Presets: resolve ``judge="auto"`` and assemble a spend-capped Resolver.

This is the machinery behind the three-verb DX layer (:mod:`langres.verbs`):
picking a judge from available API keys, building its scorer Module, wiring a
blocker by dataset size, and wrapping the scorer in a hard spend cap. It sits
strictly ABOVE :class:`~langres.core.resolver.Resolver` (which must not import
back from here -- see ``Resolver.from_schema``'s own, deliberately duplicated,
low-level judge switch) and BELOW :mod:`langres.verbs`:

    verbs.py -> core/presets.py -> Resolver -> {Blocker, Comparator, Module, Clusterer}

Nothing here is domain-specific: every function takes a Pydantic ``schema``
type and works for any schema (see ``build_judge``'s ``"string"``/``"embedding"``
branches, which derive their fields from the schema, never a hard-coded name).

Threshold semantics differ across judges (E12): ``"heuristic"`` (string),
``"sim_cos"`` (embedding), and ``"prob_llm"`` (zero_shot_llm) are three
different score scales, so one hand-picked ``threshold=0.7`` is not
comparable across them. :data:`_DEFAULT_THRESHOLDS` gives each judge kind its
own sane default; pass ``threshold=`` explicitly to override, and calibrate a
real one with :func:`~langres.core.calibration.derive_threshold` once you have
labels.

Spend cap (adopted CEO decision #8 + Eng E1/E9): every judge -- including the
free ones -- is wrapped in :class:`_SpendCappedModule`, a small
:class:`~langres.core.module.Module` that tallies each judgement's
``provenance["cost_usd"]`` through a :class:`~langres.clients.openrouter.SpendMonitor`
and raises :class:`~langres.clients.openrouter.BudgetExceeded` the moment
cumulative spend would cross ``budget_usd`` (default
:data:`DEFAULT_BUDGET_USD`). The exception carries every judgement already
produced (and paid for) on ``.partial_judgements`` -- recover them with::

    try:
        result = dedupe(records, budget_usd=0.50)
    except BudgetExceeded as exc:
        already_scored = exc.partial_judgements  # list[PairwiseJudgement]
        # e.g. cluster just those, or raise budget_usd and retry

A resolver built here is NOT guaranteed ``save()``-able (the spend-cap wrapper
has no ``type_name``): for a durable artifact, build the pipeline directly
with ``Resolver.from_schema`` instead.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

from pydantic import BaseModel

from langres.clients.openrouter import BudgetExceeded, SpendMonitor, dspy_price_per_1k
from langres.clients.settings import Settings
from langres.core.blocker import Blocker
from langres.core.blockers.all_pairs import AllPairsBlocker
from langres.core.blockers.vector import VectorBlocker
from langres.core.clusterer import Clusterer
from langres.core.comparator import Comparator
from langres.core.embeddings import SentenceTransformerEmbedder
from langres.core.indexes.vector_index import FAISSIndex
from langres.core.judges.embedding_score import EmbeddingScoreJudge
from langres.core.judges.weighted_average import WeightedAverageJudge
from langres.core.models import ERCandidate, PairwiseJudgement
from langres.core.module import Module
from langres.core.resolver import Resolver

if TYPE_CHECKING:
    from collections.abc import Iterator

#: The judge names the verb layer understands, plus the "auto" meta-value that
#: :func:`choose_auto_judge` resolves down to one of the other three.
JudgeName = Literal["auto", "zero_shot_llm", "embedding", "string"]

#: Default clusterer/match threshold per resolved judge kind (D3). Different
#: ``score_type`` scales make one global constant meaningless; pass
#: ``threshold=`` explicitly to override, or derive one from labels with
#: :func:`~langres.core.calibration.derive_threshold`.
_DEFAULT_THRESHOLDS: dict[str, float] = {
    "string": 0.5,
    "embedding": 0.5,
    "zero_shot_llm": 0.7,
}

#: Default per-call spend cap for a presets-built judge (CEO decision #8),
#: overridable via ``budget_usd=``. Zero-spend judges never approach it.
DEFAULT_BUDGET_USD = 1.0

#: ``AllPairsBlocker`` is used at or below this many records; above it,
#: :func:`build_resolver` switches to a ``VectorBlocker`` (O(N*k) instead of
#: O(N^2)). An embedding judge always uses the VectorBlocker regardless of N
#: (it needs the index's similarity score to score on).
_ALL_PAIRS_MAX_N = 100
_VECTOR_K_NEIGHBORS = 10
_VECTOR_MODEL_NAME = "all-MiniLM-L6-v2"

#: judge="auto" model ids, keyed by which API key is present (choose_auto_judge).
_OPENROUTER_MODEL = "openrouter/openai/gpt-4o-mini"
_OPENAI_MODEL = "openai/gpt-5-mini"

#: Static ``PairwiseJudgement.score_type`` each judge kind emits -- used as a
#: fallback label when no judgement was actually produced (e.g. an empty or
#: single-record ``dedupe()`` call short-circuits before scoring).
_SCORE_TYPE_BY_JUDGE: dict[str, str] = {
    "string": "heuristic",
    "embedding": "sim_cos",
    "zero_shot_llm": "prob_llm",
}

#: Rough, deliberately worst-case-biased token count for the pre-scoring cost
#: estimate (``notice_pre_scoring_cost``). The real, metered cost is what the
#: spend cap actually enforces per pair as scoring happens -- this constant
#: only sizes the upfront heads-up line.
_ESTIMATED_TOKENS_PER_PAIR = 500


def _notice(message: str) -> None:
    """Emit a user-visible notice via ``warnings.warn`` (D2).

    ``logger.info`` is invisible under default logging config, so both the
    auto-judge fallback notice and the pre-scoring cost line go through this
    one channel instead -- tests assert on it with ``pytest.warns``.
    """
    warnings.warn(message, stacklevel=3)


def choose_auto_judge(settings: Settings) -> tuple[JudgeName, str | None, str | None]:
    """Resolve ``judge="auto"`` from available API keys.

    ``OPENROUTER_API_KEY`` set -> the OpenRouter gpt-4o-mini route;
    else ``OPENAI_API_KEY`` set -> the direct-OpenAI gpt-5-mini route;
    else -> the zero-spend ``"string"`` judge, with one notice (E1: a
    candidate paid judge whose price is unpinned -- $0, unmetered -- is also
    refused down to ``"string"``, since a blind cap is no cap at all).

    Args:
        settings: Loaded :class:`~langres.clients.settings.Settings` (reads
            ``openrouter_api_key`` / ``openai_api_key``).

    Returns:
        ``(resolved_judge, model, fallback_reason)`` -- ``model`` is the
        model id to use when ``resolved_judge == "zero_shot_llm"`` (``None``
        otherwise); ``fallback_reason`` is set (and a notice already emitted)
        only when resolution fell back to ``"string"``.
    """
    model: str | None
    if settings.openrouter_api_key:
        model = _OPENROUTER_MODEL
    elif settings.openai_api_key:
        model = _OPENAI_MODEL
    else:
        model = None

    if model is not None and dspy_price_per_1k(model) > 0.0:
        return "zero_shot_llm", model, None

    if model is None:
        reason = (
            'judge="auto": no OPENROUTER_API_KEY or OPENAI_API_KEY is set, so '
            "falling back to the zero-spend 'string' judge. Set one of those "
            "env vars to use an LLM judge, and calibrate its threshold with "
            "langres.core.calibration.derive_threshold once you have labels."
        )
    else:
        reason = (
            f'judge="auto": the selected model {model!r} has no pinned price in '
            "langres.clients.openrouter.PRICES_PER_1M, so its spend cap would be "
            "blind; falling back to the zero-spend 'string' judge. Pass "
            "judge='zero_shot_llm' explicitly to use it anyway, or pin a price."
        )
    _notice(reason)
    return "string", None, reason


def build_judge(
    judge: JudgeName | Module[Any],
    schema: type[BaseModel],
    *,
    model: str | None = None,
    entity_noun: str = "entity",
) -> Module[Any]:
    """Build the scorer Module for a resolved ``judge``.

    Args:
        judge: ``"zero_shot_llm"`` / ``"embedding"`` / ``"string"``, or a
            ``Module`` instance passed through verbatim -- the escape hatch
            (and the ``DummyLM``-injected-``DSPyJudge`` zero-spend test seam).
            ``"auto"`` is NOT resolved here; call :func:`choose_auto_judge`
            first (:func:`resolve_judge` does this).
        schema: The entity schema (drives ``"string"``'s comparator and
            ``"zero_shot_llm"``'s entity rendering). Works for ANY schema.
        model: Model id for ``"zero_shot_llm"``. Defaults to the OpenRouter
            gpt-4o-mini route when omitted.
        entity_noun: Domain noun woven into the LLM judge's prompt.

    Returns:
        A ready (uncapped) scorer Module.

    Raises:
        ValueError: If ``judge`` is an unrecognized string (including
            ``"auto"``, which only :func:`choose_auto_judge`/:func:`resolve_judge`
            resolve).
    """
    if isinstance(judge, Module):
        return judge
    if judge == "string":
        comparator: Comparator[Any] = Comparator.from_schema(schema)
        return WeightedAverageJudge(feature_specs=comparator.feature_specs)
    if judge == "embedding":
        return EmbeddingScoreJudge()
    if judge == "zero_shot_llm":
        # Lazy: dspy must stay out of sys.modules unless a zero_shot_llm judge
        # is actually chosen (mirrors langres.methods._make_module_builder).
        from langres.core.modules.dspy_judge import DSPyJudge

        resolved_model = model or _OPENROUTER_MODEL
        dspy_module: DSPyJudge[Any] = DSPyJudge(model=resolved_model, entity_noun=entity_noun)
        dspy_module.price_per_1k_tokens = dspy_price_per_1k(resolved_model)
        return dspy_module
    raise ValueError(
        f"unknown judge {judge!r}; choose one of 'zero_shot_llm', 'embedding', "
        "'string', 'auto', or pass a Module instance directly"
    )


class _SpendCappedModule(Module[Any]):
    """Wrap a Module, hard-stopping the moment cumulative cost crosses a budget.

    Reuses :class:`~langres.clients.openrouter.SpendMonitor` for the tally +
    threshold check, per pair, and re-raises its
    :class:`~langres.clients.openrouter.BudgetExceeded` with every judgement
    already produced (and paid for) attached as ``.partial_judgements`` (E9;
    mirrors :class:`~langres.core.benchmark.BlindCostError`'s "set by the
    catcher, not at raise time" pattern).

    Deliberately NOT :class:`~langres.core.benchmark.BudgetedModuleRunner`:
    that runner *silently truncates* past its soft cap (correct for the
    benchmark harness, wrong here -- a verb call must raise, never silently
    hand back a partially-scored, partially-clustered result).
    """

    def __init__(self, module: Module[Any], *, budget_usd: float) -> None:
        self._module = module
        self._budget_usd = budget_usd

    def forward(self, candidates: Iterator[ERCandidate[Any]]) -> Iterator[PairwiseJudgement]:
        monitor = SpendMonitor(budget_usd=self._budget_usd)
        produced: list[PairwiseJudgement] = []
        for judgement in self._module.forward(candidates):
            produced.append(judgement)
            cost = judgement.provenance.get("cost_usd", 0.0)
            monitor.add(float(cost) if cost is not None else 0.0)
            try:
                monitor.check()
            except BudgetExceeded as exc:
                exc.partial_judgements = list(produced)  # type: ignore[attr-defined]
                raise
            yield judgement

    def inspect_scores(self, judgements: list[PairwiseJudgement], sample_size: int = 10) -> Any:
        return self._module.inspect_scores(judgements, sample_size)


class ResolvedModule(NamedTuple):
    """:func:`resolve_judge`'s return: the capped scorer plus what was resolved."""

    module: Module[Any]
    judge_used: str
    model: str | None
    fallback_reason: str | None


def resolve_judge(
    judge: JudgeName | Module[Any],
    schema: type[BaseModel],
    *,
    model: str | None = None,
    entity_noun: str = "entity",
    budget_usd: float | None = None,
) -> ResolvedModule:
    """Resolve ``judge`` (including ``"auto"``) to a spend-capped scorer Module.

    Args:
        judge: ``"auto"``, one of the other :data:`JudgeName` values, or a
            ``Module`` instance (the escape hatch -- reported as
            ``judge_used="custom"``).
        schema: The entity schema.
        model: Model id override for ``"zero_shot_llm"`` (ignored otherwise).
        entity_noun: Domain noun for the LLM judge's prompt.
        budget_usd: Spend cap override; defaults to :data:`DEFAULT_BUDGET_USD`.

    Returns:
        A :class:`ResolvedModule` with the capped module, the resolved judge
        name, the resolved model (only for ``"zero_shot_llm"``), and any
        auto-fallback reason.
    """
    fallback_reason: str | None = None
    resolved_model = model

    if isinstance(judge, Module):
        judge_used = "custom"
        judge_kind: JudgeName | Module[Any] = judge
    elif judge == "auto":
        resolved_kind, resolved_model, fallback_reason = choose_auto_judge(Settings())
        judge_kind = resolved_kind
        judge_used = resolved_kind
    else:
        judge_kind = judge
        judge_used = judge

    if judge_used == "zero_shot_llm" and resolved_model is None:
        resolved_model = _OPENROUTER_MODEL

    built = build_judge(judge_kind, schema, model=resolved_model, entity_noun=entity_noun)
    capped_budget = DEFAULT_BUDGET_USD if budget_usd is None else budget_usd
    capped = _SpendCappedModule(built, budget_usd=capped_budget)
    return ResolvedModule(capped, judge_used, resolved_model, fallback_reason)


def _text_field_extractor(schema: type[BaseModel]) -> Any:
    """Concatenate every comparable string field into one blocking text.

    Schema-agnostic: derives the field list from
    :meth:`~langres.core.comparator.Comparator.from_schema` (every
    ``str | None`` field except ``id``) rather than assuming a field named
    ``"name"`` or similar.
    """
    field_names = [spec.name for spec in Comparator.from_schema(schema).feature_specs]

    def extract(entity: Any) -> str:
        parts = [str(getattr(entity, name)) for name in field_names if getattr(entity, name, None)]
        return " ".join(parts)

    return extract


def _build_vector_blocker(schema: type[BaseModel]) -> VectorBlocker[Any]:
    """Build a ``VectorBlocker`` (MiniLM + FAISS cosine) for ``schema``."""
    embedder = SentenceTransformerEmbedder(_VECTOR_MODEL_NAME)
    index = FAISSIndex(embedder=embedder, metric="cosine")
    return VectorBlocker(
        vector_index=index,
        schema=schema,
        text_field_extractor=_text_field_extractor(schema),
        k_neighbors=_VECTOR_K_NEIGHBORS,
    )


def build_embedding_candidate(
    schema: type[BaseModel], left: dict[str, Any], right: dict[str, Any]
) -> ERCandidate[Any]:
    """Build the one ``ERCandidate`` for a ``judge="embedding"`` pair, scored.

    Used by ``langres.verbs.link`` (a single pair -- no blocking needed):
    embeds both records' blocking text and attaches the cosine similarity via
    the same ``VectorBlocker``/FAISS path :func:`build_resolver` uses for
    ``dedupe()``, so the two verbs score embeddings identically.
    """
    blocker = _build_vector_blocker(schema)
    entities = [blocker.schema_factory(record) for record in (left, right)]
    texts = [blocker.text_field_extractor(entity) for entity in entities]
    blocker.vector_index.create_index(texts)
    return next(blocker.stream([left, right]))


def _estimate_n_pairs(n_records: int, *, use_vector: bool) -> int:
    """Worst-case pair-count estimate for the pre-scoring cost notice."""
    if use_vector:
        return n_records * _VECTOR_K_NEIGHBORS
    return n_records * (n_records - 1) // 2


def notice_pre_scoring_cost(model: str, n_pairs: int) -> None:
    """Emit the "about to spend money" notice before any paid scoring (D2).

    ``est_cost`` is a rough, worst-case-biased estimate
    (:data:`_ESTIMATED_TOKENS_PER_PAIR`) -- the spend cap
    (:class:`_SpendCappedModule`) meters and enforces the REAL cost live, per
    pair, as scoring happens.
    """
    price_per_1k = dspy_price_per_1k(model)
    est_cost = n_pairs * (_ESTIMATED_TOKENS_PER_PAIR / 1000.0) * price_per_1k
    _notice(f"scoring ~{n_pairs} pairs with {model!r}, est. cost ${est_cost:.4f}")


class ResolvedJudge(NamedTuple):
    """:func:`build_resolver`'s return: the pipeline plus what was resolved."""

    resolver: Resolver
    judge_used: str
    score_type: str
    fallback_reason: str | None


def build_resolver(
    schema: type[BaseModel],
    *,
    judge: JudgeName | Module[Any],
    model: str | None,
    entity_noun: str,
    threshold: float | None,
    n_records: int,
    budget_usd: float | None = None,
) -> ResolvedJudge:
    """Assemble a spend-capped Resolver for ``dedupe()``.

    Blocker rule: an embedding judge always uses a ``VectorBlocker`` (it needs
    the index's similarity score); every other judge uses ``AllPairsBlocker``
    at ``n_records <= 100`` and a ``VectorBlocker`` above it (O(N*k) instead of
    O(N^2)).

    Args:
        schema: The entity schema (any Pydantic model with an ``id`` field).
        judge: ``"auto"``, another :data:`JudgeName`, or a ``Module`` instance.
        model: Model id override for ``"zero_shot_llm"``.
        entity_noun: Domain noun for the LLM judge's prompt.
        threshold: Clusterer threshold; ``None`` resolves to the judge's
            default (:data:`_DEFAULT_THRESHOLDS`, D3).
        n_records: Size of the batch about to be resolved (drives the blocker
            choice and the pre-scoring cost estimate).
        budget_usd: Spend cap override; defaults to :data:`DEFAULT_BUDGET_USD`.

    Returns:
        A :class:`ResolvedJudge` with the assembled Resolver and judge metadata.
    """
    resolved = resolve_judge(
        judge, schema, model=model, entity_noun=entity_noun, budget_usd=budget_usd
    )

    use_vector = resolved.judge_used == "embedding" or n_records > _ALL_PAIRS_MAX_N
    blocker: Blocker[Any] = (
        _build_vector_blocker(schema) if use_vector else AllPairsBlocker(schema=schema)
    )
    comparator: Comparator[Any] | None = (
        Comparator.from_schema(schema) if resolved.judge_used == "string" else None
    )

    if resolved.judge_used == "zero_shot_llm" and resolved.model is not None:
        n_pairs_est = _estimate_n_pairs(n_records, use_vector=use_vector)
        notice_pre_scoring_cost(resolved.model, n_pairs_est)

    resolved_threshold = (
        _DEFAULT_THRESHOLDS.get(resolved.judge_used, 0.5) if threshold is None else threshold
    )
    resolver = Resolver(
        blocker=blocker,
        comparator=comparator,
        module=resolved.module,
        clusterer=Clusterer(threshold=resolved_threshold),
    )
    score_type = _SCORE_TYPE_BY_JUDGE.get(resolved.judge_used, "unknown")
    return ResolvedJudge(resolver, resolved.judge_used, score_type, resolved.fallback_reason)
