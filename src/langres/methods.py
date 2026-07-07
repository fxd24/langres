"""The competing-method registry: one uniform resolver-factory per method.

M3 races five resolution *methods* against each other on the same datasets. The
:mod:`langres.core.benchmark` harness consumes a ``resolver_factory:
Callable[[float], Resolver]`` (a clusterer threshold -> a built pipeline) and
runs it through both evaluation tracks. This module is the single place that maps
a method name to such a factory, so :func:`~langres.core.benchmark.run_method`
can race *any* method on *any* dataset.

The five methods differ **only in the scorer** (the ``module`` slot) — blocking
is held constant per dataset so the race compares judges, not blockers:

- ``rapidfuzz`` — VectorBlocker -> ``RapidfuzzModule`` (string similarity over the
  schema's comparable string fields) -> Clusterer.
- ``weighted_average`` — VectorBlocker -> ``Comparator.from_schema`` ->
  ``WeightedAverageJudge`` -> Clusterer (generalizes ``build_restaurant_resolver``
  to an arbitrary schema).
- ``embedding_cosine`` — VectorBlocker -> ``EmbeddingScoreJudge`` (passes the
  blocker's cosine similarity through) -> Clusterer. No Comparator.
- ``llm_judge`` — VectorBlocker -> ``LLMJudge`` -> Clusterer. The LLM client is
  *injected* (a mock in tests, the real frontier/GLM client in W4); it is never a
  hard import-time requirement.
- ``cascade`` — VectorBlocker -> ``CascadeModule`` (embedding early-exit, LLM only
  on the uncertain band) -> Clusterer. Its OpenAI client is injected the same way.
- ``dspy_judge`` — VectorBlocker -> ``DSPyJudge`` (a compilable DSPy
  ``ChainOfThought``) -> Clusterer (M4). Its injected client is a **DSPy LM**
  (``dspy.LM`` / ``DummyLM``), distinct from the LiteLLM/OpenAI clients above.
- ``select_judge`` — VectorBlocker -> ``SelectJudge`` (a ComEM-style set-wise
  judge: ONE LLM call per anchor group instead of one call per pair) ->
  Clusterer (W1.1). Its injected client is a **DSPy LM**, same shape as
  ``dspy_judge``.

Two more methods are dispatchable by name (:func:`_make_module_builder`
recognizes them) but are deliberately **not** members of
``ZERO_SPEND_METHODS``/``ALL_METHODS`` — the trained family (W1.2) needs an
explicit fit step before scoring, which ``run_methods``/``run_method`` cannot
provide (they rebuild the module fresh, unfit, per grid threshold). Build
+ fit + evaluate them via ``Resolver.fit(...)`` and
``evaluate_judge_on_candidates`` instead — see ``docs/EXPERIMENTS.md``.

- ``fellegi_sunter`` — VectorBlocker -> ``Comparator.from_schema`` ->
  ``FellegiSunterJudge`` -> Clusterer. Learns m/u/prior via EM with **no
  labels** (``UnsupervisedFitMixin.fit_unlabeled``, i.e.
  ``resolver.fit(records)``).
- ``random_forest`` — VectorBlocker -> ``Comparator.from_schema`` ->
  ``RFJudge`` -> Clusterer. sklearn RandomForest over comparator similarities,
  supervised (``SupervisedFitMixin.fit``, i.e.
  ``resolver.fit(records, labels=...)``).

A dataset participates by conforming to :class:`BlockingBenchmark` — exposing its
record ``schema`` plus a pinned blocking config (``blocking_k`` and a
``build_blocker`` that returns a *fresh* VectorBlocker each call). This module
imports no dataset (no ``langres.data``), so it stays free of the
``core -> data -> core`` cycle the harness was designed to avoid.
"""

import warnings
from collections.abc import Callable
from typing import Any, Protocol

# ``_cost_track`` is the harness's spend aggregator. We deliberately reuse it
# (rather than re-implement its cost-key fallback math) so cascade spend totals
# stay identical to every other method's; it is a stable same-package internal
# that the harness's own tests also import directly.
# Relocated to ``langres.clients.openrouter.dspy_price_per_1k`` (dspy-free,
# layer-neutral) so both this module and ``langres.core.presets`` can share it
# without a ``core -> methods -> core`` import cycle. Aliased to the old
# private name since existing call sites in this module reference it.
from langres.clients.openrouter import dspy_price_per_1k as _dspy_price_per_1k
from langres.core.benchmark import CostTrack, _cost_track
from langres.core.blockers.vector import VectorBlocker
from langres.core.clusterer import Clusterer
from langres.core.comparator import Comparator, StringComparator
from langres.core.judges.embedding_score import EmbeddingScoreJudge
from langres.core.judges.fellegi_sunter import FellegiSunterJudge
from langres.core.judges.weighted_average import WeightedAverageJudge
from langres.core.models import PairwiseJudgement
from langres.core.module import Module
from langres.core.modules.cascade import CASCADE_LLM_DECISION_STEP, CascadeModule
from langres.core.modules.llm_judge import LLMJudge
from langres.core.modules.rapidfuzz import RapidfuzzModule
from langres.core.modules.rf_judge import RFJudge
from langres.core.resolver import Resolver

#: Methods whose scorer makes no API call — fully deterministic and zero-spend.
ZERO_SPEND_METHODS: tuple[str, ...] = ("rapidfuzz", "weighted_average", "embedding_cosine")

#: Methods whose scorer calls an LLM — they take an injected client (mock/real).
#: ``dspy_judge`` and ``select_judge`` are LLM-backed too, but their injected
#: client is a **DSPy LM** (``dspy.LM`` / ``DummyLM``), not the LiteLLM/OpenAI
#: client the others take — see :func:`_make_module_builder`. ``select_judge``
#: (W1.1, ComEM-style set-wise) additionally makes ONE LLM call per anchor
#: GROUP instead of one call per pair — see
#: :class:`~langres.core.modules.select_judge.SelectJudge`.
LLM_METHODS: tuple[str, ...] = ("llm_judge", "cascade", "dspy_judge", "select_judge")

#: Every method the registry can build, in race order.
ALL_METHODS: tuple[str, ...] = ZERO_SPEND_METHODS + LLM_METHODS

#: Default LLM model id for the LLM/cascade methods. Overridden in W4 with the
#: real frontier/GLM model; tests inject a mock client and ignore the model.
DEFAULT_LLM_MODEL = "openrouter/openai/gpt-4o-mini"


class BlockingBenchmark(Protocol):
    """A benchmark that exposes its schema + pinned blocking config to a method.

    This is the extra contract (beyond the core
    :class:`~langres.core.benchmark.Benchmark`) a dataset must satisfy to be raced
    by this registry. ``build_blocker`` must return a **fresh, unbuilt**
    VectorBlocker on every call (its index is populated later, per resolver), so
    each ``resolver_factory(threshold)`` gets an independent blocker.

    Attributes:
        schema: The Pydantic record type (drives Comparator/rapidfuzz fields).
        blocking_k: The pinned nearest-neighbour count (blocking held constant).
    """

    schema: type[Any]
    blocking_k: int

    def build_blocker(self, k_neighbors: int) -> VectorBlocker[Any]:
        """Return a fresh VectorBlocker over the dataset's blocking text."""
        ...  # pragma: no cover


def _field_getter(field: str) -> Callable[[Any], str]:
    """A string extractor for ``field`` (missing / non-str -> empty string).

    The ``-> ""`` for a missing value is RapidfuzzModule's documented convention
    (``lambda x: x.address or ""``). Note this makes ``rapidfuzz`` score a field
    that is absent on *both* records as a perfect match (``fuzz.ratio("", "")``
    is ``100``), so missing data lifts the score — unlike the missing-aware
    ``weighted_average``, which drops absent features. That asymmetry is an
    *intrinsic* property of the classical string baseline, not a wiring bug: the
    race is meant to surface exactly such method differences, so it is left as-is.
    """

    def get(entity: Any) -> str:
        value = getattr(entity, field, None)
        return value if isinstance(value, str) else ""

    return get


def _rapidfuzz_extractors(
    schema: type[Any],
) -> dict[str, tuple[Callable[[Any], str], float]]:
    """Derive RapidfuzzModule field extractors from a schema's comparable fields.

    Reuses ``Comparator.from_schema``'s field selection (``str | None`` fields,
    ``id`` excluded) and weights, so ``rapidfuzz`` and ``weighted_average`` score
    on the *same* fields — the race isolates the scorer, not the field set. (They
    still differ in *missing-field handling*; see :func:`_field_getter`.)
    """
    specs = Comparator.from_schema(schema).feature_specs
    return {spec.name: (_field_getter(spec.name), spec.weight) for spec in specs}


def _build_cascade_module(
    *,
    llm_client: Any,
    llm_model: str,
    low_threshold: float,
    high_threshold: float,
) -> CascadeModule[Any]:
    """Build a CascadeModule with an injected LLM client.

    ``CascadeModule`` requires a non-empty ``llm_api_key`` at construction even
    when no pair escalates. We satisfy that with a placeholder and inject the real
    client (a mock in tests, the live client in W4) so no live key is ever needed
    at build time. The injected client must be **OpenAI-shaped** — cascade calls
    ``client.chat.completions.create(...)``, not the ``completion(...)`` an
    ``llm_judge`` (LiteLLM) client exposes.
    """
    with warnings.catch_warnings():
        # CascadeModule is deprecated in favor of CascadeJudge (T3), but this
        # benchmark method registry still constructs it deliberately (migration
        # tracked in TODOS.md). Suppress the DeprecationWarning at this one
        # sanctioned construction site so run_methods("cascade") stays
        # noise-free for callers.
        warnings.simplefilter("ignore", DeprecationWarning)
        module: CascadeModule[Any] = CascadeModule(
            llm_model=llm_model,
            llm_api_key="injected",
            low_threshold=low_threshold,
            high_threshold=high_threshold,
        )
    if llm_client is not None:
        module._llm_client = llm_client
    return module


def _make_module_builder(
    method: str,
    schema: type[Any],
    *,
    llm_client: Any,
    llm_model: str,
    cascade_low: float,
    cascade_high: float,
) -> tuple[Callable[[], Module[Any]], Comparator[Any] | None]:
    """Resolve a method name to its (module-builder, comparator) pair.

    The module builder is called once per resolver (fresh scorer each threshold);
    the comparator (if any) is shared across thresholds — it is stateless.
    """
    if method == "rapidfuzz":
        extractors = _rapidfuzz_extractors(schema)
        return (lambda: RapidfuzzModule(field_extractors=extractors)), None
    if method == "weighted_average":
        comparator: Comparator[Any] = Comparator.from_schema(schema)
        specs = comparator.feature_specs
        return (lambda: WeightedAverageJudge(feature_specs=specs)), comparator
    if method == "embedding_cosine":
        return (lambda: EmbeddingScoreJudge()), None
    if method == "llm_judge":
        return (lambda: LLMJudge(client=llm_client, model=llm_model)), None
    if method == "dspy_judge":
        # ``dspy_judge`` takes a **DSPy LM** as its injected client — a
        # ``dspy.LM(...)`` for real runs or a ``dspy.utils.dummies.DummyLM`` in
        # tests — NOT the LiteLLM ``client.completion(...)`` shape ``llm_judge``
        # expects. Imported lazily so building any other method's factory (and
        # plain ``import langres.methods``) never imports ``dspy``.
        from langres.core.modules.dspy_judge import DSPyJudge

        # Wire the honest-cost seam: DSPyJudge prices each pair as
        # ``tokens/1000 * price_per_1k_tokens`` into ``provenance["cost_usd"]`` that
        # the DEFAULT ``_cost_track`` reads — but its price defaults to $0, so a real
        # paid run would report $0 and the live budget-stop would never fire. Pin the
        # per-1k price from the OpenRouter table so cost is honest with no custom
        # ``cost_track_fn`` (unknown models keep $0; see ``_dspy_price_per_1k``).
        price_per_1k = _dspy_price_per_1k(llm_model)

        def build_dspy_judge() -> Module[Any]:
            judge: DSPyJudge[Any] = DSPyJudge(lm=llm_client, model=llm_model)
            judge.price_per_1k_tokens = price_per_1k
            return judge

        return build_dspy_judge, None
    if method == "select_judge":
        # ComEM-style set-wise judge (W1.1): same DSPy-LM injection contract as
        # ``dspy_judge`` (a ``dspy.LM(...)`` / ``DummyLM``), lazily imported for
        # the same import-safety reason, and priced from the same table — it
        # differs only in scoring a whole anchor GROUP per call, not one pair.
        from langres.core.modules.select_judge import SelectJudge

        select_price_per_1k = _dspy_price_per_1k(llm_model)

        def build_select_judge() -> Module[Any]:
            judge: SelectJudge[Any] = SelectJudge(lm=llm_client, model=llm_model)
            judge.price_per_1k_tokens = select_price_per_1k
            return judge

        return build_select_judge, None
    if method == "cascade":
        return (
            lambda: _build_cascade_module(
                llm_client=llm_client,
                llm_model=llm_model,
                low_threshold=cascade_low,
                high_threshold=cascade_high,
            )
        ), None
    if method == "fellegi_sunter":
        fs_comparator: StringComparator[Any] = Comparator.from_schema(schema)
        return (lambda: FellegiSunterJudge(comparator=fs_comparator)), fs_comparator
    if method == "random_forest":
        rf_comparator: StringComparator[Any] = Comparator.from_schema(schema)
        return (lambda: RFJudge(feature_specs=rf_comparator.feature_specs)), rf_comparator
    raise ValueError(
        f"unknown method {method!r}; choose one of "
        f"{ALL_METHODS + ('fellegi_sunter', 'random_forest')}"
    )


def make_resolver_factory(
    method: str,
    benchmark: BlockingBenchmark,
    *,
    llm_client: Any = None,
    llm_model: str = DEFAULT_LLM_MODEL,
    cascade_low: float = 0.3,
    cascade_high: float = 0.9,
) -> Callable[[float], Resolver]:
    """Build a ``threshold -> Resolver`` factory for ``method`` on ``benchmark``.

    The returned factory is exactly the contract
    :func:`~langres.core.benchmark.run_method` consumes: each call builds a fresh,
    independent resolver (fresh blocker + scorer) at the given clusterer
    threshold, with blocking pinned to ``benchmark.blocking_k`` so every method
    races on the identical candidate set.

    Args:
        method: One of :data:`ALL_METHODS`.
        benchmark: The dataset adapter (must expose ``schema`` + ``build_blocker``).
        llm_client: Injected LLM client for ``llm_judge`` / ``cascade`` (a mock in
            tests, the real client in W4). Ignored by zero-spend methods. Never a
            live key requirement at build time. **The two LLM methods expect
            different client shapes and are not interchangeable:** ``llm_judge``
            calls ``client.completion(...)`` (LiteLLM-shaped, e.g.
            ``langres.clients.create_llm_client``) while ``cascade`` calls
            ``client.chat.completions.create(...)`` (OpenAI-shaped) — a
            pre-existing ``CascadeModule`` constraint. W4 must pass a separate
            OpenAI-shaped client for ``cascade`` (do not reuse one client for
            both).
        llm_model: Model id for the LLM methods (overridden in W4).
        cascade_low: Cascade lower early-exit threshold (below = not a match).
        cascade_high: Cascade upper early-exit threshold (above = a match).

    Returns:
        A ``Callable[[float], Resolver]`` ready for ``run_method`` or direct use.

    Raises:
        ValueError: If ``method`` is not a known method name.
    """
    make_module, comparator = _make_module_builder(
        method,
        benchmark.schema,
        llm_client=llm_client,
        llm_model=llm_model,
        cascade_low=cascade_low,
        cascade_high=cascade_high,
    )
    k_neighbors = benchmark.blocking_k

    def factory(threshold: float) -> Resolver:
        return Resolver(
            blocker=benchmark.build_blocker(k_neighbors),
            comparator=comparator,
            module=make_module(),
            clusterer=Clusterer(threshold=threshold),
        )

    return factory


def cascade_cost_track(judgements: list[PairwiseJudgement]) -> CostTrack:
    """Aggregate cascade judgements into a CostTrack with escalation diagnostics.

    The generic :func:`~langres.core.benchmark._cost_track` cannot derive the
    cascade-only ``escalation_rate`` / ``llm_calls_per_candidate`` from a flat
    judgement list, so it leaves them ``None``. This fills them from each
    judgement's ``decision_step``: a cascade pair escalates (one LLM call) iff its
    step is ``"llm_judgment"``. Spend totals reuse ``_cost_track`` (which reads the
    ``llm_cost_usd`` provenance key cascade writes).

    Args:
        judgements: The cascade module's per-pair judgements.

    Returns:
        A :class:`~langres.core.benchmark.CostTrack` with spend totals plus the
        escalation rate and mean LLM-calls-per-candidate populated.
    """
    base = _cost_track(judgements)
    n_pairs = len(judgements)
    escalated = sum(1 for j in judgements if j.decision_step == CASCADE_LLM_DECISION_STEP)
    rate = escalated / n_pairs if n_pairs > 0 else 0.0
    return base.model_copy(update={"escalation_rate": rate, "llm_calls_per_candidate": rate})
