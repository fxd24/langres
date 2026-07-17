"""The competing-method registry: one uniform resolver-factory per method.

M3 races five resolution *methods* against each other on the same datasets. The
:mod:`langres.core.benchmark` harness consumes a ``resolver_factory:
Callable[[float], Resolver]`` (a clusterer threshold -> a built pipeline) and
runs it through both evaluation tracks. This module maps a method name to such
a factory -- resolving the name through the one
:mod:`langres.core.method_registry` (shared with the verbs and
``Resolver.from_schema``, so a name means the same thing everywhere) -- and
:func:`~langres.core.benchmark.run_method` can then race *any* method on *any*
dataset.

The five methods differ **only in the scorer** (the ``module`` slot) — blocking
is held constant per dataset so the race compares judges, not blockers:

- ``rapidfuzz`` — VectorBlocker -> ``RapidfuzzMatcher`` (string similarity over the
  schema's comparable string fields) -> Clusterer.
- ``weighted_average`` — VectorBlocker -> ``StringComparator.from_schema`` ->
  ``WeightedAverageMatcher`` -> Clusterer (generalizes ``build_restaurant_resolver``
  to an arbitrary schema).
- ``embedding_cosine`` — VectorBlocker -> ``EmbeddingScoreMatcher`` (passes the
  blocker's cosine similarity through) -> Clusterer. No Comparator.
- ``llm_judge`` — VectorBlocker -> ``LLMMatcher`` -> Clusterer. The LLM client is
  *injected* (a mock in tests, the real frontier/GLM client in W4); it is never a
  hard import-time requirement.
- ``cascade`` — VectorBlocker -> ``CascadeChainMatcher`` (embedding early-exit, LLM only
  on the uncertain band) -> Clusterer. Its OpenAI client is injected the same way.
- ``dspy_judge`` — VectorBlocker -> ``DSPyMatcher`` (a compilable DSPy
  ``ChainOfThought``) -> Clusterer (M4). Its injected client is a **DSPy LM**
  (``dspy.LM`` / ``DummyLM``), distinct from the LiteLLM/OpenAI clients above.
- ``select_judge`` — VectorBlocker -> ``SelectMatcher`` (a ComEM-style set-wise
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

- ``fellegi_sunter`` — VectorBlocker -> ``StringComparator.from_schema`` ->
  ``FellegiSunterMatcher`` -> Clusterer. Learns m/u/prior via EM with **no
  labels** (``UnsupervisedFitMixin.fit_unlabeled``, i.e.
  ``resolver.fit(records)``).
- ``random_forest`` — VectorBlocker -> ``StringComparator.from_schema`` ->
  ``RandomForestMatcher`` -> Clusterer. sklearn RandomForest over comparator similarities,
  supervised (``SupervisedFitMixin.fit``, i.e.
  ``resolver.fit(records, labels=...)``).

A dataset participates by conforming to :class:`BlockingBenchmark` — exposing its
record ``schema`` plus a pinned blocking config (``blocking_k`` and a
``build_blocker`` that returns a *fresh* VectorBlocker each call). This module
imports no dataset (no ``langres.data``), so it stays free of the
``core -> data -> core`` cycle the harness was designed to avoid.
"""

from collections.abc import Callable
from typing import Any, Protocol

from langres.clients.openrouter import DEFAULT_OPENROUTER_MODEL
from langres.core.benchmark import _cost_track
from langres.core.blockers.vector import VectorBlocker
from langres.core.clusterer import Clusterer
from langres.core.comparator import Comparator
from langres.core.comparators import StringComparator
from langres.core.method_registry import get_method
from langres.core.models import PairwiseJudgement
from langres.core.usage import CostTrack
from langres.core.matcher import Matcher
from langres.core.matchers.cascade import CASCADE_LLM_DECISION_STEP
from langres.core.resolver import Resolver

# The canonical method-name tuples live in the import-light ``_method_names``
# leaf (no heavy deps) so name-listing (``data.registry.list_methods``) and
# dispatch (this module) share one source of truth. Re-exported here so the
# public ``langres.methods.ALL_METHODS`` etc. stay stable.
from langres._method_names import ALL_METHODS, LLM_METHODS, ZERO_SPEND_METHODS

#: Default LLM model id for the LLM/cascade methods -- an alias of the one
#: shared constant (``clients.openrouter.DEFAULT_OPENROUTER_MODEL``) so the
#: benchmark harness cannot drift from ``ERModel.from_schema`` on the default.
#: Tests inject a mock client and ignore the model.
DEFAULT_LLM_MODEL = DEFAULT_OPENROUTER_MODEL


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


def _make_module_builder(
    method: str,
    schema: type[Any],
    *,
    llm_client: Any,
    llm_model: str,
    cascade_low: float,
    cascade_high: float,
) -> tuple[Callable[[], Matcher[Any]], Comparator[Any] | None]:
    """Resolve a method name to its (module-builder, comparator) pair.

    A thin adapter over the one :mod:`langres.core.method_registry` (the #55
    unification): the name means exactly what it means on the verbs' and
    ``Resolver.from_schema``'s paths, and each spec's builder owns the
    construction details this module used to hand-roll (the DSPy price pin,
    cascade's injected-client shape, lazy heavy imports).

    The module builder is called once per resolver (fresh scorer each threshold);
    the comparator (if any) is shared across thresholds — it is stateless.

    Raises:
        UnknownMethodError: (a ``ValueError``) for an unknown method name,
            with the registered names and a did-you-mean suggestion.
    """
    spec = get_method(method)
    comparator: Comparator[Any] | None = (
        StringComparator.from_schema(schema) if spec.needs_comparator else None
    )
    params: dict[str, Any] = (
        {"cascade_low": cascade_low, "cascade_high": cascade_high} if method == "cascade" else {}
    )

    def build() -> Matcher[Any]:
        return spec.build(
            schema,
            model=llm_model,
            entity_noun="entity",
            client=llm_client,
            comparator=comparator,
            **params,
        )

    return build, comparator


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
            pre-existing ``CascadeChainMatcher`` constraint. W4 must pass a separate
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
            matcher=make_module(),
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
