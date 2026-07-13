"""One method registry: judge/method name -> :class:`MethodSpec` (closes #55's debt).

Before this module, a *name-selectable* judge had to be wired into three
hand-rolled dispatch switches -- ``core/presets.py:build_judge`` (the verbs),
``core/resolver.py:_build_module_for_judge`` (``Resolver.from_schema``), and
``methods.py:_make_module_builder`` (the benchmark harness) -- and the same
string could mean different things per layer (``llm_judge`` the *method* built
``LLMJudge`` while ``zero_shot_llm`` the *judge name* built ``DSPyJudge``).
All three sites now resolve through this one registry, so a judge name maps to
exactly one construction everywhere, together with its identity metadata: the
score family it emits, its default decision threshold, and the underlying
model id that :func:`langres.link`/:func:`langres.dedupe` stamp on results
(see the model-identity design note,
``docs/research/20260713_model_identity_and_hub.md``).

The registry does NOT replace each layer's *policy*: the verbs still expose
only the judge names that are safe without an injected client or a fit step
(``presets.build_judge``), and ``Resolver.from_schema`` still refuses
``"auto"`` (a verbs-layer feature). What is unified is the construction and
the metadata -- one spec per name, looked up by everyone.

Id grammar (decided once, deliberately): **bare names are built-ins**;
``/`` is **reserved** for future HF-style author namespacing of third-party
methods (``"jdoe/ditto"``). :func:`get_method` and :func:`register_method`
both reject slashed ids today so the namespace stays clean until the
entry-points publishing seam (v0.4) defines it. ``model=`` stays an
orthogonal kwarg -- model ids contain slashes themselves
(``openrouter/openai/gpt-4o-mini``), so the model axis never rides inside the
method id.

Import discipline: this module is eager-imported by ``langres.core``, so every
builder lazy-imports its judge class inside the build function -- dspy, litellm,
scikit-learn and friends must stay out of ``sys.modules`` on a bare
``import langres`` (see ``tests/test_import_budget.py``).
"""

from __future__ import annotations

import difflib
import warnings
from collections.abc import Callable
from typing import Any, cast

from pydantic import BaseModel, ConfigDict

from langres.core.comparator import Comparator
from langres.core.module import Module

__all__ = [
    "DEFAULT_EMBEDDING_MODEL",
    "MethodSpec",
    "UnknownMethodError",
    "get_method",
    "list_methods",
    "register_method",
]

#: The sentence-transformers model behind ``judge="embedding"`` and every
#: preset-built ``VectorBlocker`` (both the verbs' and ``Resolver.from_schema``'s
#: pipelines pin it). Defined once here so the two construction sites cannot
#: drift, and so the ``"embedding"`` spec below can report it as the judge's
#: model identity on results.
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

#: Install line for the ``[llm]`` extra, woven into the actionable ImportError
#: an LLM-family builder raises when its lazy import fails.
_INSTALL_LLM_EXTRA = "`uv sync --extra llm` or `pip install 'langres[llm]'`"


class UnknownMethodError(ValueError):
    """Raised when a judge/method name is not registered (or uses reserved grammar).

    Subclasses :class:`ValueError` so pre-registry callers that caught
    ``ValueError`` from the old dispatch switches keep working. The message
    carries the available names and a ``difflib`` did-you-mean suggestion.
    """


class MethodSpec(BaseModel):
    """One name-selectable judge method: builder + identity metadata.

    The single registration seam for judges (closing issue #55's three-site
    wiring debt): registering a spec makes the name resolvable by the verbs
    (subject to their allowlist), ``Resolver.from_schema``, and the benchmark
    harness alike.

    Attributes:
        name: The registry key (bare, slash-free -- see the module docstring's
            id grammar).
        build: The module builder. Called as ``build(schema, *, model,
            entity_noun, client, comparator, **params)`` where ``model`` may be
            ``None`` (the builder falls back to its own default),
            ``client`` is an optional injected LLM client/LM (tests, the
            benchmark harness), ``comparator`` is an optional pre-built
            :class:`~langres.core.comparator.Comparator` (so custom
            weights/excludes flow into feature-spec-driven judges), and
            ``params`` are judge-specific knobs (e.g. ``prompt_template`` for
            ``"prompt_llm"``). Unknown params fail loudly (``TypeError``).
        score_type: The :class:`~langres.core.models.PairwiseJudgement`
            score-family tag this judge emits -- used as the fallback label
            when a run produces no judgements.
        default_threshold: The decision threshold used when a caller passes
            ``threshold=None`` (E12: score scales differ per family, so each
            method carries its own sane default).
        default_model: The underlying model id when the caller names none --
            the value stamped on results as ``model`` (``None`` for judges
            with no model at all, e.g. pure-string similarity).
        accepts_model: Whether the builder honors a caller ``model=``
            override. ``False`` means the caller's ``model`` is ignored and
            ``default_model`` (if any) names the fixed underlying model.
        needs_comparator: Whether the pipeline must attach per-feature
            comparison vectors for this judge to score (drives the
            ``Comparator`` slot in the assembled ``Resolver``).
        requires_extra: The optional-dependency extra this judge needs
            (``"llm"``, ``"trained"``), or ``None``. The builder raises an
            actionable ``ImportError`` naming the install line when the extra
            is missing.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    build: Callable[..., Module[Any]]
    score_type: str
    default_threshold: float = 0.5
    default_model: str | None = None
    accepts_model: bool = False
    needs_comparator: bool = False
    requires_extra: str | None = None


_METHOD_REGISTRY: dict[str, MethodSpec] = {}


def _reject_namespaced(name: str) -> None:
    """Reject a slashed method id (reserved grammar -- see module docstring)."""
    if "/" in name:
        raise UnknownMethodError(
            f"unknown method {name!r}: '/' in a method id is reserved for future "
            "author-namespaced third-party methods ('author/method'); built-in "
            "methods use bare names. If you meant to pick a model, pass it as the "
            "separate model= argument (model ids like 'openrouter/openai/gpt-4o-mini' "
            "keep their slashes there)."
        )


def register_method(spec: MethodSpec) -> None:
    """Register ``spec`` under ``spec.name``.

    Args:
        spec: The method spec to register.

    Raises:
        UnknownMethodError: If ``spec.name`` contains ``/`` (reserved for the
            future ``author/method`` namespace -- v0.4's entry-points seam
            defines it; until then every registered name is bare).
        ValueError: If ``spec.name`` is already registered (loud collision,
            mirroring ``core.registry.register``).
    """
    _reject_namespaced(spec.name)
    if spec.name in _METHOD_REGISTRY:
        raise ValueError(f"Method '{spec.name}' is already registered")
    _METHOD_REGISTRY[spec.name] = spec


def get_method(name: str) -> MethodSpec:
    """Look up a registered :class:`MethodSpec` by name.

    Raises:
        UnknownMethodError: If ``name`` is not registered (message lists the
            available names with a did-you-mean suggestion), or contains the
            reserved ``/`` namespace separator.
    """
    _reject_namespaced(name)
    spec = _METHOD_REGISTRY.get(name)
    if spec is None:
        available = sorted(_METHOD_REGISTRY)
        suggestions = difflib.get_close_matches(name, available, n=3)
        hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        raise UnknownMethodError(
            f"unknown method {name!r}.{hint} Available methods: {', '.join(available)}"
        )
    return spec


def list_methods() -> list[str]:
    """Every registered method name, sorted."""
    return sorted(_METHOD_REGISTRY)


# ---------------------------------------------------------------------------
# Built-in specs
# ---------------------------------------------------------------------------
# Builders share one calling convention (see MethodSpec.build) and lazy-import
# anything beyond the always-installed core deps (import-budget discipline).


def _llm_extra_error(name: str, missing: str) -> ImportError:
    """The actionable [llm]-extra ImportError every LLM-family builder raises."""
    return ImportError(
        f'judge "{name}" needs the [llm] extra ({missing} is not installed). '
        f"Install it with {_INSTALL_LLM_EXTRA}."
    )


def _feature_specs(schema: type[BaseModel], comparator: Comparator[Any] | None) -> Any:
    """Feature specs from the caller's comparator (custom weights win) or the schema."""
    return (comparator if comparator is not None else Comparator.from_schema(schema)).feature_specs


def _build_string(
    schema: type[BaseModel],
    *,
    model: str | None = None,
    entity_noun: str = "entity",
    client: Any = None,
    comparator: Comparator[Any] | None = None,
) -> Module[Any]:
    """``string`` / ``weighted_average``: missing-aware weighted string similarity."""
    from langres.core.judges.weighted_average import WeightedAverageJudge

    return WeightedAverageJudge(feature_specs=_feature_specs(schema, comparator))


def _build_embedding(
    schema: type[BaseModel],
    *,
    model: str | None = None,
    entity_noun: str = "entity",
    client: Any = None,
    comparator: Comparator[Any] | None = None,
) -> Module[Any]:
    """``embedding`` / ``embedding_cosine``: pass the blocker's cosine similarity through."""
    from langres.core.judges.embedding_score import EmbeddingScoreJudge

    return EmbeddingScoreJudge()


def _build_zero_shot_llm(
    schema: type[BaseModel],
    *,
    model: str | None = None,
    entity_noun: str = "entity",
    client: Any = None,
    comparator: Comparator[Any] | None = None,
) -> Module[Any]:
    """``zero_shot_llm`` / ``dspy_judge``: the DSPy ChainOfThought judge, price pinned."""
    from langres.clients.openrouter import DEFAULT_OPENROUTER_MODEL, dspy_price_per_1k

    try:
        from langres.core.modules.dspy_judge import DSPyJudge
    except ImportError as exc:
        raise _llm_extra_error("zero_shot_llm", "dspy") from exc

    resolved_model = model or DEFAULT_OPENROUTER_MODEL
    judge: DSPyJudge[Any] = DSPyJudge(lm=client, model=resolved_model, entity_noun=entity_noun)
    # Honest-cost seam: DSPyJudge prices each pair as tokens/1000 * price. Its
    # price defaults to $0, so without this pin a real paid run would report $0
    # and the verbs' spend cap could never trip (unknown models stay $0 --
    # the callers warn about that blind cap).
    judge.price_per_1k_tokens = dspy_price_per_1k(resolved_model)
    return judge


def _build_prompt_llm(
    schema: type[BaseModel],
    *,
    model: str | None = None,
    entity_noun: str = "entity",
    client: Any = None,
    comparator: Comparator[Any] | None = None,
    **params: Any,
) -> Module[Any]:
    """``prompt_llm`` / ``llm_judge``: the prompt-seam ``LLMJudge`` (issue #103).

    ``params`` flow straight into :class:`~langres.core.modules.llm_judge.LLMJudge`
    (``prompt_template``, ``system_prompt``, ``response_parser`` /
    ``record_serializer`` -- registered *names* serialize, see
    ``llm_judge.RESPONSE_PARSERS`` -- ``on_parse_error``, ``temperature``,
    ``provider``, ``confidence``); an unknown param fails loudly.
    """
    from langres.clients.openrouter import DEFAULT_OPENROUTER_MODEL

    try:
        from langres.core.modules.llm_judge import LLMJudge
    except ImportError as exc:
        raise _llm_extra_error("prompt_llm", "litellm") from exc

    judge: LLMJudge[Any] = LLMJudge(
        client=client,
        model=model or DEFAULT_OPENROUTER_MODEL,
        entity_noun=entity_noun,
        **params,
    )
    return judge


def _build_select_judge(
    schema: type[BaseModel],
    *,
    model: str | None = None,
    entity_noun: str = "entity",
    client: Any = None,
    comparator: Comparator[Any] | None = None,
) -> Module[Any]:
    """``select_judge``: ComEM-style set-wise judge (one LLM call per anchor group)."""
    from langres.clients.openrouter import DEFAULT_OPENROUTER_MODEL, dspy_price_per_1k

    try:
        from langres.core.modules.select_judge import SelectJudge
    except ImportError as exc:
        raise _llm_extra_error("select_judge", "dspy") from exc

    resolved_model = model or DEFAULT_OPENROUTER_MODEL
    judge: SelectJudge[Any] = SelectJudge(lm=client, model=resolved_model)
    judge.price_per_1k_tokens = dspy_price_per_1k(resolved_model)
    return judge


def _build_rapidfuzz(
    schema: type[BaseModel],
    *,
    model: str | None = None,
    entity_noun: str = "entity",
    client: Any = None,
    comparator: Comparator[Any] | None = None,
) -> Module[Any]:
    """``rapidfuzz``: classical string baseline over the schema's comparable fields.

    Reuses ``Comparator.from_schema``'s field selection and weights so
    ``rapidfuzz`` and ``weighted_average`` score on the *same* fields. The
    ``-> ""`` for a missing value is RapidfuzzModule's documented convention;
    it makes a field absent on *both* records a perfect match -- an intrinsic
    property of the classical baseline, deliberately preserved (the benchmark
    race is meant to surface exactly such method differences).
    """
    from langres.core.modules.rapidfuzz import RapidfuzzModule

    def field_getter(field: str) -> Callable[[Any], str]:
        def get(entity: Any) -> str:
            value = getattr(entity, field, None)
            return value if isinstance(value, str) else ""

        return get

    specs = _feature_specs(schema, comparator)
    extractors = {spec.name: (field_getter(spec.name), spec.weight) for spec in specs}
    return RapidfuzzModule(field_extractors=extractors)


def _build_cascade(
    schema: type[BaseModel],
    *,
    model: str | None = None,
    entity_noun: str = "entity",
    client: Any = None,
    comparator: Comparator[Any] | None = None,
    cascade_low: float = 0.3,
    cascade_high: float = 0.9,
) -> Module[Any]:
    """``cascade``: embedding early-exit + LLM on the uncertain band.

    ``CascadeModule`` requires a non-empty ``llm_api_key`` at construction even
    when no pair escalates; a placeholder satisfies that and the real client (a
    mock in tests, a live one in paid runs) is injected, so no live key is ever
    needed at build time. The injected client must be **OpenAI-shaped**
    (``client.chat.completions.create(...)``), unlike ``llm_judge``'s
    LiteLLM-shaped ``client.completion(...)``.
    """
    from langres.clients.openrouter import DEFAULT_OPENROUTER_MODEL

    try:
        from langres.core.modules.cascade import CascadeModule
    except ImportError as exc:
        raise _llm_extra_error("cascade", "litellm") from exc

    with warnings.catch_warnings():
        # CascadeModule is deprecated in favor of CascadeJudge (T3), but this
        # registry still constructs it deliberately (migration tracked in
        # TODOS.md). Suppress the DeprecationWarning at this one sanctioned
        # construction site so run_methods("cascade") stays noise-free.
        warnings.simplefilter("ignore", DeprecationWarning)
        module: Module[Any] = CascadeModule(
            llm_model=model or DEFAULT_OPENROUTER_MODEL,
            llm_api_key="injected",
            low_threshold=cascade_low,
            high_threshold=cascade_high,
        )
    if client is not None:
        module._llm_client = client  # type: ignore[attr-defined]
    return module


def _build_fellegi_sunter(
    schema: type[BaseModel],
    *,
    model: str | None = None,
    entity_noun: str = "entity",
    client: Any = None,
    comparator: Comparator[Any] | None = None,
) -> Module[Any]:
    """``fellegi_sunter``: unsupervised EM-fit probabilistic record linkage."""
    from langres.core.comparator import StringComparator
    from langres.core.judges.fellegi_sunter import FellegiSunterJudge

    fs_comparator = comparator if comparator is not None else Comparator.from_schema(schema)
    # FellegiSunterJudge is typed against StringComparator, the concrete class
    # Comparator.from_schema returns (Comparator is its alias-in-spirit here).
    return FellegiSunterJudge(comparator=cast(StringComparator[Any], fs_comparator))


def _build_random_forest(
    schema: type[BaseModel],
    *,
    model: str | None = None,
    entity_noun: str = "entity",
    client: Any = None,
    comparator: Comparator[Any] | None = None,
) -> Module[Any]:
    """``random_forest``: supervised sklearn forest over comparator similarities."""
    from langres.core.modules.random_forest_judge import RandomForestJudge

    return RandomForestJudge(feature_specs=_feature_specs(schema, comparator))


def _register_builtins() -> None:
    """Seed the registry with the built-in methods (module import time).

    ``default_model`` for the LLM family deliberately references
    ``clients.openrouter.DEFAULT_OPENROUTER_MODEL`` -- the same constant
    ``presets.DEFAULT_AUTO_MODEL`` aliases -- so the registry, the auto-judge
    policy, and every builder agree on one literal.
    """
    from langres.clients.openrouter import DEFAULT_OPENROUTER_MODEL

    specs = [
        # -- The verb-facing judge family (presets/from_schema allowlists) ----
        MethodSpec(
            name="string",
            build=_build_string,
            score_type="heuristic",
            default_threshold=0.5,
            needs_comparator=True,
        ),
        MethodSpec(
            name="embedding",
            build=_build_embedding,
            score_type="sim_cos",
            default_threshold=0.5,
            # The judge scores the preset-built VectorBlocker's cosine sims,
            # so the pipeline's model IS the pinned embedder.
            default_model=DEFAULT_EMBEDDING_MODEL,
        ),
        MethodSpec(
            name="zero_shot_llm",
            build=_build_zero_shot_llm,
            score_type="prob_llm",
            default_threshold=0.7,
            default_model=DEFAULT_OPENROUTER_MODEL,
            accepts_model=True,
            requires_extra="llm",
        ),
        MethodSpec(
            name="prompt_llm",
            build=_build_prompt_llm,
            score_type="prob_llm",
            default_threshold=0.7,
            default_model=DEFAULT_OPENROUTER_MODEL,
            accepts_model=True,
            requires_extra="llm",
        ),
        # -- The benchmark-harness method names (methods.py race) ------------
        # Historical aliases keep their exact pre-registry construction; each
        # name has ONE meaning everywhere now (e.g. "llm_judge" builds LLMJudge
        # on every path -- it shares prompt_llm's builder).
        MethodSpec(
            name="rapidfuzz",
            build=_build_rapidfuzz,
            score_type="heuristic",
        ),
        MethodSpec(
            name="weighted_average",
            build=_build_string,
            score_type="heuristic",
            needs_comparator=True,
        ),
        MethodSpec(
            name="embedding_cosine",
            build=_build_embedding,
            score_type="sim_cos",
            # No default_model: the benchmark path scores whatever embedder the
            # dataset's pinned blocker used -- unknown at spec level.
        ),
        MethodSpec(
            name="llm_judge",
            build=_build_prompt_llm,
            score_type="prob_llm",
            default_threshold=0.7,
            default_model=DEFAULT_OPENROUTER_MODEL,
            accepts_model=True,
            requires_extra="llm",
        ),
        MethodSpec(
            name="dspy_judge",
            build=_build_zero_shot_llm,
            score_type="prob_llm",
            default_threshold=0.7,
            default_model=DEFAULT_OPENROUTER_MODEL,
            accepts_model=True,
            requires_extra="llm",
        ),
        MethodSpec(
            name="select_judge",
            build=_build_select_judge,
            score_type="prob_group_llm",
            default_threshold=0.7,
            default_model=DEFAULT_OPENROUTER_MODEL,
            accepts_model=True,
            requires_extra="llm",
        ),
        MethodSpec(
            name="cascade",
            # Mixed emitter: early-exits are "sim_cos", escalations "prob_llm";
            # the fallback tag names the LLM family it escalates into.
            build=_build_cascade,
            score_type="prob_llm",
            default_threshold=0.7,
            default_model=DEFAULT_OPENROUTER_MODEL,
            accepts_model=True,
            requires_extra="llm",
        ),
        MethodSpec(
            name="fellegi_sunter",
            build=_build_fellegi_sunter,
            score_type="prob_fs",
            needs_comparator=True,
        ),
        MethodSpec(
            name="random_forest",
            build=_build_random_forest,
            score_type="prob_rf",
            needs_comparator=True,
            requires_extra="trained",
        ),
    ]
    for spec in specs:
        register_method(spec)


_register_builtins()
