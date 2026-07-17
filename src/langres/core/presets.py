"""Presets: resolve ``judge="auto"`` and assemble a spend-capped Resolver.

This is the machinery behind the two-verb DX layer (:mod:`langres.verbs`):
picking a judge from available API keys (failing fast with
:class:`NoMatcherAvailableError` when ``judge="auto"`` finds none -- never a
silent fallback to fuzzy matching), building its scorer Matcher, wiring a
blocker by dataset size, and wrapping the scorer in a hard spend cap. It sits
strictly ABOVE :class:`~langres.core.resolver.Resolver` (which must not import
back from here -- ``Resolver.from_schema`` keeps its own thin policy switch
over the shared :mod:`~langres.core.method_registry`) and BELOW
:mod:`langres.verbs`:

    verbs.py -> core/presets.py -> Resolver -> {Blocker, Comparator, Matcher, Clusterer}

Nothing here is domain-specific: every function takes a Pydantic ``schema``
type and works for any schema (the registry's ``"string"``/``"embedding"``
builders derive their fields from the schema, never a hard-coded name).

Threshold semantics differ across judges (E12): ``"heuristic"`` (string),
``"sim_cos"`` (embedding), and ``"prob_llm"`` (zero_shot_llm / prompt_llm) are
different score scales, so one hand-picked ``threshold=0.7`` is not comparable
across them. Each judge's :class:`~langres.core.method_registry.MethodSpec`
carries its own sane default (``default_threshold``); pass ``threshold=``
explicitly to override, and calibrate a real one with
:func:`~langres.core.calibration.derive_threshold` once you have labels.

Judge construction is NOT hand-rolled here anymore: every name resolves
through the one :mod:`langres.core.method_registry` (the #55 unification), so
a judge name means the same thing on this path, on
``Resolver.from_schema``'s, and on the benchmark harness's. This layer keeps
only its *policy*: which names the verbs expose (:data:`MatcherName`), the
``"auto"`` key-based resolution, and the spend cap.

Spend cap (adopted CEO decision #8 + Eng E1/E9): every judge -- including the
free ones -- is wrapped in
:class:`~langres.core.spend_cap.SpendCappedMatcher` (re-exported here as
``_SpendCappedMatcher``), a small :class:`~langres.core.matcher.Matcher` that
tallies each judgement's ``provenance["cost_usd"]`` through a
:class:`~langres.core.spend.SpendMonitor` and raises
:class:`~langres.core.spend.BudgetExceeded` the moment cumulative spend crosses
``budget_usd`` (default :data:`~langres.core.spend_cap.DEFAULT_BUDGET_USD`).
Spend is bounded by the budget plus at most one further call -- an LLM call's
cost is only knowable once made. The exception carries every judgement already
produced (and paid for) on ``.partial_judgements`` -- recover them with::

    try:
        result = dedupe(records, budget_usd=0.50)
    except BudgetExceeded as exc:
        already_scored = exc.partial_judgements  # list[PairwiseJudgement]
        # e.g. cluster just those, or raise budget_usd and retry

Where the cap is applied differs by verb, and deliberately:
:func:`resolve_judge` wraps the scorer it returns (``link`` scores one pair
itself, with no Resolver in the picture), while :func:`build_resolver` hands the
*unwrapped* scorer to :class:`~langres.core.resolver.Resolver` along with
``budget_usd`` and lets the Resolver's own per-instance cap enforce it (B1).
There is exactly ONE cap on the ``dedupe`` path, not two.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

from pydantic import BaseModel

from langres.clients.openrouter import (
    DEFAULT_OPENROUTER_MODEL,
    dspy_price_per_1k,
)
from langres.clients.settings import Settings
from langres.core.blocker import Blocker
from langres.core.blockers.all_pairs import AllPairsBlocker
from langres.core.clusterer import Clusterer
from langres.core.comparator import Comparator
from langres.core.comparators import StringComparator
from langres.core.method_registry import (
    DEFAULT_EMBEDDING_MODEL,
    UnknownMethodError,
    get_method,
)
from langres.core.models import ERCandidate
from langres.core.matcher import Matcher
from langres.core.resolver import Resolver

from langres.core.spend_cap import DEFAULT_BUDGET_USD as DEFAULT_BUDGET_USD
from langres.core.spend_cap import SpendCappedMatcher, effective_budget

# The spend cap MOVED to the core leaf `langres.core.spend_cap` (B1) so
# `Resolver` -- which this module sits above and imports -- can enforce a budget
# too. These aliases keep the historical private names importable from here:
# `_SpendCappedMatcher` and `_effective_budget` are used by verbs.py,
# benchmark.py, and tests/docs/examples across the repo. Plain assignments, not
# `import X as _X` -- mypy only treats `import X as X` as an explicit re-export.
_SpendCappedMatcher = SpendCappedMatcher
_effective_budget = effective_budget

if TYPE_CHECKING:
    # [semantic] extra (faiss/sentence-transformers/torch) -- imported lazily
    # inside _build_vector_blocker (W0.4) so a core-only `import langres`
    # never pulls faiss/torch in for a caller who never picks judge="embedding"
    # or crosses the AllPairsBlocker -> VectorBlocker size threshold.
    from langres.core.blockers.vector import VectorBlocker
    from langres.core.embeddings import SentenceTransformerEmbedder
    from langres.core.indexes.vector_index import FAISSIndex

#: The judge names the verb layer understands, plus the "auto" meta-value that
#: :func:`choose_auto_judge` resolves down to one of the others.
MatcherName = Literal["auto", "zero_shot_llm", "prompt_llm", "embedding", "string"]

#: The concrete judge names the verbs expose (:data:`MatcherName` minus
#: ``"auto"``) -- the verb-layer *policy* allowlist over the shared
#: :mod:`~langres.core.method_registry`. Benchmark-only method names
#: (``rapidfuzz``, ``cascade``, the fit-requiring trained family, ...) resolve
#: through the same registry but are deliberately NOT reachable here: they
#: need an injected client or an explicit fit step the verbs cannot provide.
_VERB_JUDGE_NAMES: tuple[str, ...] = ("string", "embedding", "zero_shot_llm", "prompt_llm")

#: The verb judges whose scoring makes PAID API calls -- gates the pre-scoring
#: cost notice in :func:`build_resolver` and ``langres.link``.
PAID_JUDGE_NAMES: tuple[str, ...] = ("zero_shot_llm", "prompt_llm")

#: **The pinned default model for ``judge="auto"``** (and the fallback for an
#: explicit ``judge="zero_shot_llm"``/``"prompt_llm"`` with no ``model=``).
#:
#: Policy: this is a deliberate, documented default -- an alias of
#: ``langres.clients.openrouter.DEFAULT_OPENROUTER_MODEL`` (defined once so the
#: verbs, ``Resolver.from_schema``, the method registry's LLM specs, and the
#: benchmark harness can never drift on the literal). ``judge="auto"`` resolves
#: to it when an ``OPENROUTER_API_KEY`` is discovered (the preferred route);
#: with only an ``OPENAI_API_KEY`` set, auto routes direct-to-OpenAI via
#: :data:`_OPENAI_MODEL` instead. Every result reports the resolved model
#: (``LinkVerdict.model`` / ``DedupeResult.model``), so what ran is never
#: invisible. **Changing this constant is a user-facing behavior change**
#: (different quality, cost, and privacy posture for every default
#: ``link()``/``dedupe()`` call) **and requires a CHANGELOG entry**.
DEFAULT_AUTO_MODEL: str = DEFAULT_OPENROUTER_MODEL

#: The direct-OpenAI route ``judge="auto"`` falls back to when only
#: ``OPENAI_API_KEY`` is set (see :data:`DEFAULT_AUTO_MODEL`'s policy note --
#: the same changelog-entry rule applies).
_OPENAI_MODEL = "openai/gpt-5-mini"

#: ``AllPairsBlocker`` is used at or below this many records; above it,
#: :func:`build_resolver` switches to a ``VectorBlocker`` (O(N*k) instead of
#: O(N^2)). An embedding judge always uses the VectorBlocker regardless of N
#: (it needs the index's similarity score to score on). The blocker's embedder
#: is the shared ``method_registry.DEFAULT_EMBEDDING_MODEL``.
_ALL_PAIRS_MAX_N = 100
_VECTOR_K_NEIGHBORS = 10


def default_threshold_for(judge_used: str) -> float:
    """The decision threshold for ``judge_used`` when the caller passed ``None``.

    Reads the judge's :class:`~langres.core.method_registry.MethodSpec`
    (``default_threshold`` -- D3/E12: score scales differ per family, so each
    method carries its own default). A name outside the registry (``"custom"``
    for an injected ``Matcher``) falls back to ``0.5``.
    """
    try:
        return get_method(judge_used).default_threshold
    except UnknownMethodError:
        return 0.5


def _score_type_for(judge_used: str) -> str:
    """The static ``PairwiseJudgement.score_type`` label for ``judge_used``.

    Used as a fallback when no judgement was actually produced (e.g. a blocker
    that yields no candidate pairs). ``"unknown"`` for names outside the
    registry (an injected ``Matcher``).
    """
    try:
        return get_method(judge_used).score_type
    except UnknownMethodError:
        return "unknown"


#: Rough, deliberately worst-case-biased token count for the pre-scoring cost
#: estimate (``notice_pre_scoring_cost``). The real, metered cost is what the
#: spend cap actually enforces per pair as scoring happens -- this constant
#: only sizes the upfront heads-up line.
_ESTIMATED_TOKENS_PER_PAIR = 500


def _notice(message: str) -> None:
    """Emit a user-visible notice via ``warnings.warn`` (D2).

    ``logger.info`` is invisible under default logging config, so both the
    auto-judge selection notice and the pre-scoring cost line go through this
    one channel instead -- tests assert on it with ``pytest.warns``.
    """
    warnings.warn(message, stacklevel=3)


class NoMatcherAvailableError(RuntimeError):
    """``judge="auto"`` refused to pick a judge it cannot run safely.

    Raised (never a silent fallback) when no LLM API key is set, when
    ``LANGRES_OFFLINE`` is truthy (the deterministic keyless switch -- every
    key is treated as absent), or when the selected model's price is unpinned
    so the spend cap would be blind. The message carries the exact fixes; the
    offline escape hatch is an explicit ``judge="string"``. Root-exported as
    ``langres.NoMatcherAvailableError``.
    """


#: Install line for the ``[llm]`` extra, shared by every error message that
#: funnels a user onto the LLM-judge path (the keyed path dead-ends without it:
#: ``dspy`` is imported at dspy_judge.py module level but ships in the extra).
_INSTALL_LLM_EXTRA = "`uv sync --extra llm` or `pip install 'langres[llm]'`"
_GETTING_STARTED_URL = "https://github.com/fxd24/langres/blob/main/docs/GETTING_STARTED.md"


def choose_auto_judge(
    settings: Settings, *, model: str | None = None, budget_usd: float | None = None
) -> tuple[MatcherName, str]:
    """Resolve ``judge="auto"`` from available API keys -- or refuse, loudly.

    ``LANGRES_OFFLINE`` truthy -> :class:`NoMatcherAvailableError` (every key is
    treated as absent); else ``OPENROUTER_API_KEY`` set -> the OpenRouter
    gpt-4o-mini route; else ``OPENAI_API_KEY`` set -> the direct-OpenAI
    gpt-5-mini route; else -> :class:`NoMatcherAvailableError`. There is
    deliberately no silent fallback to ``"string"``: unsupervised fuzzy
    matching over-merges on unlabeled data, so the offline judge is an
    explicit opt-in, never a default. A caller-supplied ``model=`` overrides
    the key-derived pick (and runs the same pinned-price check); a model whose
    price is unpinned -- $0-metered, so the spend cap would be blind -- is
    refused too (E1; explicit ``judge="zero_shot_llm"`` remains the blind-cap
    escape hatch).

    Key discovery (what "set" means): each key/flag is read from ``settings``,
    which resolves, per field, constructor kwargs > process env > ``.env`` in
    the current working directory (CWD-relative, no parent-directory walk-up
    -- see :class:`~langres.clients.settings.Settings`). Two consequences:

    - Merely UNSETTING ``OPENROUTER_API_KEY``/``OPENAI_API_KEY`` does not
      produce a keyless run inside a project whose ``.env`` carries a key --
      the ``.env`` refills it. To force the keyless fail-fast path
      deterministically, set ``LANGRES_OFFLINE=1`` (process-wide switch), or
      set the key variables to the EMPTY string (an empty env var wins over
      ``.env`` and counts as absent).
    - The decision is made from ``settings`` alone, BEFORE litellm/dspy is
      imported -- litellm's own import-time ``load_dotenv()`` (which walks up
      the directory tree) can never influence it.

    ``LANGRES_OFFLINE`` is scoped to this auto path: an explicit
    ``judge="zero_shot_llm"``/``"string"``/``"embedding"`` or a ``Matcher``
    instance in code bypasses it (explicit code beats ambient environment).

    The happy path emits one selection notice via :func:`_notice` -- which
    model was picked, that paid API calls follow, and the cap -- BEFORE any
    paid call is made.

    Args:
        settings: Loaded :class:`~langres.clients.settings.Settings` (reads
            ``langres_offline`` and ``openrouter_api_key`` /
            ``openai_api_key``).
        model: Caller's model-id override (honored instead of the
            key-derived default).
        budget_usd: Spend cap named in the selection notice; ``None``
            resolves to :data:`DEFAULT_BUDGET_USD`.

    Returns:
        ``("zero_shot_llm", model)`` -- the resolved judge and model id.

    Raises:
        NoMatcherAvailableError: ``LANGRES_OFFLINE`` is truthy, no API key is
            set, or the selected model has no pinned price in
            ``PRICES_PER_1M``.
    """
    if settings.langres_offline:
        raise NoMatcherAvailableError(
            'judge="auto" is disabled: LANGRES_OFFLINE is set, so every API key is '
            "treated as absent (deterministic keyless mode -- the process env beats "
            "any .env file).\n"
            "Fix A: unset LANGRES_OFFLINE (or set it to 0) to allow key discovery.\n"
            'Fix B: pass judge="string" to opt into offline fuzzy matching (lower '
            "quality; calibrate its threshold with "
            "langres.core.calibration.derive_threshold).\n"
            f"Guide: {_GETTING_STARTED_URL}"
        )
    if not settings.openrouter_api_key and not settings.openai_api_key:
        raise NoMatcherAvailableError(
            'judge="auto" found no API key (OPENROUTER_API_KEY / OPENAI_API_KEY are unset).\n'
            "langres refuses to fall back silently: unsupervised fuzzy string matching "
            "over-merges on unlabeled data.\n"
            f"Fix A: export OPENROUTER_API_KEY=... and install the LLM extra "
            f"({_INSTALL_LLM_EXTRA}).\n"
            'Fix B: pass judge="string" to opt into offline fuzzy matching (lower quality; '
            "calibrate its threshold with langres.core.calibration.derive_threshold).\n"
            f"LLM spend is capped at ${DEFAULT_BUDGET_USD:.2f} by default (budget_usd=). "
            f"Guide: {_GETTING_STARTED_URL}"
        )
    resolved_model = model or (DEFAULT_AUTO_MODEL if settings.openrouter_api_key else _OPENAI_MODEL)
    if dspy_price_per_1k(resolved_model) <= 0.0:
        raise NoMatcherAvailableError(
            f'judge="auto" selected {resolved_model!r}, but it has no pinned price in '
            "langres.clients.openrouter.PRICES_PER_1M -- its spend cap would be blind "
            "($0-metered), and a blind cap is no cap at all.\n"
            "Fix A: pin the model's price in PRICES_PER_1M, or use a model that already is.\n"
            'Fix B: pass judge="zero_shot_llm" explicitly to run it anyway (unmetered), or '
            'judge="string" for offline fuzzy matching.\n'
            f"Guide: {_GETTING_STARTED_URL}"
        )
    _notice(
        f'judge="auto" selected the LLM judge {resolved_model!r}: scoring makes PAID '
        f"API calls, capped at ${_effective_budget(budget_usd):.2f} (budget_usd=). The cap "
        "is enforced between calls, so one in-flight call can overrun it by its own cost."
    )
    return "zero_shot_llm", resolved_model


def build_judge(
    judge: MatcherName | Matcher[Any],
    schema: type[BaseModel],
    *,
    model: str | None = None,
    entity_noun: str = "entity",
    judge_params: dict[str, Any] | None = None,
) -> Matcher[Any]:
    """Build the scorer Matcher for a resolved ``judge``.

    Construction is delegated to the judge's
    :class:`~langres.core.method_registry.MethodSpec` (the one registry all
    three dispatch sites share); this function keeps the verb-layer policy:
    only the :data:`_VERB_JUDGE_NAMES` are reachable here.

    Args:
        judge: ``"zero_shot_llm"`` / ``"prompt_llm"`` / ``"embedding"`` /
            ``"string"``, or a ``Matcher`` instance passed through verbatim --
            the escape hatch (and the ``DummyLM``-injected-``DSPyMatcher``
            zero-spend test seam). ``"auto"`` is NOT resolved here; call
            :func:`choose_auto_judge` first (:func:`resolve_judge` does this).
        schema: The entity schema (drives ``"string"``'s comparator and the
            LLM judges' entity rendering). Works for ANY schema.
        model: Model id for the LLM judges. Defaults to
            :data:`DEFAULT_AUTO_MODEL` when omitted; ignored by the others.
        entity_noun: Domain noun woven into the LLM judge's prompt.
        judge_params: Judge-specific construction knobs, forwarded to the
            spec's builder -- for ``"prompt_llm"``: ``prompt_template``,
            ``system_prompt``, ``response_parser`` / ``record_serializer``
            (registered names serialize; see
            ``langres.core.matchers.llm_judge.RESPONSE_PARSERS``). An unknown
            param fails loudly (``TypeError``).

    Returns:
        A ready (uncapped) scorer Matcher.

    Raises:
        ValueError: If ``judge`` is an unrecognized string (including
            ``"auto"``, which only :func:`choose_auto_judge`/:func:`resolve_judge`
            resolve).
        ImportError: For an LLM judge when the ``[llm]`` extra is not
            installed -- raised by the builder with the install line.
    """
    if isinstance(judge, Matcher):
        return judge
    if judge not in _VERB_JUDGE_NAMES:
        raise ValueError(
            f"unknown judge {judge!r}; choose one of 'zero_shot_llm', 'prompt_llm', "
            "'embedding', 'string', 'auto', or pass a Matcher instance directly"
        )
    spec = get_method(judge)
    return spec.build(
        schema,
        model=model,
        entity_noun=entity_noun,
        client=None,
        comparator=None,
        **(judge_params or {}),
    )


class ResolvedModule(NamedTuple):
    """:func:`resolve_judge`'s return: the capped scorer plus what was resolved.

    ``model`` is the id of the underlying model that will actually score --
    the resolved LLM id for the LLM judges, the pinned embedder
    (:data:`~langres.core.method_registry.DEFAULT_EMBEDDING_MODEL`) for
    ``"embedding"``, ``None`` for pure-string similarity, and an injected
    ``Matcher``'s own ``model`` attribute (when it has a string one) for
    ``"custom"``. The verbs stamp it on every result.
    """

    module: Matcher[Any]
    judge_used: str
    model: str | None


def _module_model(module: Matcher[Any]) -> str | None:
    """An injected ``Matcher``'s self-reported model id (its ``model`` attribute).

    Both LLM judge families expose ``model: str`` (``LLMMatcher`` / ``DSPyMatcher``
    and friends); anything else -- absent or non-string -- honestly reports
    ``None`` rather than fabricating an identity.
    """
    candidate = getattr(module, "model", None)
    return candidate if isinstance(candidate, str) else None


def _resolve_judge_uncapped(
    judge: MatcherName | Matcher[Any],
    schema: type[BaseModel],
    *,
    model: str | None = None,
    entity_noun: str = "entity",
    budget_usd: float | None = None,
    judge_params: dict[str, Any] | None = None,
) -> ResolvedModule:
    """:func:`resolve_judge`'s resolution WITHOUT the spend cap.

    Split out so :func:`build_resolver` can hand the raw scorer to
    :class:`~langres.core.resolver.Resolver`, which applies its own
    per-instance cap (B1) -- wrapping here as well would put two monitors on
    the same stream. Every caller that is not building a Resolver wants
    :func:`resolve_judge`, which is this plus the cap.

    ``budget_usd`` is used only for the ``"auto"`` selection notice's text; it
    caps nothing here.
    """
    if isinstance(judge, Matcher):
        return ResolvedModule(judge, "custom", _module_model(judge))

    resolved_model = model
    if judge == "auto":
        judge_used, resolved_model = choose_auto_judge(
            Settings(), model=model, budget_usd=budget_usd
        )
    else:
        judge_used = judge

    # The name is validated by build_judge below, but the backbone must be checked
    # BEFORE construction: a method that cannot run this ref should say so here,
    # not fail deep inside litellm (or, worse, drop the argument in silence).
    spec = get_method(judge_used)
    spec.check_backbone(resolved_model)

    built = build_judge(
        judge_used, schema, model=resolved_model, entity_noun=entity_noun, judge_params=judge_params
    )
    # The spec is the identity authority: a method with no model slot reports its
    # fixed default_model, and one WITH a slot reports the caller's override.
    # ``check_backbone`` above already rejected a model= the method cannot honor,
    # so this can no longer stamp an identity that did not run.
    resolved_model = (
        (resolved_model or spec.default_model) if spec.accepted_kinds else spec.default_model
    )
    return ResolvedModule(built, judge_used, resolved_model)


def resolve_judge(
    judge: MatcherName | Matcher[Any],
    schema: type[BaseModel],
    *,
    model: str | None = None,
    entity_noun: str = "entity",
    budget_usd: float | None = None,
    judge_params: dict[str, Any] | None = None,
) -> ResolvedModule:
    """Resolve ``judge`` (including ``"auto"``) to a spend-capped scorer Matcher.

    The cap binds for the returned wrapper's whole lifetime, not per
    ``forward()`` call, and bounds spend at ``budget_usd`` plus at most one
    further call (see :mod:`langres.core.spend_cap`).

    Args:
        judge: ``"auto"``, one of the other :data:`MatcherName` values, or a
            ``Matcher`` instance (the escape hatch -- reported as
            ``judge_used="custom"``).
        schema: The entity schema.
        model: Model id override for the LLM judges and ``"auto"`` (ignored
            otherwise). On the auto path the caller's model wins over
            :func:`choose_auto_judge`'s key-derived pick.
        entity_noun: Domain noun for the LLM judge's prompt.
        budget_usd: Spend cap override; defaults to
            :data:`~langres.core.spend_cap.DEFAULT_BUDGET_USD`.
        judge_params: Judge-specific construction knobs (see
            :func:`build_judge`).

    Returns:
        A :class:`ResolvedModule` with the capped module, the resolved judge
        name, and the resolved underlying model id (see the class docstring
        for what ``model`` means per judge).

    Raises:
        NoMatcherAvailableError: On the ``"auto"`` path when no API key is set
            or the selected model's price is unpinned (see
            :func:`choose_auto_judge`).
    """
    resolved = _resolve_judge_uncapped(
        judge,
        schema,
        model=model,
        entity_noun=entity_noun,
        budget_usd=budget_usd,
        judge_params=judge_params,
    )
    return resolved._replace(
        module=_SpendCappedMatcher(resolved.module, budget_usd=_effective_budget(budget_usd))
    )


def _text_field_extractor(schema: type[BaseModel]) -> Any:
    """Concatenate every comparable string field into one blocking text.

    Schema-agnostic: derives the field list from
    :meth:`~langres.core.comparator.Comparator.from_schema` (every
    ``str | None`` field except ``id``) rather than assuming a field named
    ``"name"`` or similar.
    """
    field_names = [spec.name for spec in StringComparator.from_schema(schema).feature_specs]

    def extract(entity: Any) -> str:
        parts = [str(getattr(entity, name)) for name in field_names if getattr(entity, name, None)]
        return " ".join(parts)

    return extract


def _build_vector_blocker(
    schema: type[BaseModel], *, model: str | None = None
) -> VectorBlocker[Any]:
    """Build a ``VectorBlocker`` (FAISS cosine) for ``schema`` over an embedder backbone.

    ``model`` names the sentence-transformers backbone; ``None`` pins
    :data:`~langres.core.model_ref.DEFAULT_EMBEDDING_MODEL`. This is the seam that
    makes ``matcher="embedding", model=...`` mean something: the ``embedding``
    method's model slot is the *blocker's* embedder (its matcher only passes the
    blocker's cosine similarity through), so the caller's backbone has to land
    here or nowhere.
    """
    # Lazy: faiss/sentence-transformers ([semantic] extra) must stay out of
    # sys.modules unless a VectorBlocker is actually built (mirrors the
    # zero_shot_llm branch's lazy dspy import right below).
    from langres.core.blockers.vector import VectorBlocker
    from langres.core.embeddings import SentenceTransformerEmbedder
    from langres.core.indexes.vector_index import FAISSIndex

    embedder = SentenceTransformerEmbedder(model or DEFAULT_EMBEDDING_MODEL)
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

    Raises:
        RuntimeError: If the blocker yields no candidate for the pair (should
            never happen for a two-record ``VectorBlocker.stream()`` call --
            guarded explicitly rather than letting a bare ``StopIteration``
            leak out of ``next()``, mirroring ``link()``'s identical
            no-judgement guard for the string/LLM path).
    """
    blocker = _build_vector_blocker(schema)
    entities = [blocker.schema_factory(record) for record in (left, right)]
    texts = [blocker.text_field_extractor(entity) for entity in entities]
    blocker.vector_index.create_index(texts)
    candidates = list(blocker.stream([left, right]))
    if not candidates:
        raise RuntimeError(
            "the embedding blocker produced no candidate for this pair; a "
            "VectorBlocker over exactly two records must always yield one. "
            "This indicates a bug in the vector index/blocker construction."
        )
    return candidates[0]


def _estimate_n_pairs(n_records: int, *, use_vector: bool) -> int:
    """Worst-case pair-count estimate for the pre-scoring cost notice."""
    if use_vector:
        return n_records * _VECTOR_K_NEIGHBORS
    return n_records * (n_records - 1) // 2


def notice_pre_scoring_cost(
    model: str, n_pairs: int, *, budget_usd: float = DEFAULT_BUDGET_USD
) -> None:
    """Emit the "about to spend money" notice before any paid scoring (D2).

    ``est_cost`` is a rough, worst-case-biased estimate
    (:data:`_ESTIMATED_TOKENS_PER_PAIR`) -- the spend cap
    (:class:`_SpendCappedMatcher`) meters and enforces the REAL cost live, per
    pair, as scoring happens.

    If ``model`` has no pinned price in :data:`PRICES_PER_1M`, DSPyMatcher
    self-reports ``$0`` per pair -- printing "est. cost $0.0000" here would be
    actively misleading: the spend cap tallies that same ``$0`` and can NEVER
    trip, so an unpinned model that OpenRouter really bills for would run
    uncapped in practice while looking capped. Warn honestly about the blind
    cap instead of the reassuring (and false) zero estimate.
    """
    price_per_1k = dspy_price_per_1k(model)
    if price_per_1k == 0.0:
        _notice(
            f"model {model!r} has no pinned price in "
            "langres.clients.openrouter.PRICES_PER_1M, so it self-reports $0/pair "
            f"cost -- the ${budget_usd:.2f} spend cap CANNOT enforce a limit for it "
            "and will not stop a runaway bill. Pin its price in PRICES_PER_1M, or "
            "use a model that already is, to get real spend-cap protection."
        )
        return
    est_cost = n_pairs * (_ESTIMATED_TOKENS_PER_PAIR / 1000.0) * price_per_1k
    _notice(f"scoring ~{n_pairs} pairs with {model!r}, est. cost ${est_cost:.4f}")


class ResolvedJudge(NamedTuple):
    """:func:`build_resolver`'s return: the pipeline plus what was resolved.

    ``model`` carries the resolved underlying model id (see
    :class:`ResolvedModule`) so ``dedupe()`` can stamp it on the result
    instead of dropping it here.
    """

    resolver: Resolver
    judge_used: str
    score_type: str
    model: str | None


def build_resolver(
    schema: type[BaseModel],
    *,
    judge: MatcherName | Matcher[Any],
    model: str | None,
    entity_noun: str,
    threshold: float | None,
    n_records: int,
    budget_usd: float | None = None,
    judge_params: dict[str, Any] | None = None,
) -> ResolvedJudge:
    """Assemble a spend-capped Resolver for ``dedupe()``.

    Blocker rule: an embedding judge always uses a ``VectorBlocker`` (it needs
    the index's similarity score); every other judge uses ``AllPairsBlocker``
    at ``n_records <= 100`` and a ``VectorBlocker`` above it (O(N*k) instead of
    O(N^2)).

    Args:
        schema: The entity schema (any Pydantic model with an ``id`` field).
        judge: ``"auto"``, another :data:`MatcherName`, or a ``Matcher`` instance.
        model: Model id override for the LLM judges.
        entity_noun: Domain noun for the LLM judge's prompt.
        threshold: Clusterer threshold; ``None`` resolves to the judge's
            default (its ``MethodSpec.default_threshold``, D3).
        n_records: Size of the batch about to be resolved (drives the blocker
            choice and the pre-scoring cost estimate).
        budget_usd: Spend cap override; defaults to :data:`DEFAULT_BUDGET_USD`.
        judge_params: Judge-specific construction knobs (see
            :func:`build_judge`).

    Returns:
        A :class:`ResolvedJudge` with the assembled Resolver and judge metadata.
    """
    # Uncapped on purpose: the Resolver built below owns the cap for this
    # pipeline (B1), so wrapping here too would meter the same stream twice.
    resolved = _resolve_judge_uncapped(
        judge,
        schema,
        model=model,
        entity_noun=entity_noun,
        budget_usd=budget_usd,
        judge_params=judge_params,
    )

    use_vector = resolved.judge_used == "embedding" or n_records > _ALL_PAIRS_MAX_N
    # The caller's model= names the EMBEDDER backbone only for the "embedding"
    # method, whose model slot IS this blocker (see _build_vector_blocker). For any
    # other judge, model= names that judge's LLM backbone and the blocker -- built
    # here only as a scaling choice above _ALL_PAIRS_MAX_N -- keeps its own default.
    blocker: Blocker[Any] = (
        _build_vector_blocker(
            schema, model=resolved.model if resolved.judge_used == "embedding" else None
        )
        if use_vector
        else AllPairsBlocker(schema=schema)
    )
    comparator: Comparator[Any] | None = (
        StringComparator.from_schema(schema) if resolved.judge_used == "string" else None
    )

    if resolved.judge_used in PAID_JUDGE_NAMES and resolved.model is not None:
        n_pairs_est = _estimate_n_pairs(n_records, use_vector=use_vector)
        notice_pre_scoring_cost(
            resolved.model, n_pairs_est, budget_usd=_effective_budget(budget_usd)
        )

    resolved_threshold = (
        default_threshold_for(resolved.judge_used) if threshold is None else threshold
    )
    resolver = Resolver(
        blocker=blocker,
        comparator=comparator,
        matcher=resolved.module,
        clusterer=Clusterer(threshold=resolved_threshold),
        budget_usd=budget_usd,
    )
    return ResolvedJudge(
        resolver, resolved.judge_used, _score_type_for(resolved.judge_used), resolved.model
    )
