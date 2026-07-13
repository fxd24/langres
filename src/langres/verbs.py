"""The two-verb DX layer: ``link``, ``dedupe``, and their result types.

``link(left, right)`` and ``dedupe(records)`` are the thin, schema-optional
convenience layer on top of :mod:`langres.core.presets` (judge resolution +
spend cap) and :class:`~langres.core.resolver.Resolver` (blocking + scoring +
clustering). Schema-optional: pass ``schema=<YourModel>`` for a durable,
type-checked entity, or omit it and let :func:`dedupe`/:func:`link` infer an
ephemeral one from the records' own keys.

Both verbs share one small contract:

- ``judge="auto"`` (default) picks an LLM judge from the available API key
  (``OPENROUTER_API_KEY``/``OPENAI_API_KEY``) and emits one selection notice
  -- which model, that paid calls follow, the cap -- BEFORE any paid call.
  With no key -- or with ``LANGRES_OFFLINE=1``, the deterministic keyless
  switch -- it raises :class:`~langres.core.presets.NoJudgeAvailableError`
  (root-exported as ``langres.NoJudgeAvailableError``) instead of silently
  falling back: unsupervised fuzzy matching over-merges on unlabeled data,
  so the offline zero-spend ``"string"`` judge is an explicit opt-in. Key
  discovery order (process env > CWD ``.env``; empty string counts as
  absent) is documented on :func:`~langres.core.presets.choose_auto_judge`.
- Every judge, including the free ones, runs under a default $1 spend cap
  (override with ``budget_usd=``); a cap breach raises
  :class:`~langres.clients.openrouter.BudgetExceeded` carrying every
  judgement already produced on ``.partial_judgements`` (E9) -- see
  :mod:`langres.core.presets`'s module docstring for the resume recipe.
- Results are self-describing (D2): every verb reports which judge actually
  ran (``judge_used``), what its raw score means (``score_type`` -- see the
  threshold-semantics note below), and the effective ``threshold`` the match
  cut used -- so downstream flywheel steps (``select_for_review``,
  ``EvalReport``) can read the cut off the result instead of the caller
  remembering the float.
- Threshold semantics differ across ``score_type`` scales (E12): a
  ``"heuristic"`` score, a cosine ``"sim_cos"``, and an LLM ``"prob_llm"`` are
  not comparable on the same 0..1 cut. ``threshold=None`` (the default)
  resolves to a sane per-judge default; pass ``threshold=`` explicitly to
  override, or derive one from labels with
  :func:`~langres.core.calibration.derive_threshold`.
- ``log=`` (opt-in, ``None`` by default -- zero overhead) records every judge
  call to a JSONL file via :class:`~langres.core.judgement_log.JudgementLog`
  -- the flywheel inlet later harvested (W2.4) into training pairs for
  :func:`~langres.core.calibration.derive_threshold` and ``fit()``. See
  :mod:`langres.core.judgement_log` for the record shape and the
  ``features=True`` opt-in (PII note).

Known limitation: an *inferred* schema is ephemeral (a dynamically created
class, not importable by name) -- a ``dedupe()``/``link()`` call built on one
works fine in-process, but reloading a saved artifact referencing it in a
FRESH process raises the registry's existing ``SchemaNotRegistered`` error.
Pass ``schema=<YourModel>`` explicitly for anything you intend to persist.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, create_model

from langres.core.blockers.all_pairs import schema_to_factory
from langres.core.comparator import Comparator
from langres.core.judgement_log import JudgementLog, LoggingModule
from langres.core.models import (
    ERCandidate,
    JudgeAbstainedError,
    PairwiseJudgement,
    predicted_match,
)
from langres.core.module import Module
from langres.core.presets import (
    JudgeName,
    build_embedding_candidate,
    build_resolver,
    notice_pre_scoring_cost,
    resolve_judge,
)
from langres.core.presets import _DEFAULT_THRESHOLDS as _THRESHOLDS
from langres.core.presets import _effective_budget

__all__ = ["DedupeResult", "LinkVerdict", "dedupe", "link"]


class LinkVerdict(BaseModel):
    """The result of :func:`link`: a match decision with full provenance.

    Truthy iff ``match`` (``if link(a, b): ...``); ``repr`` shows the verdict,
    score, and judge for a friendly REPL/notebook experience (D2/D10).
    """

    match: bool
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    reasoning: str | None = None
    judge_used: str
    score_type: str
    #: The effective match cut ``match`` was decided at (the caller's
    #: ``threshold=``, or the judge's default when it was ``None``) -- feed it
    #: to ``select_for_review(threshold=...)`` instead of remembering the float.
    threshold: float
    judgement: PairwiseJudgement

    def __bool__(self) -> bool:
        return self.match

    def __repr__(self) -> str:
        verdict = "MATCH" if self.match else "NO MATCH"
        score = "n/a" if self.score is None else f"{self.score:.3f}"
        return f"LinkVerdict({verdict}, score={score}, judge={self.judge_used!r})"


class DedupeResult(list[set[str]]):
    """The clusters :func:`dedupe` returns -- a plain ``list[set[str]]``, self-describing.

    Behaves exactly like the list :meth:`~langres.core.resolver.Resolver.resolve`
    returns; additionally carries ``judge_used``, ``score_type`` and the
    effective ``threshold`` (D2) so a caller can inspect what actually ran --
    and feed ``threshold`` straight to
    :func:`~langres.core.review.select_for_review` / ``EvalReport`` -- without
    a separate call or a remembered constant. ``threshold`` is ``None`` only
    for the ``len(records) < 2`` short-circuit, where no judge (and hence no
    cut) was ever resolved.
    """

    def __init__(
        self,
        clusters: Iterable[set[str]],
        *,
        judge_used: str,
        score_type: str,
        threshold: float | None,
    ) -> None:
        super().__init__(clusters)
        self.judge_used = judge_used
        self.score_type = score_type
        self.threshold = threshold

    def __repr__(self) -> str:
        return (
            f"DedupeResult({list.__repr__(self)}, judge_used={self.judge_used!r}, "
            f"score_type={self.score_type!r}, threshold={self.threshold!r})"
        )


# ---------------------------------------------------------------------------
# Schema-optional inference (E8, D11)
# ---------------------------------------------------------------------------

_INFERRED_SCHEMA_CACHE: dict[frozenset[str], type[BaseModel]] = {}


def _inferred_schema_name(field_names: frozenset[str]) -> str:
    """Deterministic ``Inferred_<sha8>`` name from a field-set (memoization key)."""
    digest = sha256("|".join(sorted(field_names)).encode()).hexdigest()[:8]
    return f"Inferred_{digest}"


def _infer_schema(field_names: frozenset[str]) -> type[BaseModel]:
    """Build (or reuse) an ephemeral all-``str | None`` schema for ``field_names``.

    Memoized by field-set so repeated ``dedupe()``/``link()`` calls over the
    same record shape reuse one class instead of minting a new one each time.
    """
    cached = _INFERRED_SCHEMA_CACHE.get(field_names)
    if cached is not None:
        return cached
    fields: dict[str, Any] = {"id": (str, ...)}
    for name in sorted(field_names):
        fields[name] = (str | None, None)
    schema: type[BaseModel] = create_model(_inferred_schema_name(field_names), **fields)
    _INFERRED_SCHEMA_CACHE[field_names] = schema
    return schema


def _coerce_scalar(value: Any) -> str | None:
    """Coerce one raw field value for an inferred (all-``str | None``) schema.

    ``None`` and ``float('nan')`` both become ``None`` -- never the string
    ``"nan"``, which would silently poison string-similarity scoring (D11). A
    nested ``list``/``dict`` value cannot be represented by a flat inferred
    field, so it raises with guidance rather than being silently stringified
    (E8).
    """
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (list, dict)):
        raise ValueError(
            f"cannot infer a schema field from a nested {type(value).__name__} "
            f"value ({value!r}). Pass schema=<YourModel> explicitly to control "
            "how nested fields are compared."
        )
    return str(value)


def _field_union(records: Sequence[dict[str, Any]]) -> frozenset[str]:
    """Every key across ``records`` except ``"id"`` (handled separately)."""
    fields: set[str] = set()
    for record in records:
        fields.update(key for key in record if key != "id")
    return frozenset(fields)


def _resolve_ids(records: Sequence[dict[str, Any]]) -> list[str]:
    """Per-record id: explicit ``"id"`` key if EVERY record has one, else positional.

    Raises:
        ValueError: If some records have an ``"id"`` key and others don't
            (ambiguous -- which source should win?).
    """
    has_id = ["id" in record for record in records]
    if all(has_id):
        return [str(record["id"]) for record in records]
    if not any(has_id):
        return [str(i) for i in range(len(records))]
    raise ValueError(
        "some records have an 'id' key and some don't -- schema inference "
        "needs consistent id presence across all records. Add 'id' to every "
        "record (or none), or pass schema=<YourModel> explicitly."
    )


def _infer(records: Sequence[dict[str, Any]]) -> tuple[type[BaseModel], list[dict[str, Any]]]:
    """Infer an ephemeral schema + coerced records for ``records``.

    No id-uniqueness check here -- ``link(a, a)`` (comparing an entity to
    itself) is well-defined and must not raise; :func:`dedupe` enforces
    uniqueness itself via :func:`_check_no_duplicate_ids`.
    """
    field_names = _field_union(records)
    ids = _resolve_ids(records)
    schema = _infer_schema(field_names)
    coerced = [
        {"id": rid, **{name: _coerce_scalar(record.get(name)) for name in field_names}}
        for record, rid in zip(records, ids, strict=True)
    ]
    return schema, coerced


def _with_resolved_ids(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach a resolved ``"id"`` to each record, via the same rule ``_infer``
    uses (:func:`_resolve_ids`): every record already has one -> keep it
    (normalized to ``str``); none do -> assign positional ids; a mix ->
    raises. Used by ``dedupe()``'s explicit-``schema=`` path so it can't
    mistake "no id" for "duplicate id" (``str(record.get("id"))`` on two
    id-less records both reads as the string ``"None"`` -- a false collision).
    """
    ids = _resolve_ids(records)
    return [{**record, "id": rid} for record, rid in zip(records, ids, strict=True)]


def _check_no_duplicate_ids(ids: Sequence[str]) -> None:
    """Raise if ``ids`` contains a repeat (dedupe()'s batch-uniqueness contract)."""
    if len(set(ids)) == len(ids):
        return
    seen: set[str] = set()
    dupes: set[str] = set()
    for i in ids:
        if i in seen:
            dupes.add(i)
        seen.add(i)
    raise ValueError(
        f"duplicate ids in input: {sorted(dupes)}; every record must have a unique id."
    )


def _resolved_threshold(judge_used: str, threshold: float | None) -> float:
    return _THRESHOLDS.get(judge_used, 0.5) if threshold is None else threshold


def _coerce_log(log: JudgementLog | str | Path | None) -> JudgementLog | None:
    """Normalize ``log=`` to a :class:`JudgementLog` (W0.2): ``None`` stays
    ``None`` (zero overhead -- no wrap); a path is wrapped in a fresh,
    default (``features=False``) :class:`JudgementLog`; an existing
    :class:`JudgementLog` (e.g. constructed with ``features=True``) passes
    through unchanged."""
    if log is None or isinstance(log, JudgementLog):
        return log
    return JudgementLog(log)


# ---------------------------------------------------------------------------
# link()
# ---------------------------------------------------------------------------


def link(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    judge: JudgeName | Module[Any] = "auto",
    schema: type[BaseModel] | None = None,
    model: str | None = None,
    entity_noun: str = "entity",
    threshold: float | None = None,
    budget_usd: float | None = None,
    log: JudgementLog | str | Path | None = None,
) -> LinkVerdict:
    """Decide whether two records refer to the same real-world entity.

    Args:
        left: The first record (a plain dict).
        right: The second record. ``link(a, a)`` (the same record twice) is
            well-defined -- it scores the entity against itself.
        judge: ``"auto"`` (default; picks an LLM judge from the available API
            key and raises ``NoJudgeAvailableError`` when none is set),
            ``"zero_shot_llm"``, ``"embedding"``, ``"string"`` (the explicit
            offline opt-in), or a ``Module`` instance (e.g. an injected
            ``DSPyJudge(lm=DummyLM(...))`` for a zero-spend test).
        schema: Optional explicit Pydantic schema. Omit to infer an ephemeral
            one from ``left``/``right``'s own keys.
        model: Model id override for ``"zero_shot_llm"``.
        entity_noun: Domain noun woven into the LLM judge's prompt.
        threshold: Match cutoff; ``None`` resolves to the judge's default --
            for ``judge="string"`` that is ``0.5`` on its ``"heuristic"``
            score (see ``presets._DEFAULT_THRESHOLDS``). The effective value
            is reported back on :attr:`LinkVerdict.threshold`.
        budget_usd: Spend cap override (default $1; see
            :mod:`langres.core.presets`).
        log: Opt-in signal-log sink -- a :class:`~langres.core.judgement_log.JudgementLog`
            or a path (wrapped in a default one). ``None`` (default): no
            logging, zero overhead. See :mod:`langres.core.judgement_log`.

    Returns:
        A :class:`LinkVerdict`.

    Raises:
        ValueError: On schema-inference errors (nested values, inconsistent
            id presence).
        NoJudgeAvailableError: With ``judge="auto"`` and no API key set,
            ``LANGRES_OFFLINE=1``, or an unpinned-price model -- never a
            silent fallback.
        BudgetExceeded: If scoring this pair would cross the spend cap.
    """
    if schema is None:
        resolved_schema, (left_record, right_record) = _infer([left, right])
    else:
        resolved_schema, left_record, right_record = schema, left, right

    module, judge_used, resolved_model = resolve_judge(
        judge, resolved_schema, model=model, entity_noun=entity_noun, budget_usd=budget_usd
    )

    if judge_used == "zero_shot_llm" and resolved_model is not None:
        notice_pre_scoring_cost(resolved_model, 1, budget_usd=_effective_budget(budget_usd))

    candidate: ERCandidate[Any]
    if judge_used == "embedding":
        candidate = build_embedding_candidate(resolved_schema, left_record, right_record)
    else:
        factory = schema_to_factory(resolved_schema)
        left_entity = factory(left_record)
        right_entity = factory(right_record)
        candidate = ERCandidate(left=left_entity, right=right_entity, blocker_name="link")
        if judge_used == "string":
            comparator: Comparator[Any] = Comparator.from_schema(resolved_schema)
            candidate = candidate.model_copy(
                update={"comparison": comparator.compare(left_entity, right_entity)}
            )

    resolved_threshold = _resolved_threshold(judge_used, threshold)
    log_sink = _coerce_log(log)
    scorer_module: Module[Any] = module
    if log_sink is not None:
        scorer_module = LoggingModule(module, log=log_sink, threshold=resolved_threshold)

    judgements = list(scorer_module.forward(iter([candidate])))
    if not judgements:
        raise RuntimeError(
            f"the {judge_used!r} judge produced no judgement for this pair; every "
            "candidate must yield exactly one PairwiseJudgement. This indicates a "
            "bug in an injected judge= Module."
        )
    judgement = judgements[0]

    predicted = predicted_match(judgement, resolved_threshold)
    if predicted is None:
        # A judge that neither scored nor decided abstained; link() owes the
        # caller a match/no-match verdict and cannot honestly fabricate one.
        raise JudgeAbstainedError(
            f"the {judge_used!r} judge abstained (no decision and no score) on "
            "this pair, so link() cannot return a match verdict. An LLMJudge "
            "abstains when its response fails to parse (the default "
            "on_parse_error='abstain'); pass on_parse_error='raise' to surface "
            "the parse failure itself, or catch JudgeAbstainedError."
        )

    return LinkVerdict(
        match=predicted,
        score=judgement.score,
        reasoning=judgement.reasoning,
        judge_used=judge_used,
        score_type=judgement.score_type,
        threshold=resolved_threshold,
        judgement=judgement,
    )


# ---------------------------------------------------------------------------
# dedupe()
# ---------------------------------------------------------------------------


def dedupe(
    records: list[dict[str, Any]],
    *,
    judge: JudgeName | Module[Any] = "auto",
    schema: type[BaseModel] | None = None,
    model: str | None = None,
    entity_noun: str = "entity",
    threshold: float | None = None,
    budget_usd: float | None = None,
    log: JudgementLog | str | Path | None = None,
) -> DedupeResult:
    """Group a batch of records into entity clusters.

    Abstentions are handled differently here than in :func:`link`. ``link``
    judges one pair and *raises* :class:`~langres.core.models.JudgeAbstainedError`
    on an abstain, because a single caller needs a verdict. ``dedupe`` judges
    many pairs to build clusters, and an abstained pair is left **unmerged** (the
    conservative default -- the same as "not a match" for edge-building) rather
    than aborting the whole batch: one unparseable judgement among thousands
    should not sink an entire dedupe run. Inspect the ``log`` if you need to see
    which pairs the judge declined.

    Args:
        records: The records to dedupe (plain dicts). ``[]`` -> ``[]``; a
            single record -> ``[]`` (no pair possible) -- both short-circuit
            BEFORE judge resolution, so neither can raise
            ``NoJudgeAvailableError``. Every record must have a unique
            ``"id"`` (or none at all -- positional ids are assigned); a
            duplicate ``"id"`` raises.
        judge: ``"auto"`` (default; picks an LLM judge from the available API
            key and raises ``NoJudgeAvailableError`` when none is set),
            ``"zero_shot_llm"``, ``"embedding"``, ``"string"`` (the explicit
            offline opt-in), or a ``Module`` instance.
        schema: Optional explicit Pydantic schema. Omit to infer an ephemeral
            one from the records' own keys.
        model: Model id override for ``"zero_shot_llm"``.
        entity_noun: Domain noun woven into the LLM judge's prompt.
        threshold: Clusterer threshold; ``None`` resolves to the judge's
            default -- for ``judge="string"`` that is ``0.5`` on its
            ``"heuristic"`` score (see ``presets._DEFAULT_THRESHOLDS``). The
            effective value is reported back on the result's ``threshold``,
            ready for ``select_for_review(threshold=...)``.
        budget_usd: Spend cap override (default $1).
        log: Opt-in signal-log sink -- a :class:`~langres.core.judgement_log.JudgementLog`
            or a path (wrapped in a default one). ``None`` (default): no
            logging, zero overhead. See :mod:`langres.core.judgement_log`.

    Returns:
        A :class:`DedupeResult` (a ``list[set[str]]`` of entity-id clusters).

    Raises:
        ValueError: Duplicate ids, inconsistent id presence, or a nested
            value under schema inference.
        NoJudgeAvailableError: With ``judge="auto"`` and no API key set,
            ``LANGRES_OFFLINE=1``, or an unpinned-price model -- never a
            silent fallback.
        BudgetExceeded: If scoring would cross the spend cap; the exception
            carries the judgements already produced on ``.partial_judgements``.
    """
    if len(records) < 2:
        # [] -> [] and [x] -> [] (no pair possible): short-circuit BEFORE
        # judge resolution so a keyless empty/single-record call never raises
        # NoJudgeAvailableError -- zero spend is possible either way. No judge
        # was resolved, so there is no effective threshold to report.
        return DedupeResult([], judge_used="none", score_type="none", threshold=None)

    if schema is None:
        resolved_schema, resolved_records = _infer(records)
    else:
        resolved_schema, resolved_records = schema, _with_resolved_ids(records)

    _check_no_duplicate_ids([record["id"] for record in resolved_records])

    resolved = build_resolver(
        resolved_schema,
        judge=judge,
        model=model,
        entity_noun=entity_noun,
        threshold=threshold,
        n_records=len(resolved_records),
        budget_usd=budget_usd,
    )
    log_sink = _coerce_log(log)
    if log_sink is not None:
        resolved.resolver.module = LoggingModule(
            resolved.resolver.module,
            log=log_sink,
            threshold=resolved.resolver.clusterer.threshold,
        )
    judgements = resolved.resolver.predict(resolved_records)
    clusters = resolved.resolver.clusterer.cluster(judgements)
    score_type = judgements[0].score_type if judgements else resolved.score_type
    return DedupeResult(
        clusters,
        judge_used=resolved.judge_used,
        score_type=score_type,
        # The clusterer's threshold IS the effective cut (threshold=None was
        # resolved to the judge's default inside build_resolver) -- the same
        # value the LoggingModule above stamps on every logged verdict.
        threshold=resolved.resolver.clusterer.threshold,
    )
