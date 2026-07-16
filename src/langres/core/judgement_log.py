"""Opt-in JSONL judgement log: the flywheel inlet (harvested by W2.4).

``JudgementLog`` is a small file-backed sink -- it is NOT a ``Matcher``.
Wire it via ``log=`` on :func:`langres.link`/:func:`langres.dedupe`; those
verbs wrap the resolved judge in :class:`LoggingMatcher` below, which appends
one JSON line per :class:`~langres.core.models.PairwiseJudgement` **as it
streams past**. It never buffers or materializes the full judgement stream,
so laziness and memory behavior are unaffected (Eng finding E10: an explicit
boundary component wrapping a ``Matcher``, composed the same way
:class:`~langres.core.presets._SpendCappedMatcher` wraps one -- never a
monkey-patch of ``Matcher.forward``).

Zero overhead when omitted: ``log=None`` (the default on both verbs) skips
the wrap entirely -- no file, no extra generator layer, byte-identical to
pre-W0.2 behavior.

Privacy (adopted DX): record content is OFF by default. Each line carries
only ids, score, the judge's own ``decision`` plus the caller's ``verdict``, the
judge's ``confidence``/``confidence_source``, model, cost, the typed ``usage``
token vector (``LLMUsage.model_dump()``, or ``null`` for non-LLM judges),
decision_step, timestamp, the enclosing run's ``run_id`` (the active
``capture_run`` attempt id, or ``null`` outside one -- the join key to the
``RunRecord``/trace, W1 S5), and the schema-version field ``"v": 3`` -- never
the underlying record fields or the judge's free-text reasoning. The ``usage``
vector is non-PII (token
counts only), so it belongs in the default row alongside ``cost_usd``/``model``
-- capturing token spend is the whole point of the log. Pass ``features=True``
to additionally log ``reasoning`` and the judge's raw ``provenance`` dict
(comparison levels, similarities, ...): **this may contain PII** -- the record
content a judge reasoned over, verbatim -- and JSONL is plaintext on disk.

Serialization: excluded from Resolver artifacts (decided for W0.2, per E10).
``LoggingMatcher`` has no registry ``type_name`` and is applied by the verbs
layer to an already-built ``Resolver.module`` *after* construction -- it is
never part of a ``resolver.save()``'d pipeline (``link``/``dedupe`` never
persist their internal resolver). A future milestone can register it if a
durable, logging-enabled artifact is ever needed.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langres.clients.openrouter import BudgetExceeded
from langres.core.inspection import _ensure_inspectable
from langres.core.models import ERCandidate, PairwiseJudgement, predicted_match
from langres.core.matcher import Matcher
from langres.core.reports import ScoreInspectionReport
from langres.core.runs import current_run

__all__ = ["JudgementLog", "LoggingMatcher"]

#: Schema-version tag written into every line (CEO #15) -- lets a future
#: harvester (W2.4) or format-migration branch on it instead of guessing.
#: Bumped 1 -> 2 when the default row gained the ``usage`` token-usage vector,
#: 2 -> 3 when it gained the decision-contract columns (``decision`` /
#: ``confidence`` / ``confidence_source``).
_SCHEMA_VERSION = 3

#: The schema version at which :meth:`JudgementLog.append` began writing the
#: decision-contract columns natively. :meth:`JudgementLog.read` trusts any row
#: at or above this version to carry them (a missing column there is a genuine
#: corruption, surfaced on access -- never silently defaulted) and backfills them
#: onto older rows from ``verdict``. Anchored at 3 rather than ``_SCHEMA_VERSION``
#: so a later additive bump cannot retro-backfill and clobber a real logged
#: ``decision``.
_DECISION_CONTRACT_VERSION = 3

#: Provenance keys carrying a judgement's USD cost, in priority order (first
#: present wins). Twin of ``benchmark._COST_KEYS`` -- a deliberate local copy, not
#: an import, to avoid a ``judgement_log`` <- ``benchmark`` cycle (``benchmark``
#: imports log-adjacent things). ``CascadeChainMatcher`` writes cost under
#: ``llm_cost_usd``, so a plain ``.get("cost_usd")`` would persist 0.0 for every
#: cascade row.
_COST_KEYS: tuple[str, ...] = ("cost_usd", "llm_cost_usd")


def _judgement_cost(judgement: PairwiseJudgement) -> float:
    """Measured USD cost of one judgement from its provenance.

    Reads :data:`_COST_KEYS` in order (``"cost_usd"`` first, then
    ``"llm_cost_usd"`` -- the key ``CascadeChainMatcher`` writes). Zero-spend judges
    set neither, so this returns ``0.0`` for them. Twin of
    ``benchmark._judgement_cost``.
    """
    prov = judgement.provenance
    for key in _COST_KEYS:
        if key in prov:
            return float(prov[key])
    return 0.0


def _backfill_decision_contract(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize a read-back row to the v3 decision-contract shape (in place).

    A row at :data:`_DECISION_CONTRACT_VERSION` or newer carries ``decision`` /
    ``confidence`` / ``confidence_source`` natively (written by
    :meth:`JudgementLog.append`) and is returned untouched -- a row that new
    *missing* one of those columns is a genuine corruption, surfaced on access
    rather than masked by a silent default. Older (v1/v2, or unversioned) rows
    predate the contract: ``decision`` is backfilled from the logged ``verdict``
    (``bool(verdict)`` for a real bool, else ``None`` -- an honest abstain, never
    a coerced ``False``), and ``confidence`` / ``confidence_source`` default to
    ``None`` / ``"none"``.
    """
    v = row.get("v")
    if isinstance(v, int) and v >= _DECISION_CONTRACT_VERSION:
        return row
    verdict = row.get("verdict")
    row["decision"] = bool(verdict) if isinstance(verdict, bool) else None
    row.setdefault("confidence", None)
    row.setdefault("confidence_source", "none")
    return row


class JudgementLog:
    """JSONL-file-backed sink for judge-call signals.

    Args:
        path: Where to append JSON lines. Parent directories are created on
            first :meth:`append` if missing.
        features: When ``True``, additionally record ``reasoning`` and the
            judge's raw ``provenance`` dict on every line -- may contain PII
            (see module docstring). Default ``False``.
    """

    def __init__(self, path: str | Path, *, features: bool = False) -> None:
        self.path = Path(path)
        self.features = features

    def append(
        self, judgement: PairwiseJudgement, *, verdict: bool | None, model: str | None = None
    ) -> None:
        """Append one JSON line for ``judgement`` (called by :class:`LoggingMatcher`).

        ``verdict`` is the caller's predicted-match decision (see
        :func:`~langres.core.models.predicted_match`); ``None`` records an
        abstention honestly rather than coercing it to a match/no-match.
        ``model`` is the caller's resolved pipeline model id, used as the
        row's ``model`` when the judgement's own ``provenance["model"]`` is
        absent -- the judge's per-call stamp wins (e.g. a cascade's per-step
        models), so the logged identity is always what actually ran, and a
        non-LLM pipeline (the embedding judge) still logs the model the verbs
        report on the result.
        """
        row: dict[str, Any] = {
            "v": _SCHEMA_VERSION,
            # The enclosing tracking run (S5): joins this row to its RunRecord
            # and any LLM trace on the attempt id; ``None`` outside a capture_run.
            "run_id": current_run.get(),
            "left_id": judgement.left_id,
            "right_id": judgement.right_id,
            "score": judgement.score,
            # The decision-contract columns (v3): the judge's own ``decision``
            # (``None`` when it only ranked or abstained) plus the ``confidence``
            # it earned and where that came from -- logged alongside, and never
            # derived from, the caller's ``verdict`` (its ``predicted_match``).
            "decision": judgement.decision,
            "verdict": verdict,
            "confidence": judgement.confidence,
            "confidence_source": judgement.confidence_source,
            "model": judgement.provenance.get("model") or model,
            # First of _COST_KEYS present wins: CascadeChainMatcher logs ``llm_cost_usd``,
            # so a bare ``.get("cost_usd")`` would persist 0.0 for every cascade row.
            "cost_usd": _judgement_cost(judgement),
            # The typed token-usage vector (LLMUsage.model_dump()) when the judge
            # is an LLM; ``None`` for non-LLM judges. Non-PII counts, so it stays
            # in the DEFAULT (features=False) row alongside cost_usd/model.
            "usage": judgement.provenance.get("usage"),
            "decision_step": judgement.decision_step,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        if self.features:
            row["reasoning"] = judgement.reasoning
            row["provenance"] = judgement.provenance
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")

    def read(self) -> list[dict[str, Any]]:
        """Reload every line written so far -- the round-trip reader.

        Each line is independently valid JSON (one row per :meth:`append`
        call); this parses them back into a list of plain dicts in write order,
        version-dispatching each through :func:`_backfill_decision_contract` so
        pre-v3 rows read back with ``decision`` / ``confidence`` /
        ``confidence_source`` present (backfilled from ``verdict``). Returns
        ``[]`` if the file was never created (e.g. a ``dedupe()``/``link()`` call
        that scored zero pairs).
        """
        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if stripped:
                    rows.append(_backfill_decision_contract(json.loads(stripped)))
        return rows


class LoggingMatcher(Matcher[Any]):
    """Boundary component: wraps a scorer ``Matcher``, logging each judgement as it streams past.

    Deliberately NOT a monkey-patch of ``module.forward`` (E10) -- a small
    wrapper ``Matcher`` in the same family as
    :class:`~langres.core.presets._SpendCappedMatcher`, composing
    transparently with any ``Matcher`` -- including a future
    ``GroupwiseMatcher`` (W1.0): both yield ``PairwiseJudgement`` one at a
    time, so wrapping and logging is identical either way.

    ``verdict`` is computed per judgement from ``threshold`` -- the same
    match cutoff the calling verb (``link``/``dedupe``) already resolved for
    its own ``score >= threshold`` decision, so the logged verdict always
    agrees with what the caller acted on.

    Wrapping a spend-capped module (e.g.
    :class:`~langres.core.presets._SpendCappedMatcher`, as ``link``/``dedupe``
    do): the judgement that trips the cap is recorded on the raised
    ``BudgetExceeded.partial_judgements`` but never yielded (the cap raises
    *before* yielding it -- E9's "set by the catcher, not at raise time"
    pattern). A ``LoggingMatcher`` sitting outside that cap would otherwise
    silently drop exactly the paid call the flywheel most needs. ``forward``
    catches ``BudgetExceeded`` and logs any trailing ``partial_judgements``
    entries not already logged (tracked by count, so nothing is logged
    twice) before re-raising the exception unmodified.
    """

    def __init__(
        self,
        module: Matcher[Any],
        *,
        log: JudgementLog,
        threshold: float,
        model: str | None = None,
    ) -> None:
        """Wrap ``module``, logging every judgement to ``log``.

        ``threshold`` and ``model`` are the calling verb's *resolved* values --
        the same ones it reports on the result -- so log rows and results
        cannot drift apart. ``model`` only backfills rows whose judgement
        carries no ``provenance["model"]`` of its own (see
        :meth:`JudgementLog.append`).
        """
        self._module = module
        self._log = log
        self._threshold = threshold
        self._model = model

    def forward(self, candidates: Iterator[ERCandidate[Any]]) -> Iterator[PairwiseJudgement]:
        logged = 0
        try:
            for judgement in self._module.forward(candidates):
                self._log.append(
                    judgement,
                    verdict=predicted_match(judgement, self._threshold),
                    model=self._model,
                )
                logged += 1
                yield judgement
        except BudgetExceeded as exc:
            for judgement in exc.partial_judgements[logged:]:
                self._log.append(
                    judgement,
                    verdict=predicted_match(judgement, self._threshold),
                    model=self._model,
                )
            raise

    def inspect_scores(
        self, judgements: list[PairwiseJudgement], sample_size: int = 10
    ) -> ScoreInspectionReport:
        """Delegate to the wrapped matcher, which must opt into ``Inspectable``."""
        return _ensure_inspectable(self._module).inspect_scores(judgements, sample_size)
