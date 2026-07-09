"""Opt-in JSONL judgement log: the flywheel inlet (harvested by W2.4).

``JudgementLog`` is a small file-backed sink -- it is NOT a ``Module``.
Wire it via ``log=`` on :func:`langres.link`/:func:`langres.dedupe`; those
verbs wrap the resolved judge in :class:`LoggingModule` below, which appends
one JSON line per :class:`~langres.core.models.PairwiseJudgement` **as it
streams past**. It never buffers or materializes the full judgement stream,
so laziness and memory behavior are unaffected (Eng finding E10: an explicit
boundary component wrapping a ``Module``, composed the same way
:class:`~langres.core.presets._SpendCappedModule` wraps one -- never a
monkey-patch of ``Module.forward``).

Zero overhead when omitted: ``log=None`` (the default on both verbs) skips
the wrap entirely -- no file, no extra generator layer, byte-identical to
pre-W0.2 behavior.

Privacy (adopted DX): record content is OFF by default. Each line carries
only ids, score, verdict, model, cost, the typed ``usage`` token vector
(``LLMUsage.model_dump()``, or ``null`` for non-LLM judges), decision_step,
timestamp, the enclosing run's ``run_id`` (the active ``capture_run`` attempt
id, or ``null`` outside one -- the join key to the ``RunRecord``/trace, W1 S5),
and the schema-version field ``"v": 2`` -- never the underlying record fields
or the judge's free-text reasoning. The ``usage`` vector is non-PII (token
counts only), so it belongs in the default row alongside ``cost_usd``/``model``
-- capturing token spend is the whole point of the log. Pass ``features=True``
to additionally log ``reasoning`` and the judge's raw ``provenance`` dict
(comparison levels, similarities, ...): **this may contain PII** -- the record
content a judge reasoned over, verbatim -- and JSONL is plaintext on disk.

Serialization: excluded from Resolver artifacts (decided for W0.2, per E10).
``LoggingModule`` has no registry ``type_name`` and is applied by the verbs
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
from langres.core.models import ERCandidate, PairwiseJudgement
from langres.core.module import Module
from langres.core.reports import ScoreInspectionReport
from langres.core.runs import current_run

__all__ = ["JudgementLog", "LoggingModule"]

#: Schema-version tag written into every line (CEO #15) -- lets a future
#: harvester (W2.4) or format-migration branch on it instead of guessing.
#: Bumped 1 -> 2 when the default row gained the ``usage`` token-usage vector.
_SCHEMA_VERSION = 2


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

    def append(self, judgement: PairwiseJudgement, *, verdict: bool) -> None:
        """Append one JSON line for ``judgement`` (called by :class:`LoggingModule`)."""
        row: dict[str, Any] = {
            "v": _SCHEMA_VERSION,
            # The enclosing tracking run (S5): joins this row to its RunRecord
            # and any LLM trace on the attempt id; ``None`` outside a capture_run.
            "run_id": current_run.get(),
            "left_id": judgement.left_id,
            "right_id": judgement.right_id,
            "score": judgement.score,
            "verdict": verdict,
            "model": judgement.provenance.get("model"),
            "cost_usd": judgement.provenance.get("cost_usd", 0.0),
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
        call); this just parses them back into a list of plain dicts in
        write order. Returns ``[]`` if the file was never created (e.g. a
        ``dedupe()``/``link()`` call that scored zero pairs).
        """
        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if stripped:
                    rows.append(json.loads(stripped))
        return rows


class LoggingModule(Module[Any]):
    """Boundary component: wraps a scorer ``Module``, logging each judgement as it streams past.

    Deliberately NOT a monkey-patch of ``module.forward`` (E10) -- a small
    wrapper ``Module`` in the same family as
    :class:`~langres.core.presets._SpendCappedModule`, composing
    transparently with any ``Module`` -- including a future
    ``GroupwiseModule`` (W1.0): both yield ``PairwiseJudgement`` one at a
    time, so wrapping and logging is identical either way.

    ``verdict`` is computed per judgement from ``threshold`` -- the same
    match cutoff the calling verb (``link``/``dedupe``) already resolved for
    its own ``score >= threshold`` decision, so the logged verdict always
    agrees with what the caller acted on.

    Wrapping a spend-capped module (e.g.
    :class:`~langres.core.presets._SpendCappedModule`, as ``link``/``dedupe``
    do): the judgement that trips the cap is recorded on the raised
    ``BudgetExceeded.partial_judgements`` but never yielded (the cap raises
    *before* yielding it -- E9's "set by the catcher, not at raise time"
    pattern). A ``LoggingModule`` sitting outside that cap would otherwise
    silently drop exactly the paid call the flywheel most needs. ``forward``
    catches ``BudgetExceeded`` and logs any trailing ``partial_judgements``
    entries not already logged (tracked by count, so nothing is logged
    twice) before re-raising the exception unmodified.
    """

    def __init__(self, module: Module[Any], *, log: JudgementLog, threshold: float) -> None:
        self._module = module
        self._log = log
        self._threshold = threshold

    def forward(self, candidates: Iterator[ERCandidate[Any]]) -> Iterator[PairwiseJudgement]:
        logged = 0
        try:
            for judgement in self._module.forward(candidates):
                self._log.append(judgement, verdict=judgement.score >= self._threshold)
                logged += 1
                yield judgement
        except BudgetExceeded as exc:
            for judgement in exc.partial_judgements[logged:]:
                self._log.append(judgement, verdict=judgement.score >= self._threshold)
            raise

    def inspect_scores(
        self, judgements: list[PairwiseJudgement], sample_size: int = 10
    ) -> ScoreInspectionReport:
        return self._module.inspect_scores(judgements, sample_size)
