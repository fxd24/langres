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
only ids, score, verdict, model, cost, decision_step, timestamp, and the
schema-version field ``"v": 1`` -- never the underlying record fields or the
judge's free-text reasoning. Pass ``features=True`` to additionally log
``reasoning`` and the judge's raw ``provenance`` dict (comparison levels,
similarities, token counts, ...): **this may contain PII** -- the record
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

from langres.core.models import ERCandidate, PairwiseJudgement
from langres.core.module import Module
from langres.core.reports import ScoreInspectionReport

__all__ = ["JudgementLog", "LoggingModule"]

#: Schema-version tag written into every line (CEO #15) -- lets a future
#: harvester (W2.4) or format-migration branch on it instead of guessing.
_SCHEMA_VERSION = 1


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
            "left_id": judgement.left_id,
            "right_id": judgement.right_id,
            "score": judgement.score,
            "verdict": verdict,
            "model": judgement.provenance.get("model"),
            "cost_usd": judgement.provenance.get("cost_usd", 0.0),
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
    """

    def __init__(self, module: Module[Any], *, log: JudgementLog, threshold: float) -> None:
        self._module = module
        self._log = log
        self._threshold = threshold

    def forward(self, candidates: Iterator[ERCandidate[Any]]) -> Iterator[PairwiseJudgement]:
        for judgement in self._module.forward(candidates):
            self._log.append(judgement, verdict=judgement.score >= self._threshold)
            yield judgement

    def inspect_scores(
        self, judgements: list[PairwiseJudgement], sample_size: int = 10
    ) -> ScoreInspectionReport:
        return self._module.inspect_scores(judgements, sample_size)
