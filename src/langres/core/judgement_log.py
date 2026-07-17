"""Back-compat shim: moved to ``langres.tracking.judgement_log``.

# TEMPORARY: deleted by the W2 sweep

Judgement logging is observability, not ER modelling, so it now lives in
``langres.tracking`` beside ``core`` rather than inside it. Import from
``langres.tracking.judgement_log`` (or the unchanged ``langres.core`` facade,
which still re-exports these names).
"""

from langres.tracking.judgement_log import JudgementLog, LoggingMatcher

__all__ = ["JudgementLog", "LoggingMatcher"]
