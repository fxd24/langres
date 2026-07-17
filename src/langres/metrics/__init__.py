"""``langres.metrics``: the evaluation and diagnostics metrics for entity resolution.

A leaf package that sits *beside* ``langres.core`` (not in it): metrics *score*
a resolution, they are not part of the modelling contract a pipeline is written
against. The dependency is one-way — ``metrics`` imports the ``core`` data models
(``ERCandidate``/``PairwiseJudgement``) and the ``core.reports`` value types, and
``core`` never imports ``metrics`` at module top (the few consumers reach it
function-locally; see ``tests/test_import_tangle.py``).

Four modules, each imported directly from where it lives (this ``__init__`` is a
docstring-only aggregator, import-light on purpose — ``numpy`` + the core deps
only, ``ranx`` stays lazy inside ``evaluate_blocking_with_ranking``)::

    import langres.metrics.metrics       # BCubed / pairwise / ranking ER metrics
    from langres.metrics.debugging import PipelineDebugger
    from langres.metrics.analysis import evaluate_blocker_detailed
    from langres.metrics.diagnostics import DiagnosticExamples

The curated public entry points stay where users already reach them: the ER
metric functions are re-exported (lazily) by the ``langres.eval`` facade. The old
``langres.core.metrics`` / ``.debugging`` / ``.analysis`` / ``.diagnostics``
import paths keep working via back-compat shims until the refactor's final sweep.
"""
