"""Autoresearch loop for entity resolution: propose → run → evaluate → keep-if-better.

This package is the blocking-search **engine**, sitting outside ``langres.core``
(which is ER *modelling*, not search) and depending on it one-way — nothing in
``core`` imports back into here. Its public entry point is the ``langres.optimize``
facade, which composes these parts into a one-call search:

- :mod:`~langres.autoresearch.objective` — P-A, the immutable keep-if-better scorer.
- :mod:`~langres.autoresearch.search_space` — P-B, the declarative config grid.
- :mod:`~langres.autoresearch.factory` — P-B, config → runnable blocker. **Heavy**
  (faiss/sentence-transformers at module top); import it lazily only.
- :mod:`~langres.autoresearch.loop` — P-C, the ``propose → run → evaluate → keep``
  driver over ``core.runs`` persistence.
- :mod:`~langres.autoresearch.blocker_optimizer` — the separate Optuna study
  (``BlockerOptimizer``); optuna is a dev-only dep, so it too is lazy-only.

See epic #145. Submodules are imported by dotted path — this package intentionally
exports nothing, which is also what keeps it import-light: ``factory`` and
``blocker_optimizer`` carry heavy/dev-only deps at module top, so an eager
re-export here would drag faiss / sentence-transformers / torch / optuna into
every bare ``import langres`` (the facade is root-exported). Keep this ``__init__``
empty of imports.

**Why the engine does not live under ``langres.optimize``.** ``langres.optimize``
is a *callable* — ``langres/_exports/_optimize.py`` binds the attribute to the
``optimize`` **function**, which is the public API. Making ``optimize`` a package
therefore put these submodules under a name attribute traversal can never reach.
Against the old ``langres.optimize.loop`` path, ``from ... import`` resolved, but
``import langres.optimize.loop as l`` raised ``ImportError`` and
``langres.optimize.loop.run_loop`` raised ``AttributeError`` ('function' object
has no attribute 'loop'). The engine keeps its own un-shadowed name here; the
facade stays a module at ``langres/optimize.py``.
"""
