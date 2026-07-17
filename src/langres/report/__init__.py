"""``langres.report``: the shared ``$0`` rendering seam.

The HTML/SVG primitives every langres tearsheet renders through, and the
``EvalReport`` tearsheet itself. This is presentation, not entity-resolution
modelling -- it lives beside :mod:`langres.core` rather than inside it.

**Why it is not in ``core``.** These modules were in ``langres.core`` by accident
of history, not by design: they landed there because ``EvalReport`` was written
first and ``_svg`` was its backend. The import graph shows what they actually
are -- ``_report_html`` had **zero** importers in ``core`` (all eight were
:mod:`langres.data.data_profile`), and ``_svg``'s only ``core`` importer was
``eval_report`` itself. They are shared rendering primitives that ``langres.data``
uses more than ``core`` ever did. ``core`` is the contracts floor; a chart
renderer is not a contract.

**Layering.** The dependency runs one way: ``report -> core`` (``eval_report``
reads ``core.benchmark``/``core.metrics``/``core.models``; ``_report_html`` reads
``core.metrics``). Nothing in ``core`` imports ``report``, so the seam cannot
knot -- see ``tests/test_import_tangle.py``, the ratchet that measures it.

**This ``__init__`` is deliberately empty of imports.** Consumers reach a module
directly (``from langres.report import _svg``), and an empty package module keeps
that free: re-exporting ``EvalReport`` here would make every
``data_profile`` import of ``_svg`` transitively execute ``eval_report`` ->
``core.benchmark`` -> ``core.metrics``, taxing the ``$0`` profile path with
modules it never calls. ``EvalReport``'s public home is :mod:`langres.eval` (and
the root ``langres`` namespace), both of which resolve it lazily.

Import weight: stdlib + numpy (via ``core.metrics``) only. The tearsheets are
dependency-free by construction -- inline SVG, no matplotlib, no CDN, no ML
stack -- so a ``$0`` report is always buildable on a bare core-only install.
``tests/test_import_budget.py`` locks that guarantee.
"""
