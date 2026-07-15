"""Composable data-profile report: the ``ProfileSection`` bag + its container.

Wave 0 ships the seam only -- the frozen :class:`ProfileSection` base and the
:class:`DataProfileReport` container (:mod:`langres.core.data_profile.base`),
plus the render scaffold (:mod:`langres.core._report_html`) and the streaming
accumulators (:mod:`langres.core.data_profile.accumulators`).

Later waves add, without touching the frozen base:
- Wave 1/2 -- concrete sections (``LabelStructureSection``, ``CorpusFieldSection``,
  ``SeparabilitySection``, ``EmbeddingSection``, ...), the ``EmbeddingSource``
  protocol, and the profiler functions that build them.
- Wave 2 -- ``builders`` (the module ``DataProfileReport.from_benchmark`` /
  ``from_records`` delegate to) so the convenience constructors light up.

A leaf module: import-light (numpy + stdlib + ``core.metrics``), guarded by
``tests/test_import_budget.py``.
"""

from langres.core.data_profile.base import DataProfileReport, ProfileSection

__all__ = ["DataProfileReport", "ProfileSection"]
