"""The human-in-the-loop labelling surface: pick the margin, harvest, hold, canonicalise.

Curation is where a human's judgement enters the loop, and where the labels it
produces are kept:

- **Pick the uncertain margin** -- :mod:`langres.curation.review`
  (``select_for_review`` / ``ReviewQueue``).
- **Harvest the corrections** -- :mod:`langres.curation.harvest`
  (``Correction`` / ``CorrectionLog``, ``harvest_labeled_pairs``,
  ``derive_threshold_from_pairs``).
- **Hold the anchors** -- :mod:`langres.curation.anchor_store`.
- **Canonicalise** -- :mod:`langres.curation.canonicalizer`.
- **Cold-start a gold set** (the former ``langres.bootstrap``, dissolved into
  this package): the data contract (:class:`GoldPair`, :class:`GoldSet`), the
  :class:`Miner` / :class:`Labeler` interfaces, :class:`HardNegativeMiner`
  (stratified sampling), the labelers (:class:`GroundTruthLabeler` and
  :class:`FakeLabeler`, both zero-spend; :class:`TeacherLabeler`, the
  budget-capped LLM teacher), the :class:`Bootstrapper` orchestrator (block ->
  filter -> mine -> label) and :class:`BootstrapReport` (coverage + calibration
  health check).

**Five names here are lazy, and that is load-bearing.** Unlike the old
``langres.bootstrap`` -- which nothing eager ever imported, so its facade could
pull in anything it liked -- this package *is* on the eager path:
``langres.core``'s ``_exports/_flywheel`` fragment imports ``curation.harvest``,
and Python runs a parent package's ``__init__`` before any submodule of it. An
eager import here therefore lands in every bare ``import langres``.
:class:`Bootstrapper` reaches faiss (via ``core.blockers.vector`` ->
``core.indexes.vector_index``) and the labelers reach litellm (via
``core.matchers.llm_judge``), so both resolve via PEP 562 instead -- the same
pattern as ``langres.core.matchers``. litellm in particular calls
``load_dotenv()`` as an import side effect, which makes this a *spend* hazard and
not merely a slow import; ``tests/test_import_budget.py`` is the gate.

The eager names (:class:`GoldPair`, :class:`Miner`, :class:`HardNegativeMiner`,
:class:`BootstrapReport`, ...) reach only stdlib/pydantic + ``core.models`` /
``core.metrics``. None of the bootstrap components are ``@register``-ed -- they
carry no serializable config -- so nothing here needs an eager import to keep
``Resolver.load`` working in a fresh process.
"""

import importlib
from typing import TYPE_CHECKING, Any

from langres.curation.base import Labeler, Miner
from langres.curation.miners import HardNegativeMiner
from langres.curation.models import GoldPair, GoldPairSource, GoldSet
from langres.curation.report import BootstrapReport

if TYPE_CHECKING:
    # Never executed at runtime -- keeps the lazy names visible to `mypy --strict`
    # without pulling faiss / litellm into a bare `import langres`.
    from langres.curation.bootstrapper import Bootstrapper
    from langres.curation.labelers import (
        BlindCostError,
        FakeLabeler,
        GroundTruthLabeler,
        TeacherLabeler,
    )

__all__ = [
    "BlindCostError",
    "BootstrapReport",
    "Bootstrapper",
    "FakeLabeler",
    "GoldPair",
    "GoldPairSource",
    "GoldSet",
    "GroundTruthLabeler",
    "HardNegativeMiner",
    "Labeler",
    "Miner",
    "TeacherLabeler",
]

_LAZY: dict[str, tuple[str, str]] = {
    "Bootstrapper": (
        "langres.curation.bootstrapper",
        "pip install 'langres[semantic]'",
    ),
    "BlindCostError": ("langres.curation.labelers", "pip install 'langres[llm]'"),
    "FakeLabeler": ("langres.curation.labelers", "pip install 'langres[llm]'"),
    "GroundTruthLabeler": ("langres.curation.labelers", "pip install 'langres[llm]'"),
    "TeacherLabeler": ("langres.curation.labelers", "pip install 'langres[llm]'"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_path, install_hint = _LAZY[name]
    try:
        value = getattr(importlib.import_module(module_path), name)
    except ImportError as exc:
        raise ImportError(
            f"langres.curation.{name} requires an optional dependency: {install_hint}"
        ) from exc
    globals()[name] = value  # cache: subsequent access skips __getattr__
    return value
