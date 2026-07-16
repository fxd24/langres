"""Evaluation + debugging tooling: the pipeline debugger and the eval submodules.

See ``langres.core._exports`` for the fragment contract.
"""

from typing import TYPE_CHECKING

from langres.core.debugging import (
    CandidateStats,
    ClusterStats,
    ErrorExample,
    PipelineDebugger,
    ScoreStats,
)

if TYPE_CHECKING:
    # Never executed at runtime -- keeps the lazy submodules visible to
    # `mypy --strict` without pulling ranx/optuna/wandb into a bare
    # `import langres`.
    from langres.core import benchmark, metrics, optimizers

__all__ = [
    "CandidateStats",
    "ClusterStats",
    "ErrorExample",
    "PipelineDebugger",
    "ScoreStats",
]

#: Unlike ``LAZY_SYMBOLS``, the value ``__getattr__`` binds for these is the
#: imported *module* itself (``langres.core.benchmark``, not an attribute of
#: it). All three eventually need ranx (``metrics``, and ``benchmark`` which
#: imports it) or optuna/wandb (``optimizers``) -- dev/eval tooling, not part
#: of the link()/dedupe() runtime path, and not distributed as a pip extra
#: (hence no ``EXTRA_BY_SYMBOL`` entries).
LAZY_SUBMODULES: tuple[str, ...] = ("benchmark", "metrics", "optimizers")

LAZY_SYMBOLS: dict[str, str] = {}
EXTRA_BY_SYMBOL: dict[str, str] = {}

NAMES: tuple[str, ...] = (*__all__, *LAZY_SUBMODULES, *LAZY_SYMBOLS)
