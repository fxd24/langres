"""The data-prep surface at the root.

``DataProfileReport`` is import-light (stdlib + numpy + the import-light
``core.metrics``; it consumes precomputed embeddings and never generates them)
but lives outside the eager import graph on purpose, so it stays lazy and needs
no extra -- an ImportError from it is a genuine bug and must propagate
unchanged.

See ``langres._exports`` for the fragment contract.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Never executed at runtime -- keeps the lazy name visible to `mypy --strict`
    # without pulling the data-profile module into a bare `import langres`.
    from langres.data.data_profile import DataProfileReport

#: Nothing is eager here by design -- see the module docstring.
__all__: list[str] = []

LAZY_SYMBOLS: dict[str, str] = {
    "DataProfileReport": "langres.data.data_profile",
}

EXTRA_BY_SYMBOL: dict[str, str] = {}

NAMES: tuple[str, ...] = (*__all__, *LAZY_SYMBOLS)
