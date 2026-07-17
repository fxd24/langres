"""The package version string -- a pure-stdlib leaf that imports no langres.

**Nothing in this module may import from ``langres``, and that is the point.**

``langres.core.resolver`` stamps ``langres_version`` into every saved artifact
and compares it on load. Reading that one string used to cost it a toplevel
``import langres`` -- the floor importing the ceiling -- which was the single
edge closing the package's only runtime import cycle::

    langres -> _exports._core -> langres.core -> core._exports._resolver
             -> core.resolver -> langres

Twelve modules were strongly connected on a bare ``import langres`` solely so a
version string could be read (measured: ``tools/import_graph.py kinds``). The
cycle was benign only because the read happens inside a method body; a single
module-scope use of ``langres.__version__`` in ``resolver`` would have turned it
into an ImportError at a partially-initialized ``langres``.

Sourcing the string from here instead leaves that cycle with no edge to close:
``resolver`` and ``cli`` depend on this leaf, the root re-exports it, and none of
them depends on the root. ``tests/test_import_tangle.py`` pins the result (the
runtime tangle is empty and must stay empty).

The version depends on nothing inside langres -- it comes from installed
metadata -- which is exactly why it was extractable.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _metadata_version

# Single source of truth is pyproject.toml; resolved from installed metadata so
# a version bump can never miss this string again. Falls back for source trees
# imported without installation.
try:
    __version__ = _metadata_version("langres")
except PackageNotFoundError:  # pragma: no cover - only hit on uninstalled source trees
    __version__ = "0.0.0.dev0"
