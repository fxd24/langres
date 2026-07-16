"""The export-fragment contract (W-1a): what keeps the split honest.

``langres/__init__.py`` and ``langres/core/__init__.py`` are thin aggregators
over per-domain fragments (``langres/_exports/``, ``langres/core/_exports/``).
The split exists to let parallel work-streams edit disjoint files instead of
one sorted ~100-name ``__all__``. But a contract enforced only by convention
drifts -- and it drifts hardest under exactly the concurrent-edit pressure the
split invites, especially since ``.gitattributes`` marks the fragments
``merge=union`` (git keeps BOTH sides' lines rather than raising a conflict).

So the contract is tested, not trusted. These are cheap, offline, structural
checks that fail loudly in CI instead of surfacing as a confusing traceback or
a silently wrong ``pip install`` hint months later.

Fragments are DISCOVERED (``pkgutil``), never listed, so a newly added fragment
is covered automatically -- including the case where someone adds a fragment
file but forgets to wire it into its aggregator.
"""

from __future__ import annotations

import importlib
import pkgutil
from types import ModuleType

import pytest

import langres
import langres._exports
import langres.core
import langres.core._exports


def _fragments(pkg: ModuleType) -> list[ModuleType]:
    """Every ``_*.py`` fragment module in an ``_exports`` package."""
    return [
        importlib.import_module(f"{pkg.__name__}.{m.name}")
        for m in pkgutil.iter_modules(pkg.__path__)
        if m.name.startswith("_")
    ]


_CORE_FRAGMENTS = _fragments(langres.core._exports)
_ROOT_FRAGMENTS = _fragments(langres._exports)
_ALL_FRAGMENTS = _CORE_FRAGMENTS + _ROOT_FRAGMENTS


def _lazy_submodules(frag: ModuleType) -> tuple[str, ...]:
    """Root fragments declare no ``LAZY_SUBMODULES`` -- only ``core`` has any."""
    return tuple(getattr(frag, "LAZY_SUBMODULES", ()))


def _ids(frags: list[ModuleType]) -> list[str]:
    return [f.__name__.removeprefix("langres.") for f in frags]


def test_fragments_are_discovered() -> None:
    """Guard the guard: these tests are worthless if discovery finds nothing."""
    assert len(_CORE_FRAGMENTS) >= 10
    assert len(_ROOT_FRAGMENTS) >= 5


@pytest.mark.parametrize("frag", _ALL_FRAGMENTS, ids=_ids(_ALL_FRAGMENTS))
class TestFragmentContract:
    """Each fragment declares the contract in ``_exports/__init__.py``'s docstring."""

    def test_declares_the_contract(self, frag: ModuleType) -> None:
        for attr in ("__all__", "LAZY_SYMBOLS", "EXTRA_BY_SYMBOL", "NAMES"):
            assert hasattr(frag, attr), f"{frag.__name__} is missing {attr}"

    def test_names_is_derived_not_hand_maintained(self, frag: ModuleType) -> None:
        """``NAMES == (*__all__, *LAZY_SUBMODULES, *LAZY_SYMBOLS)``.

        Hand-editing ``NAMES`` is how a name silently drops out of (or sneaks
        into) the public surface.
        """
        expected = (*frag.__all__, *_lazy_submodules(frag), *frag.LAZY_SYMBOLS)
        assert tuple(frag.NAMES) == expected

    def test_extras_only_describe_lazy_symbols(self, frag: ModuleType) -> None:
        """An extra for a non-lazy name is dead config -- ``__getattr__`` never reads it."""
        assert set(frag.EXTRA_BY_SYMBOL) <= set(frag.LAZY_SYMBOLS)

    def test_eager_names_are_actually_defined(self, frag: ModuleType) -> None:
        """``__all__`` drives ``from ._x import *`` -- an undefined entry is an AttributeError.

        This is the loud failure mode ``merge=union`` trades for: if a removal
        and an edit interleave, union can resurrect an ``__all__`` entry whose
        import is gone. It must die here, at import, not in a user's process.
        """
        for name in frag.__all__:
            assert hasattr(frag, name), f"{frag.__name__}.__all__ has undefined {name!r}"

    def test_lazy_names_are_not_eagerly_defined(self, frag: ModuleType) -> None:
        """A lazy name must NOT exist at fragment module scope.

        Two failures at once if it does: the star-import would bind it eagerly
        (blowing the import budget -- fragments are eagerly imported), and the
        aggregator's ``__getattr__`` would never be consulted for it.
        """
        for name in frag.LAZY_SYMBOLS:
            assert name not in vars(frag), (
                f"{frag.__name__} defines lazy symbol {name!r} at module scope"
            )


@pytest.mark.parametrize(
    ("frags", "pkg"),
    [(_CORE_FRAGMENTS, langres.core._exports), (_ROOT_FRAGMENTS, langres._exports)],
    ids=["core", "root"],
)
class TestNoCollisionsBetweenFragments:
    """Exactly one fragment owns any given name.

    ``**``-merging means a name declared by two fragments is silently
    last-wins -- which would resolve it from the wrong module, or hint the
    wrong extra. Nothing about the merge itself would complain.
    """

    def test_no_name_is_owned_by_two_fragments(
        self, frags: list[ModuleType], pkg: ModuleType
    ) -> None:
        seen: dict[str, str] = {}
        for frag in frags:
            for name in frag.NAMES:
                assert name not in seen, (
                    f"{name!r} is declared by BOTH {seen[name]} and {frag.__name__}"
                )
                seen[name] = frag.__name__

    def test_no_lazy_symbol_is_mapped_twice(self, frags: list[ModuleType], pkg: ModuleType) -> None:
        seen: dict[str, str] = {}
        for frag in frags:
            for name in frag.LAZY_SYMBOLS:
                assert name not in seen, (
                    f"lazy {name!r} is mapped by BOTH {seen[name]} and {frag.__name__}"
                )
                seen[name] = frag.__name__

    def test_every_fragment_is_wired_into_the_aggregate(
        self, frags: list[ModuleType], pkg: ModuleType
    ) -> None:
        """A fragment file that no aggregator composes is invisible, silently."""
        for frag in frags:
            assert set(frag.NAMES) <= set(pkg.NAMES), (
                f"{frag.__name__} is not composed by {pkg.__name__} -- forgot to wire it?"
            )


class TestAggregatesMatchTheirFragments:
    """The composed surface is exactly the union of the fragments' slices."""

    def test_core_all_is_the_union_of_fragment_names(self) -> None:
        assert set(langres.core.__all__) == {n for f in _CORE_FRAGMENTS for n in f.NAMES}

    def test_root_all_is_the_union_of_fragment_names(self) -> None:
        assert set(langres.__all__) == {n for f in _ROOT_FRAGMENTS for n in f.NAMES}

    @pytest.mark.parametrize("mod", [langres, langres.core], ids=["root", "core"])
    def test_all_has_no_duplicates(self, mod: ModuleType) -> None:
        assert len(mod.__all__) == len(set(mod.__all__))

    @pytest.mark.parametrize("mod", [langres, langres.core], ids=["root", "core"])
    def test_every_exported_name_actually_resolves(self, mod: ModuleType) -> None:
        """The end-to-end promise: every ``__all__`` name is gettable.

        Eager names resolve from the namespace; lazy ones go through
        ``__getattr__``. A lazy name whose owning module needs an uninstalled
        extra raises the actionable ImportError -- that IS the contract, so it
        counts as resolving.
        """
        for name in mod.__all__:
            try:
                getattr(mod, name)
            except ImportError:
                pass  # missing optional extra -- the documented, actionable path
            except AttributeError:  # pragma: no cover - a real regression
                pytest.fail(f"{mod.__name__}.{name} is in __all__ but does not resolve")


class TestCoreExtraBySymbolIsTotal:
    """``core.__getattr__`` INDEXES ``_EXTRA_BY_SYMBOL[name]`` -- so it must be total.

    The pre-split code built the map with a dict comprehension defaulting to
    ``"semantic"``, which made totality automatic. The fragments declare extras
    explicitly instead (clearer -- no magic default), which means a future lazy
    symbol added WITHOUT an extra would raise ``KeyError`` from inside the
    ``except ImportError`` handler: a confusing chained traceback in place of
    the actionable "pip install langres[...]" hint. Locked here.

    The root package needs no such rule -- its ``__getattr__`` uses ``.get()``
    and deliberately re-raises unchanged for symbols that need no extra.
    """

    def test_every_core_lazy_symbol_has_an_extra(self) -> None:
        missing = set(langres.core._LAZY_SYMBOLS) - set(langres.core._EXTRA_BY_SYMBOL)
        assert not missing, f"core lazy symbols without a [extra] hint: {sorted(missing)}"

    def test_core_submodules_are_not_given_extras(self) -> None:
        """The lazy submodules are dev/eval tooling, not pip extras."""
        assert not (langres.core._LAZY_SUBMODULES & set(langres.core._EXTRA_BY_SYMBOL))
