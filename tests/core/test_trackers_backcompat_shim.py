"""Back-compat: the W2-sweep shim at the old ``langres.core.trackers`` path.

The tracker layer moved to ``langres.tracking.trackers`` (observability, not ER
modelling). The shim left behind is *not* a plain re-export: the three backend
adapters must stay lazy, so it mirrors the real module's PEP 562 ``__getattr__``
rather than importing them. That conditional resolution is real behavior, which
is why this shim is covered by tests instead of carrying the ``# pragma: no
cover`` its plain-re-export siblings use (see ``test_training_backcompat_shims``
for the sibling pattern).

The lazy-path tests resolve against a stand-in injected into the real module's
``__dict__``: asserting against the genuine ``MlflowTracker`` would import
``mlflow`` (it sits at that adapter module's top level), which is exactly the
heavyweight import the laziness exists to avoid.

``monkeypatch.setitem(real.__dict__, ...)`` -- NOT ``monkeypatch.setattr(real,
..., raising=False)``. That distinction is load-bearing and was found the hard
way: ``setattr`` records the previous value with ``getattr``, which on a PEP 562
module *invokes* ``__getattr__`` (importing the backend), and its teardown then
writes that class back into the module ``__dict__``. The name would be cached
there permanently, ``__getattr__`` would never run for it again, and every later
test asserting the missing-extra ``ImportError`` path would fail. ``setitem``
reads with ``dict.get``, which does not trigger ``__getattr__``, and deletes the
key on teardown -- leaving the module exactly as it was found.
"""

from __future__ import annotations

from typing import Any

import pytest

import langres.core.trackers as shim
import langres.tracking.trackers as real

#: The names the shim re-exports eagerly (no backend import involved).
_EAGER = ["ExperimentTracker", "MultiTracker", "NoOpTracker", "TrackerSpec", "resolve_tracker"]


def test_eager_reexports_are_the_new_objects() -> None:
    """Every eagerly re-exported name is identity-equal to its new home."""
    for name in _EAGER:
        assert getattr(shim, name) is getattr(real, name), name
    assert sorted(shim.__all__) == sorted(_EAGER)


def test_lazy_adapter_names_match_the_real_adapter_table() -> None:
    """The shim's lazy-name set cannot drift from the real module's table.

    The shim hardcodes the adapter names; the real module derives them from
    ``_ADAPTERS``. If a fourth backend is added there and not here, the shim
    would silently raise ``AttributeError`` for a name that does resolve at its
    real home -- so pin the two together.
    """
    assert shim._LAZY_ADAPTERS == {cls_name for _mod, cls_name in real._ADAPTERS.values()}


@pytest.mark.parametrize("name", sorted({"MlflowTracker", "WandbTracker", "TrackioTracker"}))
def test_lazy_adapter_resolves_through_the_new_module(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Accessing an adapter name delegates to ``langres.tracking.trackers``."""
    sentinel = object()
    # Inject into the real module's __dict__ so the shim's `getattr` finds it
    # directly -- exercising the shim's lazy branch without importing the
    # backend, and without perturbing the real module's __getattr__ for any
    # later test (see this module's docstring: setitem, never setattr).
    monkeypatch.setitem(real.__dict__, name, sentinel)
    assert getattr(shim, name) is sentinel


def test_unknown_attribute_raises_attribute_error() -> None:
    """A name that is neither re-exported nor a lazy adapter still fails cleanly."""
    with pytest.raises(AttributeError) as excinfo:
        _: Any = shim.NoSuchTracker  # type: ignore[attr-defined]
    message = str(excinfo.value)
    assert "langres.core.trackers" in message
    assert "NoSuchTracker" in message
