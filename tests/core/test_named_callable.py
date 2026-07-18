"""Unit tests for the shared named-callable resolution seam.

``resolve_named`` is the one mechanism both the ``[llm]`` matcher
(``response_parser``/``record_serializer``) and the ``[semantic]`` blocker
(``text_field_extractor``) use to accept a callable by name, by value, or by
default -- and to serialize it back by name. These pin every branch directly,
independent of either consumer.
"""

from typing import Any

import pytest

from langres.core.named_callable import resolve_named


def _a(x: str) -> str:  # pragma: no cover - identity, never called here
    return x


def _b(x: str) -> str:  # pragma: no cover - identity, never called here
    return x


REGISTRY: dict[str, Any] = {"a": _a, "b": _b}


def test_registered_name_resolves_to_its_callable_and_serializes_as_the_name() -> None:
    fn, name = resolve_named("a", REGISTRY, kind="thing")
    assert fn is _a
    assert name == "a"


def test_registered_callable_is_reverse_looked_up_to_its_name() -> None:
    """Passing the function object still serializes (by reverse lookup)."""
    fn, name = resolve_named(_b, REGISTRY, kind="thing")
    assert fn is _b
    assert name == "b"


def test_unregistered_callable_resolves_but_serializes_as_none() -> None:
    """A custom callable works at runtime but has no serializable name."""

    def custom(x: str) -> str:  # pragma: no cover - identity, never called
        return x

    fn, name = resolve_named(custom, REGISTRY, kind="thing")
    assert fn is custom
    assert name is None


def test_none_with_default_falls_back_to_the_default_entry() -> None:
    fn, name = resolve_named(None, REGISTRY, kind="thing", default_name="a")
    assert fn is _a
    assert name == "a"


def test_none_without_default_raises() -> None:
    """A seam with no default (the blocker extractor) treats None as an error."""
    with pytest.raises(ValueError, match="a thing is required"):
        resolve_named(None, REGISTRY, kind="thing")


def test_unknown_name_raises_and_lists_the_registered_names() -> None:
    with pytest.raises(ValueError, match="unknown thing name 'zzz'") as exc:
        resolve_named("zzz", REGISTRY, kind="thing")
    # The message lists the registered names, sorted, to guide the caller.
    assert "a, b" in str(exc.value)
