"""``resolve_named``: the named-callable serialization seam.

A langres component that takes a pluggable *callable* -- an LLM
``response_parser`` / ``record_serializer``
(:mod:`langres.core.matchers.llm_judge`), a blocker ``text_field_extractor``
(:mod:`langres.core.blockers.vector`) -- faces one problem when it has to
``save``/``load``: a bare callable cannot round-trip through JSON config. The
house pattern -- shared here so every such seam resolves the SAME way, rather
than each inventing its own -- is a module-level ``{name: callable}`` registry
plus this resolver:

- a **registered name** (a ``str``) resolves to its callable and serializes back
  as that name;
- a **registered callable** (the function object) is reverse-looked-up to its
  name, so passing the function still serializes;
- an **unregistered callable** works at runtime but serializes as ``None`` -- it
  reverts to the caller's default on load (documented, never silent);
- ``None`` falls back to ``default_name`` (the registry's default entry), for a
  seam that has one.

This is a stdlib-only leaf -- it imports nothing from ``langres`` -- so both the
``[llm]`` matcher and the ``[semantic]`` blocker reuse ONE mechanism without
coupling the two extras' modules to each other.
"""

from collections.abc import Callable
from typing import Any


def resolve_named(
    value: Callable[..., Any] | str | None,
    registry: dict[str, Callable[..., Any]],
    *,
    kind: str,
    default_name: str | None = None,
) -> tuple[Callable[..., Any], str | None]:
    """Resolve a callable given by registered name, callable, or ``None``.

    Returns ``(callable, name)`` where ``name`` is the registered name to
    serialize in the component's ``config`` -- ``None`` for a custom callable
    that is not in ``registry`` (documented as non-serializable: it reverts to
    the default on load).

    Args:
        value: A registered name, a callable, or ``None``.
        registry: The ``{name: callable}`` registry to resolve against.
        kind: Human-readable noun for error messages (e.g. ``"response_parser"``).
        default_name: The registry key used when ``value`` is ``None``. Pass it
            for a seam that HAS a default (parsers/serializers); omit it for a
            seam where the callable is required and ``None`` is a caller error
            (the blocker extractor, whose absence routes to ``text_field=``
            instead and so never reaches this resolver with ``None``).

    Raises:
        ValueError: For an unknown name (listing the registered ones), or for
            ``value=None`` with no ``default_name``.
    """
    if value is None:
        if default_name is None:
            raise ValueError(f"a {kind} is required (no default registered)")
        return registry[default_name], default_name
    if isinstance(value, str):
        resolved = registry.get(value)
        if resolved is None:
            raise ValueError(
                f"unknown {kind} name {value!r}; registered names: "
                f"{', '.join(sorted(registry))}. Pass a callable for a custom "
                f"{kind} (it will not serialize in config)."
            )
        return resolved, value
    name = next((n for n, fn in registry.items() if fn is value), None)
    return value, name
