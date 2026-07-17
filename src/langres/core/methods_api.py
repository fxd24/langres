"""The ``method=`` object seam: a training strategy as a first-class object.

``Resolver.fit(..., method=...)`` takes a :class:`Method` -- a small,
declarative object that names *how* to train (its ``kind``) and *what it costs*
(its :meth:`Method.describe`), and routes to the matching fit path. This extends
the blessed "a strategy is a meta-object you pass" pattern: one ``method=``
argument replaces a growing pile of per-strategy keyword flags, and each concrete
strategy (prompt-optimize, fine-tune, calibrate) is its own subclass declaring a
distinct ``kind``. It is deliberately *not* the rejected ``TrainingRecipe`` noun
-- a ``Method`` is a value passed at the call site, not a config document.

The base is import-light by construction -- Pydantic only, no torch/litellm/
scikit-learn -- so ``import langres`` never pays for a training backend it may
never use (locked by ``tests/test_import_budget.py``). Concrete methods land in
later PRs (MIPRO prompt-optimization, QLoRA fine-tune, Platt calibration) and
pull their heavy dependencies lazily in their own modules.

.. note::
   This is intentionally **not** :mod:`langres.methods` -- that module is the
   benchmark-method registry (``_make_module_builder``), an unrelated concept
   that happens to share the word "method". :class:`Method` here is the
   training-strategy object :meth:`langres.core.resolver.Resolver.fit` dispatches
   on via ``.kind``.
"""

from typing import ClassVar

from pydantic import BaseModel

__all__ = ["Method", "UnsupportedMethodKind"]


class UnsupportedMethodKind(TypeError):
    """Raised when an architecture is asked to fit with a ``Method`` kind it rejects.

    See ``Resolver.accepted_method_kinds`` for the declaration this enforces and
    why the base :class:`~langres.core.resolver.Resolver` never raises it.

    Subclasses :class:`TypeError` rather than :class:`ValueError` because
    :attr:`Method.kind` is a ``ClassVar`` -- an identity of the strategy *type*,
    not a per-instance value (see :class:`Method`) -- so rejecting a kind is
    rejecting a type of argument. It also matches the sibling ``TypeError`` that
    ``Resolver._fit_finetune`` already raises for a method-object mismatch
    ("method kind 'finetune' requires a QLoRA method"), so one ``except
    TypeError`` catches both shapes of "this method does not belong here".
    """


class Method(BaseModel):
    """A training strategy passed to ``Resolver.fit(method=...)``.

    A ``Method`` carries the two things the fit call site needs: a ``kind`` that
    selects the fit path (the Resolver dispatches on it), and a
    :meth:`describe` one-liner that renders "what + cost" for
    ``Resolver.describe()`` and the ``FitReport``. Concrete subclasses declare a
    distinct ``kind`` and add their own Pydantic-validated configuration fields;
    this base only fixes the contract the Resolver dispatches against.

    ``kind`` is a ``ClassVar`` -- an identity of the strategy *type*, not a
    per-instance field -- so it never becomes serialized config and every
    instance of a subclass shares it. Subclasses set it (e.g.
    ``kind = "finetune"``); the base leaves it unset because it is not meant to
    be instantiated on its own.
    """

    kind: ClassVar[str]

    def describe(self) -> str:
        """Return a one-line "what this method does + what it costs" summary.

        Rendered at the fit call site by ``Resolver.describe()`` and the
        ``FitReport`` so a caller sees the chosen strategy and its cost profile
        without running it. The base returns the bare ``kind``; subclasses
        override with their specifics (e.g. ``"fine-tune (QLoRA, ~GPU-seconds)"``).
        """
        return self.kind
