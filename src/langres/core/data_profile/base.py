"""The composable seam of the data-profile report: ``ProfileSection`` + ``DataProfileReport``.

This is the load-bearing contract from the data-layer plan (§2): the report is a
*bag of sections*, not a monolith. Each metric family (label structure, field
stats, separability, embeddings, mining readiness -- all landing in later waves)
is a self-contained :class:`ProfileSection`; :class:`DataProfileReport` is a pure
container that holds whatever sections it is given and renders exactly those.
Adding a metric family is a new subclass + a profiler function -- the container
is never touched (open/closed).

**Frozen contract.** This module's public surface is frozen after Wave 0: later
waves add *subclasses* and *profiler functions* elsewhere, never edits here. The
two convenience constructors (:meth:`DataProfileReport.from_benchmark` /
:meth:`~DataProfileReport.from_records`) are deliberately *thin delegators* to a
``builders`` module that lands in Wave 2 -- imported locally so this module stays
valid now and those methods light up later without a signature change.

**Layering.** A leaf, like :mod:`langres.core.eval_report`: it renders through the
shared :mod:`langres.core._report_html` scaffold and pulls no heavy dependency
(an import-budget test locks it). Text-first is primary -- ``print``/markdown/dict
are the default surfaces; ``to_html`` is the optional ``$0`` tearsheet.
"""

from __future__ import annotations

import abc
import importlib
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, SerializeAsAny

from langres.core import _report_html


class ProfileSection(BaseModel, abc.ABC):
    """One self-contained metric block of a :class:`DataProfileReport`.

    A frozen Pydantic model that every concrete profiler section subclasses.
    Each subclass reports exactly one metric family and exposes the same render
    ladder (text / dict / tabular / HTML panel), so the container can compose an
    arbitrary subset without knowing what any of them compute.

    Subclasses must set a distinct :attr:`kind` (a discriminator, so a dumped
    report is re-identifiable) and implement :meth:`to_markdown`,
    :attr:`summary`, :meth:`rows`, and :meth:`panels`.

    Attributes:
        title: Human-readable section heading; also the key
            :meth:`DataProfileReport.__getitem__` looks the section up by.
        kind: A stable discriminator string. The base declares it as ``str``;
            each subclass pins a ``Literal[...]`` with a default so instances
            self-identify in a dumped/serialised report.
    """

    model_config = ConfigDict(frozen=True)

    title: str
    kind: str

    @abc.abstractmethod
    def to_markdown(self) -> str:
        """This section's Markdown block (heading + body). Primary text surface."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def summary(self) -> dict[str, Any]:
        """This section's headline numbers as a flat dict (log it, assert on it)."""
        raise NotImplementedError

    @abc.abstractmethod
    def rows(self) -> list[dict[str, Any]]:
        """This section's tabular rows -- ``pd.DataFrame(section.rows())``, no pandas dep."""
        raise NotImplementedError

    @abc.abstractmethod
    def panels(self) -> list[str]:
        """This section's HTML ``<section>`` panels (inline SVG/tables); HTML render only."""
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        """This section as a plain dict (``model_dump()``); machine / JSON surface."""
        return self.model_dump()

    def __repr__(self) -> str:
        """Render as Markdown so a section prints cleanly in a REPL/notebook."""
        return self.to_markdown()


class DataProfileReport(BaseModel):
    """A frozen container over the :class:`ProfileSection`\\ s you compose.

    Holds whatever sections it is given and renders exactly those -- it computes
    nothing itself (composition lives at the edges). Text-first: ``print`` /
    :meth:`to_markdown` / :meth:`to_dict` / :attr:`summary` are the primary
    surfaces; :meth:`to_html` is the optional ``$0`` tearsheet.

    Construct it directly from a list of sections (positional, as in the plan)::

        report = DataProfileReport([label_section, field_section])

    or via the convenience constructors :meth:`from_benchmark` /
    :meth:`from_records` (which delegate to the Wave 2 ``builders`` module).

    Attributes:
        sections: The composed sections, in render order. Typed with
            ``SerializeAsAny`` so ``model_dump()`` keeps each concrete
            subclass's own fields rather than narrowing to the base.
    """

    model_config = ConfigDict(frozen=True)

    sections: list[SerializeAsAny[ProfileSection]]

    def __init__(self, sections: list[ProfileSection] | None = None, **data: Any) -> None:
        """Accept ``sections`` positionally (``DataProfileReport([...])``).

        Pydantic's generated ``__init__`` is keyword-only; the plan's ergonomics
        (and the acceptance check ``DataProfileReport([])``) want a positional
        list. Forward it into the field, then defer to normal validation.
        """
        if sections is not None:
            data["sections"] = sections
        super().__init__(**data)

    # ------------------------------------------------------------- text surfaces
    def to_markdown(self) -> str:
        """Concatenate the sections' Markdown blocks under a report heading."""
        parts: list[str] = ["# Data profile", ""]
        if not self.sections:
            parts.append("_No sections._")
        else:
            for section in self.sections:
                parts.append(section.to_markdown())
                parts.append("")
        return "\n".join(parts).rstrip() + "\n"

    def __repr__(self) -> str:
        """Render as Markdown so ``print(report)`` just works in a REPL/notebook."""
        return self.to_markdown()

    def to_dict(self) -> dict[str, Any]:
        """The report as a plain dict (``model_dump()``).

        One-way (dump-out): the sections serialise with their concrete
        subclass fields (``SerializeAsAny``), but reloading a dumped report
        coerces each section back to the :class:`ProfileSection` base -- so this
        is for persistence / diffing / dashboards, not round-tripping.
        """
        return self.model_dump()

    @property
    def summary(self) -> dict[str, Any]:
        """The sections' headline numbers flattened into one dict."""
        out: dict[str, Any] = {}
        for section in self.sections:
            out.update(section.summary)
        return out

    def __getitem__(self, title: str) -> SerializeAsAny[ProfileSection]:
        """Pull one section out by its :attr:`~ProfileSection.title`.

        Raises:
            KeyError: If no section has that title.
        """
        for section in self.sections:
            if section.title == title:
                return section
        raise KeyError(title)

    # -------------------------------------------------------------- html surface
    def to_html(self, *, title: str = "Data profile") -> str:
        """Render a single self-contained HTML document over the present sections.

        Only the sections you gave it contribute -- each section's
        :meth:`~ProfileSection.panels` are emitted in order through the shared
        :mod:`langres.core._report_html` scaffold. An empty report renders a
        valid, section-less page (never an error), matching the plan's
        graceful-degradation contract.

        Args:
            title: Page and document title.

        Returns:
            A complete ``<!doctype html>...`` string with inline styles and
            inline SVG only (no external assets).
        """
        body_parts: list[str] = []
        for section in self.sections:
            body_parts.extend(section.panels())
        return _report_html.document(title, "\n".join(body_parts))

    # ---------------------------------------------------- convenience constructors
    @classmethod
    def from_benchmark(cls, *args: Any, **kwargs: Any) -> DataProfileReport:
        """Profile a benchmark (labels + fields + ... ) with sensible defaults.

        A thin delegator to the ``builders`` module (Wave 2): the heavy lifting
        -- composing the default section set, honouring optional
        ``embeddings=``/``blocker=`` inputs, applying an ``include=`` subset --
        lives there so this frozen base never changes as that logic firms up.
        Resolved *dynamically* (``importlib``) rather than a static ``from ...
        import``: ``builders`` does not exist until Wave 2, and resolving it
        dynamically keeps this frozen module type-clean both before and after it
        lands (a static import would need a ``# type: ignore`` that Wave 2 would
        then have to remove -- editing a module that must stay frozen).
        """
        builders = importlib.import_module("langres.core.data_profile.builders")
        return cast("DataProfileReport", builders.from_benchmark(*args, **kwargs))

    @classmethod
    def from_records(cls, *args: Any, **kwargs: Any) -> DataProfileReport:
        """Profile raw records (+ optional gold / embeddings) with sensible defaults.

        The bring-your-own-data counterpart of :meth:`from_benchmark`; delegates
        to the same Wave 2 ``builders`` module, resolved dynamically for the same
        frozen-module reason documented there.
        """
        builders = importlib.import_module("langres.core.data_profile.builders")
        return cast("DataProfileReport", builders.from_records(*args, **kwargs))
