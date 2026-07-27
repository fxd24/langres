"""Vocabulary-overlap profile section: do the two sources even share words?

The one measurement the rest of the profiler does not make. Every other section
looks at *one* corpus (field sparsity, string length) or at *labels*
(cluster shape, separability). This one looks across the **two sources of a
linkage benchmark** and asks the question a string comparator implicitly bets
on: *are the two sides written in the same vocabulary at all?*

Two records that mean the same thing but share no tokens ("Canon EOS 400D" vs
"Canon Digital Rebel XTi") are invisible to rapidfuzz no matter how the
threshold is tuned. A benchmark whose sides share almost all their vocabulary is
one where a free string method is already near its ceiling; a benchmark whose
sides barely overlap is one where the string floor is structurally low and any
lexical method is measuring the *encoding gap*, not entity resolution. Neither
is visible in prevalence, AUC, or field sparsity.

Two numbers carry the finding, and they answer different questions:

- **Jaccard** over the token *types* -- how much of the combined dictionary is
  common. Type-weighted, so a long tail of one-off tokens (model numbers, OCR
  noise) dominates it.
- **Token coverage** per side -- the fraction of that side's token
  *occurrences* whose type also occurs on the other side. Occurrence-weighted,
  so it says what a comparator actually experiences while reading a record.
  A set can have low Jaccard and high coverage (a shared common core plus two
  disjoint long tails) -- and that combination is the interesting one.

Generic by construction, like every other profiler here: :func:`profile_vocabulary_overlap`
takes two plain sequences of strings, never a benchmark-coupled type. A
``None``/empty side means "there is no second source" (a single-corpus dedup
set), and the profiler returns ``None`` so the section is simply absent from the
report -- the same graceful degradation as a missing gold clustering.
"""

from __future__ import annotations

import html
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from langres.report import _report_html
from langres.data.data_profile.base import ProfileSection

#: Default row cap for the shared-token table.
_DEFAULT_TOP_N = 25

#: Tokenizer: maximal runs of word characters, Unicode-aware (so accented names
#: in a person set tokenize as one token, not three). Deliberately the crudest
#: thing that matches what a string comparator sees -- this section measures the
#: data, not a tokenization strategy.
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Case-fold ``text`` and split it into word-character tokens.

    Args:
        text: Any string (a field value, a concatenated record).

    Returns:
        The lower-cased tokens, in order, with duplicates kept (the counts are
        what the occurrence-weighted coverage is computed from).
    """
    return _TOKEN_RE.findall(text.casefold())


class TokenOverlap(BaseModel):
    """One shared token and how often each side uses it.

    Attributes:
        token: The case-folded token type.
        left_count: Occurrences on the left source.
        right_count: Occurrences on the right source.
    """

    model_config = ConfigDict(frozen=True)

    token: str
    left_count: int
    right_count: int


class VocabularyOverlapSection(ProfileSection):
    """Lexical overlap between the two sources of a linkage corpus.

    A frozen :class:`ProfileSection` holding the type-level overlap (Jaccard,
    overlap coefficient) and the occurrence-level coverage per side, plus the
    most-shared tokens. Build it with :func:`profile_vocabulary_overlap`.

    Attributes:
        left_name: Display name of the left source (e.g. ``"abt"``).
        right_name: Display name of the right source.
        n_left_docs: Documents (records) contributing to the left vocabulary.
        n_right_docs: Documents contributing to the right vocabulary.
        n_left_tokens: Total token *occurrences* on the left.
        n_right_tokens: Total token occurrences on the right.
        n_left_types: Distinct token types on the left.
        n_right_types: Distinct token types on the right.
        n_shared_types: Types present on both sides.
        n_union_types: Types present on either side.
        jaccard: ``shared / union`` over types; ``None`` when both sides are
            empty (undefined, never a fabricated ``0.0``).
        overlap_coefficient: ``shared / min(left_types, right_types)``
            (Szymkiewicz-Simpson) -- insensitive to one side being far larger;
            ``None`` when either side has no types.
        left_token_coverage: Fraction of the left's token *occurrences* whose
            type also occurs on the right; ``None`` when the left is empty.
        right_token_coverage: The mirror image; ``None`` when the right is empty.
        top_shared: The most-shared tokens, ranked by ``min(left, right)`` count
            (a token both sides use heavily), capped by ``top_n``.
    """

    kind: Literal["vocabulary_overlap"] = "vocabulary_overlap"

    left_name: str
    right_name: str
    n_left_docs: int
    n_right_docs: int
    n_left_tokens: int
    n_right_tokens: int
    n_left_types: int
    n_right_types: int
    n_shared_types: int
    n_union_types: int
    jaccard: float | None
    overlap_coefficient: float | None
    left_token_coverage: float | None
    right_token_coverage: float | None
    top_shared: list[TokenOverlap]

    _HEADERS = ("token", "left count", "right count")

    # ------------------------------------------------------------- shared render
    def _metrics_kv(self) -> list[tuple[str, str]]:
        """The headline metrics as ``(label, display)`` pairs (markdown + HTML share this)."""
        return [
            ("sources", f"{self.left_name} vs {self.right_name}"),
            ("documents", f"{self.n_left_docs:,} / {self.n_right_docs:,}"),
            ("token occurrences", f"{self.n_left_tokens:,} / {self.n_right_tokens:,}"),
            ("token types", f"{self.n_left_types:,} / {self.n_right_types:,}"),
            ("shared types", f"{self.n_shared_types:,}"),
            ("union types", f"{self.n_union_types:,}"),
            ("type Jaccard", _report_html._num(self.jaccard)),
            ("type overlap coefficient", _report_html._num(self.overlap_coefficient)),
            (
                f"token coverage ({self.left_name})",
                _report_html._num(self.left_token_coverage),
            ),
            (
                f"token coverage ({self.right_name})",
                _report_html._num(self.right_token_coverage),
            ),
        ]

    # ------------------------------------------------------------ text surfaces
    def to_markdown(self) -> str:
        """Markdown: the metrics table plus the most-shared-token table."""
        lines = [f"## {self.title}", "", "| metric | value |", "|---|---|"]
        lines += [
            f"| {_report_html._md_cell(k)} | {_report_html._md_cell(v)} |"
            for k, v in self._metrics_kv()
        ]
        lines += [
            "",
            "### Most-shared tokens",
            "",
            f"| token | {self.left_name} count | {self.right_name} count |",
            "|---|---|---|",
        ]
        if self.top_shared:
            lines += [
                f"| {_report_html._md_cell(t.token)} | {t.left_count} | {t.right_count} |"
                for t in self.top_shared
            ]
        else:
            lines.append("| _(none)_ | 0 | 0 |")
        return "\n".join(lines)

    @property
    def summary(self) -> dict[str, Any]:
        """Headline numbers as a flat, title-namespaced dict (collision-free per report)."""
        return {
            f"{self.title}.jaccard": self.jaccard,
            f"{self.title}.overlap_coefficient": self.overlap_coefficient,
            f"{self.title}.left_token_coverage": self.left_token_coverage,
            f"{self.title}.right_token_coverage": self.right_token_coverage,
            f"{self.title}.n_shared_types": self.n_shared_types,
        }

    def rows(self) -> list[dict[str, Any]]:
        """One row per shared token -- ``pd.DataFrame(section.rows())``-ready."""
        return [
            {
                "token": t.token,
                "left_count": t.left_count,
                "right_count": t.right_count,
            }
            for t in self.top_shared
        ]

    # -------------------------------------------------------------- html panel
    def panels(self) -> list[str]:
        """A single ``<section>``: the metrics KV table above the shared-token table."""
        kv = "".join(
            f"<tr><th>{html.escape(k)}</th><td>{html.escape(v)}</td></tr>"
            for k, v in self._metrics_kv()
        )
        head = "<tr>" + "".join(f"<th>{html.escape(h)}</th>" for h in self._HEADERS) + "</tr>"
        body_rows = "".join(
            "<tr>"
            + f"<td>{html.escape(t.token)}</td>"
            + f"<td>{t.left_count}</td><td>{t.right_count}</td>"
            + "</tr>"
            for t in self.top_shared
        )
        table = f'<table class="errors">{head}{body_rows}</table>'
        body = f'<table class="kv">{kv}</table>{table}'
        return [_report_html.section(self.title, body)]


def profile_vocabulary_overlap(
    left_texts: Sequence[str] | None,
    right_texts: Sequence[str] | None,
    *,
    left_name: str = "A",
    right_name: str = "B",
    top_n: int = _DEFAULT_TOP_N,
    title: str = "Vocabulary overlap",
) -> VocabularyOverlapSection | None:
    """Profile the lexical overlap between two document sets.

    Args:
        left_texts: The left source's documents (one string per record --
            typically that record's string fields joined). ``None`` means there
            is no second source to compare against and the profiler returns
            ``None`` (the section is omitted).
        right_texts: The right source's documents. ``None`` behaves as above.
        left_name: Display name for the left source.
        right_name: Display name for the right source.
        top_n: Row cap on the most-shared-token table (ranked by the smaller of
            the two counts, so the table shows tokens *both* sides lean on).
        title: Section heading; also the key the report looks it up by, and the
            namespace for this section's :attr:`~VocabularyOverlapSection.summary`
            keys.

    Returns:
        A :class:`VocabularyOverlapSection`, or ``None`` when either side is
        ``None`` (no second source). Two present-but-empty sides are a valid,
        degenerate input and render with ``None`` ratios rather than raising.
    """
    if left_texts is None or right_texts is None:
        return None

    left_counts = _count_tokens(left_texts)
    right_counts = _count_tokens(right_texts)

    left_types = set(left_counts)
    right_types = set(right_counts)
    shared = left_types & right_types
    union = left_types | right_types

    n_left_tokens = sum(left_counts.values())
    n_right_tokens = sum(right_counts.values())

    jaccard = len(shared) / len(union) if union else None
    smaller = min(len(left_types), len(right_types))
    overlap_coefficient = len(shared) / smaller if smaller else None

    left_shared_tokens = sum(left_counts[t] for t in shared)
    right_shared_tokens = sum(right_counts[t] for t in shared)
    left_coverage = left_shared_tokens / n_left_tokens if n_left_tokens else None
    right_coverage = right_shared_tokens / n_right_tokens if n_right_tokens else None

    # Rank by the smaller of the two counts: a token both sides use heavily is
    # far more informative than one side's stopword that happens to appear once
    # on the other. Token as tie-break keeps the table deterministic.
    ranked = sorted(shared, key=lambda t: (-min(left_counts[t], right_counts[t]), t))
    top_shared = [
        TokenOverlap(token=t, left_count=left_counts[t], right_count=right_counts[t])
        for t in ranked[:top_n]
    ]

    return VocabularyOverlapSection(
        title=title,
        left_name=left_name,
        right_name=right_name,
        n_left_docs=len(left_texts),
        n_right_docs=len(right_texts),
        n_left_tokens=n_left_tokens,
        n_right_tokens=n_right_tokens,
        n_left_types=len(left_types),
        n_right_types=len(right_types),
        n_shared_types=len(shared),
        n_union_types=len(union),
        jaccard=jaccard,
        overlap_coefficient=overlap_coefficient,
        left_token_coverage=left_coverage,
        right_token_coverage=right_coverage,
        top_shared=top_shared,
    )


def _count_tokens(texts: Iterable[str]) -> Counter[str]:
    """Token-occurrence counter over a document set."""
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(tokenize(text))
    return counts
