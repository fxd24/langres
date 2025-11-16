"""Diagnostic models for blocker evaluation.

Provides concrete failure examples for manual inspection, similar to
scikit-learn's pattern of extracting misclassified samples.
"""

from pydantic import BaseModel, Field


class MissedMatchExample(BaseModel):
    """A true match that the blocker failed to find.

    These are entity pairs that SHOULD have been candidates (they're in the
    same gold cluster) but the blocker didn't retrieve them.

    Example:
        >>> example = MissedMatchExample(
        ...     left_id="e1",
        ...     left_text="Acme Corp",
        ...     right_id="e2",
        ...     right_text="Acme Corporation",
        ...     cluster_id=0,
        ... )
        >>> print(f"Missed: {example.left_text} <-> {example.right_text}")
    """

    left_id: str = Field(description="ID of left entity")
    left_text: str = Field(description="Display text of left entity")
    right_id: str = Field(description="ID of right entity")
    right_text: str = Field(description="Display text of right entity")
    cluster_id: int = Field(description="Gold cluster ID this pair belongs to")
    baseline_similarity: float | None = Field(
        default=None,
        description="Similarity by baseline metric (e.g., rapidfuzz) for severity ranking",
    )


class FalsePositiveExample(BaseModel):
    """A non-match that the blocker ranked highly.

    These are entity pairs that were retrieved as candidates (high similarity
    score) but are NOT in the same gold cluster.

    Example:
        >>> example = FalsePositiveExample(
        ...     left_id="e1",
        ...     left_text="Apple Inc",
        ...     right_id="e3",
        ...     right_text="Apple Fruit",
        ...     score=0.92,
        ... )
        >>> print(f"False positive: {example.left_text} <-> {example.right_text} (score: {example.score})")
    """

    left_id: str = Field(description="ID of left entity")
    left_text: str = Field(description="Display text of left entity")
    right_id: str = Field(description="ID of right entity")
    right_text: str = Field(description="Display text of right entity")
    score: float = Field(description="Blocker similarity score")


class DiagnosticExamples(BaseModel):
    """Collection of diagnostic examples for manual inspection.

    Similar to scikit-learn's pattern of showing misclassified samples,
    this provides concrete failure cases for understanding blocker behavior.

    Example:
        >>> examples = DiagnosticExamples(
        ...     missed_matches=[...],
        ...     false_positives=[...],
        ... )
        >>> for ex in examples.missed_matches[:10]:
        ...     print(f"Missed: {ex.left_text} <-> {ex.right_text}")
    """

    missed_matches: list[MissedMatchExample] = Field(
        default_factory=list,
        description="True matches the blocker failed to find",
    )
    false_positives: list[FalsePositiveExample] = Field(
        default_factory=list,
        description="Non-matches the blocker ranked highly",
    )

    def to_markdown(self) -> str:
        """Generate markdown report with examples.

        Returns:
            Markdown-formatted string with categorized examples.
        """
        lines = ["# Diagnostic Examples\n"]

        # Missed matches
        lines.append(f"## Missed Matches ({len(self.missed_matches)})\n")
        if self.missed_matches:
            for i, missed in enumerate(self.missed_matches[:20], 1):
                lines.append(f"{i}. **{missed.left_text}** ↔ **{missed.right_text}**")
                lines.append(f"   - Cluster: {missed.cluster_id}")
                if missed.baseline_similarity is not None:
                    lines.append(f"   - Baseline similarity: {missed.baseline_similarity:.2f}")
                lines.append("")
        else:
            lines.append("No missed matches.\n")

        # False positives
        lines.append(f"## False Positives ({len(self.false_positives)})\n")
        if self.false_positives:
            for i, fp in enumerate(self.false_positives[:20], 1):
                lines.append(f"{i}. **{fp.left_text}** ↔ **{fp.right_text}**")
                lines.append(f"   - Score: {fp.score:.2f}")
                lines.append("")
        else:
            lines.append("No false positives in top results.\n")

        return "\n".join(lines)
