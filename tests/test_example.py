"""Example test to verify test infrastructure and top-level imports work."""

import langres


def test_version() -> None:
    """Test that version is defined."""
    assert hasattr(langres, "__version__")
    assert isinstance(langres.__version__, str)


def test_top_level_imports() -> None:
    """Test that core models are exported at top level."""
    # Verify models are accessible from langres namespace
    assert hasattr(langres, "CompanySchema")
    assert hasattr(langres, "ERCandidate")
    assert hasattr(langres, "PairwiseJudgement")

    # Verify they work
    company = langres.CompanySchema(id="test", name="Test Corp")
    assert company.id == "test"
    assert company.name == "Test Corp"
