"""Tests for the shared canonical pair-key helper."""

from langres.bootstrap._pairs import canonical_pair_key


def test_canonical_pair_key_is_order_independent() -> None:
    # Both argument orders map to the same sorted tuple (covers both branches).
    assert canonical_pair_key("a", "b") == ("a", "b")
    assert canonical_pair_key("b", "a") == ("a", "b")


def test_canonical_pair_key_equal_ids() -> None:
    assert canonical_pair_key("x", "x") == ("x", "x")
