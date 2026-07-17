"""Shared record-pair identity helper for bootstrapping.

Both the miner (deduplication) and the ground-truth labeler need an
order-independent key for a record pair so that ``(a, b)`` and ``(b, a)`` are
treated as the same pair. Keeping that logic in one place prevents the two
call sites from drifting apart.
"""


def canonical_pair_key(left_id: str, right_id: str) -> tuple[str, str]:
    """Return an order-independent key for a record pair.

    ``(a, b)`` and ``(b, a)`` both map to the same tuple, so the key identifies
    a pair regardless of which side each record was placed on.

    Args:
        left_id: One record id.
        right_id: The other record id.

    Returns:
        The two ids as a sorted 2-tuple.
    """
    return (left_id, right_id) if left_id <= right_id else (right_id, left_id)
