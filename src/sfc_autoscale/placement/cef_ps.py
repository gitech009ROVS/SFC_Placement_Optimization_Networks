"""CEF-PS: Cost Efficiency Factor Priority Sorting (Paper 1, Section IV-B)."""
from __future__ import annotations

from ..core.entities import SFCRequest


def cef_priority_sort(requests: list[SFCRequest]) -> list[SFCRequest]:
    """Sort requests by CEF (revenue/data_rate) desc, ties broken by revenue desc.

    Embedding high-CEF requests first maximises revenue captured per unit of
    scarce bandwidth/compute, which is the profit-maximising ordering.
    """
    ordered = sorted(requests, key=lambda r: (-r.cef, -r.revenue))
    for rank, req in enumerate(ordered):
        req.priority = float(len(ordered) - rank)  # higher = earlier
    return ordered
