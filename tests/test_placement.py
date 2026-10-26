"""Tests for Phase-1 placement (CEF-PS, K-CMP, DAVE)."""
from __future__ import annotations

from sfc_autoscale.core.entities import SFCRequest
from sfc_autoscale.placement import InitialPlacement, cef_priority_sort
from sfc_autoscale.placement.k_cmp import KCMP


def test_cef_sort_orders_by_efficiency():
    a = SFCRequest("A", data_rate=100, ingress_node="N0", egress_node="N1",
                   vnf_chain=["NAT"], max_delay=100, revenue=1000)  # CEF 10
    b = SFCRequest("B", data_rate=100, ingress_node="N0", egress_node="N1",
                   vnf_chain=["NAT"], max_delay=100, revenue=500)   # CEF 5
    ordered = cef_priority_sort([b, a])
    assert [r.request_id for r in ordered] == ["A", "B"]
    assert ordered[0].priority > ordered[1].priority


def test_kcmp_finds_paths(network, requests):
    paths = KCMP(network, k=5).find_paths(requests[0])
    assert paths, "expected at least one ingress->egress path"
    cost, path = paths[0]
    assert path[0] == requests[0].ingress_node
    assert path[-1] == requests[0].egress_node


def test_initial_placement_commits_resources(network, requests):
    result = InitialPlacement(network).place_all(requests)
    assert len(result.placed) >= 1
    for req in result.placed:
        assert req.is_active
        assert req.placed_vnf_instances
        assert network.e2e_delay(req) <= req.max_delay
    # Allocation actually happened somewhere.
    assert any(n.cpu_used > 0 for n in network.nodes.values())


def test_acceptance_ratio_bounds(network, requests):
    result = InitialPlacement(network).place_all(requests)
    assert 0.0 <= result.acceptance_ratio <= 1.0
