"""Tests for substrate state, allocation, and the simulate-and-rollback check."""
from __future__ import annotations

from sfc_autoscale.core.entities import PhysicalLink, PhysicalNode
from sfc_autoscale.core.topology import NetworkTopology


def test_bidirectional_links_and_neighbors():
    net = NetworkTopology()
    net.add_node(PhysicalNode("A", cpu_capacity=1000))
    net.add_node(PhysicalNode("B", cpu_capacity=1000))
    net.add_link(PhysicalLink("A", "B", bandwidth_capacity=100, distance_km=10))
    assert net.get_link("A", "B") is not None
    assert net.get_link("B", "A") is not None
    assert net.neighbors("A") == ["B"]
    assert net.neighbors("B") == ["A"]


def test_allocation_roundtrip():
    net = NetworkTopology()
    net.add_node(PhysicalNode("A", cpu_capacity=1000))
    net.allocate_on_node("A", 300)
    assert net.nodes["A"].cpu_used == 300
    assert net.nodes["A"].cpu_available == 700
    net.deallocate_on_node("A", 1000)  # clamps at zero
    assert net.nodes["A"].cpu_used == 0


def test_simulate_delays_does_not_mutate_state(network, requests):
    from sfc_autoscale.placement import InitialPlacement

    InitialPlacement(network).place_all(requests)
    node_id = next(iter(network.nodes))
    before = network.nodes[node_id].cpu_used
    _ = network.simulate_delays(node_id, extra_cpu=5000)
    assert network.nodes[node_id].cpu_used == before  # rolled back
