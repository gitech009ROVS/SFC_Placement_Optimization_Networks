"""Reusable substrate + VNF-catalogue builders for experiments."""
from __future__ import annotations

from ..core.entities import PhysicalLink, PhysicalNode, VNFType
from ..core.topology import NetworkTopology

# A small catalogue of 5G-core VNF types (Paper 1 Table-style parameters).
DEFAULT_VNF_TYPES: dict[str, VNFType] = {
    "NAT": VNFType("NAT", "NAT", cpu_per_unit_traffic=2.0, min_cpu=200, max_cpu=2000),
    "FW": VNFType("FW", "Firewall", cpu_per_unit_traffic=3.0, min_cpu=300, max_cpu=2500),
    "TM": VNFType("TM", "Traffic Monitor", cpu_per_unit_traffic=1.5, min_cpu=150, max_cpu=1500),
    "VOC": VNFType("VOC", "Video Optimizer", cpu_per_unit_traffic=4.0, min_cpu=400, max_cpu=3000),
    "IDPS": VNFType("IDPS", "IDS/IPS", cpu_per_unit_traffic=3.5, min_cpu=350, max_cpu=2800),
}


def register_vnf_types(net: NetworkTopology, types: dict[str, VNFType] | None = None) -> None:
    for vt in (types or DEFAULT_VNF_TYPES).values():
        net.add_vnf_type(vt)


def build_sample_topology(
    num_nodes: int = 10,
    node_cpu: float = 10000.0,
    link_bw: float = 10000.0,
    switch_nodes: tuple[int, ...] = (2, 5),
) -> NetworkTopology:
    """A 10-node ring-with-chords core network (the paper's demo substrate)."""
    net = NetworkTopology()
    register_vnf_types(net)
    for i in range(num_nodes):
        net.add_node(PhysicalNode(
            node_id=f"N{i}",
            cpu_capacity=node_cpu,
            cost_per_cpu_unit=1.0 + (i % 3) * 0.5,
            is_computing=(i not in switch_nodes),
        ))
    link_pairs = [
        (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6),
        (6, 7), (7, 8), (8, 9), (4, 9), (1, 6), (3, 8),
    ]
    for a, b in link_pairs:
        if a < num_nodes and b < num_nodes:
            net.add_link(PhysicalLink(
                src=f"N{a}", dst=f"N{b}",
                bandwidth_capacity=link_bw,
                distance_km=50.0 + (a + b) * 10,
            ))
    return net


def build_stress_topology(
    num_nodes: int = 8,
    node_cpu: float = 200_000.0,
    link_bw: float = 100_000.0,
    service_delay_ms: float = 3.0,
    switch_nodes: tuple[int, ...] = (3,),
) -> NetworkTopology:
    """A QoS-tight substrate for the policy comparison.

    The scarcity here is the *delay budget*, not CPU: nodes have ample capacity so
    scaling is always feasible, while a larger per-VNF service time makes the
    instance-queue term dominate the end-to-end delay. Under-provisioning during a
    traffic rise then pushes a VNF's utilisation toward saturation and breaches the
    budget — which is exactly what scaling (and especially *proactive* scaling)
    must prevent. VNF types are registered with the given ``service_delay_ms``.
    """
    net = NetworkTopology()
    stress_types = {
        tid: VNFType(vt.type_id, vt.name, vt.cpu_per_unit_traffic,
                     vt.min_cpu, max_cpu=20_000.0, service_delay_ms=service_delay_ms)
        for tid, vt in DEFAULT_VNF_TYPES.items()
    }
    for vt in stress_types.values():
        net.add_vnf_type(vt)
    for i in range(num_nodes):
        net.add_node(PhysicalNode(
            node_id=f"N{i}",
            cpu_capacity=node_cpu,
            cost_per_cpu_unit=1.0 + (i % 3) * 0.5,
            is_computing=(i not in switch_nodes),
        ))
    link_pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7),
                  (0, 4), (1, 5), (2, 6)]
    for a, b in link_pairs:
        if a < num_nodes and b < num_nodes:
            net.add_link(PhysicalLink(src=f"N{a}", dst=f"N{b}",
                                      bandwidth_capacity=link_bw, distance_km=20.0))
    return net


def build_fat_tree(k: int = 4, node_cpu: float = 8000.0, link_bw: float = 10000.0) -> NetworkTopology:
    """A k-ary fat-tree (common DC/edge topology). Core+agg switch-only; edge hosts compute."""
    net = NetworkTopology()
    register_vnf_types(net)
    pods = k
    half = k // 2

    def add(node_id: str, computing: bool) -> None:
        net.add_node(PhysicalNode(node_id, cpu_capacity=node_cpu,
                                  cost_per_cpu_unit=1.0, is_computing=computing))

    core = [f"core{i}" for i in range((k // 2) ** 2)]
    for c in core:
        add(c, computing=False)
    for p in range(pods):
        aggs = [f"agg{p}_{i}" for i in range(half)]
        edges = [f"edge{p}_{i}" for i in range(half)]
        for a in aggs:
            add(a, computing=False)
        for e in edges:
            add(e, computing=True)
        for ai, a in enumerate(aggs):
            for e in edges:
                net.add_link(PhysicalLink(a, e, bandwidth_capacity=link_bw, distance_km=5))
            for ci in range(half):
                net.add_link(PhysicalLink(a, core[ai * half + ci],
                                          bandwidth_capacity=link_bw, distance_km=20))
    return net
