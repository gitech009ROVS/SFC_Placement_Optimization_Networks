"""DAVE: Delay-Aware VNF Embedding (Paper 1, Section IV-D).

Given candidate paths from K-CMP, map the VNF chain onto computing nodes along
each path, check resource + delay feasibility, and return the cheapest feasible
embedding.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..core.entities import SFCRequest
from ..core.topology import NetworkTopology

INF = float("inf")


@dataclass
class Embedding:
    """A concrete, feasible placement of a request's chain onto a path."""

    placement: list[tuple[str, str]]  # (vnf_type_id, node_id) in chain order
    path: list[str]
    cost: float
    delay: float
    cpu_by_node: dict[str, float] = field(default_factory=dict)


class DAVE:
    def __init__(self, network: NetworkTopology, bw_cost_per_unit: float = 1.0):
        self.network = network
        self.bw_cost_per_unit = bw_cost_per_unit

    def _map_chain_to_nodes(
        self, vnf_chain: list[str], computing_nodes: list[str]
    ) -> list[tuple[str, str]] | None:
        """Assign each VNF in order to a computing node along the path.

        If nodes outnumber VNFs, one VNF per node; otherwise VNFs are co-located
        as evenly as possible while preserving chain order.
        """
        if not computing_nodes:
            return None
        n_vnf, n_nodes = len(vnf_chain), len(computing_nodes)
        placement: list[tuple[str, str]] = []
        if n_nodes >= n_vnf:
            for i, vnf in enumerate(vnf_chain):
                placement.append((vnf, computing_nodes[i]))
        else:
            per_node, remainder = divmod(n_vnf, n_nodes)
            idx = 0
            for i, node in enumerate(computing_nodes):
                count = per_node + (1 if i < remainder else 0)
                for _ in range(count):
                    if idx < n_vnf:
                        placement.append((vnf_chain[idx], node))
                        idx += 1
        return placement

    def embed_on_path(self, request: SFCRequest, path: list[str]) -> Embedding | None:
        net = self.network
        computing = [n for n in path if net.nodes[n].is_computing]
        placement = self._map_chain_to_nodes(request.vnf_chain, computing)
        if placement is None:
            return None

        # Aggregate CPU demand per node and check capacity.
        cpu_by_node: dict[str, float] = {}
        for vnf_id, node_id in placement:
            vt = net.vnf_types[vnf_id]
            cpu_by_node[node_id] = cpu_by_node.get(node_id, 0.0) + vt.cpu_for_rate(request.data_rate)
        for node_id, need in cpu_by_node.items():
            if net.nodes[node_id].cpu_available < need:
                return None

        # Link bandwidth feasibility.
        for a, b in zip(path, path[1:], strict=False):
            link = net.get_link(a, b)
            if link is None or link.bandwidth_available < request.data_rate:
                return None

        # Cost = compute OPEX + bandwidth cost.
        cost = sum(
            net.vnf_types[vnf_id].cpu_for_rate(request.data_rate)
            * net.nodes[node_id].cost_per_cpu_unit
            for vnf_id, node_id in placement
        )
        cost += (len(path) - 1) * request.data_rate * self.bw_cost_per_unit

        # Pre-allocation delay estimate.
        delay = 0.0
        for vnf_id, node_id in placement:
            cpu = net.vnf_types[vnf_id].cpu_for_rate(request.data_rate)
            delay += net.processing_delay(node_id, request.data_rate, cpu)
        for a, b in zip(path, path[1:], strict=False):
            delay += net.link_delay(a, b, request.data_rate, request.data_rate)

        if delay > request.max_delay:
            return None

        return Embedding(placement=placement, path=path, cost=cost, delay=delay,
                         cpu_by_node=cpu_by_node)

    def select_best(
        self, request: SFCRequest, candidate_paths: list[tuple[float, list[str]]]
    ) -> Embedding | None:
        feasible = [
            emb
            for _, path in candidate_paths
            if (emb := self.embed_on_path(request, path)) is not None
        ]
        if not feasible:
            return None
        return min(feasible, key=lambda e: e.cost)
