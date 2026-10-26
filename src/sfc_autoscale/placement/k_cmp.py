"""K-CMP: K Cost-Minimising Paths (Paper 1, Section IV-C).

Finds up to K lowest-cost ingress->egress paths after pruning links that lack
bandwidth for the request. Link cost blends a base bandwidth price with a
contention penalty so congested links are avoided even when feasible.
"""
from __future__ import annotations

import heapq

from ..core.entities import SFCRequest
from ..core.topology import NetworkTopology

INF = float("inf")


class KCMP:
    def __init__(self, network: NetworkTopology, k: int = 5, bw_cost_per_unit: float = 1.0):
        self.network = network
        self.k = k
        self.bw_cost_per_unit = bw_cost_per_unit

    def _edge_cost(self, src: str, dst: str, data_rate: float) -> float:
        link = self.network.get_link(src, dst)
        if link is None or link.bandwidth_available < data_rate:
            return INF
        cost = data_rate * self.bw_cost_per_unit
        if link.load > 0:
            cost *= 1.0 + link.load / (1.0 - min(link.load, 0.99))
        return cost

    def find_paths(self, request: SFCRequest) -> list[tuple[float, list[str]]]:
        """Return up to K ``(cost, path)`` pairs, cheapest first (loopless)."""
        source, target = request.ingress_node, request.egress_node
        if source not in self.network.nodes or target not in self.network.nodes:
            return []

        results: list[tuple[float, list[str]]] = []
        seen_paths: set[tuple[str, ...]] = set()
        # Priority queue of (cost, path); tie-break on path to keep heap stable.
        frontier: list[tuple[float, list[str]]] = [(0.0, [source])]

        while frontier and len(results) < self.k:
            cost, path = heapq.heappop(frontier)
            node = path[-1]

            if node == target:
                key = tuple(path)
                if key not in seen_paths:
                    seen_paths.add(key)
                    results.append((cost, path))
                continue

            for nbr in self.network.neighbors(node):
                if nbr in path:  # loopless
                    continue
                ec = self._edge_cost(node, nbr, request.data_rate)
                if ec == INF:
                    continue
                heapq.heappush(frontier, (cost + ec, path + [nbr]))

        return results
