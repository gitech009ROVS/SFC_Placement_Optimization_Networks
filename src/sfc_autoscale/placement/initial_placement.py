"""Phase 1 orchestration: CEF-PS -> K-CMP -> DAVE -> commit."""
from __future__ import annotations

from dataclasses import dataclass, field

from ..core.entities import SFCRequest, VNFInstance, VNFState
from ..core.topology import NetworkTopology
from .cef_ps import cef_priority_sort
from .dave import DAVE, Embedding
from .k_cmp import KCMP


@dataclass
class PlacementResult:
    placed: list[SFCRequest] = field(default_factory=list)
    rejected: list[SFCRequest] = field(default_factory=list)
    total_profit: float = 0.0

    @property
    def acceptance_ratio(self) -> float:
        total = len(self.placed) + len(self.rejected)
        return 0.0 if total == 0 else len(self.placed) / total


class InitialPlacement:
    """Place a batch of SFC requests under constant data rates (Paper 1)."""

    def __init__(self, network: NetworkTopology, k: int = 5, bw_cost_per_unit: float = 1.0):
        self.network = network
        self.k_cmp = KCMP(network, k=k, bw_cost_per_unit=bw_cost_per_unit)
        self.dave = DAVE(network, bw_cost_per_unit=bw_cost_per_unit)

    def place_all(self, requests: list[SFCRequest]) -> PlacementResult:
        result = PlacementResult()
        for request in cef_priority_sort(requests):
            paths = self.k_cmp.find_paths(request)
            embedding = self.dave.select_best(request, paths) if paths else None
            if embedding is None:
                result.rejected.append(request)
                continue
            self._commit(request, embedding)
            result.placed.append(request)
            result.total_profit += request.revenue - embedding.cost
        return result

    def _commit(self, request: SFCRequest, embedding: Embedding) -> None:
        net = self.network
        instance_ids: list[str] = []
        for vnf_id, node_id in embedding.placement:
            vt = net.vnf_types[vnf_id]
            cpu = vt.cpu_for_rate(request.data_rate)
            inst = VNFInstance(
                instance_id=f"{request.request_id}:{vnf_id}@{node_id}",
                vnf_type=vt,
                node_id=node_id,
                allocated_cpu=cpu,
                state=VNFState.RUNNING,
                current_load=request.data_rate,
                serving_requests=[request.request_id],
            )
            net.vnf_instances[inst.instance_id] = inst
            net.allocate_on_node(node_id, cpu)
            instance_ids.append(inst.instance_id)

        path = embedding.path
        for a, b in zip(path, path[1:], strict=False):
            net.allocate_on_link(a, b, request.data_rate)

        request.placed_vnf_instances = instance_ids
        request.placed_path = path
        request.is_active = True
        net.active_requests[request.request_id] = request
