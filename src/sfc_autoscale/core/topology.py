"""The physical 5G-core substrate and the state it carries.

:class:`NetworkTopology` owns nodes, links, VNF instances and active requests,
and exposes resource (de)allocation plus end-to-end delay computation. It is the
single source of truth that every other layer reads from and writes to.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from . import delay as delay_models
from .entities import PhysicalLink, PhysicalNode, SFCRequest, VNFInstance, VNFType

INF = float("inf")


class NetworkTopology:
    def __init__(self) -> None:
        self.nodes: dict[str, PhysicalNode] = {}
        self.links: dict[tuple[str, str], PhysicalLink] = {}
        self.vnf_instances: dict[str, VNFInstance] = {}
        self.vnf_types: dict[str, VNFType] = {}
        self.active_requests: dict[str, SFCRequest] = {}
        self._adjacency: dict[str, set[str]] = {}

    # ------------------------------------------------------------------ build
    def add_vnf_type(self, vnf_type: VNFType) -> None:
        self.vnf_types[vnf_type.type_id] = vnf_type

    def add_node(self, node: PhysicalNode) -> None:
        self.nodes[node.node_id] = node
        self._adjacency.setdefault(node.node_id, set())

    def add_link(self, link: PhysicalLink) -> None:
        """Register a bidirectional link (stored as two directed entries)."""
        self.links[(link.src, link.dst)] = link
        self.links[(link.dst, link.src)] = PhysicalLink(
            src=link.dst,
            dst=link.src,
            bandwidth_capacity=link.bandwidth_capacity,
            bandwidth_used=link.bandwidth_used,
            distance_km=link.distance_km,
            propagation_speed_km_s=link.propagation_speed_km_s,
        )
        self._adjacency.setdefault(link.src, set()).add(link.dst)
        self._adjacency.setdefault(link.dst, set()).add(link.src)

    # ------------------------------------------------------------- accessors
    def get_link(self, src: str, dst: str) -> PhysicalLink | None:
        return self.links.get((src, dst))

    def neighbors(self, node_id: str) -> list[str]:
        return sorted(self._adjacency.get(node_id, set()))

    def computing_nodes(self) -> list[PhysicalNode]:
        return [n for n in self.nodes.values() if n.is_computing]

    # ------------------------------------------------------------ allocation
    def allocate_on_node(self, node_id: str, cpu: float, memory: float = 0.0) -> None:
        node = self.nodes[node_id]
        node.cpu_used += cpu
        node.memory_used += memory

    def deallocate_on_node(self, node_id: str, cpu: float, memory: float = 0.0) -> None:
        node = self.nodes[node_id]
        node.cpu_used = max(0.0, node.cpu_used - cpu)
        node.memory_used = max(0.0, node.memory_used - memory)

    def allocate_on_link(self, src: str, dst: str, bandwidth: float) -> None:
        link = self.get_link(src, dst)
        if link is not None:
            link.bandwidth_used += bandwidth

    def deallocate_on_link(self, src: str, dst: str, bandwidth: float) -> None:
        link = self.get_link(src, dst)
        if link is not None:
            link.bandwidth_used = max(0.0, link.bandwidth_used - bandwidth)

    # ------------------------------------------------------------- delay API
    def processing_delay(self, node_id: str, data_rate: float, allocated_cpu: float) -> float:
        return delay_models.processing_delay(self.nodes[node_id].load, data_rate, allocated_cpu)

    def link_delay(self, src: str, dst: str, data_rate: float, allocated_bw: float) -> float:
        link = self.get_link(src, dst)
        if link is None:
            return INF
        return delay_models.link_delay(
            link.load, link.propagation_delay_ms, data_rate, allocated_bw
        )

    def _offered_share(self, request: SFCRequest) -> dict[str, float]:
        """Offered rate seen by each instance, splitting a request's load evenly
        across same-type replicas (H_OUT)."""
        counts = Counter(
            self.vnf_instances[i].vnf_type.type_id
            for i in request.placed_vnf_instances
            if i in self.vnf_instances
        )
        shares: dict[str, float] = {}
        for i in request.placed_vnf_instances:
            inst = self.vnf_instances.get(i)
            if inst is None:
                continue
            n = counts[inst.vnf_type.type_id] or 1
            shares[i] = request.data_rate / n
        return shares

    def e2e_delay(self, request: SFCRequest) -> float:
        """End-to-end delay for a placed request (Paper 1, Eq.10) with an
        instance-level M/M/1 processing term (the demand-sensitive part)."""
        if not request.placed_path or not request.placed_vnf_instances:
            return INF

        total = 0.0
        shares = self._offered_share(request)
        for inst_id in request.placed_vnf_instances:
            inst = self.vnf_instances.get(inst_id)
            if inst is None:
                continue
            offered = shares.get(inst_id, request.data_rate)
            mt = inst.max_throughput
            rho = 1.0 if mt <= 0 else offered / mt
            total += delay_models.instance_queue_delay(rho, inst.vnf_type.service_delay_ms)
            if total == INF:
                return INF

        path = request.placed_path
        for a, b in zip(path, path[1:], strict=False):
            total += self.link_delay(a, b, request.data_rate, allocated_bw=request.data_rate)
            if total == INF:
                return INF
        return total

    def resize_instance(self, instance_id: str, new_cpu: float) -> bool:
        """Vertically resize an instance to ``new_cpu`` MIPS, updating node usage.

        Returns False if the host node lacks the headroom to grow. Used by the
        provisioning controllers in the policy comparison.
        """
        inst = self.vnf_instances.get(instance_id)
        if inst is None:
            return False
        delta = new_cpu - inst.allocated_cpu
        if delta > 0 and self.nodes[inst.node_id].cpu_available < delta:
            return False
        if delta >= 0:
            self.allocate_on_node(inst.node_id, delta)
        else:
            self.deallocate_on_node(inst.node_id, -delta)
        inst.allocated_cpu = new_cpu
        return True

    def set_request_rate(self, request_id: str, rate: float) -> None:
        """Update a request's data rate *and* each serving instance's current load
        (split across same-type replicas), keeping utilisation bookkeeping honest."""
        req = self.active_requests.get(request_id)
        if req is None:
            return
        req.data_rate = rate
        shares = self._offered_share(req)
        for inst_id, offered in shares.items():
            inst = self.vnf_instances.get(inst_id)
            if inst is not None:
                inst.current_load = offered

    # ---------------------------------------------------- impact assessment
    def simulate_delays(
        self,
        node_id: str,
        extra_cpu: float,
        extra_bw_links: Iterable[tuple[str, str, float]] = (),
    ) -> dict[str, float]:
        """Return each active request's E2E delay *if* the extra load were applied.

        Uses a save/apply/measure/rollback pattern so the real substrate state is
        never mutated — this is the "simulate-and-rollback" non-disruption check.
        """
        node = self.nodes[node_id]
        saved_cpu = node.cpu_used
        node.cpu_used += extra_cpu

        saved_bw: dict[tuple[str, str], float] = {}
        for src, dst, bw in extra_bw_links:
            link = self.get_link(src, dst)
            if link is not None:
                saved_bw[(src, dst)] = link.bandwidth_used
                link.bandwidth_used += bw

        try:
            return {rid: self.e2e_delay(req) for rid, req in self.active_requests.items()}
        finally:
            node.cpu_used = saved_cpu
            for (src, dst), orig in saved_bw.items():
                link = self.get_link(src, dst)
                if link is not None:
                    link.bandwidth_used = orig

    def would_violate_existing(
        self,
        node_id: str,
        extra_cpu: float,
        extra_bw_links: Iterable[tuple[str, str, float]] = (),
    ) -> list[str]:
        """IDs of active requests whose delay budget would be breached by the extra load."""
        delays = self.simulate_delays(node_id, extra_cpu, extra_bw_links)
        return [
            rid
            for rid, d in delays.items()
            if d > self.active_requests[rid].max_delay
        ]

    # --------------------------------------------------------------- summary
    def utilization(self) -> dict[str, float]:
        cap = sum(n.cpu_capacity for n in self.nodes.values())
        used = sum(n.cpu_used for n in self.nodes.values())
        return {
            "cpu_used": used,
            "cpu_capacity": cap,
            "cpu_fraction": 0.0 if cap == 0 else used / cap,
            "num_instances": len(self.vnf_instances),
            "num_active_requests": len(self.active_requests),
        }
