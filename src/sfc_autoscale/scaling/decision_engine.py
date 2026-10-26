"""Phase 3 — the scaling decision engine (core contribution).

For one SFC request whose predicted rate has moved beyond a threshold, the engine
enumerates candidate actions per VNF instance, costs each, filters by resource
feasibility and non-disruption (simulate-and-rollback), and returns the cheapest
survivor. It operates on **one request at a time**; the orchestrator feeds
requests sequentially so each decision sees the substrate left by the previous.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..core.entities import SFCRequest
from ..core.enums import ScalingAction, VNFState
from ..core.topology import NetworkTopology
from .candidates import ScalingCandidate
from .cost_model import CostConfig, ScalingCostModel


@dataclass
class ScalingConfig:
    change_threshold: float = 0.05   # ignore moves smaller than 5%
    h_out_replicas: int = 1          # replicas added per H_OUT
    cost: CostConfig | None = None


class ScalingDecisionEngine:
    def __init__(self, network: NetworkTopology, config: ScalingConfig | None = None):
        self.network = network
        self.config = config or ScalingConfig()
        self.cost_model = ScalingCostModel(self.config.cost)

    # ------------------------------------------------------------- public
    def decide(
        self, request_id: str, current_rate: float, predicted_rate: float
    ) -> ScalingCandidate | None:
        request = self.network.active_requests.get(request_id)
        if request is None:
            return None

        rel_change = abs(predicted_rate - current_rate) / max(current_rate, 1e-9)
        if rel_change < self.config.change_threshold:
            return ScalingCandidate(
                action=ScalingAction.REUSE,
                target_vnf_instance_id="",
                target_request_id=request_id,
                estimated_cost=0.0,
            )

        if predicted_rate > current_rate:
            candidates = self._scale_up_candidates(request, predicted_rate)
        else:
            candidates = self._scale_down_candidates(request, predicted_rate)

        return self._select(candidates)

    # ------------------------------------------------------------ selection
    @staticmethod
    def _select(candidates: list[ScalingCandidate]) -> ScalingCandidate | None:
        # Prefer feasible + non-disruptive; fall back to least-disruptive feasible.
        clean = [c for c in candidates if c.feasible and not c.disrupted_requests]
        pool = clean or sorted(
            (c for c in candidates if c.feasible), key=lambda c: len(c.disrupted_requests)
        )
        if not pool:
            return None
        return min(pool, key=lambda c: c.estimated_cost)

    # --------------------------------------------------------- scale up
    def _scale_up_candidates(
        self, request: SFCRequest, predicted_rate: float
    ) -> list[ScalingCandidate]:
        net = self.network
        out: list[ScalingCandidate] = []
        delta_rate = predicted_rate - request.data_rate

        for inst_id in list(request.placed_vnf_instances):
            inst = net.vnf_instances.get(inst_id)
            if inst is None:
                continue
            node = net.nodes[inst.node_id]
            vt = inst.vnf_type
            needed_cpu = vt.cpu_for_rate(predicted_rate)

            # 1) REUSE — spare headroom already covers the increase.
            if inst.remaining_capacity >= delta_rate:
                out.append(ScalingCandidate(
                    action=ScalingAction.REUSE,
                    target_vnf_instance_id=inst_id,
                    target_request_id=request.request_id,
                    estimated_cost=self.cost_model.reuse(),
                ))

            # 2) V_UP — grow this instance on its node.
            new_cpu = max(needed_cpu, inst.allocated_cpu)
            cpu_delta = new_cpu - inst.allocated_cpu
            if cpu_delta > 0 and new_cpu <= vt.max_cpu and node.cpu_available >= cpu_delta:
                out.append(ScalingCandidate(
                    action=ScalingAction.V_UP,
                    target_vnf_instance_id=inst_id,
                    target_request_id=request.request_id,
                    new_cpu=new_cpu,
                    estimated_cost=self.cost_model.v_up(inst.allocated_cpu, new_cpu, node),
                    disrupted_requests=net.would_violate_existing(node.node_id, cpu_delta),
                    downtime_ms=self.cost_model.cfg.restart_ms,
                ))

            # 3) H_OUT — add a replica on the first node with room.
            replica_cpu = vt.cpu_for_rate(predicted_rate / 2)
            for cand in net.computing_nodes():
                if cand.node_id == node.node_id:
                    continue
                if cand.cpu_available >= replica_cpu:
                    out.append(ScalingCandidate(
                        action=ScalingAction.H_OUT,
                        target_vnf_instance_id=inst_id,
                        target_request_id=request.request_id,
                        new_node_id=cand.node_id,
                        additional_instances=self.config.h_out_replicas,
                        estimated_cost=self.cost_model.h_out(vt, predicted_rate / 2, cand),
                        disrupted_requests=net.would_violate_existing(cand.node_id, replica_cpu),
                    ))
                    break

            # 4) REPLACE — migrate to a cheaper node with room.
            for cand in net.computing_nodes():
                if cand.node_id == node.node_id:
                    continue
                if cand.cpu_available >= needed_cpu and cand.cost_per_cpu_unit < node.cost_per_cpu_unit:
                    out.append(ScalingCandidate(
                        action=ScalingAction.REPLACE,
                        target_vnf_instance_id=inst_id,
                        target_request_id=request.request_id,
                        new_node_id=cand.node_id,
                        new_cpu=needed_cpu,
                        estimated_cost=self.cost_model.replace(inst, node, cand, needed_cpu),
                        disrupted_requests=net.would_violate_existing(cand.node_id, needed_cpu),
                        downtime_ms=self.cost_model.cfg.migrate_ms,
                    ))
                    break
        return out

    # ------------------------------------------------------- scale down
    def _scale_down_candidates(
        self, request: SFCRequest, predicted_rate: float
    ) -> list[ScalingCandidate]:
        net = self.network
        out: list[ScalingCandidate] = []
        chain_len = len(request.vnf_chain)
        instance_count = len(request.placed_vnf_instances)

        for inst_id in list(request.placed_vnf_instances):
            inst = net.vnf_instances.get(inst_id)
            if inst is None or inst.state != VNFState.RUNNING:
                continue
            node = net.nodes[inst.node_id]
            vt = inst.vnf_type
            needed_cpu = vt.cpu_for_rate(predicted_rate)

            # 1) REUSE — keep over-provisioned (ongoing waste as its cost).
            out.append(ScalingCandidate(
                action=ScalingAction.REUSE,
                target_vnf_instance_id=inst_id,
                target_request_id=request.request_id,
                estimated_cost=self.cost_model.reuse(
                    idle_cpu_cost=inst.allocated_cpu * node.cost_per_cpu_unit
                ),
            ))

            # 2) V_DOWN — shrink the instance.
            if needed_cpu < inst.allocated_cpu:
                out.append(ScalingCandidate(
                    action=ScalingAction.V_DOWN,
                    target_vnf_instance_id=inst_id,
                    target_request_id=request.request_id,
                    new_cpu=needed_cpu,
                    estimated_cost=self.cost_model.v_down(inst.allocated_cpu, needed_cpu, node),
                    downtime_ms=self.cost_model.cfg.restart_ms,
                ))

            # 3) H_IN — retire a replica created by an earlier H_OUT.
            if inst.is_replica and instance_count > chain_len:
                out.append(ScalingCandidate(
                    action=ScalingAction.H_IN,
                    target_vnf_instance_id=inst_id,
                    target_request_id=request.request_id,
                    remove_instances=1,
                    estimated_cost=self.cost_model.h_in(inst, node),
                ))

            # 4) REPLACE — only if migrating to a cheaper node actually saves.
            for cand in net.computing_nodes():
                if cand.node_id == node.node_id:
                    continue
                if cand.cpu_available >= needed_cpu and cand.cost_per_cpu_unit < node.cost_per_cpu_unit:
                    cost = self.cost_model.replace(inst, node, cand, needed_cpu)
                    if cost < 0:
                        out.append(ScalingCandidate(
                            action=ScalingAction.REPLACE,
                            target_vnf_instance_id=inst_id,
                            target_request_id=request.request_id,
                            new_node_id=cand.node_id,
                            new_cpu=needed_cpu,
                            estimated_cost=cost,
                            downtime_ms=self.cost_model.cfg.migrate_ms,
                        ))
                    break
        return out
