"""Phase 4 — apply a chosen scaling action to the substrate."""
from __future__ import annotations

from ..core.entities import VNFInstance, VNFState
from ..core.enums import ScalingAction
from ..core.topology import NetworkTopology
from .candidates import ScalingCandidate


class ScalingExecutor:
    def __init__(self, network: NetworkTopology):
        self.network = network
        self._replica_counter = 0

    def execute(self, decision: ScalingCandidate) -> bool:
        if not decision.feasible:
            return False
        handler = {
            ScalingAction.REUSE: self._reuse,
            ScalingAction.V_UP: self._v_up,
            ScalingAction.V_DOWN: self._v_down,
            ScalingAction.H_OUT: self._h_out,
            ScalingAction.H_IN: self._h_in,
            ScalingAction.REPLACE: self._replace,
        }.get(decision.action)
        return bool(handler and handler(decision))

    # ------------------------------------------------------------ handlers
    def _reuse(self, _: ScalingCandidate) -> bool:
        return True

    def _v_up(self, d: ScalingCandidate) -> bool:
        inst = self.network.vnf_instances.get(d.target_vnf_instance_id)
        if inst is None:
            return False
        delta = d.new_cpu - inst.allocated_cpu
        if self.network.nodes[inst.node_id].cpu_available < delta:
            return False
        inst.state = VNFState.PENDING
        self.network.allocate_on_node(inst.node_id, delta)
        inst.allocated_cpu = d.new_cpu
        inst.state = VNFState.RUNNING
        return True

    def _v_down(self, d: ScalingCandidate) -> bool:
        inst = self.network.vnf_instances.get(d.target_vnf_instance_id)
        if inst is None:
            return False
        delta = inst.allocated_cpu - d.new_cpu
        inst.state = VNFState.PENDING
        self.network.deallocate_on_node(inst.node_id, delta)
        inst.allocated_cpu = d.new_cpu
        inst.state = VNFState.RUNNING
        return True

    def _h_out(self, d: ScalingCandidate) -> bool:
        parent = self.network.vnf_instances.get(d.target_vnf_instance_id)
        new_node = self.network.nodes.get(d.new_node_id) if d.new_node_id else None
        req = self.network.active_requests.get(d.target_request_id)
        if parent is None or new_node is None or req is None:
            return False

        vt = parent.vnf_type
        replica_cpu = vt.cpu_for_rate(req.data_rate / 2)
        self._replica_counter += 1
        replica = VNFInstance(
            instance_id=f"{parent.instance_id}#rep{self._replica_counter}@{new_node.node_id}",
            vnf_type=vt,
            node_id=new_node.node_id,
            allocated_cpu=replica_cpu,
            state=VNFState.RUNNING,
            current_load=req.data_rate / 2,
            serving_requests=[d.target_request_id],
            is_replica=True,
        )
        self.network.vnf_instances[replica.instance_id] = replica
        self.network.allocate_on_node(new_node.node_id, replica_cpu)
        req.placed_vnf_instances.append(replica.instance_id)
        # Shed half the load from the parent onto the replica.
        parent.current_load = max(0.0, parent.current_load - replica.current_load)
        return True

    def _h_in(self, d: ScalingCandidate) -> bool:
        inst = self.network.vnf_instances.get(d.target_vnf_instance_id)
        if inst is None:
            return False
        self.network.deallocate_on_node(inst.node_id, inst.allocated_cpu)
        inst.state = VNFState.WAITING  # cache rather than destroy (Paper 2)
        inst.allocated_cpu = 0.0
        inst.current_load = 0.0
        req = self.network.active_requests.get(d.target_request_id)
        if req and inst.instance_id in req.placed_vnf_instances:
            req.placed_vnf_instances.remove(inst.instance_id)
        return True

    def _replace(self, d: ScalingCandidate) -> bool:
        inst = self.network.vnf_instances.get(d.target_vnf_instance_id)
        if inst is None or d.new_node_id is None:
            return False
        self.network.deallocate_on_node(inst.node_id, inst.allocated_cpu)
        self.network.allocate_on_node(d.new_node_id, d.new_cpu)
        inst.node_id = d.new_node_id
        inst.allocated_cpu = d.new_cpu
        inst.state = VNFState.RUNNING
        return True
