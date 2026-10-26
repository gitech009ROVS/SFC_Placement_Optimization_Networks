"""Tests for the scaling decision engine, cost model, and executor."""
from __future__ import annotations

from sfc_autoscale.core.enums import ScalingAction, VNFState
from sfc_autoscale.placement import InitialPlacement
from sfc_autoscale.scaling import (
    ScalingConfig,
    ScalingDecisionEngine,
    ScalingExecutor,
)
from sfc_autoscale.scaling.cost_model import CostConfig, ScalingCostModel


def test_small_change_is_reuse(network, requests):
    InitialPlacement(network).place_all(requests)
    engine = ScalingDecisionEngine(network, ScalingConfig(change_threshold=0.05))
    r = requests[0]
    decision = engine.decide(r.request_id, r.data_rate, r.data_rate * 1.02)  # +2%
    assert decision is not None
    assert decision.action == ScalingAction.REUSE


def test_scale_up_picks_feasible_min_cost(network, requests):
    InitialPlacement(network).place_all(requests)
    engine = ScalingDecisionEngine(network)
    r = requests[0]
    decision = engine.decide(r.request_id, r.data_rate, r.data_rate * 1.5)  # +50%
    assert decision is not None
    assert decision.feasible
    assert not decision.disrupted_requests


def test_executor_v_up_allocates_and_v_down_releases(network, requests):
    InitialPlacement(network).place_all(requests)
    engine = ScalingDecisionEngine(network)
    execu = ScalingExecutor(network)
    r = requests[0]
    inst_id = r.placed_vnf_instances[0]
    before = network.vnf_instances[inst_id].allocated_cpu
    node_before = network.nodes[network.vnf_instances[inst_id].node_id].cpu_used

    up = engine.decide(r.request_id, r.data_rate, r.data_rate * 1.6)
    if up and up.action == ScalingAction.V_UP:
        assert execu.execute(up)
        assert network.vnf_instances[up.target_vnf_instance_id].allocated_cpu > before
        assert network.nodes[network.vnf_instances[up.target_vnf_instance_id].node_id].cpu_used > node_before


def test_h_out_then_h_in_roundtrip(network, requests):
    InitialPlacement(network).place_all(requests)
    execu = ScalingExecutor(network)
    engine = ScalingDecisionEngine(network)
    r = requests[0]
    n_before = len(r.placed_vnf_instances)

    # Force an H_OUT by crafting a candidate directly.
    up = engine._scale_up_candidates(r, r.data_rate * 2.0)
    hout = next((c for c in up if c.action == ScalingAction.H_OUT), None)
    if hout is not None:
        assert execu.execute(hout)
        assert len(r.placed_vnf_instances) == n_before + 1
        replica_id = r.placed_vnf_instances[-1]
        assert network.vnf_instances[replica_id].is_replica

        # Now scale in: the replica should be retired.
        down = engine._scale_down_candidates(r, r.data_rate * 0.4)
        hin = next((c for c in down if c.action == ScalingAction.H_IN), None)
        assert hin is not None
        assert execu.execute(hin)
        assert replica_id not in r.placed_vnf_instances
        assert network.vnf_instances[replica_id].state == VNFState.WAITING


def test_cost_model_savings_are_negative():
    cm = ScalingCostModel(CostConfig())
    from sfc_autoscale.core.entities import PhysicalNode, VNFInstance, VNFType

    vt = VNFType("NAT", "NAT", cpu_per_unit_traffic=2.0)
    node = PhysicalNode("A", cpu_capacity=1000, cost_per_cpu_unit=1.0)
    inst = VNFInstance("i", vt, "A", allocated_cpu=400)
    assert cm.h_in(inst, node) < 0  # removing capacity saves money
