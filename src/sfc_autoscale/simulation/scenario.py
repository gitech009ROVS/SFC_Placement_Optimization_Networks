"""Scenario = substrate + request set + workload + policy knobs.

A scenario is the fully-specified, reproducible input to a simulation run.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..core.entities import SFCRequest
from ..core.topology import NetworkTopology
from ..scaling import CostConfig, ScalingConfig
from ..topologies import build_sample_topology, build_stress_topology
from .workload import Workload, build_workload


def default_requests() -> list[SFCRequest]:
    """A small, heterogeneous request set for demos and tests."""
    return [
        SFCRequest("R1", data_rate=100.0, ingress_node="N0", egress_node="N9",
                   vnf_chain=["NAT", "FW", "TM", "VOC", "IDPS"],
                   max_delay=500.0, revenue=1000.0),
        SFCRequest("R2", data_rate=64.0, ingress_node="N0", egress_node="N4",
                   vnf_chain=["NAT", "FW", "TM"], max_delay=300.0, revenue=800.0),
        SFCRequest("R3", data_rate=120.0, ingress_node="N1", egress_node="N9",
                   vnf_chain=["NAT", "FW", "VOC", "IDPS"], max_delay=400.0, revenue=1500.0),
        SFCRequest("R4", data_rate=50.0, ingress_node="N3", egress_node="N7",
                   vnf_chain=["NAT", "FW", "TM"], max_delay=350.0, revenue=600.0),
    ]


def stress_requests(max_delay: float = 19.0) -> list[SFCRequest]:
    """Short-path requests with *tight* delay budgets. Paths are only ~2 hops so
    the fixed link delay is a small constant and the demand-sensitive instance
    processing term dominates the budget — the part scaling can actually control.
    """
    return [
        SFCRequest("R1", data_rate=120.0, ingress_node="N0", egress_node="N2",
                   vnf_chain=["NAT", "FW", "VOC"], max_delay=max_delay, revenue=1600.0),
        SFCRequest("R2", data_rate=90.0, ingress_node="N1", egress_node="N2",
                   vnf_chain=["NAT", "FW"], max_delay=max_delay, revenue=1100.0),
        SFCRequest("R3", data_rate=100.0, ingress_node="N0", egress_node="N2",
                   vnf_chain=["NAT", "VOC", "IDPS"], max_delay=max_delay, revenue=1300.0),
    ]


@dataclass
class Scenario:
    network: NetworkTopology
    requests: list[SFCRequest]
    workload: Workload
    num_steps: int = 48
    k_paths: int = 5
    bw_cost_per_unit: float = 1.0
    retrain_every: int = 10
    scaling_config: ScalingConfig = field(default_factory=ScalingConfig)

    @classmethod
    def default(
        cls,
        workload_name: str = "diurnal_spike",
        num_steps: int = 48,
        seed: int = 0,
        change_threshold: float = 0.05,
    ) -> Scenario:
        return cls(
            network=build_sample_topology(),
            requests=default_requests(),
            workload=build_workload(workload_name, seed=seed),
            num_steps=num_steps,
            scaling_config=ScalingConfig(change_threshold=change_threshold, cost=CostConfig()),
        )

    @classmethod
    def stress(
        cls,
        num_steps: int = 120,
        seed: int = 0,
        amplitude: float = 0.7,
        period: int = 24,
        max_delay: float = 19.0,
        change_threshold: float = 0.05,
    ) -> Scenario:
        """A QoS-tight scenario: ample-CPU substrate + steep diurnal demand +
        tight delay budgets.

        Built to make the proactive-vs-reactive difference observable. On each
        daily ramp, demand climbs fast enough that a policy which sizes only for
        what it has already seen falls one step behind and breaches QoS, while a
        policy that provisions for the *forecast* stays ahead. CPU is plentiful,
        so the difference is about *timing*, not capacity.
        """
        return cls(
            network=build_stress_topology(),
            requests=stress_requests(max_delay=max_delay),
            workload=build_workload("diurnal", seed=seed, period=period,
                                    amplitude=amplitude, noise=0.02),
            num_steps=num_steps,
            scaling_config=ScalingConfig(change_threshold=change_threshold, cost=CostConfig()),
        )
