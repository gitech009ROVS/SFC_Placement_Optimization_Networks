"""Cost model for scaling actions.

All constants are injected via :class:`CostConfig` so they can be tuned from a
YAML file or calibrated against measured data without touching code. Costs are
in abstract "cost units"; scale-down actions can return negative costs (savings).
"""
from __future__ import annotations

from dataclasses import dataclass

from ..core.entities import PhysicalNode, VNFInstance, VNFType


@dataclass
class CostConfig:
    deploy_new_vnf: float = 10.0      # one-time spin-up of a VNF instance
    per_cpu_unit: float = 1.0         # OPEX per MIPS per step
    downtime_per_ms: float = 0.5      # penalty per ms of service interruption
    migration: float = 15.0           # one-time VNF migration cost
    load_balancer: float = 3.0        # cost of an LB in front of H_OUT replicas
    restart_ms: float = 500.0         # downtime of a vertical resize / restart
    migrate_ms: float = 1000.0        # downtime of a migration
    hin_risk_per_instance: float = 2.0  # risk penalty for removing capacity


class ScalingCostModel:
    def __init__(self, config: CostConfig | None = None):
        self.cfg = config or CostConfig()

    def reuse(self, idle_cpu_cost: float = 0.0) -> float:
        """REUSE on scale-up is free; on scale-down it is the retained waste."""
        return idle_cpu_cost

    def h_out(self, vnf_type: VNFType, data_rate: float, target: PhysicalNode,
              num_new: int = 1) -> float:
        cpu = vnf_type.cpu_for_rate(data_rate)
        deployment = self.cfg.deploy_new_vnf * num_new
        ongoing = cpu * target.cost_per_cpu_unit * num_new
        return deployment + self.cfg.load_balancer + ongoing

    def v_up(self, current_cpu: float, new_cpu: float, node: PhysicalNode) -> float:
        delta = max(0.0, new_cpu - current_cpu)
        return delta * node.cost_per_cpu_unit + self.cfg.restart_ms * self.cfg.downtime_per_ms

    def v_down(self, current_cpu: float, new_cpu: float, node: PhysicalNode) -> float:
        savings = max(0.0, current_cpu - new_cpu) * node.cost_per_cpu_unit
        return self.cfg.restart_ms * self.cfg.downtime_per_ms - savings

    def replace(self, instance: VNFInstance, old: PhysicalNode, new: PhysicalNode,
                new_cpu: float) -> float:
        new_ongoing = new_cpu * new.cost_per_cpu_unit
        old_savings = instance.allocated_cpu * old.cost_per_cpu_unit
        return (self.cfg.migration + new_ongoing - old_savings
                + self.cfg.migrate_ms * self.cfg.downtime_per_ms)

    def h_in(self, instance: VNFInstance, node: PhysicalNode, num_removed: int = 1) -> float:
        savings = instance.allocated_cpu * node.cost_per_cpu_unit
        risk = self.cfg.hin_risk_per_instance * num_removed
        return -(savings - risk)  # negative = net saving
