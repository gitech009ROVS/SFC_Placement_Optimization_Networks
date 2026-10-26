"""The scaling-candidate value object."""
from __future__ import annotations

from dataclasses import dataclass, field

from ..core.enums import ScalingAction


@dataclass
class ScalingCandidate:
    """One possible scaling action with its estimated cost and impact.

    The decision engine builds a flat list of these (up to ~4 per VNF instance),
    filters by feasibility + non-disruption, then picks the minimum-cost survivor.
    """

    action: ScalingAction
    target_vnf_instance_id: str
    target_request_id: str
    new_node_id: str | None = None
    new_cpu: float = 0.0
    additional_instances: int = 0
    remove_instances: int = 0
    estimated_cost: float = 0.0
    disrupted_requests: list[str] = field(default_factory=list)
    downtime_ms: float = 0.0
    feasible: bool = True

    @property
    def is_disruptive(self) -> bool:
        return bool(self.disrupted_requests)

    def describe(self) -> str:
        bits = [f"{self.action.name} on {self.target_request_id}",
                f"cost={self.estimated_cost:.1f}"]
        if self.new_node_id:
            bits.append(f"node={self.new_node_id}")
        if self.new_cpu:
            bits.append(f"cpu={self.new_cpu:.0f}")
        if self.downtime_ms:
            bits.append(f"downtime={self.downtime_ms:.0f}ms")
        return ", ".join(bits)
