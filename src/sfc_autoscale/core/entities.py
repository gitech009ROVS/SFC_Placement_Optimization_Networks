"""Domain entities: physical substrate, VNF templates/instances, and SFC requests.

All entities are plain dataclasses with light invariant checking in
``__post_init__``. They carry no behaviour that depends on the rest of the
network — relationships (delay, contention) live in :mod:`topology`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .enums import VNFState


@dataclass
class PhysicalNode:
    """A computing (or switching-only) node in the 5G core substrate."""

    node_id: str
    cpu_capacity: float              # Total compute capacity (MIPS)
    cpu_used: float = 0.0            # Currently allocated MIPS
    memory_capacity: float = 0.0     # Total memory (MB)
    memory_used: float = 0.0
    cost_per_cpu_unit: float = 1.0   # OPEX per MIPS per time step
    latitude: float = 0.0
    longitude: float = 0.0
    is_computing: bool = True        # False => switching-only (no VNF hosting)

    def __post_init__(self) -> None:
        if self.cpu_capacity < 0:
            raise ValueError(f"{self.node_id}: cpu_capacity must be >= 0")
        if self.cost_per_cpu_unit < 0:
            raise ValueError(f"{self.node_id}: cost_per_cpu_unit must be >= 0")

    @property
    def cpu_available(self) -> float:
        return self.cpu_capacity - self.cpu_used

    @property
    def load(self) -> float:
        """Fractional CPU load, used by the contention model (Paper 1, Eq.9)."""
        if self.cpu_capacity == 0:
            return 0.0
        return self.cpu_used / self.cpu_capacity


@dataclass
class PhysicalLink:
    """A directed fibre link between two nodes (stored per direction)."""

    src: str
    dst: str
    bandwidth_capacity: float        # Total bandwidth (Mbps)
    bandwidth_used: float = 0.0
    distance_km: float = 0.0
    propagation_speed_km_s: float = 2.0e5  # ~speed of light in fibre

    def __post_init__(self) -> None:
        if self.bandwidth_capacity < 0:
            raise ValueError(f"{self.src}->{self.dst}: bandwidth_capacity must be >= 0")

    @property
    def bandwidth_available(self) -> float:
        return self.bandwidth_capacity - self.bandwidth_used

    @property
    def load(self) -> float:
        """Fractional bandwidth load (Paper 1, Eq.7)."""
        if self.bandwidth_capacity == 0:
            return 0.0
        return self.bandwidth_used / self.bandwidth_capacity

    @property
    def propagation_delay_ms(self) -> float:
        """Propagation delay in ms (Paper 1, Eq.5)."""
        if self.propagation_speed_km_s == 0:
            return 0.0
        return (self.distance_km / self.propagation_speed_km_s) * 1000.0


#: Default provisioning target utilisation — size instances so that, at the rate
#: they are provisioned for, they run at this utilisation (leaving headroom).
DEFAULT_TARGET_UTIL = 0.6


@dataclass(frozen=True)
class VNFType:
    """Template describing a class of VNF (e.g. NAT, FW, IDPS)."""

    type_id: str
    name: str
    cpu_per_unit_traffic: float      # MIPS required per Mbps of throughput
    min_cpu: float = 200.0           # Floor for a functioning instance
    max_cpu: float = 2000.0          # Ceiling for vertical scaling
    base_processing_rate: float = 100.0
    service_delay_ms: float = 0.5    # nominal per-instance processing time (M/M/1 t_service)

    def __post_init__(self) -> None:
        if self.cpu_per_unit_traffic <= 0:
            raise ValueError(f"{self.type_id}: cpu_per_unit_traffic must be > 0")
        if self.min_cpu > self.max_cpu:
            raise ValueError(f"{self.type_id}: min_cpu must be <= max_cpu")

    def cpu_for_rate(self, data_rate: float, target_util: float = DEFAULT_TARGET_UTIL) -> float:
        """CPU to serve ``data_rate`` Mbps at ``target_util`` utilisation.

        Dividing by ``target_util`` provisions headroom, so a freshly sized
        instance sits at ρ=``target_util`` (not ρ=1), leaving room to absorb
        normal jitter before a scaling action is needed.
        """
        raw = self.cpu_per_unit_traffic * data_rate / max(1e-6, target_util)
        return min(self.max_cpu, max(self.min_cpu, raw))

    def max_throughput(self, allocated_cpu: float) -> float:
        """Maximum Mbps an instance with ``allocated_cpu`` MIPS can serve."""
        return allocated_cpu / self.cpu_per_unit_traffic


@dataclass
class VNFInstance:
    """A concrete VNF instance deployed on a node."""

    instance_id: str
    vnf_type: VNFType
    node_id: str
    allocated_cpu: float
    allocated_memory: float = 0.0
    state: VNFState = VNFState.RUNNING
    current_load: float = 0.0        # Mbps currently handled by this instance
    serving_requests: list[str] = field(default_factory=list)
    is_replica: bool = False         # True if created by an H_OUT action

    @property
    def max_throughput(self) -> float:
        return self.vnf_type.max_throughput(self.allocated_cpu)

    @property
    def remaining_capacity(self) -> float:
        return max(0.0, self.max_throughput - self.current_load)

    @property
    def utilization(self) -> float:
        mt = self.max_throughput
        return 0.0 if mt == 0 else self.current_load / mt


@dataclass
class SFCRequest:
    """A service-function-chain request from a tenant/user."""

    request_id: str
    data_rate: float                 # λ_r : current data rate (Mbps)
    ingress_node: str
    egress_node: str
    vnf_chain: list[str]             # Ordered VNF type IDs
    max_delay: float                 # E2E delay budget (ms)
    revenue: float                   # Revenue if embedded

    priority: float = 0.0            # Set by CEF-PS
    placed_vnf_instances: list[str] = field(default_factory=list)
    placed_path: list[str] = field(default_factory=list)
    is_active: bool = False

    def __post_init__(self) -> None:
        if self.data_rate < 0:
            raise ValueError(f"{self.request_id}: data_rate must be >= 0")
        if not self.vnf_chain:
            raise ValueError(f"{self.request_id}: vnf_chain must be non-empty")

    @property
    def cef(self) -> float:
        """Cost Efficiency Factor (Paper 1, Eq.22): revenue / data_rate."""
        if self.data_rate == 0:
            return float("inf")
        return self.revenue / self.data_rate
