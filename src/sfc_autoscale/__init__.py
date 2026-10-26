"""sfc-autoscale: predictive, contention-aware SFC embedding with multi-strategy
VNF auto-scaling for NFV-based 5G core networks.

Public API re-exports the pieces most users need; submodules hold the rest.
"""
from __future__ import annotations

__version__ = "0.1.0"

from .core import (
    NetworkTopology,
    PhysicalLink,
    PhysicalNode,
    ScalingAction,
    SFCRequest,
    VNFInstance,
    VNFState,
    VNFType,
)
from .orchestration import SFCOrchestrator, StepReport
from .placement import InitialPlacement, PlacementResult
from .prediction import available_predictors, build_predictor
from .scaling import ScalingConfig, ScalingDecisionEngine, ScalingExecutor
from .simulation import Scenario, SimulationRunner, build_workload
from .topologies import build_sample_topology

__all__ = [
    "__version__",
    "NetworkTopology",
    "PhysicalNode",
    "PhysicalLink",
    "VNFType",
    "VNFInstance",
    "SFCRequest",
    "VNFState",
    "ScalingAction",
    "InitialPlacement",
    "PlacementResult",
    "build_predictor",
    "available_predictors",
    "ScalingDecisionEngine",
    "ScalingExecutor",
    "ScalingConfig",
    "SFCOrchestrator",
    "StepReport",
    "Scenario",
    "SimulationRunner",
    "build_workload",
    "build_sample_topology",
]
