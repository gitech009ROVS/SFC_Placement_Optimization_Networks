"""Core domain model: entities, delay physics, and the network substrate."""
from __future__ import annotations

from .entities import (
    PhysicalLink,
    PhysicalNode,
    SFCRequest,
    VNFInstance,
    VNFType,
)
from .enums import ScalingAction, VNFState
from .topology import NetworkTopology

__all__ = [
    "PhysicalNode",
    "PhysicalLink",
    "VNFType",
    "VNFInstance",
    "SFCRequest",
    "VNFState",
    "ScalingAction",
    "NetworkTopology",
]
