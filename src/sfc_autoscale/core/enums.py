"""Enumerations shared across the system."""
from __future__ import annotations

from enum import Enum


class VNFState(str, Enum):
    """Lifecycle states for a VNF instance (Paper 2 + extensions).

    Using ``str`` as a mixin makes the values JSON/YAML/log friendly.
    """

    PENDING = "pending"          # Being initialised, not yet ready
    RUNNING = "running"          # Actively processing traffic
    WAITING = "waiting"          # Cached/idle, minimal resources, quick to reactivate
    TERMINATING = "terminating"  # Being shut down


class ScalingAction(str, Enum):
    """All scaling actions the decision engine can choose from."""

    REUSE = "reuse"        # Existing capacity is sufficient; no change
    H_OUT = "h_out"        # Horizontal scale-out: add VNF instance(s)
    H_IN = "h_in"          # Horizontal scale-in: remove excess instance(s)
    V_UP = "v_up"          # Vertical scale-up: grow an instance's resources
    V_DOWN = "v_down"      # Vertical scale-down: shrink an instance's resources
    REPLACE = "replace"    # Migrate a VNF to a more cost-efficient node

    @property
    def is_scale_up(self) -> bool:
        return self in (ScalingAction.H_OUT, ScalingAction.V_UP)

    @property
    def is_scale_down(self) -> bool:
        return self in (ScalingAction.H_IN, ScalingAction.V_DOWN)
