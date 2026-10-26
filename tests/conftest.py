"""Shared pytest fixtures."""
from __future__ import annotations

import pytest

from sfc_autoscale.core.entities import SFCRequest
from sfc_autoscale.simulation.scenario import default_requests
from sfc_autoscale.topologies import build_sample_topology


@pytest.fixture
def network():
    return build_sample_topology()


@pytest.fixture
def requests() -> list[SFCRequest]:
    return default_requests()
