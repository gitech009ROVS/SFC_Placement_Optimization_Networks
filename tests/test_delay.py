"""Unit tests for the contention-aware delay models."""
from __future__ import annotations

import math

import pytest

from sfc_autoscale.core import delay


def test_contention_factor_monotonic_and_bounds():
    assert delay.contention_factor(0.0) == 0.0
    assert delay.contention_factor(0.5) == pytest.approx(1.0)
    assert delay.contention_factor(0.9) == pytest.approx(9.0)
    # Diverges at saturation.
    assert delay.contention_factor(1.0) == math.inf
    assert delay.contention_factor(1.5) == math.inf
    # Monotonic increasing in (0, 1).
    assert delay.contention_factor(0.3) < delay.contention_factor(0.7)


def test_processing_delay_scales_with_load():
    low = delay.processing_delay(0.2, data_rate=100, allocated_cpu=1000)
    high = delay.processing_delay(0.8, data_rate=100, allocated_cpu=1000)
    assert high > low
    assert delay.processing_delay(0.5, 100, 0) == math.inf


def test_link_delay_components():
    d = delay.link_delay(link_load=0.0, propagation_delay_ms=5.0,
                         data_rate=100, allocated_bw=100)
    # No queuing at zero load; transmission = 100/100 = 1 ms; prop = 5 ms.
    assert d == pytest.approx(6.0)


def test_transmission_delay_zero_bw_is_inf():
    assert delay.transmission_delay(100, 0) == math.inf
