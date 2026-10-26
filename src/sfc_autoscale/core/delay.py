"""Contention-aware delay models (Paper 1, Section III).

These are pure functions so they can be unit-tested in isolation and reused by
the placement, scaling, and validation layers without pulling in topology state.
"""
from __future__ import annotations

INF = float("inf")


def contention_factor(load: float) -> float:
    """M/M/1-style contention multiplier ``ρ / (1 - ρ)`` (Paper 1, Eqs.6, 8).

    As utilisation approaches 1 the factor diverges, capturing the sharp delay
    growth of a saturated resource. A fully loaded resource returns +inf.
    """
    if load >= 1.0:
        return INF
    if load <= 0.0:
        return 0.0
    return load / (1.0 - load)


def processing_delay(node_load: float, data_rate: float, allocated_cpu: float) -> float:
    """Processing delay keyed off *node* CPU load (Paper 1, Eq.8).

    ``delay = (ρ / (1 - ρ)) * (λ / C_cpu)``. Retained for Paper-1 fidelity and
    used as a secondary term; the demand-sensitive term is
    :func:`instance_queue_delay` below.
    """
    if allocated_cpu <= 0:
        return INF
    cf = contention_factor(node_load)
    if cf == INF:
        return INF
    return cf * (data_rate / allocated_cpu)


def instance_queue_delay(utilization: float, service_delay_ms: float) -> float:
    """Queueing delay at a single VNF *instance* (M/M/1 waiting time).

    ``delay = (ρ / (1 - ρ)) * t_service`` where ``ρ = offered_rate / capacity``
    is the instance's own utilisation. This is the term that makes scaling matter:
    as offered load approaches an instance's provisioned throughput, ρ→1 and delay
    diverges, so under-provisioning during a traffic rise breaches the delay
    budget — while scaling up (more CPU) or out (load split across replicas) pulls
    ρ back down and restores QoS.
    """
    cf = contention_factor(utilization)
    if cf == INF:
        return INF
    return cf * service_delay_ms


def transmission_delay(data_rate: float, allocated_bw: float) -> float:
    """Transmission delay on a link (Paper 1, Eq.4): ``λ / C_bw``."""
    if allocated_bw <= 0:
        return INF
    return data_rate / allocated_bw


def queuing_delay(link_load: float, data_rate: float, allocated_bw: float) -> float:
    """Queuing delay on a link (Paper 1, Eq.6): contention * transmission."""
    cf = contention_factor(link_load)
    if cf == INF:
        return INF
    return cf * transmission_delay(data_rate, allocated_bw)


def link_delay(
    link_load: float,
    propagation_delay_ms: float,
    data_rate: float,
    allocated_bw: float,
) -> float:
    """Total single-link delay = transmission + propagation + queuing."""
    t = transmission_delay(data_rate, allocated_bw)
    q = queuing_delay(link_load, data_rate, allocated_bw)
    if t == INF or q == INF:
        return INF
    return t + propagation_delay_ms + q
