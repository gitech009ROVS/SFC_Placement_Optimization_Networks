"""Synthetic traffic generators — the free, reproducible testbed.

Each generator turns a request's base data rate into a full time series of
"ground-truth" rates. Patterns are seeded for exact reproducibility, so an
experiment re-runs identically on any machine with no external data or cost.

A generator may also load a real public trace (see :class:`TraceWorkload`) when
a CSV is supplied, but the defaults need nothing but NumPy.
"""
from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod

import numpy as np

from ..core.entities import SFCRequest


class Workload(ABC):
    """Produces per-request ground-truth rate series of length ``num_steps``."""

    def __init__(self, seed: int = 0):
        self.seed = seed

    @abstractmethod
    def build(self, requests: list[SFCRequest], num_steps: int) -> dict[str, np.ndarray]:
        ...

    def _rng(self, request_id: str) -> np.random.Generator:
        # Deterministic per-request stream derived from the global seed.
        # NB: use a *stable* hash (sha256), not Python's built-in hash(), which is
        # randomised per process (PYTHONHASHSEED) and would make "seeded" runs
        # differ across invocations and machines — fatal for reproducibility.
        digest = hashlib.sha256(f"{request_id}:{self.seed}".encode()).digest()
        mix = int.from_bytes(digest[:8], "little")
        return np.random.default_rng(mix)


class ConstantWorkload(Workload):
    """Flat rate — useful as a control (no scaling should ever fire)."""

    def build(self, requests, num_steps):
        return {r.request_id: np.full(num_steps, r.data_rate, dtype=float) for r in requests}


class DiurnalWorkload(Workload):
    """Daily sinusoidal demand + Gaussian noise (classic telecom pattern).

    rate(t) = base * (1 + amplitude * sin(2π (t + phase) / period)) + noise
    """

    def __init__(self, seed: int = 0, period: int = 24, amplitude: float = 0.6,
                 noise: float = 0.05):
        super().__init__(seed)
        self.period = period
        self.amplitude = amplitude
        self.noise = noise

    def build(self, requests, num_steps):
        out: dict[str, np.ndarray] = {}
        t = np.arange(num_steps)
        for r in requests:
            rng = self._rng(r.request_id)
            phase = rng.uniform(0, self.period)
            seasonal = 1.0 + self.amplitude * np.sin(2 * np.pi * (t + phase) / self.period)
            noise = rng.normal(0.0, self.noise, num_steps)
            out[r.request_id] = np.maximum(1.0, r.data_rate * (seasonal + noise))
        return out


class SpikeWorkload(Workload):
    """Baseline demand with occasional multiplicative spikes (flash crowds)."""

    def __init__(self, seed: int = 0, spike_prob: float = 0.08,
                 spike_mult: tuple[float, float] = (1.8, 3.0), decay: float = 0.5):
        super().__init__(seed)
        self.spike_prob = spike_prob
        self.spike_mult = spike_mult
        self.decay = decay

    def build(self, requests, num_steps):
        out: dict[str, np.ndarray] = {}
        for r in requests:
            rng = self._rng(r.request_id)
            series = np.full(num_steps, r.data_rate, dtype=float)
            boost = 0.0
            for t in range(num_steps):
                if rng.random() < self.spike_prob:
                    boost = max(boost, rng.uniform(*self.spike_mult) - 1.0)
                series[t] = r.data_rate * (1.0 + boost) + rng.normal(0, 0.03 * r.data_rate)
                boost *= self.decay
            out[r.request_id] = np.maximum(1.0, series)
        return out


class RandomWalkWorkload(Workload):
    """Bounded multiplicative random walk — noisy, trendy, no seasonality."""

    def __init__(self, seed: int = 0, step: float = 0.08, lo: float = 0.3, hi: float = 2.5):
        super().__init__(seed)
        self.step = step
        self.lo = lo
        self.hi = hi

    def build(self, requests, num_steps):
        out: dict[str, np.ndarray] = {}
        for r in requests:
            rng = self._rng(r.request_id)
            mult = 1.0
            series = np.empty(num_steps, dtype=float)
            for t in range(num_steps):
                mult = float(np.clip(mult * (1.0 + rng.normal(0, self.step)), self.lo, self.hi))
                series[t] = r.data_rate * mult
            out[r.request_id] = np.maximum(1.0, series)
        return out


class DiurnalSpikeWorkload(DiurnalWorkload):
    """Diurnal seasonality with sparse spikes — the most realistic default."""

    def __init__(self, seed: int = 0, period: int = 24, amplitude: float = 0.5,
                 noise: float = 0.04, spike_prob: float = 0.05,
                 spike_mult: tuple[float, float] = (1.6, 2.4)):
        super().__init__(seed, period, amplitude, noise)
        self._spikes = SpikeWorkload(seed, spike_prob, spike_mult)

    def build(self, requests, num_steps):
        base = super().build(requests, num_steps)        # seasonal component
        spikes = self._spikes.build(requests, num_steps)  # baseline + spikes
        base_rate = {r.request_id: r.data_rate for r in requests}
        out: dict[str, np.ndarray] = {}
        for rid in base:
            # Add the spike *excess* (amount above the flat baseline) on top of
            # the seasonal curve, so both seasonality and bursts are present.
            excess = np.maximum(0.0, spikes[rid] - base_rate[rid])
            out[rid] = np.maximum(1.0, base[rid] + excess)
        return out


_REGISTRY: dict[str, type[Workload]] = {
    "constant": ConstantWorkload,
    "diurnal": DiurnalWorkload,
    "spike": SpikeWorkload,
    "random_walk": RandomWalkWorkload,
    "diurnal_spike": DiurnalSpikeWorkload,
}


def build_workload(name: str, **kwargs) -> Workload:
    key = name.lower()
    if key not in _REGISTRY:
        raise KeyError(f"unknown workload '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[key](**kwargs)


def available_workloads() -> list[str]:
    return sorted(_REGISTRY)
