"""Prediction interfaces.

Every predictor consumes a stream of per-request traffic observations and emits
a one-step-ahead forecast per request. Crucially, forecasts are produced from
*observed* history only — a predictor never feeds its own predictions back into
the input window — which keeps errors from compounding in the closed loop.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict, deque


class History:
    """Per-request sliding ring buffers of observed data rates."""

    def __init__(self, window_size: int, max_len: int | None = None):
        self.window_size = window_size
        self._max_len = max_len or max(1000, window_size * 50)
        self._series: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=self._max_len)
        )
        self._t: int = 0

    def record(self, observations: dict[str, float]) -> None:
        for rid, value in observations.items():
            self._series[rid].append(float(value))
        self._t += 1

    def last(self, request_id: str) -> float | None:
        s = self._series.get(request_id)
        return s[-1] if s else None

    def window(self, request_id: str) -> list[float] | None:
        """The most recent ``window_size`` observations, or None if too short."""
        s = self._series.get(request_id)
        if s is None or len(s) < self.window_size:
            return None
        return list(s)[-self.window_size :]

    def series(self, request_id: str) -> list[float]:
        return list(self._series.get(request_id, ()))

    def all_series(self) -> dict[str, list[float]]:
        return {rid: list(s) for rid, s in self._series.items()}

    @property
    def t(self) -> int:
        return self._t


class TrafficPredictor(ABC):
    """Abstract one-step-ahead traffic predictor."""

    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.history = History(window_size)

    def observe(self, observations: dict[str, float]) -> None:
        """Record one time step of real observations for all active requests."""
        self.history.record(observations)

    @abstractmethod
    def fit(self) -> None:
        """(Re)train on buffered history. May be a no-op for stateless models."""

    @abstractmethod
    def predict(self, request_ids: list[str]) -> dict[str, float]:
        """Return the forecast data rate for the next step for each request."""

    @property
    def name(self) -> str:
        return type(self).__name__
