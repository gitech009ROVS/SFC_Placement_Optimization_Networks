"""Classical, dependency-free baseline predictors.

These are useful both as honest comparison baselines for the GRU and as a
zero-dependency fallback so the simulator always runs.
"""
from __future__ import annotations

from .base import TrafficPredictor


class LastValuePredictor(TrafficPredictor):
    """Persistence model: forecast = last observed value (random walk)."""

    def fit(self) -> None:  # stateless
        return None

    def predict(self, request_ids: list[str]) -> dict[str, float]:
        out: dict[str, float] = {}
        for rid in request_ids:
            last = self.history.last(rid)
            out[rid] = 0.0 if last is None else last
        return out


class MovingAveragePredictor(TrafficPredictor):
    """Simple moving average over the sliding window."""

    def fit(self) -> None:
        return None

    def predict(self, request_ids: list[str]) -> dict[str, float]:
        out: dict[str, float] = {}
        for rid in request_ids:
            win = self.history.window(rid)
            if win:
                out[rid] = sum(win) / len(win)
            else:
                last = self.history.last(rid)
                out[rid] = 0.0 if last is None else last
        return out


class ExponentialSmoothingPredictor(TrafficPredictor):
    """Holt's linear trend method (level + trend) — captures ramps the MA misses."""

    def __init__(self, window_size: int = 20, alpha: float = 0.5, beta: float = 0.3):
        super().__init__(window_size)
        self.alpha = alpha
        self.beta = beta

    def fit(self) -> None:
        return None

    def predict(self, request_ids: list[str]) -> dict[str, float]:
        out: dict[str, float] = {}
        for rid in request_ids:
            series = self.history.series(rid)
            if not series:
                out[rid] = 0.0
                continue
            if len(series) == 1:
                out[rid] = series[0]
                continue
            level = series[0]
            trend = series[1] - series[0]
            for value in series[1:]:
                prev_level = level
                level = self.alpha * value + (1 - self.alpha) * (level + trend)
                trend = self.beta * (level - prev_level) + (1 - self.beta) * trend
            out[rid] = max(0.0, level + trend)
        return out
