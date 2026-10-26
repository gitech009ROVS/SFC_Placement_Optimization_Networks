"""Metrics collection and aggregation for a simulation run."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from ..core.enums import ScalingAction
from ..core.topology import NetworkTopology
from ..orchestration.orchestrator import StepReport


@dataclass
class StepMetrics:
    time_step: int
    cpu_fraction: float
    num_instances: int
    num_actions: int
    num_violations: int
    action_counts: dict[str, int] = field(default_factory=dict)
    prediction_rmse: float | None = None
    delay_satisfaction: float = 1.0  # fraction of requests meeting their delay budget


@dataclass
class RunMetrics:
    steps: list[StepMetrics] = field(default_factory=list)
    action_totals: Counter = field(default_factory=Counter)
    total_violations: int = 0
    rmse_samples: list[float] = field(default_factory=list)

    def record(self, step: StepMetrics) -> None:
        self.steps.append(step)
        self.action_totals.update(step.action_counts)
        self.total_violations += step.num_violations
        if step.prediction_rmse is not None:
            self.rmse_samples.append(step.prediction_rmse)

    # ----------------------------------------------------------- summaries
    @property
    def avg_cpu_fraction(self) -> float:
        return _mean([s.cpu_fraction for s in self.steps])

    @property
    def avg_delay_satisfaction(self) -> float:
        return _mean([s.delay_satisfaction for s in self.steps])

    @property
    def total_actions(self) -> int:
        return sum(self.action_totals.values())

    @property
    def prediction_rmse(self) -> float:
        return _mean(self.rmse_samples)

    def summary(self) -> dict[str, float | int | dict]:
        return {
            "steps": len(self.steps),
            "total_actions": self.total_actions,
            "action_breakdown": dict(self.action_totals),
            "total_violations": self.total_violations,
            "avg_cpu_utilization": round(self.avg_cpu_fraction, 4),
            "avg_delay_satisfaction": round(self.avg_delay_satisfaction, 4),
            "prediction_rmse": round(self.prediction_rmse, 4),
        }


class MetricsCollector:
    """Turns per-step orchestrator reports into structured metrics."""

    def __init__(self, network: NetworkTopology):
        self.network = network
        self.run = RunMetrics()

    def observe_step(
        self,
        report: StepReport,
        truth_next: dict[str, float],
        predictions: dict[str, float],
    ) -> StepMetrics:
        counts: Counter = Counter(a.action.name for a in report.actions)
        # Ensure all action keys present for stable CSV columns.
        for action in ScalingAction:
            counts.setdefault(action.name, 0)

        rmse = _rmse(predictions, truth_next)
        sat = self._delay_satisfaction()

        sm = StepMetrics(
            time_step=report.time_step,
            cpu_fraction=report.utilization.get("cpu_fraction", 0.0),
            num_instances=report.utilization.get("num_instances", 0),
            num_actions=len(report.actions),
            num_violations=len(report.violations),
            action_counts=dict(counts),
            prediction_rmse=rmse,
            delay_satisfaction=sat,
        )
        self.run.record(sm)
        return sm

    def _delay_satisfaction(self) -> float:
        reqs = list(self.network.active_requests.values())
        if not reqs:
            return 1.0
        ok = sum(1 for r in reqs if self.network.e2e_delay(r) <= r.max_delay)
        return ok / len(reqs)


def _mean(xs: list[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def _rmse(pred: dict[str, float], truth: dict[str, float]) -> float | None:
    keys = [k for k in pred if k in truth]
    if not keys:
        return None
    se = [(pred[k] - truth[k]) ** 2 for k in keys]
    return float((sum(se) / len(se)) ** 0.5)
