"""The discrete-time simulation driver.

Ties a :class:`Scenario`, a predictor, and the orchestrator together into a
reproducible closed-loop run, collecting metrics each tick. This is the free,
practical testbed: no emulator, no hardware, no external services.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..orchestration.orchestrator import SFCOrchestrator
from ..placement import PlacementResult
from ..prediction.base import TrafficPredictor
from ..prediction.factory import build_predictor
from .metrics import MetricsCollector, RunMetrics
from .scenario import Scenario


@dataclass
class RunResult:
    placement: PlacementResult
    metrics: RunMetrics
    predictor_name: str

    def summary(self) -> dict:
        return {
            "predictor": self.predictor_name,
            "placed": len(self.placement.placed),
            "rejected": len(self.placement.rejected),
            "acceptance_ratio": round(self.placement.acceptance_ratio, 4),
            "placement_profit": round(self.placement.total_profit, 2),
            **self.metrics.summary(),
        }


class SimulationRunner:
    def __init__(
        self,
        scenario: Scenario,
        predictor: TrafficPredictor | None = None,
        predictor_name: str = "gru_numpy",
        predictor_kwargs: dict | None = None,
    ):
        self.scenario = scenario
        if predictor is None:
            kwargs = {"window_size": 20, **(predictor_kwargs or {})}
            predictor = build_predictor(predictor_name, **kwargs)
        self.predictor = predictor
        self.predictor_name = predictor.name

    def run(self, verbose: bool = False) -> RunResult:
        sc = self.scenario
        net = sc.network

        # Ground-truth traffic series for the whole horizon (reproducible).
        truth = sc.workload.build(sc.requests, sc.num_steps + 1)

        orch = SFCOrchestrator(
            net, self.predictor,
            k=sc.k_paths, bw_cost_per_unit=sc.bw_cost_per_unit,
            scaling_config=sc.scaling_config, retrain_every=sc.retrain_every,
        )

        # Phase 1: initial placement at base rates.
        placement = orch.initial_placement(sc.requests)
        collector = MetricsCollector(net)

        if verbose:
            print(f"[placement] placed={len(placement.placed)} "
                  f"rejected={len(placement.rejected)} "
                  f"profit={placement.total_profit:.1f}")

        placed_ids = {r.request_id for r in placement.placed}

        # Closed loop.
        for t in range(sc.num_steps):
            observed = {rid: float(truth[rid][t]) for rid in placed_ids if rid in truth}
            report = orch.step(observed)
            truth_next = {rid: float(truth[rid][t + 1]) for rid in placed_ids if rid in truth}
            sm = collector.observe_step(report, truth_next, report.predictions)
            if verbose and (report.actions or sm.num_violations):
                acts = ", ".join(a.describe() for a in report.actions) or "—"
                print(f"[t={report.time_step:02d}] cpu={sm.cpu_fraction*100:4.1f}% "
                      f"sat={sm.delay_satisfaction*100:5.1f}% actions: {acts}")

        return RunResult(placement=placement, metrics=collector.run,
                         predictor_name=self.predictor_name)
