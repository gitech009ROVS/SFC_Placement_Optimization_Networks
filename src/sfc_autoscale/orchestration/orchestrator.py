"""The closed-loop orchestrator (Monitor -> Predict -> Decide -> Execute -> Validate)."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ..core.entities import SFCRequest
from ..core.enums import ScalingAction
from ..core.topology import NetworkTopology
from ..placement import InitialPlacement, PlacementResult
from ..prediction.base import TrafficPredictor
from ..scaling import ScalingCandidate, ScalingConfig, ScalingDecisionEngine, ScalingExecutor

logger = logging.getLogger(__name__)


@dataclass
class StepReport:
    """Everything that happened in one control-loop tick."""

    time_step: int
    predictions: dict[str, float] = field(default_factory=dict)
    actions: list[ScalingCandidate] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)
    utilization: dict[str, float] = field(default_factory=dict)


class SFCOrchestrator:
    """Runs Phase 1 once, then one closed-loop tick per :meth:`step` call.

    The predictor and scaling policy are injected, so swapping the GRU for a
    baseline — or the greedy engine for another policy — needs no changes here.
    """

    def __init__(
        self,
        network: NetworkTopology,
        predictor: TrafficPredictor,
        *,
        k: int = 5,
        bw_cost_per_unit: float = 1.0,
        scaling_config: ScalingConfig | None = None,
        retrain_every: int = 10,
    ):
        self.network = network
        self.predictor = predictor
        self.placer = InitialPlacement(network, k=k, bw_cost_per_unit=bw_cost_per_unit)
        self.decision_engine = ScalingDecisionEngine(network, scaling_config)
        self.executor = ScalingExecutor(network)
        self.retrain_every = retrain_every
        self.time_step = 0

    # --------------------------------------------------------------- Phase 1
    def initial_placement(self, requests: list[SFCRequest]) -> PlacementResult:
        return self.placer.place_all(requests)

    # ----------------------------------------------------------- closed loop
    def step(self, observed_rates: dict[str, float]) -> StepReport:
        """Advance the control loop by one tick.

        ``observed_rates`` is the *ground-truth* traffic measured this interval
        (from the workload generator or a live monitor). It is recorded as the
        predictor's new observation; the predictor then forecasts t+1, and the
        engine scales toward that forecast.
        """
        self.time_step += 1

        # Phase 2a: monitor — record ground truth for every active request.
        active_ids = list(self.network.active_requests.keys())
        obs = {rid: observed_rates.get(rid, self.network.active_requests[rid].data_rate)
               for rid in active_ids}
        self.predictor.observe(obs)

        # Keep each request's "current" rate (and instance loads) aligned with reality.
        for rid, rate in obs.items():
            self.network.set_request_rate(rid, rate)

        # Phase 2b: (re)train periodically — the closed loop of Paper 2.
        if self.time_step % self.retrain_every == 1:
            self.predictor.fit()

        # Phase 2c: predict t+1.
        predictions = self.predictor.predict(active_ids)

        # Phase 3+4: decide and execute, one request at a time.
        report = StepReport(time_step=self.time_step, predictions=predictions)
        for rid in active_ids:
            req = self.network.active_requests.get(rid)
            if req is None:
                continue
            decision = self.decision_engine.decide(rid, req.data_rate, predictions[rid])
            if decision is None or decision.action == ScalingAction.REUSE:
                continue
            if self.executor.execute(decision):
                report.actions.append(decision)
                if self.network.e2e_delay(req) > req.max_delay:
                    report.violations.append(rid)
                    logger.warning(
                        "t=%d %s on %s left delay above budget",
                        self.time_step, decision.action.name, rid,
                    )

        report.utilization = self.network.utilization()
        return report
