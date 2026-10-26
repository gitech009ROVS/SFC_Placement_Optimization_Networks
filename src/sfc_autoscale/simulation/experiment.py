"""Controlled comparison of scaling policies under the *same* traffic.

Three policies share the identical action repertoire, cost model, and substrate;
they differ only in **what rate they size for** and **when** they act:

* ``static``    — no scaling after initial placement (control).
* ``reactive``  — resize toward the rate just observed (HPA-style: acts *after*
  demand moves, so it is always one step behind a ramp).
* ``proactive`` — resize toward the GRU's forecast of the next step (acts
  *before* demand arrives).

Timing is the crux. Each tick we first let the realised demand arrive and
**measure QoS against the allocation decided on the previous tick** (did we
prepare correctly?), and only then adapt for the next tick. This exposes the
one-step lag that proactive scaling removes and reactive scaling cannot.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..placement import InitialPlacement
from ..prediction.factory import build_predictor
from .scenario import Scenario

POLICIES = ("static", "reactive", "proactive")


@dataclass
class PolicyMetrics:
    policy: str
    predictor: str = "—"
    violation_steps: int = 0        # ticks with >=1 request over budget
    request_violations: int = 0     # summed per-request budget breaches
    scale_ups: int = 0
    scale_downs: int = 0
    mean_delay_satisfaction: float = 1.0
    worst_recovery: int = 0         # longest run of consecutive violation ticks
    sat_trace: list[float] = field(default_factory=list)

    @property
    def total_actions(self) -> int:
        return self.scale_ups + self.scale_downs

    def summary(self) -> dict:
        return {
            "policy": self.policy,
            "predictor": self.predictor,
            "violation_steps": self.violation_steps,
            "request_violations": self.request_violations,
            "worst_recovery_steps": self.worst_recovery,
            "mean_delay_satisfaction": round(self.mean_delay_satisfaction, 4),
            "total_actions": self.total_actions,
        }


def _measure(network) -> tuple[float, int]:
    """Return (delay-satisfaction fraction, #requests over budget) right now."""
    reqs = list(network.active_requests.values())
    if not reqs:
        return 1.0, 0
    over = sum(1 for r in reqs if network.e2e_delay(r) > r.max_delay)
    return (len(reqs) - over) / len(reqs), over


def _provision_toward(net, request_id: str, target_rate: float, threshold: float,
                      target_util: float = 0.6) -> int:
    """Vertically resize a request's instances to serve ``target_rate`` at
    ``target_util`` utilisation. Returns +1/-1/0 for a net up/down/no change.

    This is the controller the reactive and proactive policies share; they differ
    only in the ``target_rate`` they pass (current observation vs GRU forecast),
    which isolates the value of prediction.
    """
    req = net.active_requests.get(request_id)
    if req is None:
        return 0
    direction = 0
    for inst_id in list(req.placed_vnf_instances):
        inst = net.vnf_instances.get(inst_id)
        if inst is None:
            continue
        new_cpu = inst.vnf_type.cpu_for_rate(target_rate, target_util)
        if abs(new_cpu - inst.allocated_cpu) / max(inst.allocated_cpu, 1e-9) < threshold:
            continue
        if net.resize_instance(inst_id, new_cpu):
            direction = 1 if new_cpu > inst.allocated_cpu or direction == 1 else -1
    return direction


def run_policy(
    scenario: Scenario,
    policy: str,
    predictor_name: str = "gru_numpy",
    predictor_kwargs: dict | None = None,
) -> PolicyMetrics:
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}")
    net = scenario.network
    truth = scenario.workload.build(scenario.requests, scenario.num_steps + 1)

    InitialPlacement(net, k=scenario.k_paths,
                     bw_cost_per_unit=scenario.bw_cost_per_unit).place_all(scenario.requests)
    placed_ids = [rid for rid, r in net.active_requests.items()]

    predictor = None
    if policy == "proactive":
        kwargs = {"window_size": 20, **(predictor_kwargs or {})}
        predictor = build_predictor(predictor_name, **kwargs)

    threshold = scenario.scaling_config.change_threshold
    m = PolicyMetrics(policy=policy, predictor=(predictor.name if predictor else "—"))
    run_len = 0

    for t in range(scenario.num_steps):
        demand = {rid: float(truth[rid][t]) for rid in placed_ids}

        # (1) Demand arrives. Measure QoS against the allocation prepared last tick.
        for rid, rate in demand.items():
            net.set_request_rate(rid, rate)
        sat, over = _measure(net)
        m.sat_trace.append(sat)
        m.mean_delay_satisfaction += sat
        if over:
            m.violation_steps += 1
            m.request_violations += over
            run_len += 1
            m.worst_recovery = max(m.worst_recovery, run_len)
        else:
            run_len = 0

        # (2) Adapt for the next tick — differ only in the target rate.
        if policy == "static":
            pass
        else:
            if policy == "reactive":
                targets = demand  # react to what was just observed
            else:  # proactive
                predictor.observe(demand)
                if t % scenario.retrain_every == 0:
                    predictor.fit()
                targets = predictor.predict(placed_ids)  # anticipate next step
            for rid in placed_ids:
                d = _provision_toward(net, rid, targets[rid], threshold)
                if d > 0:
                    m.scale_ups += 1
                elif d < 0:
                    m.scale_downs += 1

    m.mean_delay_satisfaction /= max(1, scenario.num_steps)
    return m


@dataclass
class ComparisonResult:
    results: dict[str, PolicyMetrics]

    def table(self) -> str:
        cols = ["policy", "predictor", "violation_steps", "request_violations",
                "worst_recovery_steps", "mean_delay_satisfaction", "total_actions"]
        lines = ["  ".join(f"{c:>22s}" for c in cols)]
        lines.append("-" * len(lines[0]))
        for pol in POLICIES:
            if pol in self.results:
                s = self.results[pol].summary()
                lines.append("  ".join(f"{str(s.get(c, '')):>22s}" for c in cols))
        return "\n".join(lines)


def compare_policies(
    scenario_factory,
    predictor_name: str = "gru_numpy",
    predictor_kwargs: dict | None = None,
) -> ComparisonResult:
    """Run all three policies, each on a fresh identical scenario.

    ``scenario_factory`` is a zero-arg callable returning a *new* Scenario, so each
    policy starts from an untouched substrate but the same seeded workload.
    """
    out: dict[str, PolicyMetrics] = {}
    for policy in POLICIES:
        out[policy] = run_policy(scenario_factory(), policy,
                                 predictor_name=predictor_name,
                                 predictor_kwargs=predictor_kwargs)
    return ComparisonResult(results=out)
