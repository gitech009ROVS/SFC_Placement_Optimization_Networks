"""End-to-end integration tests over the full closed loop."""
from __future__ import annotations

import os
import subprocess
import sys

from sfc_autoscale.config import ExperimentConfig
from sfc_autoscale.simulation import Scenario, SimulationRunner

_REPRO_SNIPPET = (
    "from sfc_autoscale.simulation import Scenario, SimulationRunner;"
    "s=Scenario.default(num_steps=40, seed=0);"
    "r=SimulationRunner(s, predictor_name='gru_numpy', predictor_kwargs={'seed':0}).run();"
    "print('%.6f|%d' % (r.summary()['prediction_rmse'], r.summary()['total_actions']))"
)


def _run_in_subprocess(hashseed: str) -> str:
    env = {**os.environ, "PYTHONHASHSEED": hashseed}
    out = subprocess.check_output([sys.executable, "-c", _REPRO_SNIPPET], env=env, text=True)
    return out.strip()


def test_full_run_is_stable_and_reproducible():
    s1 = Scenario.default(workload_name="diurnal_spike", num_steps=40, seed=7)
    r1 = SimulationRunner(s1, predictor_name="gru_numpy",
                          predictor_kwargs={"seed": 7}).run()

    s2 = Scenario.default(workload_name="diurnal_spike", num_steps=40, seed=7)
    r2 = SimulationRunner(s2, predictor_name="gru_numpy",
                          predictor_kwargs={"seed": 7}).run()

    # Deterministic given the seed.
    assert r1.summary() == r2.summary()
    # Sanity: everything placed, delays respected most of the time.
    assert r1.summary()["placed"] >= 1
    assert r1.summary()["avg_delay_satisfaction"] >= 0.9


def test_reproducible_across_processes_and_hashseeds():
    """Same seed must yield identical results in separate processes, regardless of
    PYTHONHASHSEED — guards against seeding the workload with Python's salted hash().
    """
    a = _run_in_subprocess("0")
    b = _run_in_subprocess("12345")
    c = _run_in_subprocess("random")
    assert a == b == c


def test_runner_with_classical_backend():
    scenario = Scenario.default(num_steps=30, seed=1)
    result = SimulationRunner(scenario, predictor_name="moving_average").run()
    assert result.summary()["steps"] == 30


def test_config_roundtrip_builds_runnable_scenario(tmp_path):
    cfg_dict = {
        "seed": 3,
        "simulation": {"num_steps": 25, "retrain_every": 10},
        "topology": {"type": "sample"},
        "workload": {"type": "diurnal"},
        "predictor": {"backend": "exp_smoothing", "window_size": 15},
        "scaling": {"change_threshold": 0.1},
    }
    cfg = ExperimentConfig.from_dict(cfg_dict)
    scenario = cfg.build_scenario()
    backend, kwargs = cfg.predictor_spec()
    result = SimulationRunner(scenario, predictor_name=backend,
                              predictor_kwargs=kwargs).run()
    assert backend == "exp_smoothing"
    assert result.summary()["steps"] == 25


def test_proactive_beats_reactive_beats_static_on_stress():
    """The headline result: on the QoS-tight stress scenario, proactive (GRU)
    scaling incurs fewer delay-budget violations than reactive, which beats static.
    """
    from sfc_autoscale.simulation import Scenario, compare_policies

    def factory():
        return Scenario.stress(num_steps=96, seed=0)

    res = compare_policies(factory, predictor_name="gru_numpy",
                           predictor_kwargs={"seed": 0, "epochs": 60})
    static = res.results["static"].violation_steps
    reactive = res.results["reactive"].violation_steps
    proactive = res.results["proactive"].violation_steps
    assert proactive < reactive < static
    # Proactive should also keep delay satisfaction clearly higher.
    assert res.results["proactive"].mean_delay_satisfaction > \
        res.results["reactive"].mean_delay_satisfaction


def test_constant_workload_triggers_no_scaling():
    scenario = Scenario.default(workload_name="constant", num_steps=30, seed=0)
    result = SimulationRunner(scenario, predictor_name="last_value").run()
    # Flat traffic => every decision is within threshold => no actions.
    assert result.summary()["total_actions"] == 0
