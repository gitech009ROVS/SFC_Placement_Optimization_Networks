"""Minimal library usage: build a scenario, run the closed loop, inspect metrics.

Run:  python examples/quickstart.py
"""
from __future__ import annotations

from sfc_autoscale.simulation import Scenario, SimulationRunner


def main() -> None:
    # 1. A fully-specified, reproducible scenario (substrate + requests + traffic).
    scenario = Scenario.default(workload_name="diurnal_spike", num_steps=48, seed=0)

    # 2. Pick a predictor backend by name (dependency injection).
    #    Try "last_value", "moving_average", "exp_smoothing", "gru_numpy", "gru_torch".
    runner = SimulationRunner(
        scenario,
        predictor_name="gru_numpy",
        predictor_kwargs={"window_size": 20, "hidden_size": 16, "epochs": 60, "seed": 0},
    )

    # 3. Run the discrete-time closed loop and summarise.
    result = runner.run(verbose=True)

    print("\nSummary:")
    for key, value in result.summary().items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
