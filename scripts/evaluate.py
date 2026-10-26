"""Evaluation harness: run several predictor backends over several workloads and
write per-run summaries + per-step traces to CSV (for plotting in the paper).

Run:  python scripts/evaluate.py --out results
Requires only the standard library + numpy (pandas/matplotlib optional).
"""
from __future__ import annotations

import argparse
import csv
import os

from sfc_autoscale.simulation import Scenario, SimulationRunner

PREDICTORS = ["last_value", "moving_average", "exp_smoothing", "gru_numpy"]
WORKLOADS = ["diurnal", "spike", "random_walk", "diurnal_spike"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--steps", type=int, default=72)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    summary_rows: list[dict] = []
    for workload in WORKLOADS:
        for backend in PREDICTORS:
            scenario = Scenario.default(workload_name=workload, num_steps=args.steps,
                                        seed=args.seed)
            try:
                runner = SimulationRunner(
                    scenario, predictor_name=backend,
                    predictor_kwargs={"window_size": 20, "seed": args.seed},
                )
                result = runner.run()
            except ImportError as exc:
                print(f"skip {backend}: {exc}")
                continue

            row = {"workload": workload, **result.summary()}
            row.pop("action_breakdown", None)
            summary_rows.append(row)

            # Per-step trace for this (workload, backend).
            trace_path = os.path.join(args.out, f"trace_{workload}_{backend}.csv")
            with open(trace_path, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["t", "cpu_fraction", "num_instances", "num_actions",
                            "num_violations", "prediction_rmse", "delay_satisfaction"])
                for s in result.metrics.steps:
                    w.writerow([s.time_step, s.cpu_fraction, s.num_instances, s.num_actions,
                                s.num_violations, s.prediction_rmse, s.delay_satisfaction])
            print(f"[{workload:14s} / {backend:16s}] "
                  f"rmse={result.summary()['prediction_rmse']:7.3f} "
                  f"actions={result.summary()['total_actions']:3d} "
                  f"violations={result.summary()['total_violations']:3d}")

    # Combined summary CSV.
    summary_path = os.path.join(args.out, "summary.csv")
    with open(summary_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nWrote {summary_path} and per-step traces to {args.out}/")


if __name__ == "__main__":
    main()
