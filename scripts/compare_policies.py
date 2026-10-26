"""Policy comparison for the paper: static vs reactive vs proactive(GRU).

Runs the QoS-tight stress scenario and writes a summary CSV plus per-step
delay-satisfaction traces (for plotting the "proactive stays ahead of the ramp"
figure).

Run:  python scripts/compare_policies.py --out results
"""
from __future__ import annotations

import argparse
import csv
import os

from sfc_autoscale.simulation import Scenario, compare_policies


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amplitude", type=float, default=0.7)
    ap.add_argument("--budget", type=float, default=19.0)
    ap.add_argument("--epochs", type=int, default=80)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    def factory():
        return Scenario.stress(num_steps=args.steps, seed=args.seed,
                               amplitude=args.amplitude, max_delay=args.budget)

    result = compare_policies(factory, predictor_name="gru_numpy",
                              predictor_kwargs={"seed": args.seed, "epochs": args.epochs})

    # Summary CSV.
    summary_path = os.path.join(args.out, "policy_summary.csv")
    rows = [result.results[p].summary() for p in ("static", "reactive", "proactive")]
    with open(summary_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Per-step delay-satisfaction traces (one column per policy).
    trace_path = os.path.join(args.out, "policy_sat_trace.csv")
    traces = {p: result.results[p].sat_trace for p in ("static", "reactive", "proactive")}
    n = max(len(t) for t in traces.values())
    with open(trace_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t", "static", "reactive", "proactive"])
        for i in range(n):
            w.writerow([i] + [f"{traces[p][i]:.4f}" if i < len(traces[p]) else ""
                              for p in ("static", "reactive", "proactive")])

    print(result.table())
    print(f"\nWrote {summary_path} and {trace_path}")


if __name__ == "__main__":
    main()
