"""Command-line entry point: ``sfc-autoscale <command>``."""
from __future__ import annotations

import argparse
import json
import logging
import sys

from .config import ExperimentConfig
from .prediction.factory import available_predictors
from .simulation.experiment import compare_policies
from .simulation.runner import SimulationRunner
from .simulation.scenario import Scenario
from .simulation.workload import available_workloads


def _print_summary(title: str, summary: dict) -> None:
    print(f"\n=== {title} ===")
    for key, value in summary.items():
        if isinstance(value, dict):
            inner = "  ".join(f"{k}={v}" for k, v in value.items())
            print(f"  {key:22s}: {inner}")
        else:
            print(f"  {key:22s}: {value}")


def cmd_demo(args: argparse.Namespace) -> int:
    scenario = Scenario.default(
        workload_name=args.workload, num_steps=args.steps,
        seed=args.seed, change_threshold=args.threshold,
    )
    runner = SimulationRunner(
        scenario, predictor_name=args.predictor,
        predictor_kwargs={"window_size": 20, "seed": args.seed},
    )
    result = runner.run(verbose=True)
    _print_summary(f"Run summary ({result.predictor_name})", result.summary())
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    cfg = ExperimentConfig.from_yaml(args.config)
    scenario = cfg.build_scenario()
    backend, kwargs = cfg.predictor_spec()
    kwargs.setdefault("seed", cfg.seed)
    runner = SimulationRunner(scenario, predictor_name=backend, predictor_kwargs=kwargs)
    result = runner.run(verbose=args.verbose)
    _print_summary(f"Run summary ({result.predictor_name})", result.summary())
    if args.json:
        print("\nJSON:")
        print(json.dumps(result.summary(), indent=2))
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Run several predictor backends on the *same* workload and compare them."""
    cfg = ExperimentConfig.from_yaml(args.config) if args.config else None
    backends = args.predictors or ["last_value", "moving_average", "exp_smoothing", "gru_numpy"]

    rows: list[dict] = []
    for backend in backends:
        # Fresh scenario per backend so each starts from an identical substrate.
        if cfg is not None:
            scenario = cfg.build_scenario()
            seed = cfg.seed
        else:
            scenario = Scenario.default(workload_name=args.workload, num_steps=args.steps,
                                        seed=args.seed)
            seed = args.seed
        try:
            runner = SimulationRunner(
                scenario, predictor_name=backend,
                predictor_kwargs={"window_size": 20, "seed": seed},
            )
            summary = runner.run(verbose=False).summary()
            rows.append({"predictor": backend, **summary})
        except ImportError as exc:
            print(f"  (skipping {backend}: {exc})")

    # Tabular comparison of the headline metrics.
    cols = ["predictor", "prediction_rmse", "total_actions", "total_violations",
            "avg_cpu_utilization", "avg_delay_satisfaction"]
    print("\n=== Predictor comparison (same workload & substrate) ===")
    header = "  ".join(f"{c:>22s}" for c in cols)
    print(header)
    print("-" * len(header))
    for row in rows:
        print("  ".join(f"{str(row.get(c, '')):>22s}" for c in cols))
    return 0


def cmd_policies(args: argparse.Namespace) -> int:
    """Compare static vs reactive vs proactive(GRU) scaling on the stress scenario."""
    def factory():
        return Scenario.stress(num_steps=args.steps, seed=args.seed,
                               amplitude=args.amplitude, max_delay=args.budget)

    result = compare_policies(factory, predictor_name=args.predictor,
                              predictor_kwargs={"seed": args.seed, "epochs": args.epochs})
    print("\n=== Scaling-policy comparison (QoS-tight stress scenario) ===")
    print("Lower violations = better. Proactive should beat reactive should beat static.\n")
    print(result.table())
    prox = result.results["proactive"].violation_steps
    react = result.results["reactive"].violation_steps
    if react > 0:
        print(f"\nProactive cut QoS-violation steps by "
              f"{100 * (react - prox) / react:.0f}% vs reactive.")
    return 0


def cmd_info(_: argparse.Namespace) -> int:
    print("Available predictors:", ", ".join(available_predictors()))
    print("Available workloads: ", ", ".join(available_workloads()))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sfc-autoscale", description=__doc__)
    p.add_argument("--log", default="WARNING", help="log level (DEBUG/INFO/WARNING)")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("demo", help="run the built-in demo scenario, verbosely")
    d.add_argument("--predictor", default="gru_numpy")
    d.add_argument("--workload", default="diurnal_spike")
    d.add_argument("--steps", type=int, default=48)
    d.add_argument("--threshold", type=float, default=0.05)
    d.add_argument("--seed", type=int, default=0)
    d.set_defaults(func=cmd_demo)

    r = sub.add_parser("run", help="run an experiment from a YAML config")
    r.add_argument("--config", required=True)
    r.add_argument("--verbose", action="store_true")
    r.add_argument("--json", action="store_true")
    r.set_defaults(func=cmd_run)

    c = sub.add_parser("compare", help="compare predictor backends on one workload")
    c.add_argument("--config", default=None)
    c.add_argument("--predictors", nargs="*", default=None)
    c.add_argument("--workload", default="diurnal_spike")
    c.add_argument("--steps", type=int, default=48)
    c.add_argument("--seed", type=int, default=0)
    c.set_defaults(func=cmd_compare)

    pol = sub.add_parser("policies",
                         help="compare static vs reactive vs proactive(GRU) scaling")
    pol.add_argument("--steps", type=int, default=120)
    pol.add_argument("--seed", type=int, default=0)
    pol.add_argument("--amplitude", type=float, default=0.7)
    pol.add_argument("--budget", type=float, default=19.0, help="E2E delay budget (ms)")
    pol.add_argument("--predictor", default="gru_numpy")
    pol.add_argument("--epochs", type=int, default=80)
    pol.set_defaults(func=cmd_policies)

    i = sub.add_parser("info", help="list available predictors and workloads")
    i.set_defaults(func=cmd_info)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log.upper(), logging.WARNING),
                        format="%(levelname)s %(name)s: %(message)s")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
