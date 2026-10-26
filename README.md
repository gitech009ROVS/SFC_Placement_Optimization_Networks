# sfc-autoscale

**Predictive, contention-aware SFC embedding with multi-strategy VNF auto-scaling
for NFV-based 5G core networks.**

This is a research implementation and a **free, reproducible simulation testbed**
for a system that combines two IEEE works into one closed loop:

- **Paper 1** — *Profit-Maximizing SFC Embedding in NFV-Based 5G Core Networks*
  (contention-aware placement: **CEF-PS → K-CMP → DAVE**).
- **Paper 2** — *Traffic-Prediction-Based VNF Auto-Scaling* (a **GRU** forecaster
  driving a closed control loop).
- **New contribution** — a **multi-strategy scaling decision engine** that, for
  each predicted traffic change, chooses among six actions
  (`REUSE, H_OUT, H_IN, V_UP, V_DOWN, REPLACE`) by cost, under feasibility and
  non-disruption constraints.

No cloud, no hardware, no paid services are needed: the whole system runs as a
discrete-time simulation driven by seeded synthetic workloads. The predictive
core even trains **without any deep-learning framework** (a from-scratch NumPy
GRU with verified backprop), with an optional PyTorch backend for scale.

---

## Why this design is testable for free

| Concern | How it's handled |
|---|---|
| Network substrate | In-memory model (`NetworkTopology`) — nodes, links, VNFs |
| Traffic | Seeded synthetic generators (diurnal, spike, random-walk, …) — reproducible, no datasets required |
| Prediction | NumPy GRU (zero deps) **or** optional PyTorch GRU |
| "Infrastructure" | Pure-Python discrete-time loop; deterministic given a seed |
| Evaluation | Metrics collector + baseline comparison + CSV export |

Everything heavy is an **optional extra**, so `pip install -e .` + `numpy` is enough.

---

## Architecture

```
sfc_autoscale/
├── core/            # entities, contention-aware delay physics, substrate state
├── placement/       # Phase 1: CEF-PS, K-CMP, DAVE  (Paper 1)
├── prediction/      # Phase 2: Predictor ABC + NumPy GRU + Torch GRU + baselines (Paper 2)
├── scaling/         # Phase 3/4: candidates, cost model, decision engine, executor (new)
├── orchestration/   # the closed loop: monitor → predict → decide → execute → validate
├── simulation/      # workloads, scenarios, runner, metrics, policy experiment (the free testbed)
├── topologies/      # reusable substrate builders (sample core, fat-tree)
├── config.py        # YAML-driven experiments
└── cli.py           # `sfc-autoscale {demo,run,compare,info}`
```

The loop is **dependency-injected**: the predictor and scaling policy are passed
into `SFCOrchestrator`, so swapping the GRU for a baseline — or the greedy engine
for another policy — needs no changes to the orchestrator.

### The closed loop (one tick)

```
observed λ(t)  ──▶  Predictor.observe ──▶  Predictor.fit (every N ticks)
                                      └──▶  Predictor.predict  →  λ̂(t+1)
                                                         │
     per request (sequentially):                        ▼
       DecisionEngine.decide(λ(t), λ̂(t+1))
         1. threshold filter (ignore <5% moves)
         2. enumerate ≤4 candidates per VNF instance
         3. cost each; filter infeasible; filter disruptive (simulate-and-rollback)
         4. pick min-cost survivor
       Executor.execute(action)  →  mutate substrate
       validate E2E delay ≤ budget
```

---

## Install

```bash
# Core (free, runs anywhere)
pip install -e .

# Optional extras
pip install -e ".[torch]"   # PyTorch GRU backend
pip install -e ".[plot]"    # pandas / matplotlib for analysis
pip install -e ".[dev]"     # pytest, ruff, mypy
```

Python ≥ 3.10. Core deps: `numpy`, `networkx`, `pyyaml`.

---

## Quickstart

### CLI

```bash
sfc-autoscale info                       # list predictors + workloads
sfc-autoscale demo                       # verbose run of the built-in scenario
sfc-autoscale demo --predictor moving_average --workload diurnal
sfc-autoscale run --config configs/default.yaml --verbose --json
sfc-autoscale compare --workload diurnal # GRU vs baselines, same traffic
sfc-autoscale policies                   # static vs reactive vs proactive (headline result)
```

### Library

```python
from sfc_autoscale.simulation import Scenario, SimulationRunner

scenario = Scenario.default(workload_name="diurnal_spike", num_steps=48, seed=0)
runner = SimulationRunner(scenario, predictor_name="gru_numpy",
                          predictor_kwargs={"window_size": 20, "hidden_size": 16})
result = runner.run(verbose=True)
print(result.summary())
```

### Full evaluation (writes CSVs for plotting)

```bash
python scripts/evaluate.py --out results --steps 72
# -> results/summary.csv and results/trace_<workload>_<predictor>.csv

python scripts/compare_policies.py --out results
# -> results/policy_summary.csv and results/policy_sat_trace.csv (for the paper figure)
```

---

## Representative results

`scripts/evaluate.py` (seed 0, 72 steps) — one-step-ahead forecast RMSE:

| Workload | last_value | moving_avg | exp_smoothing | **GRU (numpy)** |
|---|---:|---:|---:|---:|
| diurnal        | 11.01 | 33.35 | 15.24 | **9.16** |
| diurnal + spike| 16.07 | 31.56 | 19.35 | **15.62** |
| spike          | **27.29** | 29.89 | 30.83 | 31.97 |
| random walk    | **7.50** | 16.52 | 8.90 | 9.40 |

The GRU is the best predictor on **structured/seasonal** traffic (the realistic
5G-core case). On pure random-walk or pure-spike traffic, persistence is close to
theoretically optimal and the GRU matches rather than beats it — an honest,
expected baseline relationship.

### Headline result — proactive vs reactive scaling

`sfc-autoscale policies` runs the **QoS-tight stress scenario** (ample CPU so
scaling is always feasible; tight delay budgets so under-provisioning breaks QoS)
and compares three policies that share the same mechanism and differ only in the
rate they provision for:

| Policy | Provisions for | Violation-steps ↓ | Delay satisfaction ↑ |
|---|---|---:|---:|
| static (no scaling) | fixed (initial) | 94 | 56% |
| reactive (HPA-style) | the rate just observed | 63 | 79% |
| **proactive (GRU)** | the forecast next rate | **17** | **95%** |

**Proactive scaling cuts QoS-violation steps by ~73% versus reactive**, because it
provisions a step *ahead* of each traffic ramp while reactive is always a step
behind. This is the result that justifies the predictive contribution. The
comparison uses correct *measure-then-adapt* timing: each tick, QoS is scored
against the allocation decided on the **previous** tick, so the one-step lag that
proactive removes is actually visible.

---

## Configuration

Experiments are declared in YAML (see `configs/default.yaml`): topology, request
set, workload, predictor backend + hyperparameters, scaling thresholds, and cost
constants. The cost model constants are all injectable, so they can be calibrated
to measured data without touching code.

---

## The scaling algorithm (what's novel)

Processes **one SFC at a time**, sequentially, so each decision sees the substrate
left by the previous. For each VNF instance it builds a **flat list of ≤4
candidate actions** (not the 4^V joint combination — a deliberate greedy
simplification), costs each, discards any that break a resource/delay constraint
or would push another request over its delay budget (checked via
*simulate-and-rollback*), and executes the cheapest survivor. Complexity per tick:
`O(A · V · (N + A·K))` for `A` requests, `V` instances/request, `N` nodes, `K`
avg chain+path length.

---

## Testing

```bash
pytest            # 27 tests: delay physics, topology, placement, GRU gradient
                  # check + learning, scaling, end-to-end, cross-process
                  # reproducibility, and proactive<reactive<static on the stress scenario
```

The GRU's backprop is verified against numerical gradients (max relative error
~1e-9), and a learning test asserts it beats persistence on a seasonal signal.

---

## Extending

- **New predictor**: subclass `prediction.base.TrafficPredictor`, register it in
  `prediction/factory.py`.
- **New workload**: subclass `simulation.workload.Workload`, register it.
- **New scaling policy**: swap the engine passed to `SFCOrchestrator`.
- **Real emulation**: the substrate interface is small enough to back with
  Mininet/Containernet or Docker VNFs later — the orchestrator and policy code
  stay unchanged.

---

