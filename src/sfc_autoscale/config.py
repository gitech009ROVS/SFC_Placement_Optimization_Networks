"""YAML/dict-driven experiment configuration.

A config fully specifies a reproducible experiment: topology, request set,
workload, predictor backend, scaling knobs, and horizon. It is validated and
turned into a :class:`~sfc_autoscale.simulation.scenario.Scenario` plus the
predictor spec used by the runner.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .core.entities import SFCRequest
from .scaling import CostConfig, ScalingConfig
from .simulation.scenario import Scenario, default_requests
from .simulation.workload import build_workload
from .topologies import build_fat_tree, build_sample_topology


@dataclass
class ExperimentConfig:
    seed: int = 0
    num_steps: int = 48
    retrain_every: int = 10
    k_paths: int = 5
    bw_cost_per_unit: float = 1.0

    topology: dict[str, Any] = field(default_factory=lambda: {"type": "sample"})
    workload: dict[str, Any] = field(default_factory=lambda: {"type": "diurnal_spike"})
    predictor: dict[str, Any] = field(default_factory=lambda: {"backend": "gru_numpy", "window_size": 20})
    scaling: dict[str, Any] = field(default_factory=lambda: {"change_threshold": 0.05})
    requests: list[dict[str, Any]] | None = None

    # ------------------------------------------------------------- loaders
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentConfig:
        sim = data.get("simulation", {})
        return cls(
            seed=data.get("seed", 0),
            num_steps=sim.get("num_steps", 48),
            retrain_every=sim.get("retrain_every", 10),
            k_paths=sim.get("k_paths", 5),
            bw_cost_per_unit=sim.get("bw_cost_per_unit", 1.0),
            topology=data.get("topology", {"type": "sample"}),
            workload=data.get("workload", {"type": "diurnal_spike"}),
            predictor=data.get("predictor", {"backend": "gru_numpy", "window_size": 20}),
            scaling=data.get("scaling", {"change_threshold": 0.05}),
            requests=data.get("requests"),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        with open(path) as fh:
            return cls.from_dict(yaml.safe_load(fh) or {})

    # ------------------------------------------------------------- builders
    def _build_topology(self):
        spec = dict(self.topology)
        kind = spec.pop("type", "sample")
        if kind == "sample":
            return build_sample_topology(**spec)
        if kind == "fat_tree":
            return build_fat_tree(**spec)
        raise ValueError(f"unknown topology type '{kind}'")

    def _build_requests(self) -> list[SFCRequest]:
        if not self.requests:
            return default_requests()
        out: list[SFCRequest] = []
        for r in self.requests:
            out.append(SFCRequest(
                request_id=r["id"],
                data_rate=float(r["data_rate"]),
                ingress_node=r["ingress"],
                egress_node=r["egress"],
                vnf_chain=list(r["chain"]),
                max_delay=float(r["max_delay"]),
                revenue=float(r["revenue"]),
            ))
        return out

    def _build_scaling(self) -> ScalingConfig:
        spec = dict(self.scaling)
        cost_spec = spec.pop("cost", {}) or {}
        return ScalingConfig(
            change_threshold=spec.get("change_threshold", 0.05),
            h_out_replicas=spec.get("h_out_replicas", 1),
            cost=CostConfig(**cost_spec),
        )

    def build_scenario(self) -> Scenario:
        wl = dict(self.workload)
        wl_type = wl.pop("type", "diurnal_spike")
        workload = build_workload(wl_type, seed=self.seed, **wl)
        return Scenario(
            network=self._build_topology(),
            requests=self._build_requests(),
            workload=workload,
            num_steps=self.num_steps,
            k_paths=self.k_paths,
            bw_cost_per_unit=self.bw_cost_per_unit,
            retrain_every=self.retrain_every,
            scaling_config=self._build_scaling(),
        )

    def predictor_spec(self) -> tuple[str, dict[str, Any]]:
        spec = dict(self.predictor)
        backend = spec.pop("backend", "gru_numpy")
        spec.setdefault("window_size", 20)
        return backend, spec
