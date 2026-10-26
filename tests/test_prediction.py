"""Tests for predictors, including the NumPy GRU's correctness and learning."""
from __future__ import annotations

import numpy as np
import pytest

from sfc_autoscale.prediction import (
    LastValuePredictor,
    MovingAveragePredictor,
    NumpyGRUPredictor,
    build_predictor,
)
from sfc_autoscale.prediction.gru_numpy import _GRUCore


def test_last_value_predictor():
    p = LastValuePredictor(window_size=5)
    for v in [10, 20, 30]:
        p.observe({"R1": v})
    assert p.predict(["R1"])["R1"] == 30


def test_moving_average_predictor():
    p = MovingAveragePredictor(window_size=3)
    for v in [10, 20, 30]:
        p.observe({"R1": v})
    assert p.predict(["R1"])["R1"] == pytest.approx(20.0)


def test_gru_bptt_matches_numerical_gradient():
    rng = np.random.default_rng(0)
    core = _GRUCore(hidden_size=4, input_size=1, seed=1)
    X = rng.standard_normal((2, 4, 1))
    y = rng.standard_normal((2, 1))
    yhat, cache = core.forward(X)
    grads = core.backward(yhat, y, cache)

    def loss():
        yh, _ = core.forward(X)
        return float(np.mean(0.5 * (yh - y) ** 2))

    eps = 1e-5
    for key in ["W_ir", "W_hn", "W_o", "b_z"]:
        P = core.p[key]
        idx = tuple(0 for _ in P.shape)
        orig = P[idx]
        P[idx] = orig + eps
        lp = loss()
        P[idx] = orig - eps
        lm = loss()
        P[idx] = orig
        num = (lp - lm) / (2 * eps)
        assert num == pytest.approx(grads[key][idx], abs=1e-4)


def test_gru_learns_seasonal_signal_beats_persistence():
    rng = np.random.default_rng(0)
    t = np.arange(140)
    series = {f"R{i}": 400 + 200 * np.sin(2 * np.pi * (t + 7 * i) / 24) + rng.normal(0, 8, 140)
              for i in range(3)}

    gru = NumpyGRUPredictor(window_size=20, hidden_size=16, epochs=80, lr=0.02, seed=0)
    for k in range(100):
        gru.observe({r: float(series[r][k]) for r in series})
    gru.fit()

    sq_gru, sq_naive = [], []
    last = {r: float(series[r][99]) for r in series}
    for k in range(100, 139):
        pred = gru.predict(list(series))
        truth = {r: float(series[r][k + 1]) for r in series}
        for r in series:
            sq_gru.append((pred[r] - truth[r]) ** 2)
            sq_naive.append((last[r] - truth[r]) ** 2)
        obs = {r: float(series[r][k]) for r in series}
        gru.observe(obs)
        last = obs

    rmse_gru = float(np.sqrt(np.mean(sq_gru)))
    rmse_naive = float(np.sqrt(np.mean(sq_naive)))
    assert rmse_gru < rmse_naive  # the GRU must add value on a seasonal signal


def test_factory_filters_kwargs_for_simple_predictors():
    # Passing GRU-only kwargs must not break a classical predictor.
    p = build_predictor("moving_average", window_size=10, hidden_size=99, seed=3, epochs=5)
    assert isinstance(p, MovingAveragePredictor)
    assert p.window_size == 10
