"""A from-scratch GRU forecaster in pure NumPy (no deep-learning framework).

This exists so the predictive pipeline — the core research contribution — can be
trained and evaluated with **zero heavyweight dependencies**, anywhere, for free.
It follows PyTorch's GRU gate convention so results are comparable with
:mod:`gru_torch`.

GRU cell (PyTorch convention)::

    r_t = σ(x_t W_ir + h_{t-1} W_hr + b_r)
    z_t = σ(x_t W_iz + h_{t-1} W_hz + b_z)
    n_t = tanh(x_t W_in + r_t ⊙ (h_{t-1} W_hn + b_hn) + b_in)
    h_t = (1 - z_t) ⊙ n_t + z_t ⊙ h_{t-1}

A linear read-out maps the final hidden state to the one-step-ahead value.
Training is full BPTT with the Adam optimiser on windowed samples pooled across
all request series (a "traffic-matrix" style shared model, Paper 2).
"""
from __future__ import annotations

import numpy as np

from .base import TrafficPredictor


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class _MinMaxScaler:
    """Scales values to [0, 1] using bounds learned from training data."""

    def __init__(self) -> None:
        self.lo = 0.0
        self.hi = 1.0

    def fit(self, values: np.ndarray) -> _MinMaxScaler:
        self.lo = float(values.min())
        self.hi = float(values.max())
        if self.hi <= self.lo:
            self.hi = self.lo + 1.0
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.lo) / (self.hi - self.lo)

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return values * (self.hi - self.lo) + self.lo


class _GRUCore:
    """Weights, forward pass, and BPTT for a single-layer GRU + linear head."""

    def __init__(self, hidden_size: int, input_size: int = 1, seed: int = 0):
        self.H = hidden_size
        self.I = input_size
        rng = np.random.default_rng(seed)
        # Xavier-ish init keeps early gradients well scaled.
        s_i = 1.0 / np.sqrt(input_size)
        s_h = 1.0 / np.sqrt(hidden_size)

        def gi() -> np.ndarray:
            return rng.uniform(-s_i, s_i, size=(input_size, hidden_size))

        def gh() -> np.ndarray:
            return rng.uniform(-s_h, s_h, size=(hidden_size, hidden_size))

        self.p: dict[str, np.ndarray] = {
            "W_ir": gi(), "W_iz": gi(), "W_in": gi(),
            "W_hr": gh(), "W_hz": gh(), "W_hn": gh(),
            "b_r": np.zeros(hidden_size), "b_z": np.zeros(hidden_size),
            "b_in": np.zeros(hidden_size), "b_hn": np.zeros(hidden_size),
            "W_o": rng.uniform(-s_h, s_h, size=(hidden_size, 1)),
            "b_o": np.zeros(1),
        }

    # ------------------------------------------------------------- forward
    def forward(self, X: np.ndarray) -> tuple[np.ndarray, list[dict]]:
        """X: (B, T, I) -> yhat: (B, 1), plus per-step caches for backprop."""
        B, T, _ = X.shape
        h = np.zeros((B, self.H))
        cache: list[dict] = []
        p = self.p
        for t in range(T):
            x = X[:, t, :]
            r = _sigmoid(x @ p["W_ir"] + h @ p["W_hr"] + p["b_r"])
            z = _sigmoid(x @ p["W_iz"] + h @ p["W_hz"] + p["b_z"])
            hn_pre = h @ p["W_hn"] + p["b_hn"]
            n = np.tanh(x @ p["W_in"] + r * hn_pre + p["b_in"])
            h_new = (1.0 - z) * n + z * h
            cache.append({"x": x, "h_prev": h, "r": r, "z": z, "n": n, "hn_pre": hn_pre})
            h = h_new
        yhat = h @ p["W_o"] + p["b_o"]
        cache.append({"h_last": h})
        return yhat, cache

    # ------------------------------------------------------------ backward
    def backward(self, yhat: np.ndarray, y: np.ndarray, cache: list[dict]) -> dict[str, np.ndarray]:
        p = self.p
        B = yhat.shape[0]
        grads = {k: np.zeros_like(v) for k, v in p.items()}

        dyhat = (yhat - y) / B  # d(0.5*MSE)/dyhat, averaged over batch
        h_last = cache[-1]["h_last"]
        grads["W_o"] += h_last.T @ dyhat
        grads["b_o"] += dyhat.sum(axis=0)
        dh = dyhat @ p["W_o"].T  # (B, H)

        for t in range(len(cache) - 2, -1, -1):
            c = cache[t]
            x, h_prev = c["x"], c["h_prev"]
            r, z, n, hn_pre = c["r"], c["z"], c["n"], c["hn_pre"]

            dz = dh * (h_prev - n)
            dn = dh * (1.0 - z)
            dh_prev = dh * z

            da_n = dn * (1.0 - n * n)
            grads["W_in"] += x.T @ da_n
            grads["b_in"] += da_n.sum(axis=0)
            dr = da_n * hn_pre
            d_hn_pre = da_n * r
            grads["W_hn"] += h_prev.T @ d_hn_pre
            grads["b_hn"] += d_hn_pre.sum(axis=0)
            dh_prev += d_hn_pre @ p["W_hn"].T

            da_z = dz * z * (1.0 - z)
            grads["W_iz"] += x.T @ da_z
            grads["W_hz"] += h_prev.T @ da_z
            grads["b_z"] += da_z.sum(axis=0)
            dh_prev += da_z @ p["W_hz"].T

            da_r = dr * r * (1.0 - r)
            grads["W_ir"] += x.T @ da_r
            grads["W_hr"] += h_prev.T @ da_r
            grads["b_r"] += da_r.sum(axis=0)
            dh_prev += da_r @ p["W_hr"].T

            dh = dh_prev
        return grads


class _Adam:
    def __init__(self, params: dict[str, np.ndarray], lr: float = 0.01):
        self.lr = lr
        self.b1, self.b2, self.eps = 0.9, 0.999, 1e-8
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        self.t += 1
        for k in params:
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            m_hat = self.m[k] / (1 - self.b1**self.t)
            v_hat = self.v[k] / (1 - self.b2**self.t)
            params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class NumpyGRUPredictor(TrafficPredictor):
    """One-step-ahead traffic forecaster backed by the NumPy GRU."""

    def __init__(
        self,
        window_size: int = 20,
        hidden_size: int = 16,
        epochs: int = 60,
        lr: float = 0.01,
        batch_size: int = 32,
        min_train_samples: int = 16,
        refit_every: int = 10,
        grad_clip: float = 5.0,
        seed: int = 0,
    ):
        super().__init__(window_size)
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.min_train_samples = min_train_samples
        self.refit_every = refit_every
        self.grad_clip = grad_clip
        self.seed = seed

        self.core = _GRUCore(hidden_size, input_size=1, seed=seed)
        self.scaler = _MinMaxScaler()
        self._trained = False
        self._fits = 0
        self.last_train_loss: float = float("nan")

    # ----------------------------------------------------- data preparation
    def _build_windows(self) -> tuple[np.ndarray, np.ndarray] | None:
        W = self.window_size
        xs: list[list[float]] = []
        ys: list[float] = []
        pooled: list[float] = []
        for series in self.history.all_series().values():
            pooled.extend(series)
            for i in range(len(series) - W):
                xs.append(series[i : i + W])
                ys.append(series[i + W])
        if len(xs) < self.min_train_samples:
            return None
        self.scaler.fit(np.asarray(pooled, dtype=float))
        X = self.scaler.transform(np.asarray(xs, dtype=float))[:, :, None]  # (N, W, 1)
        y = self.scaler.transform(np.asarray(ys, dtype=float))[:, None]     # (N, 1)
        return X, y

    # ----------------------------------------------------------------- fit
    def fit(self) -> None:
        self._fits += 1
        # Throttle retraining to every ``refit_every`` calls after the first.
        if self._trained and (self._fits % self.refit_every != 1):
            return
        data = self._build_windows()
        if data is None:
            return
        X, y = data
        rng = np.random.default_rng(self.seed)
        opt = _Adam(self.core.p, lr=self.lr)
        n = X.shape[0]
        loss = float("nan")
        for _ in range(self.epochs):
            perm = rng.permutation(n)
            for start in range(0, n, self.batch_size):
                idx = perm[start : start + self.batch_size]
                xb, yb = X[idx], y[idx]
                yhat, cache = self.core.forward(xb)
                loss = float(np.mean(0.5 * (yhat - yb) ** 2))
                grads = self.core.backward(yhat, yb, cache)
                if self.grad_clip:
                    for g in grads.values():
                        np.clip(g, -self.grad_clip, self.grad_clip, out=g)
                opt.step(self.core.p, grads)
        self.last_train_loss = loss
        self._trained = True

    # ------------------------------------------------------------- predict
    def predict(self, request_ids: list[str]) -> dict[str, float]:
        out: dict[str, float] = {}
        windows: dict[str, list[float]] = {}
        for rid in request_ids:
            win = self.history.window(rid)
            if win is not None:
                windows[rid] = win

        if self._trained and windows:
            batch = self.scaler.transform(
                np.asarray(list(windows.values()), dtype=float)
            )[:, :, None]
            yhat, _ = self.core.forward(batch)
            preds = self.scaler.inverse(yhat[:, 0])
            for rid, p in zip(windows.keys(), preds, strict=True):
                out[rid] = float(max(0.0, p))

        # Fallback (untrained or too-short series): persistence.
        for rid in request_ids:
            if rid not in out:
                last = self.history.last(rid)
                out[rid] = 0.0 if last is None else last
        return out
