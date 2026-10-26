"""PyTorch GRU forecaster (optional extra).

API-compatible with :class:`NumpyGRUPredictor`. Selected when PyTorch is
installed and ``backend="torch"`` is configured; otherwise the NumPy GRU is used.
Import of ``torch`` is deferred to construction so the package imports fine
without it.
"""
from __future__ import annotations

import numpy as np

from .base import TrafficPredictor

try:  # pragma: no cover - exercised only when torch is present
    import torch
    from torch import nn

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    TORCH_AVAILABLE = False


class TorchGRUPredictor(TrafficPredictor):
    """One-step-ahead forecaster using ``torch.nn.GRU`` + a linear head."""

    def __init__(
        self,
        window_size: int = 20,
        hidden_size: int = 16,
        num_layers: int = 1,
        epochs: int = 60,
        lr: float = 0.01,
        batch_size: int = 32,
        min_train_samples: int = 16,
        refit_every: int = 10,
        seed: int = 0,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is not installed. Install with `pip install sfc-autoscale[torch]` "
                "or use backend='numpy'."
            )
        super().__init__(window_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.min_train_samples = min_train_samples
        self.refit_every = refit_every
        self.seed = seed

        torch.manual_seed(seed)
        self._model = _GRUNet(hidden_size, num_layers)
        self._lo = 0.0
        self._hi = 1.0
        self._trained = False
        self._fits = 0
        self.last_train_loss = float("nan")

    def _scale(self, a: np.ndarray) -> np.ndarray:
        return (a - self._lo) / (self._hi - self._lo)

    def _unscale(self, a: np.ndarray) -> np.ndarray:
        return a * (self._hi - self._lo) + self._lo

    def _build_windows(self):
        W = self.window_size
        xs, ys, pooled = [], [], []
        for series in self.history.all_series().values():
            pooled.extend(series)
            for i in range(len(series) - W):
                xs.append(series[i : i + W])
                ys.append(series[i + W])
        if len(xs) < self.min_train_samples:
            return None
        pooled_arr = np.asarray(pooled, dtype=float)
        self._lo = float(pooled_arr.min())
        self._hi = float(pooled_arr.max())
        if self._hi <= self._lo:
            self._hi = self._lo + 1.0
        X = self._scale(np.asarray(xs, dtype=np.float32))[:, :, None]
        y = self._scale(np.asarray(ys, dtype=np.float32))[:, None]
        return torch.from_numpy(X).float(), torch.from_numpy(y).float()

    def fit(self) -> None:
        self._fits += 1
        if self._trained and (self._fits % self.refit_every != 1):
            return
        data = self._build_windows()
        if data is None:
            return
        X, y = data
        opt = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        n = X.shape[0]
        g = torch.Generator().manual_seed(self.seed)
        self._model.train()
        loss_val = float("nan")
        for _ in range(self.epochs):
            perm = torch.randperm(n, generator=g)
            for start in range(0, n, self.batch_size):
                idx = perm[start : start + self.batch_size]
                opt.zero_grad()
                out = self._model(X[idx])
                loss = loss_fn(out, y[idx])
                loss.backward()
                opt.step()
                loss_val = float(loss.item())
        self.last_train_loss = loss_val
        self._trained = True

    def predict(self, request_ids: list[str]) -> dict[str, float]:
        out: dict[str, float] = {}
        windows = {}
        for rid in request_ids:
            win = self.history.window(rid)
            if win is not None:
                windows[rid] = win
        if self._trained and windows:
            self._model.eval()
            batch = self._scale(np.asarray(list(windows.values()), dtype=np.float32))[:, :, None]
            with torch.no_grad():
                preds = self._model(torch.from_numpy(batch).float()).numpy()[:, 0]
            preds = self._unscale(preds)
            for rid, p in zip(windows.keys(), preds, strict=True):
                out[rid] = float(max(0.0, p))
        for rid in request_ids:
            if rid not in out:
                last = self.history.last(rid)
                out[rid] = 0.0 if last is None else last
        return out


if TORCH_AVAILABLE:

    class _GRUNet(nn.Module):
        def __init__(self, hidden_size: int, num_layers: int):
            super().__init__()
            self.gru = nn.GRU(1, hidden_size, num_layers=num_layers, batch_first=True)
            self.head = nn.Linear(hidden_size, 1)

        def forward(self, x):  # x: (B, T, 1)
            out, _ = self.gru(x)
            return self.head(out[:, -1, :])  # last hidden -> (B, 1)

else:  # pragma: no cover

    class _GRUNet:  # type: ignore[no-redef]
        def __init__(self, *a, **k):
            raise ImportError("PyTorch not available")
