"""Build a predictor from a name + kwargs (dependency injection point)."""
from __future__ import annotations

import inspect

from .base import TrafficPredictor
from .gru_numpy import NumpyGRUPredictor
from .naive import (
    ExponentialSmoothingPredictor,
    LastValuePredictor,
    MovingAveragePredictor,
)


def _filter_kwargs(cls: type, kwargs: dict) -> dict:
    """Keep only kwargs that ``cls.__init__`` actually accepts.

    Lets callers pass one shared config dict (window_size, seed, hidden_size,
    epochs, ...) regardless of which backend is selected; each backend takes what
    it understands and ignores the rest.
    """
    params = inspect.signature(cls.__init__).parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in params}

_REGISTRY: dict[str, type[TrafficPredictor]] = {
    "last_value": LastValuePredictor,
    "moving_average": MovingAveragePredictor,
    "exp_smoothing": ExponentialSmoothingPredictor,
    "gru_numpy": NumpyGRUPredictor,
    "gru": NumpyGRUPredictor,  # default GRU alias -> numpy (always available)
}


def available_predictors() -> list[str]:
    names = set(_REGISTRY)
    try:
        from .gru_torch import TORCH_AVAILABLE

        if TORCH_AVAILABLE:
            names.add("gru_torch")
    except Exception:
        pass
    return sorted(names)


def build_predictor(name: str, **kwargs) -> TrafficPredictor:
    """Instantiate a predictor by registry name.

    ``gru_torch`` is resolved lazily so importing this module never requires torch.
    """
    key = name.lower()
    if key == "gru_torch":
        from .gru_torch import TORCH_AVAILABLE, TorchGRUPredictor

        if not TORCH_AVAILABLE:
            raise ImportError("backend 'gru_torch' requested but PyTorch is not installed")
        return TorchGRUPredictor(**_filter_kwargs(TorchGRUPredictor, kwargs))
    if key not in _REGISTRY:
        raise KeyError(f"unknown predictor '{name}'. Available: {available_predictors()}")
    cls = _REGISTRY[key]
    return cls(**_filter_kwargs(cls, kwargs))
