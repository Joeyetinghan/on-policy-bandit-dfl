"""Unified torch nuisance models shared across bandit algorithms."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_


def _coalesce(value, default):
    return default if value is None else value


def _row_slices(length: int, batch_size: int):
    for start in range(0, length, batch_size):
        yield slice(start, min(start + batch_size, length))


def resolve_torch_device(config_value: str):
    """Resolve the configured torch device."""
    device_name = str(config_value or "auto").lower()

    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if device_name == "cpu":
        return torch.device("cpu")

    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("torch_device='cuda' requested but CUDA is unavailable")
        return torch.device("cuda")

    if device_name == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise ValueError("torch_device='mps' requested but MPS is unavailable")
        return torch.device("mps")

    raise ValueError(f"Unknown torch_device: {config_value}")


def _parse_hidden_dims(config: dict | None) -> tuple[int, ...]:
    raw_dims = (config or {}).get("nuisance_hidden_dims")
    if raw_dims is None:
        return (256, 256)
    if not isinstance(raw_dims, Sequence) or isinstance(raw_dims, (str, bytes)):
        raise ValueError("nuisance_hidden_dims must be a sequence of positive integers")
    dims = tuple(int(dim) for dim in raw_dims)
    if not dims or any(dim <= 0 for dim in dims):
        raise ValueError("nuisance_hidden_dims must contain at least one positive integer")
    return dims


def _resolve_row_batch_size(config: dict | None) -> int:
    return max(
        1,
        int(
            _coalesce(
                (config or {}).get("nuisance_row_batch_size"),
                _coalesce((config or {}).get("shared_generative_row_batch_size"), 256),
            )
        ),
    )


def _build_mlp(input_dim: int, hidden_dims: Sequence[int], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, int(hidden_dim)))
        layers.append(nn.ReLU())
        prev_dim = int(hidden_dim)
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


class TorchNuisanceModelBase:
    """Base class for benchmark-aware nuisance networks."""

    context_mode = "global"

    def __init__(self, p: int, q: int, hidden_dims: Sequence[int], device):
        self._torch = torch
        self.p = int(p)
        self.q = int(q)
        self.hidden_dims = tuple(int(dim) for dim in hidden_dims)
        self.device = device

    def parameters(self):
        return self.net.parameters()

    def __call__(self, context):
        return self.forward(context)

    def vector_tensor(self, values):
        if isinstance(values, self._torch.Tensor):
            tensor = values.to(self.device, dtype=self._torch.float32)
        else:
            tensor = self._torch.as_tensor(values, dtype=self._torch.float32, device=self.device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 2 or tensor.shape[1] != self.q:
            raise ValueError(f"Expected nuisance vector shape ({self.q},) or (1, {self.q}), got {tuple(tensor.shape)}")
        return tensor

    def scalar_target_tensor(self, value: float):
        return self._torch.as_tensor([float(value)], dtype=self._torch.float32, device=self.device)

    def predict(self, context) -> np.ndarray:
        with self._torch.no_grad():
            return self(context).detach().cpu().numpy().reshape(-1)


class GlobalNuisanceMLP(TorchNuisanceModelBase):
    """MLP nuisance model for vector contexts."""

    context_mode = "global"

    def __init__(self, p: int, q: int, hidden_dims: Sequence[int], device):
        super().__init__(p, q, hidden_dims, device=device)
        self.net = _build_mlp(self.p, self.hidden_dims, self.q).to(self.device)

    def _context_tensor(self, context):
        if isinstance(context, self._torch.Tensor):
            tensor = context.to(self.device, dtype=self._torch.float32)
        else:
            tensor = self._torch.as_tensor(context, dtype=self._torch.float32, device=self.device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 2 or tensor.shape[1] != self.p:
            raise ValueError(f"Expected global nuisance context with trailing dim {self.p}, got {tuple(tensor.shape)}")
        return tensor

    def forward(self, context):
        return self.net(self._context_tensor(context))


class SharedLocalNuisanceMLP(TorchNuisanceModelBase):
    """Row-wise nuisance MLP for local-feature context matrices."""

    context_mode = "shared_local"

    def __init__(self, p: int, q: int, hidden_dims: Sequence[int], device, row_batch_size: int):
        super().__init__(p, q, hidden_dims, device=device)
        self.row_batch_size = max(1, int(row_batch_size))
        self.net = _build_mlp(self.p, self.hidden_dims, 1).to(self.device)

    def _context_tensor(self, context):
        if isinstance(context, self._torch.Tensor):
            tensor = context.to(self.device, dtype=self._torch.float32)
        else:
            tensor = self._torch.as_tensor(context, dtype=self._torch.float32, device=self.device)
        if tensor.ndim != 2 or tensor.shape != (self.q, self.p):
            raise ValueError(f"Expected shared-local nuisance context shape ({self.q}, {self.p}), got {tuple(tensor.shape)}")
        return tensor

    def forward(self, context):
        context_tensor = self._context_tensor(context)
        outputs = []
        for row_slice in _row_slices(self.q, self.row_batch_size):
            outputs.append(self.net(context_tensor[row_slice]).squeeze(1))
        return self._torch.cat(outputs, dim=0).unsqueeze(0)


def build_nuisance_model(config: dict, p: int, q: int) -> TorchNuisanceModelBase:
    """Construct the benchmark-specific nuisance network."""
    hidden_dims = _parse_hidden_dims(config)
    device = resolve_torch_device(config.get("torch_device", "auto"))
    benchmark = str(config.get("benchmark", "")).lower()
    model_type = str(config.get("model_type", "")).lower()
    if benchmark in {"energy", "pricing"} or model_type.startswith("shared_"):
        return SharedLocalNuisanceMLP(
            p,
            q,
            hidden_dims=hidden_dims,
            device=device,
            row_batch_size=_resolve_row_batch_size(config),
        )
    return GlobalNuisanceMLP(p, q, hidden_dims=hidden_dims, device=device)


def build_nuisance_optimizer(model: TorchNuisanceModelBase, config: dict):
    """Construct the nuisance optimizer from experiment config."""
    lr = float(_coalesce(config.get("nuisance_lr"), config.get("theta_lr", 0.01)))
    weight_decay = float(_coalesce(config.get("nuisance_lambda_reg"), 0.0))
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def nuisance_reward_from_vector(model: TorchNuisanceModelBase, vector_prediction, action, oracle=None):
    """Predict scalar reward from a nuisance vector and action."""
    if oracle is not None and hasattr(oracle, "feedback_loss_torch"):
        return -oracle.feedback_loss_torch(vector_prediction, action)
    return model._torch.sum(vector_prediction * model.vector_tensor(action), dim=1)


def nuisance_cost_from_vector(model: TorchNuisanceModelBase, vector_prediction, action, oracle=None):
    """Predict scalar cost/loss from a nuisance vector and action."""
    if oracle is not None and hasattr(oracle, "feedback_loss_torch"):
        return oracle.feedback_loss_torch(vector_prediction, action)
    return model._torch.sum(vector_prediction * model.vector_tensor(action), dim=1)


def nuisance_scalar_prediction(model: TorchNuisanceModelBase, context, action, oracle=None):
    """Predict the scalar observation associated with one action."""
    nuisance_pred = model(context)
    return nuisance_reward_from_vector(model, nuisance_pred, action, oracle=oracle)


def nuisance_cost_scalar_prediction(model: TorchNuisanceModelBase, context, action, oracle=None):
    """Predict the scalar cost/loss associated with one action."""
    nuisance_pred = model(context)
    return nuisance_cost_from_vector(model, nuisance_pred, action, oracle=oracle)


def nuisance_scalar_loss(model: TorchNuisanceModelBase, context, action, target_scalar: float, oracle=None):
    """Squared loss on one observed scalar bandit outcome."""
    target = model.scalar_target_tensor(target_scalar)
    pred = nuisance_scalar_prediction(model, context, action, oracle=oracle)
    return 0.5 * (pred - target) ** 2


def nuisance_cost_scalar_loss(model: TorchNuisanceModelBase, context, action, target_scalar: float, oracle=None):
    """Squared loss on one observed scalar cost/loss outcome."""
    target = model.scalar_target_tensor(target_scalar)
    pred = nuisance_cost_scalar_prediction(model, context, action, oracle=oracle)
    return 0.5 * (pred - target) ** 2


def nuisance_masked_vector_loss(model: TorchNuisanceModelBase, context, observed_vector, observation_mask):
    """Average squared loss over the observed coordinates in a semi-bandit/full-feedback vector."""
    torch = model._torch
    pred = model(context)
    target = model.vector_tensor(observed_vector)
    mask = model.vector_tensor(observation_mask)
    denom = torch.clamp(mask.sum(dim=1), min=1.0)
    masked_sq_error = torch.sum(mask * (pred - target) ** 2, dim=1)
    return 0.5 * masked_sq_error / denom


def linear_observation_loss_from_prediction(model: TorchNuisanceModelBase, pred, observation_matrix, observed_values):
    """Squared loss for linear measurements of a predicted score vector."""
    torch = model._torch
    pred_tensor = model.vector_tensor(pred)
    matrix = torch.as_tensor(observation_matrix, dtype=torch.float32, device=model.device)
    if matrix.ndim == 1:
        matrix = matrix.unsqueeze(0)
    if matrix.ndim != 2 or matrix.shape[1] != model.q:
        raise ValueError(
            f"Expected linear observation matrix shape (num_observations, {model.q}), got {tuple(matrix.shape)}"
        )
    target = torch.as_tensor(observed_values, dtype=torch.float32, device=model.device).reshape(1, -1)
    if target.shape[1] != matrix.shape[0]:
        raise ValueError(
            "linear observation value count must match matrix row count, "
            f"got {target.shape[1]} and {matrix.shape[0]}"
        )
    pred_values = pred_tensor @ matrix.T
    return 0.5 * torch.mean((pred_values - target) ** 2, dim=1)


def nuisance_linear_observation_loss(model: TorchNuisanceModelBase, context, observation_matrix, observed_values):
    """Squared nuisance loss for semi-bandit linear measurements."""
    return linear_observation_loss_from_prediction(model, model(context), observation_matrix, observed_values)


def fill_observed_components(imputed_vector, observed_vector, observation_mask):
    """Overwrite observed coordinates in an imputed score vector."""
    imputed = np.asarray(imputed_vector, dtype=float).reshape(-1).copy()
    observed = np.asarray(observed_vector, dtype=float).reshape(-1)
    mask = np.asarray(observation_mask, dtype=float).reshape(-1) > 0.0
    if imputed.shape != observed.shape or imputed.shape != mask.shape:
        raise ValueError(
            "fill_observed_components requires imputed_vector, observed_vector, and observation_mask to share shape."
        )
    imputed[mask] = observed[mask]
    return imputed


def nuisance_update_step(
    model: TorchNuisanceModelBase,
    optimizer,
    context,
    action,
    target_scalar: float,
    *,
    grad_clip_norm: float | None,
) -> float:
    """Take one nuisance optimizer step against an observed scalar target."""
    loss = nuisance_scalar_loss(model, context, action, target_scalar)
    optimizer.zero_grad(set_to_none=True)
    loss.mean().backward()
    if grad_clip_norm is not None:
        clip_grad_norm_(model.parameters(), float(grad_clip_norm))
    optimizer.step()
    return float(loss.mean().detach().cpu().item())
