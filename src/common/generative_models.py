"""Torch-based generative reward models used by existing bandit algorithms."""

from __future__ import annotations

import math
from itertools import chain
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_


def _coalesce(value, default):
    return default if value is None else value


def _row_slices(length: int, batch_size: int):
    for start in range(0, length, batch_size):
        yield slice(start, min(start + batch_size, length))


def resolve_torch_device(config_value: str) -> torch.device:
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


class TorchGenerativeModelBase:
    """Thin wrapper around a torch-backed conditional generative model."""

    model_family = "generative"
    uses_torch = True
    context_mode = "global"
    actor_kind = None

    def __init__(self, p: int, q: int, seed: int | None = None, config: dict | None = None):
        self.p = p
        self.q = q
        self.rng_seed = seed
        self.config = config or {}
        self.device = resolve_torch_device(self.config.get("torch_device", "auto"))
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        self.grad_clip_norm = _coalesce(
            self.config.get("generative_grad_clip_norm"),
            self.config.get("grad_clip_norm"),
        )
        self._modules_for_grad: list[nn.Module] = []
        self.optimizer = None

    def initialize_params(self):
        return None

    def parameters(self) -> Iterable[torch.nn.Parameter]:
        return chain.from_iterable(module.parameters() for module in self._modules_for_grad)

    def zero_grad(self):
        if self.optimizer is None:
            raise RuntimeError("Generative model optimizer has not been initialized")
        self.optimizer.zero_grad(set_to_none=True)

    def step(self):
        if self.optimizer is None:
            raise RuntimeError("Generative model optimizer has not been initialized")
        self.optimizer.step()

    def clip_grad(self, max_norm=None):
        clip_value = self.grad_clip_norm if max_norm is None else max_norm
        if clip_value is None:
            return
        params = [param for param in self.parameters() if param.grad is not None]
        if params:
            clip_grad_norm_(params, float(clip_value))

    def _learning_rate(self) -> float:
        return float(_coalesce(self.config.get("generative_lr"), self.config.get("theta_lr", 1e-3)))

    def _weight_decay(self) -> float:
        return float(_coalesce(self.config.get("generative_weight_decay"), 0.0))

    def _context_tensor(self, context) -> torch.Tensor:
        if isinstance(context, torch.Tensor):
            tensor = context.to(self.device, dtype=torch.float32)
        else:
            tensor = torch.as_tensor(context, dtype=torch.float32, device=self.device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    def _cost_tensor(self, cost) -> torch.Tensor:
        if isinstance(cost, torch.Tensor):
            tensor = cost.to(self.device, dtype=torch.float32)
        else:
            tensor = torch.as_tensor(cost, dtype=torch.float32, device=self.device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    @staticmethod
    def _match_batch(tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
        if tensor.shape[0] == batch_size:
            return tensor
        if tensor.shape[0] == 1:
            return tensor.expand(batch_size, -1)
        raise ValueError(f"Cannot broadcast batch {tensor.shape[0]} to {batch_size}")

    def predict(self, x, theta=None) -> np.ndarray:
        del theta
        with torch.no_grad():
            sampled_cost = self.sample_cost(x)
        return (-sampled_cost).detach().cpu().numpy().reshape(-1)

    def sample_cost(self, context_tensor) -> torch.Tensor:
        raise NotImplementedError

    def sample_for_action(self, context):
        """Sample one detached action cost and keep a differentiable log-prob proxy."""
        sampled_cost = self.sample_cost(context)
        score_info = {
            "sampled_cost": sampled_cost.detach(),
            "log_prob_proxy": self.log_prob_proxy(sampled_cost.detach(), context),
        }
        return sampled_cost, score_info

    def score_function_loss(self, score_info, advantage) -> torch.Tensor:
        """Cost-minimization score-function objective.

        Minimizing this scalar gives gradient
        advantage * grad log p_theta(sample | context).
        """
        advantage_tensor = torch.as_tensor(float(advantage), dtype=torch.float32, device=self.device)
        return advantage_tensor * score_info["log_prob_proxy"].mean()

    def sample_scenarios(self, context, K: int, keep_graph: bool = True) -> torch.Tensor:
        raise NotImplementedError

    def reparam_sample_scenarios(self, context, K: int, keep_graph: bool = True) -> torch.Tensor:
        return self.sample_scenarios(context, K=K, keep_graph=keep_graph)

    def log_prob_proxy(self, target_cost, context) -> torch.Tensor:
        return -self.nll_proxy(target_cost, context)

    def nll_proxy(self, target_cost, context) -> torch.Tensor:
        raise NotImplementedError

    def regularizer_loss(self, target_cost, context) -> torch.Tensor:
        return self.nll_proxy(target_cost, context)

    def compute_log_prob_surrogate(self, target_cost, context) -> torch.Tensor:
        return self.nll_proxy(target_cost, context)


class SharedLocalGenerativeModelBase(TorchGenerativeModelBase):
    """Torch generative model base for local-feature context matrices."""

    context_mode = "shared_local"

    def __init__(self, p: int, q: int, seed: int | None = None, config: dict | None = None):
        super().__init__(p, q, seed=seed, config=config)
        self.row_batch_size = max(1, int(_coalesce(self.config.get("shared_generative_row_batch_size"), 256)))

    def _context_tensor(self, context) -> torch.Tensor:
        if isinstance(context, torch.Tensor):
            tensor = context.to(self.device, dtype=torch.float32)
        else:
            tensor = torch.as_tensor(context, dtype=torch.float32, device=self.device)
        if tensor.ndim != 2:
            raise ValueError(f"Expected shared-local context with ndim=2, got ndim={tensor.ndim}")
        if tensor.shape != (self.q, self.p):
            raise ValueError(f"Expected shared-local context shape ({self.q}, {self.p}), got {tuple(tensor.shape)}")
        return tensor

    def _flatten_cost(self, cost) -> torch.Tensor:
        if isinstance(cost, torch.Tensor):
            tensor = cost.to(self.device, dtype=torch.float32)
        else:
            tensor = torch.as_tensor(cost, dtype=torch.float32, device=self.device)
        if tensor.ndim == 1:
            flat = tensor
        elif tensor.ndim == 2 and tensor.shape == (1, self.q):
            flat = tensor.squeeze(0)
        elif tensor.ndim == 2 and tensor.shape == (self.q, 1):
            flat = tensor.squeeze(1)
        else:
            raise ValueError(
                f"Expected shared-local cost with shape ({self.q},), (1, {self.q}), or ({self.q}, 1); "
                f"got {tuple(tensor.shape)}"
            )
        if flat.shape != (self.q,):
            raise ValueError(f"Expected shared-local cost length {self.q}, got {tuple(flat.shape)}")
        return flat

    def _cost_tensor(self, cost) -> torch.Tensor:
        return self._flatten_cost(cost).unsqueeze(0)

    def _local_cost_tensor(self, cost) -> torch.Tensor:
        return self._flatten_cost(cost).unsqueeze(1)


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal timestep embedding with fixed output width."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        timesteps = timesteps.float()
        half_dim = self.dim // 2
        if half_dim == 0:
            return timesteps.unsqueeze(1)

        scale = math.log(10000.0) / max(half_dim - 1, 1)
        freqs = torch.exp(-scale * torch.arange(half_dim, device=timesteps.device, dtype=torch.float32))
        args = timesteps.unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if emb.shape[1] < self.dim:
            emb = torch.cat([emb, torch.zeros((emb.shape[0], self.dim - emb.shape[1]), device=emb.device)], dim=1)
        return emb


class DiffusionDenoiser(nn.Module):
    """Time-conditioned MLP denoiser."""

    def __init__(self, p: int, q: int, time_embed_dim: int = 8, hidden_dim: int = 128):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        in_dim = q + time_embed_dim + p
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, q),
        )

    def forward(self, x_t: torch.Tensor, timesteps: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        t_embed = self.time_embed(timesteps)
        hidden = torch.cat([x_t, t_embed, context], dim=1)
        return self.net(hidden)


class DiffusionGenerativeModel(TorchGenerativeModelBase):
    """Conditional diffusion model with DDIM sampling and ELBO surrogate."""

    model_type = "diffusion"
    actor_kind = "diffusion"

    def __init__(self, p: int, q: int, seed: int | None = None, config: dict | None = None):
        super().__init__(p, q, seed=seed, config=config)
        time_embed_dim = int(_coalesce(self.config.get("diffusion_time_embed_dim"), 8))
        hidden_dim = int(_coalesce(self.config.get("diffusion_hidden_dim"), 128))
        self.num_steps = max(1, int(_coalesce(self.config.get("diffusion_num_steps"), 50)))
        self.inference_steps = max(1, int(_coalesce(self.config.get("diffusion_inference_steps"), self.num_steps)))
        self.ddim_eta = float(_coalesce(self.config.get("diffusion_ddim_eta"), 0.0))
        beta_start = float(_coalesce(self.config.get("diffusion_beta_start"), 1.0e-4))
        beta_end = float(_coalesce(self.config.get("diffusion_beta_end"), 2.0e-2))
        self.surrogate_num_iter = max(1, int(_coalesce(self.config.get("generative_surrogate_num_iter"), 64)))

        self.denoiser = DiffusionDenoiser(
            p=p,
            q=q,
            time_embed_dim=time_embed_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)
        self._modules_for_grad = [self.denoiser]
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self._learning_rate(),
            weight_decay=self._weight_decay(),
        )

        self.beta = torch.linspace(beta_start, beta_end, self.num_steps, device=self.device)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        # ``diffusion_loss_weighting`` selects how the regularizer weights the
        # per-timestep MSE on the noise prediction:
        #   - "simple" / "ho2020": L_simple, unweighted (Ho et al. 2020 DDPM)
        #   - "elbo": (1 - alpha_bar) / (2 sigma_t^2 alpha_bar) — variational LB
        #   - "sqrt_one_minus_alpha_bar" (legacy default): mild mid-noise emphasis
        weighting = str(_coalesce(self.config.get("diffusion_loss_weighting"), "sqrt_one_minus_alpha_bar")).lower()
        if weighting in {"simple", "ho2020", "unweighted", "l_simple"}:
            self.elbo_weight = torch.ones_like(self.alpha_bar)
        elif weighting in {"elbo", "vlb", "tight"}:
            prev_alpha_bar = torch.cat([torch.ones(1, device=self.device), self.alpha_bar[:-1]])
            sigma2 = torch.clamp(self.beta * (1.0 - prev_alpha_bar), min=1.0e-12)
            self.elbo_weight = (1.0 - self.alpha_bar) / (2.0 * sigma2 * self.alpha_bar)
        elif weighting in {"sqrt_one_minus_alpha_bar", "legacy", "default"}:
            self.elbo_weight = torch.sqrt(1.0 - self.alpha_bar)
        else:
            raise ValueError(
                "diffusion_loss_weighting must be one of {'simple','elbo','sqrt_one_minus_alpha_bar'}; "
                f"got {weighting!r}"
            )
        self._inference_schedule = self._build_inference_schedule()

    def _build_inference_schedule(self) -> list[int]:
        raw = torch.linspace(self.num_steps - 1, 0, steps=max(self.inference_steps, 1))
        schedule = []
        for value in raw.tolist():
            timestep = int(round(value))
            if not schedule or timestep != schedule[-1]:
                schedule.append(timestep)
        if schedule[-1] != 0:
            schedule.append(0)
        return schedule

    def _reverse_sample(self, context_tensor, *, num_samples: int = 1, keep_graph: bool = False) -> torch.Tensor:
        context = self._context_tensor(context_tensor)
        base_batch_size = context.shape[0]
        sample_count = max(1, int(num_samples))
        context_rep = context.repeat_interleave(sample_count, dim=0)
        batch_size = context_rep.shape[0]

        def _sample_body():
            x_t = torch.randn((batch_size, self.q), device=self.device)
            for idx, timestep in enumerate(self._inference_schedule):
                next_t = self._inference_schedule[idx + 1] if idx + 1 < len(self._inference_schedule) else -1
                t_batch = torch.full((batch_size,), timestep, device=self.device, dtype=torch.long)
                eps_hat = self.denoiser(x_t, t_batch, context_rep)
                alpha_bar_t = self.alpha_bar[timestep]
                sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
                sqrt_one_minus_t = torch.sqrt(1.0 - alpha_bar_t)
                x0_hat = (x_t - sqrt_one_minus_t * eps_hat) / torch.clamp(sqrt_alpha_bar_t, min=1.0e-12)

                if next_t < 0:
                    x_t = x0_hat
                    break

                alpha_bar_next = self.alpha_bar[next_t]
                sigma = self.ddim_eta * torch.sqrt(
                    torch.clamp(
                        ((1.0 - alpha_bar_next) / torch.clamp(1.0 - alpha_bar_t, min=1.0e-12))
                        * (1.0 - alpha_bar_t / torch.clamp(alpha_bar_next, min=1.0e-12)),
                        min=0.0,
                    )
                )
                direction_scale = torch.sqrt(torch.clamp(1.0 - alpha_bar_next - sigma**2, min=0.0))
                if float(sigma.item()) > 0.0:
                    noise = torch.randn_like(x_t)
                    x_t = torch.sqrt(alpha_bar_next) * x0_hat + direction_scale * eps_hat + sigma * noise
                else:
                    x_t = torch.sqrt(alpha_bar_next) * x0_hat + direction_scale * eps_hat
            return x_t.view(base_batch_size, sample_count, self.q)

        if keep_graph:
            return _sample_body()
        with torch.no_grad():
            return _sample_body().detach()

    def sample_cost(self, context_tensor) -> torch.Tensor:
        return self._reverse_sample(context_tensor, num_samples=1, keep_graph=False).squeeze(1)

    def sample_scenarios(self, context, K: int, keep_graph: bool = True) -> torch.Tensor:
        scenarios = self._reverse_sample(context, num_samples=K, keep_graph=keep_graph)
        return scenarios.squeeze(0) if scenarios.shape[0] == 1 else scenarios

    def reparam_sample_scenarios(self, context, K: int, keep_graph: bool = True) -> torch.Tensor:
        return self.sample_scenarios(context, K=K, keep_graph=keep_graph)

    def _diffusion_surrogate(self, target_cost, context) -> torch.Tensor:
        x0 = self._cost_tensor(target_cost).detach().clone()
        context_tensor = self._context_tensor(context)
        context_tensor = self._match_batch(context_tensor, x0.shape[0])

        expanded_x0 = x0.repeat_interleave(self.surrogate_num_iter, dim=0)
        expanded_context = context_tensor.repeat_interleave(self.surrogate_num_iter, dim=0)
        batch_size = expanded_x0.shape[0]

        timesteps = torch.randint(0, self.num_steps, (batch_size,), device=self.device)
        noise = torch.randn_like(expanded_x0)
        alpha_bar_t = self.alpha_bar[timesteps].unsqueeze(1)
        x_t = torch.sqrt(alpha_bar_t) * expanded_x0 + torch.sqrt(1.0 - alpha_bar_t) * noise
        eps_hat = self.denoiser(x_t, timesteps, expanded_context)
        raw_mse = torch.sum((eps_hat - noise) ** 2, dim=1)
        weights = self.elbo_weight[timesteps]
        return torch.mean(weights * raw_mse)

    def nll_proxy(self, target_cost, context) -> torch.Tensor:
        return self._diffusion_surrogate(target_cost, context)


class SharedDiffusionGenerativeModel(SharedLocalGenerativeModelBase):
    """Row-wise conditional scalar diffusion over local-feature contexts."""

    model_type = "shared_diffusion"
    actor_kind = "diffusion"

    def __init__(self, p: int, q: int, seed: int | None = None, config: dict | None = None):
        super().__init__(p, q, seed=seed, config=config)
        time_embed_dim = int(_coalesce(self.config.get("diffusion_time_embed_dim"), 8))
        hidden_dim = int(_coalesce(self.config.get("diffusion_hidden_dim"), 128))
        self.num_steps = max(1, int(_coalesce(self.config.get("diffusion_num_steps"), 50)))
        self.inference_steps = max(1, int(_coalesce(self.config.get("diffusion_inference_steps"), self.num_steps)))
        self.ddim_eta = float(_coalesce(self.config.get("diffusion_ddim_eta"), 0.0))
        beta_start = float(_coalesce(self.config.get("diffusion_beta_start"), 1.0e-4))
        beta_end = float(_coalesce(self.config.get("diffusion_beta_end"), 2.0e-2))
        self.surrogate_num_iter = max(1, int(_coalesce(self.config.get("generative_surrogate_num_iter"), 64)))

        self.denoiser = DiffusionDenoiser(
            p=p,
            q=1,
            time_embed_dim=time_embed_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)
        self._modules_for_grad = [self.denoiser]
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self._learning_rate(),
            weight_decay=self._weight_decay(),
        )

        self.beta = torch.linspace(beta_start, beta_end, self.num_steps, device=self.device)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        # ``diffusion_loss_weighting`` selects how the regularizer weights the
        # per-timestep MSE on the noise prediction:
        #   - "simple" / "ho2020": L_simple, unweighted (Ho et al. 2020 DDPM)
        #   - "elbo": (1 - alpha_bar) / (2 sigma_t^2 alpha_bar) — variational LB
        #   - "sqrt_one_minus_alpha_bar" (legacy default): mild mid-noise emphasis
        weighting = str(_coalesce(self.config.get("diffusion_loss_weighting"), "sqrt_one_minus_alpha_bar")).lower()
        if weighting in {"simple", "ho2020", "unweighted", "l_simple"}:
            self.elbo_weight = torch.ones_like(self.alpha_bar)
        elif weighting in {"elbo", "vlb", "tight"}:
            prev_alpha_bar = torch.cat([torch.ones(1, device=self.device), self.alpha_bar[:-1]])
            sigma2 = torch.clamp(self.beta * (1.0 - prev_alpha_bar), min=1.0e-12)
            self.elbo_weight = (1.0 - self.alpha_bar) / (2.0 * sigma2 * self.alpha_bar)
        elif weighting in {"sqrt_one_minus_alpha_bar", "legacy", "default"}:
            self.elbo_weight = torch.sqrt(1.0 - self.alpha_bar)
        else:
            raise ValueError(
                "diffusion_loss_weighting must be one of {'simple','elbo','sqrt_one_minus_alpha_bar'}; "
                f"got {weighting!r}"
            )
        self._inference_schedule = self._build_inference_schedule()

    def _build_inference_schedule(self) -> list[int]:
        raw = torch.linspace(self.num_steps - 1, 0, steps=max(self.inference_steps, 1))
        schedule = []
        for value in raw.tolist():
            timestep = int(round(value))
            if not schedule or timestep != schedule[-1]:
                schedule.append(timestep)
        if schedule[-1] != 0:
            schedule.append(0)
        return schedule

    def _sample_row_batch(self, context_batch: torch.Tensor, *, num_samples: int = 1) -> torch.Tensor:
        row_count = context_batch.shape[0]
        sample_count = max(1, int(num_samples))
        context_rep = context_batch.repeat_interleave(sample_count, dim=0)
        batch_size = context_rep.shape[0]
        x_t = torch.randn((batch_size, 1), device=self.device)
        for idx, timestep in enumerate(self._inference_schedule):
            next_t = self._inference_schedule[idx + 1] if idx + 1 < len(self._inference_schedule) else -1
            t_batch = torch.full((batch_size,), timestep, device=self.device, dtype=torch.long)
            eps_hat = self.denoiser(x_t, t_batch, context_rep)
            alpha_bar_t = self.alpha_bar[timestep]
            sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
            sqrt_one_minus_t = torch.sqrt(1.0 - alpha_bar_t)
            x0_hat = (x_t - sqrt_one_minus_t * eps_hat) / torch.clamp(sqrt_alpha_bar_t, min=1.0e-12)

            if next_t < 0:
                x_t = x0_hat
                break

            alpha_bar_next = self.alpha_bar[next_t]
            sigma = self.ddim_eta * torch.sqrt(
                torch.clamp(
                    ((1.0 - alpha_bar_next) / torch.clamp(1.0 - alpha_bar_t, min=1.0e-12))
                    * (1.0 - alpha_bar_t / torch.clamp(alpha_bar_next, min=1.0e-12)),
                    min=0.0,
                )
            )
            direction_scale = torch.sqrt(torch.clamp(1.0 - alpha_bar_next - sigma**2, min=0.0))
            if float(sigma.item()) > 0.0:
                noise = torch.randn_like(x_t)
                x_t = torch.sqrt(alpha_bar_next) * x0_hat + direction_scale * eps_hat + sigma * noise
            else:
                x_t = torch.sqrt(alpha_bar_next) * x0_hat + direction_scale * eps_hat
        return x_t.view(row_count, sample_count).transpose(0, 1)

    def _sample_rows(self, context_tensor, *, num_samples: int = 1, keep_graph: bool = False) -> torch.Tensor:
        context = self._context_tensor(context_tensor)

        def _sample_body():
            row_samples = []
            for row_slice in _row_slices(context.shape[0], self.row_batch_size):
                row_samples.append(self._sample_row_batch(context[row_slice], num_samples=num_samples))
            return torch.cat(row_samples, dim=1)

        if keep_graph:
            return _sample_body()
        with torch.no_grad():
            return _sample_body().detach()

    def sample_cost(self, context_tensor) -> torch.Tensor:
        return self._sample_rows(context_tensor, num_samples=1, keep_graph=False)

    def sample_scenarios(self, context, K: int, keep_graph: bool = True) -> torch.Tensor:
        return self._sample_rows(context, num_samples=K, keep_graph=keep_graph)

    def reparam_sample_scenarios(self, context, K: int, keep_graph: bool = True) -> torch.Tensor:
        return self.sample_scenarios(context, K=K, keep_graph=keep_graph)

    def _diffusion_surrogate(self, target_cost, context) -> torch.Tensor:
        x0 = self._local_cost_tensor(target_cost).detach().clone()
        context_tensor = self._context_tensor(context)
        total_loss = torch.zeros((), device=self.device)
        total_count = 0

        for row_slice in _row_slices(self.q, self.row_batch_size):
            x0_batch = x0[row_slice]
            context_batch = context_tensor[row_slice]
            expanded_x0 = x0_batch.repeat_interleave(self.surrogate_num_iter, dim=0)
            expanded_context = context_batch.repeat_interleave(self.surrogate_num_iter, dim=0)
            batch_size = expanded_x0.shape[0]

            timesteps = torch.randint(0, self.num_steps, (batch_size,), device=self.device)
            noise = torch.randn_like(expanded_x0)
            alpha_bar_t = self.alpha_bar[timesteps].unsqueeze(1)
            x_t = torch.sqrt(alpha_bar_t) * expanded_x0 + torch.sqrt(1.0 - alpha_bar_t) * noise
            eps_hat = self.denoiser(x_t, timesteps, expanded_context)
            raw_mse = torch.sum((eps_hat - noise) ** 2, dim=1)
            weights = self.elbo_weight[timesteps]
            total_loss = total_loss + torch.sum(weights * raw_mse)
            total_count += batch_size

        return total_loss / float(max(total_count, 1))

    def nll_proxy(self, target_cost, context) -> torch.Tensor:
        return self._diffusion_surrogate(target_cost, context)


class CouplingMLP(nn.Module):
    """Small MLP used inside affine coupling layers."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConditionalAffineCoupling(nn.Module):
    """Conditional affine coupling layer for RealNVP."""

    def __init__(self, q: int, p: int, mask: torch.Tensor, hidden_dim: int, log_scale_min: float, log_scale_max: float):
        super().__init__()
        self.register_buffer("mask", mask.view(1, q))
        input_dim = q + p
        self.scale_net = CouplingMLP(input_dim, q, hidden_dim)
        self.shift_net = CouplingMLP(input_dim, q, hidden_dim)
        self.log_scale_min = log_scale_min
        self.log_scale_max = log_scale_max

    def _scale_and_shift(self, x: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_masked = x * self.mask
        features = torch.cat([x_masked, context], dim=1)
        log_scale = torch.clamp(self.scale_net(features), self.log_scale_min, self.log_scale_max)
        shift = self.shift_net(features)
        inv_mask = 1.0 - self.mask
        return log_scale * inv_mask, shift * inv_mask

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        log_scale, shift = self._scale_and_shift(x, context)
        y = x * self.mask + (1.0 - self.mask) * (x * torch.exp(log_scale) + shift)
        log_det = torch.sum(log_scale, dim=1)
        return y, log_det

    def inverse(self, y: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        log_scale, shift = self._scale_and_shift(y, context)
        x = y * self.mask + (1.0 - self.mask) * ((y - shift) * torch.exp(-log_scale))
        log_det = -torch.sum(log_scale, dim=1)
        return x, log_det


class ConditionalRealNVPModel(TorchGenerativeModelBase):
    """Conditional RealNVP model with exact likelihood."""

    model_type = "cnf"
    actor_kind = "cnf"

    def __init__(self, p: int, q: int, seed: int | None = None, config: dict | None = None):
        super().__init__(p, q, seed=seed, config=config)
        default_layers = 2 if q > 1 else 1
        num_layers = max(default_layers, int(_coalesce(self.config.get("flow_num_coupling_layers"), default_layers)))
        hidden_dim = int(_coalesce(self.config.get("flow_hidden_dim"), 128))
        log_scale_min = float(_coalesce(self.config.get("flow_log_scale_min"), -5.0))
        log_scale_max = float(_coalesce(self.config.get("flow_log_scale_max"), 5.0))

        layers = []
        for layer_idx in range(num_layers):
            mask = torch.tensor(
                [(dim_idx + layer_idx) % 2 for dim_idx in range(q)],
                dtype=torch.float32,
                device=self.device,
            )
            layers.append(
                ConditionalAffineCoupling(
                    q=q,
                    p=p,
                    mask=mask,
                    hidden_dim=hidden_dim,
                    log_scale_min=log_scale_min,
                    log_scale_max=log_scale_max,
                )
            )
        self.flow = nn.ModuleList(layers).to(self.device)
        self._modules_for_grad = [self.flow]
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self._learning_rate(),
            weight_decay=self._weight_decay(),
        )

    def _sample_flow(self, context_tensor, *, num_samples: int = 1, keep_graph: bool = False) -> torch.Tensor:
        context = self._context_tensor(context_tensor)
        base_batch_size = context.shape[0]
        sample_count = max(1, int(num_samples))
        context_rep = context.repeat_interleave(sample_count, dim=0)
        batch_size = context_rep.shape[0]

        def _sample_body():
            x = torch.randn((batch_size, self.q), device=self.device)
            for layer in self.flow:
                x, _ = layer(x, context_rep)
            return x.view(base_batch_size, sample_count, self.q)

        if keep_graph:
            return _sample_body()
        with torch.no_grad():
            return _sample_body().detach()

    def sample_cost(self, context_tensor) -> torch.Tensor:
        return self._sample_flow(context_tensor, num_samples=1, keep_graph=False).squeeze(1)

    def sample_scenarios(self, context, K: int, keep_graph: bool = True) -> torch.Tensor:
        scenarios = self._sample_flow(context, num_samples=K, keep_graph=keep_graph)
        return scenarios.squeeze(0) if scenarios.shape[0] == 1 else scenarios

    def log_prob_cost(self, target_cost, context) -> torch.Tensor:
        x = self._cost_tensor(target_cost)
        context_tensor = self._context_tensor(context)
        context_tensor = self._match_batch(context_tensor, x.shape[0])

        z = x
        total_log_det = torch.zeros(x.shape[0], device=self.device)
        for layer in reversed(self.flow):
            z, log_det = layer.inverse(z, context_tensor)
            total_log_det = total_log_det + log_det

        base_log_prob = -0.5 * torch.sum(z**2 + math.log(2.0 * math.pi), dim=1)
        return base_log_prob + total_log_det

    def log_prob_proxy(self, target_cost, context) -> torch.Tensor:
        return self.log_prob_cost(target_cost, context)

    def nll_proxy(self, target_cost, context) -> torch.Tensor:
        return -self.log_prob_cost(target_cost, context).mean()


class ScalarConditionalAffineLayer(nn.Module):
    """1D conditional affine flow layer for shared local generative models."""

    def __init__(self, p: int, hidden_dim: int, log_scale_min: float, log_scale_max: float):
        super().__init__()
        self.conditioner = CouplingMLP(p, 2, hidden_dim)
        self.log_scale_min = log_scale_min
        self.log_scale_max = log_scale_max

    def _scale_and_shift(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        params = self.conditioner(context)
        log_scale = torch.clamp(params[:, :1], self.log_scale_min, self.log_scale_max)
        shift = params[:, 1:2]
        return log_scale, shift

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        log_scale, shift = self._scale_and_shift(context)
        y = x * torch.exp(log_scale) + shift
        return y, log_scale.squeeze(1)

    def inverse(self, y: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        log_scale, shift = self._scale_and_shift(context)
        x = (y - shift) * torch.exp(-log_scale)
        return x, (-log_scale).squeeze(1)


class SharedRealNVPModel(SharedLocalGenerativeModelBase):
    """Row-wise 1D conditional flow for local-feature benchmarks."""

    model_type = "shared_cnf"
    actor_kind = "cnf"

    def __init__(self, p: int, q: int, seed: int | None = None, config: dict | None = None):
        super().__init__(p, q, seed=seed, config=config)
        num_layers = max(1, int(_coalesce(self.config.get("flow_num_coupling_layers"), 2)))
        hidden_dim = int(_coalesce(self.config.get("flow_hidden_dim"), 128))
        log_scale_min = float(_coalesce(self.config.get("flow_log_scale_min"), -5.0))
        log_scale_max = float(_coalesce(self.config.get("flow_log_scale_max"), 5.0))

        self.flow = nn.ModuleList(
            [
                ScalarConditionalAffineLayer(
                    p=p,
                    hidden_dim=hidden_dim,
                    log_scale_min=log_scale_min,
                    log_scale_max=log_scale_max,
                )
                for _ in range(num_layers)
            ]
        ).to(self.device)
        self._modules_for_grad = [self.flow]
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self._learning_rate(),
            weight_decay=self._weight_decay(),
        )

    def _sample_rows(self, context_tensor, *, num_samples: int = 1, keep_graph: bool = False) -> torch.Tensor:
        context = self._context_tensor(context_tensor)

        def _sample_body():
            row_samples = []
            for row_slice in _row_slices(context.shape[0], self.row_batch_size):
                context_batch = context[row_slice]
                row_count = context_batch.shape[0]
                context_rep = context_batch.repeat_interleave(num_samples, dim=0)
                x = torch.randn((context_rep.shape[0], 1), device=self.device)
                for layer in self.flow:
                    x, _ = layer(x, context_rep)
                row_samples.append(x.view(row_count, num_samples).transpose(0, 1))
            return torch.cat(row_samples, dim=1)

        if keep_graph:
            return _sample_body()
        with torch.no_grad():
            return _sample_body().detach()

    def sample_cost(self, context_tensor) -> torch.Tensor:
        return self._sample_rows(context_tensor, num_samples=1, keep_graph=False)

    def sample_scenarios(self, context, K: int, keep_graph: bool = True) -> torch.Tensor:
        return self._sample_rows(context, num_samples=K, keep_graph=keep_graph)

    def log_prob_cost(self, target_cost, context) -> torch.Tensor:
        x = self._local_cost_tensor(target_cost)
        context_tensor = self._context_tensor(context)
        row_log_probs = []

        for row_slice in _row_slices(self.q, self.row_batch_size):
            z = x[row_slice]
            context_batch = context_tensor[row_slice]
            total_log_det = torch.zeros(z.shape[0], device=self.device)
            for layer in reversed(self.flow):
                z, log_det = layer.inverse(z, context_batch)
                total_log_det = total_log_det + log_det

            base_log_prob = -0.5 * (z.squeeze(1) ** 2 + math.log(2.0 * math.pi))
            row_log_probs.append(base_log_prob + total_log_det)

        return torch.cat(row_log_probs, dim=0).unsqueeze(0)

    def log_prob_proxy(self, target_cost, context) -> torch.Tensor:
        return self.log_prob_cost(target_cost, context)

    def nll_proxy(self, target_cost, context) -> torch.Tensor:
        return -self.log_prob_cost(target_cost, context).mean()


def compute_log_prob_surrogate(model, target_cost, context) -> torch.Tensor:
    """Return the differentiable surrogate corresponding to the model family."""
    if hasattr(model, "nll_proxy"):
        return model.nll_proxy(target_cost, context)
    raise ValueError(f"Unsupported generative model type: {getattr(model, 'model_type', None)}")
