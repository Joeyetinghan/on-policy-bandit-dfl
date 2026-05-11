"""DFHPG with configurable conditional distribution modes."""

from __future__ import annotations

import math

import numpy as np

from src.algos.base_algo import BaseAlgo
from src.algos.batch_updates import RoundBatchAccumulator, clip_torch_grads
from src.algos.lr_schedules import build_lr_schedule, set_optimizer_lr
from src.common.models import create_model
from src.common.generative_dfl import (
    generative_auxiliary_loss,
    generative_dfl_config,
    generative_update_objective,
    make_solution_pool,
    oracle_input_from_cost,
)
from src.common.nuisance_models import (
    build_nuisance_model,
    build_nuisance_optimizer,
    fill_observed_components,
    nuisance_cost_scalar_loss,
    nuisance_cost_scalar_prediction,
    nuisance_linear_observation_loss,
    nuisance_masked_vector_loss,
    nuisance_scalar_loss,
    nuisance_scalar_prediction,
)
from src.common.point_surrogate_losses import (
    NON_GENERATIVE_POINT_SURROGATE_LOSSES,
    POINT_SURROGATE_LOSSES,
    PointSurrogateState,
    point_surrogate_grad,
)
from src.common.surrogate_loss_names import canonical_surrogate_loss_name


def _coalesce(value, default):
    return default if value is None else value


def _canonical_distribution_mode(raw: str) -> str:
    mode = str(raw).lower()
    aliases = {
        "point": "gaussian",
        "normal": "gaussian",
        "flow": "cnf",
        "realnvp": "cnf",
        "shared_cnf": "cnf",
        "shared_diffusion": "diffusion",
    }
    return aliases.get(mode, mode)


def _benchmark_uses_shared_model(config: dict) -> bool:
    benchmark = str(config.get("benchmark", "")).lower()
    if benchmark in {"energy", "pricing"}:
        return True
    return False


def _resolve_hybrid_model_type(config: dict) -> str:
    if config.get("model_type") is not None:
        return str(config["model_type"])

    raw_mode = config.get("hybrid_distribution_mode", config.get("distribution_mode"))
    if raw_mode is None:
        return "linear"

    mode = _canonical_distribution_mode(str(raw_mode))
    shared = _benchmark_uses_shared_model(config)
    if mode == "gaussian":
        return "shared_linear" if shared else "linear"
    if mode == "cnf":
        return "shared_cnf" if shared else "cnf"
    if mode == "diffusion":
        return "shared_diffusion" if shared else "diffusion"
    raise ValueError(
        f"Unknown hybrid_distribution_mode: {mode}. "
        "Supported modes are 'gaussian', 'cnf', and 'diffusion'."
    )


def _infer_hybrid_distribution_mode(config: dict, model) -> str:
    raw_mode = config.get("hybrid_distribution_mode", config.get("distribution_mode"))
    if raw_mode is not None:
        mode = _canonical_distribution_mode(str(raw_mode))
    else:
        model_type = _canonical_distribution_mode(str(config.get("model_type", "")))
        if model_type in {"gaussian", "cnf", "diffusion"}:
            mode = model_type
        elif getattr(model, "model_family", None) == "point":
            mode = "gaussian"
        elif getattr(model, "model_family", None) == "generative":
            mode = str(getattr(model, "actor_kind", ""))
        else:
            mode = ""

    if mode == "gaussian":
        if getattr(model, "model_family", None) != "point":
            raise ValueError("hybrid_distribution_mode='gaussian' requires a point model.")
        return mode
    if mode in {"cnf", "diffusion"}:
        if getattr(model, "model_family", None) != "generative":
            raise ValueError(f"hybrid_distribution_mode='{mode}' requires a generative model.")
        if getattr(model, "actor_kind", None) != mode:
            raise ValueError(
                f"hybrid_distribution_mode='{mode}' requires actor_kind='{mode}', "
                f"got '{getattr(model, 'actor_kind', None)}'."
            )
        return mode
    raise ValueError(
        f"Unknown hybrid_distribution_mode: {mode}. "
        "Supported modes are 'gaussian', 'cnf', and 'diffusion'."
    )


class DFHPG(BaseAlgo):
    """DFHPG with configurable conditional distribution mode."""

    def __init__(self, config, oracle):
        super().__init__(config, oracle)
        self.p = config["p"]
        self.q = getattr(oracle, "d", None)
        if self.q is None:
            self.q = getattr(oracle, "q")

        model_type = _resolve_hybrid_model_type(config)
        self.model = create_model(model_type, self.p, self.q, seed=config.get("seed", 42), config=config)
        self.theta = self.model.initialize_params()
        self.model_family = self.model.model_family
        self.hybrid_distribution_mode = _infer_hybrid_distribution_mode(config, self.model)

        self.theta_lr_schedule = build_lr_schedule(config, "theta", default_lr=0.01)
        self.theta_lr = self.theta_lr_schedule.initial_lr
        self._dynamic_theta_lr = str(config.get("theta_lr_schedule", "constant")).lower() != "constant"
        self.lambda_reg = float(_coalesce(config.get("lambda_reg"), 0.0))
        self.grad_clip_norm = float(_coalesce(config.get("grad_clip_norm"), 10.0))
        self.model_update_batch_rounds = max(1, int(config.get("model_update_batch_rounds", 1)))
        self.nuisance_update_batch_rounds = max(1, int(config.get("nuisance_update_batch_rounds", 1)))
        self.hybrid_batching_mode = str(config.get("hybrid_batching_mode", "combined")).lower()
        if self.hybrid_batching_mode not in {"combined", "actor_only"}:
            raise ValueError(
                f"Unknown hybrid_batching_mode: {self.hybrid_batching_mode}. Supported: ['actor_only', 'combined']"
            )
        self.hybrid_policy_batch_rounds = max(
            1,
            int(config.get("hybrid_policy_batch_rounds", self.model_update_batch_rounds)),
        )
        self.model_batch_accumulator = RoundBatchAccumulator(self.model_update_batch_rounds)
        self.actor_batch_accumulator = RoundBatchAccumulator(self.hybrid_policy_batch_rounds)
        self.nuisance_batch_accumulator = RoundBatchAccumulator(self.nuisance_update_batch_rounds)
        self.policy_sampling_scale = max(float(_coalesce(config.get("policy_sampling_scale"), 0.1)), 1.0e-8)
        self.perturbation_distribution = str(config.get("perturbation_distribution", "normal")).lower()
        self.perturbation_student_df = float(max(_coalesce(config.get("perturbation_student_df"), 3.0), 1.01))
        self.supported_perturbation_distributions = {"normal", "laplace", "student_t", "gumbel", "cauchy"}
        if self.perturbation_distribution not in self.supported_perturbation_distributions:
            raise ValueError(
                f"Unknown perturbation_distribution: {self.perturbation_distribution}. "
                f"Supported: {sorted(self.supported_perturbation_distributions)}"
            )

        self.hybrid_loss_type = canonical_surrogate_loss_name(config.get("hybrid_loss_type", "spo_plus"))
        if self.hybrid_loss_type not in POINT_SURROGATE_LOSSES:
            raise ValueError(f"Unknown hybrid_loss_type: {self.hybrid_loss_type}")
        # Selects which estimator fills the (1 - alpha_t) branch of the hybrid
        # mixture: "surrogate" -> g^{sur}_t (eq. 7, reparameterized through a
        # differentiable proxy of the plug-in objective); "score" -> g^{plug-in}_t
        # (eq. 5, REINFORCE form with the nuisance-predicted scalar
        # f_phi(x_t)^T w_t replacing the realized feedback v_t).
        plugin_grad_mode = str(_coalesce(config.get("plugin_grad_mode"), "surrogate")).lower()
        if plugin_grad_mode in {"sur", "surr"}:
            plugin_grad_mode = "surrogate"
        if plugin_grad_mode in {"plugin", "score_function", "reinforce", "plugin_score"}:
            plugin_grad_mode = "score"
        if plugin_grad_mode not in {"surrogate", "score"}:
            raise ValueError(
                f"Unknown plugin_grad_mode: {plugin_grad_mode}. "
                "Use 'surrogate' (eq. 7) or 'score' (eq. 5)."
            )
        self.plugin_grad_mode = plugin_grad_mode
        self.hybrid_mse_weight = float(_coalesce(config.get("hybrid_mse_weight"), 0.5))
        if not 0.0 <= self.hybrid_mse_weight <= 1.0:
            raise ValueError("hybrid_mse_weight must be in [0, 1]")
        if self.hybrid_loss_type == "spo_plus":
            self.hybrid_mse_weight = 0.0
        elif self.hybrid_loss_type == "mse":
            self.hybrid_mse_weight = 1.0
        self.surrogate_solution_pool_max_size = max(2, int(config.get("surrogate_solution_pool_max_size", 128)))
        self.surrogate_pairwise_margin = float(_coalesce(config.get("surrogate_pairwise_margin"), 0.0))
        self.surrogate_listwise_temperature = float(
            max(_coalesce(config.get("surrogate_listwise_temperature"), 1.0), 1.0e-8)
        )
        self.surrogate_dbb_lambda = float(_coalesce(config.get("surrogate_dbb_lambda"), 10.0))
        self.surrogate_imle_lambda = float(_coalesce(config.get("surrogate_imle_lambda"), 10.0))
        self.surrogate_num_samples = max(1, int(config.get("surrogate_num_samples", 10)))
        self.surrogate_sigma = float(max(_coalesce(config.get("surrogate_sigma"), 1.0), 1.0e-8))
        self.surrogate_pg_sigma = float(max(_coalesce(config.get("surrogate_pg_sigma"), 0.1), 1.0e-8))
        self.surrogate_two_sides = bool(config.get("surrogate_two_sides", False))
        surrogate_seed = int(config.get("seed", 42)) + 1009
        self.point_surrogate_state = PointSurrogateState(
            self.q,
            max_pool_size=self.surrogate_solution_pool_max_size,
            seed=surrogate_seed,
        )
        self.surrogate_solution_pool = self.point_surrogate_state.solution_pool
        self.generative_dfl_cfg = generative_dfl_config(config) if self.model_family == "generative" else None
        self.generative_update_objective = (
            generative_update_objective(config) if self.model_family == "generative" else None
        )
        self.generative_solution_pool = make_solution_pool(self.q, config) if self.model_family == "generative" else None

        self.hybrid_alpha_schedule = str(config.get("hybrid_alpha_schedule", "cosine")).lower()
        self.supported_hybrid_alpha_schedules = {
            "linear",
            "cosine",
            "step",
            "constant",
            "inverse_sqrt",
            "exponential",
            "warmup_cosine",
            "warmup_exponential",
            "adaptive_nuisance_reliability",
        }
        if self.hybrid_alpha_schedule not in self.supported_hybrid_alpha_schedules:
            raise ValueError(
                f"Unknown hybrid_alpha_schedule: {self.hybrid_alpha_schedule}. "
                f"Supported: {sorted(self.supported_hybrid_alpha_schedules)}"
            )
        self.hybrid_alpha_schedule_basis = str(config.get("hybrid_alpha_schedule_basis", "round")).lower()
        if self.hybrid_alpha_schedule_basis not in {"round", "actor_update"}:
            raise ValueError(
                "Unknown hybrid_alpha_schedule_basis: "
                f"{self.hybrid_alpha_schedule_basis}. Supported: ['actor_update', 'round']"
            )

        self.hybrid_alpha_warmup_init = float(_coalesce(config.get("hybrid_alpha_warmup_init"), 0.7))
        self.hybrid_alpha_init = float(_coalesce(config.get("hybrid_alpha_init"), 0.7))
        self.hybrid_alpha_final = float(_coalesce(config.get("hybrid_alpha_final"), 0.1))
        self.hybrid_alpha_warmup_steps = max(int(_coalesce(config.get("hybrid_alpha_warmup_steps"), 0)), 0)
        if not 0.0 <= self.hybrid_alpha_warmup_init <= 1.0:
            raise ValueError("hybrid_alpha_warmup_init must be in [0, 1]")
        if not 0.0 <= self.hybrid_alpha_init <= 1.0:
            raise ValueError("hybrid_alpha_init must be in [0, 1]")
        if not 0.0 <= self.hybrid_alpha_final <= 1.0:
            raise ValueError("hybrid_alpha_final must be in [0, 1]")
        warmdown_steps = config.get("hybrid_alpha_warmdown_steps", None)
        warmdown_frac = config.get("hybrid_alpha_warmdown_frac", None)
        if warmdown_steps is not None:
            self.hybrid_alpha_warmdown_steps = max(int(warmdown_steps), 0)
            self.hybrid_alpha_warmdown_frac = None
        elif warmdown_frac is not None:
            self.hybrid_alpha_warmdown_frac = float(max(_coalesce(warmdown_frac, 0.3), 0.0))
            self.hybrid_alpha_warmdown_steps = int(
                math.ceil(self.hybrid_alpha_warmdown_frac * max(int(config.get("T", 1)), 1))
            )
        else:
            self.hybrid_alpha_warmdown_steps = 300
            self.hybrid_alpha_warmdown_frac = None
        self.hybrid_alpha_min = float(_coalesce(config.get("hybrid_alpha_min"), self.hybrid_alpha_final))
        self.hybrid_alpha_max = float(_coalesce(config.get("hybrid_alpha_max"), self.hybrid_alpha_init))
        self.hybrid_alpha_warmup_frac = float(_coalesce(config.get("hybrid_alpha_warmup_frac"), 0.05))
        self.hybrid_alpha_gate = str(_coalesce(config.get("hybrid_alpha_gate"), "warmup_frac")).lower()
        if self.hybrid_alpha_gate in {"warmup", "warmup_fraction", "frac"}:
            self.hybrid_alpha_gate = "warmup_frac"
        if self.hybrid_alpha_gate in {"exp", "exponential", "exponential_time"}:
            self.hybrid_alpha_gate = "exp_time"
        if self.hybrid_alpha_gate not in {"warmup_frac", "exp_time"}:
            raise ValueError("hybrid_alpha_gate must be one of {'warmup_frac', 'exp_time'}")
        self.hybrid_alpha_time_scale = float(_coalesce(config.get("hybrid_alpha_time_scale"), 100.0))
        self.hybrid_alpha_ema_decay = float(_coalesce(config.get("hybrid_alpha_ema_decay"), 0.98))
        self.hybrid_alpha_smooth = float(_coalesce(config.get("hybrid_alpha_smooth"), 0.05))
        self.hybrid_alpha_eps = float(max(_coalesce(config.get("hybrid_alpha_eps"), 1.0e-8), 1.0e-12))
        if not 0.0 <= self.hybrid_alpha_min <= self.hybrid_alpha_max <= 1.0:
            raise ValueError("adaptive alpha requires 0 <= hybrid_alpha_min <= hybrid_alpha_max <= 1")
        if self.hybrid_alpha_warmup_frac < 0.0:
            raise ValueError("hybrid_alpha_warmup_frac must be non-negative")
        if self.hybrid_alpha_gate == "exp_time" and self.hybrid_alpha_time_scale <= 0.0:
            raise ValueError("hybrid_alpha_time_scale must be positive when hybrid_alpha_gate='exp_time'")
        if not 0.0 <= self.hybrid_alpha_ema_decay < 1.0:
            raise ValueError("hybrid_alpha_ema_decay must be in [0, 1)")
        if not 0.0 <= self.hybrid_alpha_smooth <= 1.0:
            raise ValueError("hybrid_alpha_smooth must be in [0, 1]")
        self._adaptive_alpha = self.hybrid_alpha_max
        self._adaptive_residual_ema = None
        self._adaptive_scale_ema = None
        self._adaptive_residual_ratio_ref = None
        self.last_alpha = (
            self.hybrid_alpha_max
            if self.hybrid_alpha_schedule == "adaptive_nuisance_reliability"
            else self.hybrid_alpha_init
        )
        self.hybrid_gradient_normalization = bool(config.get("hybrid_gradient_normalization", True))
        self.hybrid_grad_norm_eps = float(max(_coalesce(config.get("hybrid_grad_norm_eps"), 1.0e-12), 1.0e-12))

        self.baseline_momentum = float(np.clip(_coalesce(config.get("baseline_momentum"), 0.95), 0.0, 1.0))
        self.utility_baseline = float(_coalesce(config.get("initial_utility_baseline"), 0.0))
        self.generative_baseline_momentum = float(
            np.clip(_coalesce(config.get("generative_baseline_momentum"), self.baseline_momentum), 0.0, 1.0)
        )
        self.policy_baseline_type = str(_coalesce(config.get("policy_baseline_type"), "ema")).lower()
        if self.policy_baseline_type not in {"ema", "none", "nuisance_induced"}:
            raise ValueError(
                f"Unknown policy_baseline_type: {self.policy_baseline_type}. "
                "Use one of none, ema, or nuisance_induced."
            )
        if self.hybrid_batching_mode != "combined" and self.model_family == "generative":
            raise ValueError("hybrid_batching_mode!='combined' is only supported for point-model DFHPG.")
        if self.model_family == "generative" and (
            self.hybrid_loss_type in NON_GENERATIVE_POINT_SURROGATE_LOSSES
        ):
            raise ValueError(
                f"hybrid_loss_type='{self.hybrid_loss_type}' is only supported for point-model DFHPG."
            )

        self.nuisance_model = build_nuisance_model(config, self.p, self.q)
        self.nuisance_optimizer = build_nuisance_optimizer(self.nuisance_model, config)
        self.nuisance_lr_schedule = build_lr_schedule(
            config,
            "nuisance",
            fallback_prefix="theta",
            default_lr=self.theta_lr,
        )
        self.nuisance_lr = self.nuisance_lr_schedule.initial_lr

        self.rng = np.random.RandomState(config.get("seed", 42))
        self._init_random_exploration(config, self.rng, self.q)
        self.t_curr = 0

        self.last_c_hat = None
        self.last_c_tilde = None
        self.last_logprob_grad_loc = None
        self.last_sampled_cost = None
        self.last_oracle_context = None
        self.last_step_diagnostics = {}

    def predict_mu(self, x, theta=None):
        if theta is None:
            theta = self.theta
        return self.model.predict(x, theta)

    def _apply_point_model_grad(self, grad):
        if grad is None:
            return
        grad = np.asarray(grad, dtype=float).reshape(-1)
        if self.lambda_reg > 0:
            grad += 2.0 * self.lambda_reg * self.theta

        grad_norm = np.linalg.norm(grad)
        if grad_norm > self.grad_clip_norm:
            grad = grad * (self.grad_clip_norm / grad_norm)

        lr = self.theta_lr_schedule.next()
        self.theta_lr = lr
        self.theta = self.theta - lr * grad
        self.last_oracle_context = None

    def _flush_point_model_batch(self):
        grad = self.model_batch_accumulator.consume_numpy_mean()
        self._apply_point_model_grad(grad)

    def _accumulate_point_model_grad(self, grad):
        self.model_batch_accumulator.add_numpy(grad)
        if self.model_batch_accumulator.should_flush(is_final_round=self._is_final_round()):
            self._flush_point_model_batch()

    def _accumulate_point_actor_grad(self, grad):
        self.actor_batch_accumulator.add_numpy(grad)

    def _consume_point_actor_grad_if_ready(self):
        if not self.actor_batch_accumulator.should_flush(is_final_round=self._is_final_round()):
            return None
        return self.actor_batch_accumulator.consume_numpy_mean()

    def _flush_torch_model_batch(self):
        if not self.model_batch_accumulator.should_flush(is_final_round=self._is_final_round()):
            return
        params = list(self.model.parameters())
        self.model_batch_accumulator.average_torch_grads_(params)
        self.model.clip_grad()
        if self._dynamic_theta_lr:
            lr = self.theta_lr_schedule.next()
            self.theta_lr = lr
            set_optimizer_lr(self.model.optimizer, lr)
        self.model.step()
        self.model.zero_grad()

    def _accumulate_weighted_torch_grads(self, actor_grads, critic_grads, alpha_t):
        params = list(self.model.parameters())
        if actor_grads is None:
            actor_grads = [None] * len(params)
        if critic_grads is None:
            critic_grads = [None] * len(params)
        if self.model_batch_accumulator.count == 0:
            self.model.zero_grad()
        combined_grads = []
        for param, actor_grad, critic_grad in zip(params, actor_grads, critic_grads):
            if actor_grad is None and critic_grad is None:
                combined_grads.append(None)
                continue
            combined = None
            if actor_grad is not None:
                combined = alpha_t * actor_grad
            if critic_grad is not None:
                combined = ((1.0 - alpha_t) * critic_grad) if combined is None else combined + (1.0 - alpha_t) * critic_grad
            if combined is None:
                combined_grads.append(None)
                continue
            combined = combined.detach()
            combined_grads.append(combined)
            if param.grad is None:
                param.grad = combined.clone()
            else:
                param.grad.add_(combined)
        self.model_batch_accumulator.mark_round()
        self._flush_torch_model_batch()
        return combined_grads

    def _accumulate_weighted_torch_branch_losses(self, score_loss, aux_total, alpha_t):
        import torch

        params = list(self.model.parameters())
        score_grads_raw = None
        score_grads = None
        aux_grads_raw = None
        aux_grads = None
        if score_loss is not None:
            score_grads_raw = list(
                torch.autograd.grad(
                    score_loss,
                    params,
                    retain_graph=aux_total is not None,
                    allow_unused=True,
                )
            )
            score_grads = self._normalize_torch_grads(score_grads_raw)
        if aux_total is not None:
            aux_grads_raw = list(torch.autograd.grad(aux_total, params, allow_unused=True))
            aux_grads = self._normalize_torch_grads(aux_grads_raw)
        combined_grads = self._accumulate_weighted_torch_grads(score_grads, aux_grads, alpha_t)
        return score_grads_raw, aux_grads_raw, score_grads, aux_grads, combined_grads

    def _accumulate_torch_model_loss(self, loss):
        if self.model_batch_accumulator.count == 0:
            self.model.zero_grad()
        loss.backward()
        self.model_batch_accumulator.mark_round()
        self._flush_torch_model_batch()

    def _mark_zero_model_round(self):
        self.model_batch_accumulator.mark_round()
        if self.model_family == "generative":
            self._flush_torch_model_batch()
        elif self.model_batch_accumulator.should_flush(is_final_round=self._is_final_round()):
            self._flush_point_model_batch()

    def _flush_nuisance_batch(self):
        if not self.nuisance_batch_accumulator.should_flush(is_final_round=self._is_final_round()):
            return
        params = list(self.nuisance_model.parameters())
        self.nuisance_batch_accumulator.average_torch_grads_(params)
        clip_torch_grads(params, self.grad_clip_norm)
        lr = self.nuisance_lr_schedule.next()
        self.nuisance_lr = lr
        set_optimizer_lr(self.nuisance_optimizer, lr)
        self.nuisance_optimizer.step()
        self.nuisance_optimizer.zero_grad(set_to_none=True)

    def _accumulate_deterministic_nuisance_update(self, context, action, reward, info=None):
        if self.nuisance_batch_accumulator.count == 0:
            self.nuisance_optimizer.zero_grad(set_to_none=True)
        observation_matrix, observation_values = self._extract_linear_observation_feedback(
            info,
            objective_values=self.model_family == "generative",
        )
        observed_score, observation_mask = self._extract_observed_score_feedback(info)
        if observation_matrix is not None:
            loss = nuisance_linear_observation_loss(
                self.nuisance_model, context, observation_matrix, observation_values
            )
        elif observed_score is not None:
            loss = nuisance_masked_vector_loss(self.nuisance_model, context, observed_score, observation_mask)
        else:
            oracle = self.oracle if hasattr(self.oracle, "feedback_loss_torch") else None
            if self.model_family == "generative":
                loss = nuisance_cost_scalar_loss(self.nuisance_model, context, action, reward, oracle=oracle)
            else:
                loss = nuisance_scalar_loss(self.nuisance_model, context, action, reward, oracle=oracle)
        loss.mean().backward()
        self.nuisance_batch_accumulator.mark_round()
        self._flush_nuisance_batch()

    def predict_nuisance(self, x):
        if self.nuisance_model is None:
            raise RuntimeError("predict_nuisance called while nuisance model is unavailable")
        return self.nuisance_model.predict(x)

    def _predict_nuisance_torch(self, context):
        return self.nuisance_model(context)

    def _oracle_solve(self, score, oracle_context=None):
        return self._cached_oracle_solve(score, oracle_context=oracle_context)

    def _alpha_schedule_rounds(self):
        if self.hybrid_alpha_schedule_basis == "round":
            return 1
        if self.hybrid_batching_mode == "actor_only":
            return self.hybrid_policy_batch_rounds
        return self.model_update_batch_rounds

    def _alpha_schedule_index(self, round_idx=None):
        if round_idx is None:
            round_idx = self.t_curr
        round_idx = max(int(round_idx), 1)
        batch_rounds = max(self._alpha_schedule_rounds(), 1)
        if self.hybrid_alpha_schedule_basis == "round":
            return round_idx
        return ((round_idx - 1) // batch_rounds) + 1

    def _adaptive_alpha_warmup_steps(self):
        # Use an absolute round count so the adaptive schedule does not depend
        # on the horizon T. The default 100 matches T=2000 with warmup_frac=0.05.
        const = self.config.get("hybrid_alpha_warmup_const_steps")
        if const is not None:
            return max(int(const), 0)
        configured_steps = getattr(self, "hybrid_alpha_warmup_steps", 0)
        if configured_steps and int(configured_steps) > 0:
            return int(configured_steps)
        return 100

    def _adaptive_alpha_time_gate(self):
        round_idx = max(int(self.t_curr), 1)
        return float(math.exp(-float(round_idx - 1) / max(self.hybrid_alpha_time_scale, self.hybrid_alpha_eps)))

    def _alpha_for_round(self, round_idx=None):
        if self.hybrid_alpha_schedule == "adaptive_nuisance_reliability":
            return float(np.clip(self._adaptive_alpha, self.hybrid_alpha_min, self.hybrid_alpha_max))
        round_idx = self._alpha_schedule_index(round_idx)
        if self.hybrid_alpha_schedule == "constant":
            return self.hybrid_alpha_init
        if self.hybrid_alpha_schedule == "inverse_sqrt":
            return float(
                self.hybrid_alpha_final
                + (self.hybrid_alpha_init - self.hybrid_alpha_final) / math.sqrt(max(round_idx, 1))
            )
        if self.hybrid_alpha_schedule in {"warmup_cosine", "warmup_exponential"}:
            if self.hybrid_alpha_warmup_steps > 0 and round_idx <= self.hybrid_alpha_warmup_steps:
                if self.hybrid_alpha_warmup_steps == 1:
                    return self.hybrid_alpha_init
                progress = np.clip((round_idx - 1) / float(self.hybrid_alpha_warmup_steps - 1), 0.0, 1.0)
                return float(
                    self.hybrid_alpha_warmup_init
                    + progress * (self.hybrid_alpha_init - self.hybrid_alpha_warmup_init)
                )
            decay_round_idx = max(round_idx - self.hybrid_alpha_warmup_steps, 1)
            if self.hybrid_alpha_warmdown_steps <= 0:
                return self.hybrid_alpha_final
            if self.hybrid_alpha_warmdown_steps == 1:
                return self.hybrid_alpha_init if decay_round_idx <= 1 else self.hybrid_alpha_final
            progress = np.clip((decay_round_idx - 1) / float(self.hybrid_alpha_warmdown_steps - 1), 0.0, 1.0)
            if self.hybrid_alpha_schedule == "warmup_cosine":
                cosine_weight = 0.5 * (1.0 + math.cos(math.pi * progress))
                return float(self.hybrid_alpha_final + (self.hybrid_alpha_init - self.hybrid_alpha_final) * cosine_weight)
            decay = 5.0
            numerator = math.exp(-decay * progress) - math.exp(-decay)
            denominator = 1.0 - math.exp(-decay)
            exp_weight = numerator / denominator if denominator > 0.0 else 0.0
            return float(self.hybrid_alpha_final + (self.hybrid_alpha_init - self.hybrid_alpha_final) * exp_weight)
        if self.hybrid_alpha_warmdown_steps <= 0:
            return self.hybrid_alpha_final
        if self.hybrid_alpha_warmdown_steps == 1:
            return self.hybrid_alpha_init if round_idx <= 1 else self.hybrid_alpha_final
        progress = np.clip((round_idx - 1) / float(self.hybrid_alpha_warmdown_steps - 1), 0.0, 1.0)
        if self.hybrid_alpha_schedule == "linear":
            return float(self.hybrid_alpha_init + progress * (self.hybrid_alpha_final - self.hybrid_alpha_init))
        if self.hybrid_alpha_schedule == "cosine":
            cosine_weight = 0.5 * (1.0 + math.cos(math.pi * progress))
            return float(self.hybrid_alpha_final + (self.hybrid_alpha_init - self.hybrid_alpha_final) * cosine_weight)
        if self.hybrid_alpha_schedule == "exponential":
            decay = 5.0
            numerator = math.exp(-decay * progress) - math.exp(-decay)
            denominator = 1.0 - math.exp(-decay)
            exp_weight = numerator / denominator if denominator > 0.0 else 0.0
            return float(self.hybrid_alpha_final + (self.hybrid_alpha_init - self.hybrid_alpha_final) * exp_weight)
        if self.hybrid_alpha_schedule == "step":
            return self.hybrid_alpha_init if round_idx <= self.hybrid_alpha_warmdown_steps else self.hybrid_alpha_final
        raise ValueError(f"Unknown hybrid_alpha_schedule: {self.hybrid_alpha_schedule}")

    def _checked_alpha_for_round(self, round_idx=None) -> float:
        alpha_t = float(self._alpha_for_round(round_idx))
        if not np.isfinite(alpha_t) or alpha_t < -self.hybrid_alpha_eps or alpha_t > 1.0 + self.hybrid_alpha_eps:
            raise ValueError(f"Hybrid alpha must be in [0, 1], got {alpha_t}")
        return float(np.clip(alpha_t, 0.0, 1.0))

    def _alpha_is_zero(self, alpha_t: float) -> bool:
        return float(alpha_t) <= self.hybrid_alpha_eps

    def _alpha_is_one(self, alpha_t: float) -> bool:
        return 1.0 - float(alpha_t) <= self.hybrid_alpha_eps

    def _adaptive_critic_prediction(self, context, action):
        if self.hybrid_alpha_schedule != "adaptive_nuisance_reliability":
            return None
        if self.nuisance_model is None:
            return None
        return self._critic_scalar_prediction(context, action)

    def _update_adaptive_alpha(self, observed_scalar, predicted_scalar):
        if self.hybrid_alpha_schedule != "adaptive_nuisance_reliability" or predicted_scalar is None:
            return
        observed = float(observed_scalar)
        predicted = float(predicted_scalar)
        if not np.isfinite(observed) or not np.isfinite(predicted):
            return

        residual_sq = (observed - predicted) ** 2
        scale_sq = observed**2
        if self._adaptive_residual_ema is None:
            self._adaptive_residual_ema = residual_sq
            self._adaptive_scale_ema = scale_sq
        else:
            beta = self.hybrid_alpha_ema_decay
            self._adaptive_residual_ema = beta * self._adaptive_residual_ema + (1.0 - beta) * residual_sq
            self._adaptive_scale_ema = beta * self._adaptive_scale_ema + (1.0 - beta) * scale_sq

        residual_ratio = float(
            self._adaptive_residual_ema / max(float(self._adaptive_scale_ema), self.hybrid_alpha_eps)
        )
        if self.hybrid_alpha_gate == "exp_time":
            if self._adaptive_residual_ratio_ref is None:
                self._adaptive_residual_ratio_ref = max(residual_ratio, self.hybrid_alpha_eps)
            else:
                ref = float(self._adaptive_residual_ratio_ref)
                time_gate = self._adaptive_alpha_time_gate()
                if residual_ratio > ref:
                    ref = ref + time_gate * (residual_ratio - ref)
                self._adaptive_residual_ratio_ref = max(ref, self.hybrid_alpha_eps)

            nuisance_unreliability = np.clip(
                residual_ratio / max(float(self._adaptive_residual_ratio_ref), self.hybrid_alpha_eps),
                0.0,
                1.0,
            )
            reliability_alpha = self.hybrid_alpha_min + (
                self.hybrid_alpha_max - self.hybrid_alpha_min
            ) * nuisance_unreliability
            time_gate = self._adaptive_alpha_time_gate()
            target_alpha = time_gate * self.hybrid_alpha_max + (1.0 - time_gate) * reliability_alpha
        else:
            warmup_steps = self._adaptive_alpha_warmup_steps()
            if self.t_curr <= warmup_steps:
                if self._adaptive_residual_ratio_ref is None:
                    self._adaptive_residual_ratio_ref = max(residual_ratio, self.hybrid_alpha_eps)
                else:
                    self._adaptive_residual_ratio_ref = max(
                        float(self._adaptive_residual_ratio_ref),
                        max(residual_ratio, self.hybrid_alpha_eps),
                    )
                target_alpha = self.hybrid_alpha_max
            else:
                if self._adaptive_residual_ratio_ref is None:
                    self._adaptive_residual_ratio_ref = max(residual_ratio, self.hybrid_alpha_eps)
                nuisance_unreliability = np.clip(
                    residual_ratio / max(float(self._adaptive_residual_ratio_ref), self.hybrid_alpha_eps),
                    0.0,
                    1.0,
                )
                target_alpha = self.hybrid_alpha_min + (
                    self.hybrid_alpha_max - self.hybrid_alpha_min
                ) * nuisance_unreliability

        smooth = self.hybrid_alpha_smooth
        self._adaptive_alpha = float(
            np.clip(
                (1.0 - smooth) * self._adaptive_alpha + smooth * target_alpha,
                self.hybrid_alpha_min,
                self.hybrid_alpha_max,
            )
        )

    def get_step_diagnostics(self):
        return dict(self.last_step_diagnostics)

    @staticmethod
    def _array_norm(value):
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            sq_norm = 0.0
            found = False
            for item in value:
                if item is None:
                    continue
                if hasattr(item, "detach"):
                    term = float((item.detach() ** 2).sum().cpu().item())
                else:
                    array = np.asarray(item, dtype=float)
                    term = float(np.sum(array**2))
                if np.isfinite(term):
                    sq_norm += term
                    found = True
            if not found:
                return None
            norm = float(math.sqrt(sq_norm))
            return norm if np.isfinite(norm) else None
        norm = float(np.linalg.norm(value))
        return norm if np.isfinite(norm) else None

    @staticmethod
    def _finite_or_none(value):
        if value is None:
            return None
        value = float(value)
        return value if np.isfinite(value) else None

    def _record_step_diagnostics(
        self,
        *,
        alpha_t,
        scalar_feedback,
        scalar_loss=None,
        actor_grad_raw=None,
        critic_grad_raw=None,
        actor_grad=None,
        critic_grad=None,
        combined_grad=None,
        critic_prediction=None,
        score_loss=None,
        plugin_loss=None,
    ):
        if scalar_loss is None:
            scalar_loss = -float(scalar_feedback)
        nuisance_loss = None
        if critic_prediction is not None:
            nuisance_loss = (float(scalar_feedback) - float(critic_prediction)) ** 2
        diagnostics = {
            "alpha_t": self._finite_or_none(alpha_t),
            "scalar_feedback": self._finite_or_none(scalar_feedback),
            "scalar_loss": self._finite_or_none(scalar_loss),
            "nuisance_loss": self._finite_or_none(nuisance_loss),
            "score_loss": self._finite_or_none(score_loss),
            "plugin_loss": self._finite_or_none(plugin_loss),
            "score_grad_norm": self._array_norm(actor_grad_raw),
            "plugin_grad_norm": self._array_norm(critic_grad_raw),
            "score_grad_norm_normalized": self._array_norm(actor_grad),
            "plugin_grad_norm_normalized": self._array_norm(critic_grad),
            "combined_grad_norm": self._array_norm(combined_grad),
            "adaptive_residual_ema": self._finite_or_none(self._adaptive_residual_ema),
            "adaptive_scale_ema": self._finite_or_none(self._adaptive_scale_ema),
            "adaptive_residual_ratio_ref": self._finite_or_none(self._adaptive_residual_ratio_ref),
        }
        self.last_step_diagnostics = {key: value for key, value in diagnostics.items() if value is not None}

    def _sample_perturbed_prediction(self, c_hat):
        if self.perturbation_distribution == "normal":
            noise = self.rng.randn(self.q)
            return c_hat + self.policy_sampling_scale * noise
        if self.perturbation_distribution == "laplace":
            noise = self.rng.laplace(loc=0.0, scale=1.0, size=self.q)
            return c_hat + self.policy_sampling_scale * noise
        if self.perturbation_distribution == "student_t":
            noise = self.rng.standard_t(self.perturbation_student_df, size=self.q)
            return c_hat + self.policy_sampling_scale * noise
        if self.perturbation_distribution == "gumbel":
            return self.rng.gumbel(loc=c_hat, scale=self.policy_sampling_scale, size=self.q)
        if self.perturbation_distribution == "cauchy":
            noise = self.rng.standard_cauchy(size=self.q)
            return c_hat + self.policy_sampling_scale * noise
        raise ValueError(f"Unknown perturbation_distribution: {self.perturbation_distribution}")

    def _score_function_grad_loc(self, c_tilde, c_hat):
        delta = c_tilde - c_hat
        if self.perturbation_distribution == "normal":
            return delta / (self.policy_sampling_scale**2)
        if self.perturbation_distribution == "laplace":
            return np.sign(delta) / self.policy_sampling_scale
        if self.perturbation_distribution == "student_t":
            nu = self.perturbation_student_df
            eta = delta / self.policy_sampling_scale
            return ((nu + 1.0) * eta / (nu + eta**2)) / self.policy_sampling_scale
        if self.perturbation_distribution == "gumbel":
            eta = np.clip(delta / self.policy_sampling_scale, -50.0, 50.0)
            return (1.0 - np.exp(-eta)) / self.policy_sampling_scale
        if self.perturbation_distribution == "cauchy":
            return (2.0 * delta) / (self.policy_sampling_scale**2 + delta**2)
        raise ValueError(f"Unknown perturbation_distribution: {self.perturbation_distribution}")

    def _select_action_gaussian(self, context, oracle_context=None):
        c_hat = self.predict_mu(context)
        perturbed_score = self._sample_perturbed_prediction(c_hat)
        action_score = self._sample_action_score(perturbed_score)
        if self.last_random_exploration:
            score_grad_loc = np.zeros_like(c_hat)
        else:
            score_grad_loc = self._score_function_grad_loc(action_score, c_hat)
        action = self._oracle_solve(action_score, oracle_context=oracle_context)

        self.last_c_hat = c_hat
        self.last_c_tilde = action_score
        self.last_logprob_grad_loc = score_grad_loc
        self.last_sampled_cost = None
        self.last_oracle_context = oracle_context
        return action

    def _select_action_generative(self, context, oracle_context=None):
        sampled_cost, _ = self.model.sample_for_action(context)
        sampled_cost_np = sampled_cost.detach().cpu().numpy().reshape(-1)
        default_oracle_input = oracle_input_from_cost(self.oracle, sampled_cost_np)
        action_score = self._sample_action_score(default_oracle_input)
        action = self._oracle_solve(action_score, oracle_context=oracle_context)
        self.last_sampled_cost = sampled_cost.detach()
        self.last_c_hat = default_oracle_input
        self.last_c_tilde = action_score
        self.last_logprob_grad_loc = None
        self.last_oracle_context = oracle_context
        return action

    def select_action(self, context, oracle_context=None, true_reward=None):
        del true_reward
        self.t_curr += 1
        self._reset_oracle_solution_cache()
        if self.hybrid_distribution_mode == "gaussian":
            return self._select_action_gaussian(context, oracle_context=oracle_context)
        return self._select_action_generative(context, oracle_context=oracle_context)

    def _build_dm_synthetic_vector(
        self,
        b_hat,
        observed_score_vector=None,
        observation_mask=None,
    ):
        base_with_observations = np.asarray(b_hat, dtype=float).reshape(-1).copy()
        if observed_score_vector is not None and observation_mask is not None:
            base_with_observations = fill_observed_components(
                base_with_observations, observed_score_vector, observation_mask
            )
        return base_with_observations

    def _has_full_observed_score(self, observed_score_vector, observation_mask) -> bool:
        if observed_score_vector is None or observation_mask is None:
            return False
        mask = np.asarray(observation_mask, dtype=float).reshape(-1)
        observed = np.asarray(observed_score_vector, dtype=float).reshape(-1)
        return mask.shape == (self.q,) and observed.shape == (self.q,) and bool(np.all(mask > 0.5))

    def _critic_scalar_prediction(self, context, action):
        oracle = self.oracle if hasattr(self.oracle, "feedback_loss_torch") else None
        if self.model_family == "generative":
            pred = nuisance_cost_scalar_prediction(self.nuisance_model, context, action, oracle=oracle)
        else:
            pred = nuisance_scalar_prediction(self.nuisance_model, context, action, oracle=oracle)
        return float(pred.detach().cpu().item())

    def _critic_score_prediction(self, context):
        return np.asarray(self.predict_nuisance(context), dtype=float).reshape(-1)

    def _critic_reward_from_score(self, critic_score, action):
        critic_score_vec = np.asarray(critic_score, dtype=float).reshape(-1)
        action_vec = np.asarray(action, dtype=float).reshape(-1)
        if hasattr(self.oracle, "decision_loss"):
            return -float(self.oracle.decision_loss(critic_score_vec, action_vec))
        return float(np.dot(critic_score_vec, action_vec))

    def _point_nuisance_induced_baseline(self, context, *, critic_score=None):
        if critic_score is None:
            critic_score = self._critic_score_prediction(context)
        greedy_action = self._oracle_solve(critic_score, oracle_context=self.last_oracle_context)
        return self._critic_reward_from_score(critic_score, greedy_action)

    def _generative_nuisance_induced_baseline_cost(self, context, *, critic_score=None):
        if critic_score is None:
            critic_score = self._critic_score_prediction(context)
        critic_cost = np.asarray(critic_score, dtype=float).reshape(-1)
        greedy_action = self._oracle_solve(
            oracle_input_from_cost(self.oracle, critic_cost),
            oracle_context=self.last_oracle_context,
        )
        oracle = self.oracle if hasattr(self.oracle, "feedback_loss_torch") else None
        baseline = nuisance_cost_scalar_prediction(self.nuisance_model, context, greedy_action, oracle=oracle)
        return float(baseline.detach().cpu().item())

    def _normalize_numpy_grad(self, grad):
        if not self.hybrid_gradient_normalization:
            return grad
        grad_norm = np.linalg.norm(grad)
        if not np.isfinite(grad_norm) or grad_norm <= self.hybrid_grad_norm_eps:
            return grad
        return grad / grad_norm

    def _normalize_torch_grads(self, grads):
        if not self.hybrid_gradient_normalization:
            return grads
        import torch

        sq_norm = None
        for grad in grads:
            if grad is None:
                continue
            term = torch.sum(grad.detach() ** 2)
            sq_norm = term if sq_norm is None else sq_norm + term
        if sq_norm is None:
            return grads
        grad_norm = torch.sqrt(sq_norm)
        if float(grad_norm.detach().cpu().item()) <= self.hybrid_grad_norm_eps:
            return grads
        return [None if grad is None else grad / grad_norm for grad in grads]

    def _point_actor_grad_descent(self, context, reward, baseline_value):
        if self.last_logprob_grad_loc is None:
            raise RuntimeError("select_action must be called before update")
        utility = float(reward)
        advantage = utility - float(baseline_value)
        return self._point_actor_grad_from_advantage(context, advantage), utility

    def _point_actor_grad_from_advantage(self, context, advantage):
        if self.last_logprob_grad_loc is None:
            raise RuntimeError("select_action must be called before update")
        g_c = advantage * self.last_logprob_grad_loc
        actor_grad_ascent = self.model.compute_grad_r(context, g_c, self.theta)
        return -actor_grad_ascent

    def _point_actor_baseline(
        self,
        context,
        action,
        reward,
        *,
        nuisance_induced_baseline=None,
    ):
        del reward
        if self.policy_baseline_type == "none":
            return 0.0
        if self.policy_baseline_type == "ema":
            return float(self.utility_baseline)
        if self.policy_baseline_type == "nuisance_induced":
            if nuisance_induced_baseline is not None:
                return float(nuisance_induced_baseline)
            return self._point_nuisance_induced_baseline(context)
        raise RuntimeError(f"Unsupported internal policy baseline type: {self.policy_baseline_type}")

    def _update_point_utility_baseline(self, reward):
        if self.policy_baseline_type != "ema":
            return
        utility = float(reward)
        beta = self.baseline_momentum
        self.utility_baseline = beta * self.utility_baseline + (1.0 - beta) * utility

    def _point_actor_advantage(
        self,
        context,
        action,
        reward,
        *,
        nuisance_induced_baseline=None,
    ):
        utility = float(reward)
        if self.policy_baseline_type == "none":
            return utility
        if self.policy_baseline_type == "ema":
            return utility - float(self.utility_baseline)
        if self.policy_baseline_type == "nuisance_induced":
            actor_baseline = self._point_actor_baseline(
                context,
                action,
                reward,
                nuisance_induced_baseline=nuisance_induced_baseline,
            )
            return utility - float(actor_baseline)
        raise RuntimeError(f"Unsupported internal policy baseline type: {self.policy_baseline_type}")

    def _generative_actor_advantage(self, action, reward, critic_score, critic_reward_baseline):
        utility = float(reward)
        if self.policy_baseline_type == "none":
            return utility
        if self.policy_baseline_type == "ema":
            return utility - float(self.utility_baseline)
        if self.policy_baseline_type == "nuisance_induced":
            return utility - float(critic_reward_baseline)
        raise RuntimeError(f"Unsupported internal policy baseline type: {self.policy_baseline_type}")

    def _point_critic_grad_descent(self, context, y_tilde):
        mu_hat = self.predict_mu(context)
        target_sol = self._oracle_solve(y_tilde, oracle_context=self.last_oracle_context)
        g_c = point_surrogate_grad(
            loss_type=self.hybrid_loss_type,
            pred_vec=mu_hat,
            target_vec=y_tilde,
            oracle_solve=self._oracle_solve,
            state=self.point_surrogate_state,
            oracle_context=self.last_oracle_context,
            target_sol=target_sol,
            mse_weight=self.hybrid_mse_weight,
            pairwise_margin=self.surrogate_pairwise_margin,
            listwise_temperature=self.surrogate_listwise_temperature,
            dbb_lambda=self.surrogate_dbb_lambda,
            imle_lambda=self.surrogate_imle_lambda,
            num_samples=self.surrogate_num_samples,
            sigma=self.surrogate_sigma,
            pg_sigma=self.surrogate_pg_sigma,
            two_sides=self.surrogate_two_sides,
        )
        return self.model.compute_grad_r(context, g_c, self.theta)

    def _update_generative_hybrid(self, context, action, reward, info=None):
        self.last_step_diagnostics = {}
        if self.nuisance_model is None or self.nuisance_optimizer is None:
            raise RuntimeError("Generative nuisance model is not initialized")
        if self.last_sampled_cost is None:
            raise RuntimeError("select_action must be called before update")

        observed_cost = -float(reward)
        alpha_t = self._checked_alpha_for_round()
        self.last_alpha = alpha_t
        actor_only = self._alpha_is_one(alpha_t)
        surrogate_only = self._alpha_is_zero(alpha_t)
        plugin_uses_score = self.plugin_grad_mode == "score"
        plugin_active = not actor_only
        surrogate_aux_active = plugin_active and not plugin_uses_score
        plugin_score_active = plugin_active and plugin_uses_score
        adaptive_pred = None if actor_only else self._adaptive_critic_prediction(context, action)
        nuisance_induced_baseline_cost = None
        # The score-form plug-in (eq. 5) reuses the same baseline subtraction
        # that g^{score}_t uses, so the nuisance_induced baseline cost must be
        # available whenever the (1 - alpha) plug-in branch is alive — even at
        # alpha = 0, where the surrogate-form path would have skipped it.
        needs_nuisance_baseline = (
            self.policy_baseline_type == "nuisance_induced"
            and (
                (not actor_only and not surrogate_only)
                or plugin_score_active
            )
        )
        if needs_nuisance_baseline:
            nuisance_induced_baseline_cost = self._generative_nuisance_induced_baseline_cost(context)

        y_imputed = None
        if not actor_only:
            self._accumulate_deterministic_nuisance_update(context, action, observed_cost, info=info)
            y_imputed = self._predict_nuisance_torch(context).detach()

        if self.generative_update_objective == "nll":
            if y_imputed is None:
                self._mark_zero_model_round()
                self._update_point_utility_baseline(reward)
                self._record_step_diagnostics(
                    alpha_t=alpha_t,
                    scalar_feedback=observed_cost,
                    scalar_loss=observed_cost,
                    critic_prediction=adaptive_pred,
                )
                return
            loss = self.model.regularizer_loss(y_imputed, context)
            self._accumulate_torch_model_loss(loss)
            self._update_point_utility_baseline(reward)
            self._update_adaptive_alpha(observed_cost, adaptive_pred)
            self._record_step_diagnostics(
                alpha_t=alpha_t,
                scalar_feedback=observed_cost,
                scalar_loss=observed_cost,
                critic_prediction=adaptive_pred,
                plugin_loss=float(loss.detach().cpu().item()),
            )
            return

        has_aux_path = (
            surrogate_aux_active
            and self.generative_update_objective != "score"
            and self.generative_dfl_cfg is not None
        )
        aux_loss_value = None
        score_loss = None
        aux_total = None
        score_grads_raw = None
        aux_grads_raw = None
        score_grads = None
        aux_grads = None
        combined_grads = None

        if surrogate_only:
            pass
        elif self.last_random_exploration:
            score_loss = self.model.log_prob_proxy(self.last_sampled_cost, context).sum() * 0.0
        else:
            if actor_only:
                advantage = observed_cost + float(self.utility_baseline)
            elif self.policy_baseline_type == "none":
                advantage = observed_cost
            elif self.policy_baseline_type == "ema":
                advantage = observed_cost + float(self.utility_baseline)
            elif self.policy_baseline_type == "nuisance_induced":
                if self.policy_baseline_type == "nuisance_induced" and nuisance_induced_baseline_cost is not None:
                    baseline_cost = nuisance_induced_baseline_cost
                else:
                    oracle = self.oracle if hasattr(self.oracle, "feedback_loss_torch") else None
                    baseline_cost = float(
                        nuisance_cost_scalar_prediction(self.nuisance_model, context, action, oracle=oracle)
                        .detach()
                        .cpu()
                        .item()
                    )
                advantage = observed_cost - baseline_cost
            else:
                raise RuntimeError(f"Unsupported internal policy baseline type: {self.policy_baseline_type}")
            score_info = {"log_prob_proxy": self.model.log_prob_proxy(self.last_sampled_cost, context)}
            score_loss = self.model.score_function_loss(score_info, advantage)

        if plugin_score_active:
            # g^{plug-in}_t (eq. 5): REINFORCE form whose scalar weight replaces
            # the realized cost v_t with the nuisance-predicted cost
            # f_phi(x_t)^T w_t. The played decision and sampled cost are reused
            # from select_action — no extra oracle calls and no fresh sampling.
            oracle = self.oracle if hasattr(self.oracle, "feedback_loss_torch") else None
            plugin_cost = float(
                nuisance_cost_scalar_prediction(self.nuisance_model, context, action, oracle=oracle)
                .detach()
                .cpu()
                .item()
            )
            if self.last_random_exploration:
                plugin_loss_tensor = self.model.log_prob_proxy(self.last_sampled_cost, context).sum() * 0.0
            else:
                if self.policy_baseline_type == "none":
                    plugin_advantage = plugin_cost
                elif self.policy_baseline_type == "ema":
                    plugin_advantage = plugin_cost + float(self.utility_baseline)
                elif self.policy_baseline_type == "nuisance_induced":
                    if nuisance_induced_baseline_cost is None:
                        raise RuntimeError(
                            "Plug-in score branch requires nuisance_induced_baseline_cost when "
                            "policy_baseline_type='nuisance_induced'."
                        )
                    plugin_advantage = plugin_cost - float(nuisance_induced_baseline_cost)
                else:
                    raise RuntimeError(
                        f"Unsupported internal policy baseline type: {self.policy_baseline_type}"
                    )
                plugin_score_info = {
                    "log_prob_proxy": self.model.log_prob_proxy(self.last_sampled_cost, context)
                }
                plugin_loss_tensor = self.model.score_function_loss(plugin_score_info, plugin_advantage)
            aux_loss_value = float(plugin_loss_tensor.detach().cpu().item())
            aux_total = plugin_loss_tensor
        elif has_aux_path:
            if y_imputed is None:
                raise RuntimeError("Surrogate generative update requested without nuisance prediction.")
            aux_loss = generative_auxiliary_loss(
                actor=self.model,
                context=context,
                c_tilde=y_imputed,
                oracle=self.oracle,
                oracle_context=self.last_oracle_context,
                solution_pool=self.generative_solution_pool,
                cfg=self.generative_dfl_cfg,
                oracle_solve=self._oracle_solve,
                point_state=self.point_surrogate_state,
            )
            aux_loss_value = float(aux_loss.detach().cpu().item())
            reg_loss = self.model.regularizer_loss(y_imputed, context)
            aux_total = self.generative_dfl_cfg.beta_dfl * aux_loss + self.generative_dfl_cfg.lambda_gen * reg_loss

        if score_loss is None and aux_total is None:
            self._mark_zero_model_round()
            self._update_point_utility_baseline(reward)
            self._record_step_diagnostics(
                alpha_t=alpha_t,
                scalar_feedback=observed_cost,
                scalar_loss=observed_cost,
                critic_prediction=adaptive_pred,
            )
            return
        branch_alpha = alpha_t if aux_total is not None else 1.0
        (
            score_grads_raw,
            aux_grads_raw,
            score_grads,
            aux_grads,
            combined_grads,
        ) = self._accumulate_weighted_torch_branch_losses(score_loss, aux_total, branch_alpha)
        self._update_point_utility_baseline(reward)
        self._update_adaptive_alpha(observed_cost, adaptive_pred)
        self._record_step_diagnostics(
            alpha_t=alpha_t,
            scalar_feedback=observed_cost,
            scalar_loss=observed_cost,
            actor_grad_raw=score_grads_raw,
            critic_grad_raw=aux_grads_raw,
            actor_grad=score_grads,
            critic_grad=aux_grads,
            combined_grad=combined_grads,
            critic_prediction=adaptive_pred,
            score_loss=None if score_loss is None else float(score_loss.detach().cpu().item()),
            plugin_loss=aux_loss_value,
        )

    def _update_generative(self, context, action, reward, info=None):
        return self._update_generative_hybrid(context, action, reward, info=info)

    def _update_gaussian_hybrid(self, context, action, reward, info=None):
        self.last_step_diagnostics = {}
        alpha_t = self._checked_alpha_for_round()
        self.last_alpha = alpha_t
        actor_only = self._alpha_is_one(alpha_t)
        surrogate_only = self._alpha_is_zero(alpha_t)
        plugin_uses_score = self.plugin_grad_mode == "score"
        plugin_active = not actor_only
        # Surrogate-form (eq. 7) plug-in is gated by plugin_grad_mode.
        surrogate_critic_active = plugin_active and not plugin_uses_score
        # The score-baseline subtraction is reused by g^{score}_t and, when the
        # (1 - alpha) branch is the score-form plug-in (eq. 5), by g^{plug-in}_t
        # as well. Compute it whenever either branch needs it.
        score_baseline_active = (not surrogate_only) or (plugin_active and plugin_uses_score)
        observed_score, observation_mask = self._extract_observed_score_feedback(info)
        full_observed_score = self._has_full_observed_score(observed_score, observation_mask)
        adaptive_pred = None if actor_only or full_observed_score else self._adaptive_critic_prediction(context, action)
        nuisance_induced_baseline = None
        critic_grad_raw = None
        critic_grad = None
        b_hat = None
        y_tilde = None
        if (
            score_baseline_active
            and self.policy_baseline_type == "nuisance_induced"
            and full_observed_score
        ):
            nuisance_induced_baseline = self._point_nuisance_induced_baseline(context, critic_score=observed_score)
        elif score_baseline_active and self.policy_baseline_type == "nuisance_induced":
            nuisance_induced_baseline = self._point_nuisance_induced_baseline(context)
        if plugin_active:
            if full_observed_score:
                b_hat = np.asarray(observed_score, dtype=float).reshape(self.q)
            else:
                self._accumulate_deterministic_nuisance_update(context, action, reward, info=info)
                b_hat = self.predict_nuisance(context)
            if surrogate_critic_active:
                if full_observed_score:
                    y_tilde = b_hat
                else:
                    y_tilde = self._build_dm_synthetic_vector(
                        b_hat,
                        observed_score_vector=observed_score,
                        observation_mask=observation_mask,
                    )

        actor_grad_raw = None
        actor_grad = None
        actor_baseline = 0.0
        if score_baseline_active:
            actor_advantage = self._point_actor_advantage(
                context,
                action,
                reward,
                nuisance_induced_baseline=nuisance_induced_baseline,
            )
            actor_baseline = float(reward) - float(actor_advantage)

        if not surrogate_only:
            actor_grad_raw, _ = self._point_actor_grad_descent(context, reward, actor_baseline)
            actor_grad = self._normalize_numpy_grad(actor_grad_raw)

        if plugin_active:
            if plugin_uses_score:
                # g^{plug-in}_t (eq. 5): scalar weight = f_phi(x_t)^T w_t,
                # baseline subtraction matches g^{score}_t. The played decision
                # w_t and sampled cost c_hat_t (encoded in last_logprob_grad_loc)
                # are reused with no extra oracle calls or fresh sampling.
                predicted_reward = float(self._critic_reward_from_score(b_hat, action))
                plugin_advantage = predicted_reward - float(actor_baseline)
                critic_grad_raw = self._point_actor_grad_from_advantage(context, plugin_advantage)
            else:
                critic_grad_raw = self._point_critic_grad_descent(context, y_tilde)
            critic_grad = self._normalize_numpy_grad(critic_grad_raw)

        if actor_only:
            grad = actor_grad
            if self.hybrid_batching_mode == "combined":
                self._accumulate_point_model_grad(grad)
            else:
                self._accumulate_point_actor_grad(grad)
                actor_batch_grad = self._consume_point_actor_grad_if_ready()
                if actor_batch_grad is not None:
                    self._apply_point_model_grad(actor_batch_grad)
        elif surrogate_only:
            grad = critic_grad
            self._accumulate_point_model_grad(grad)
        elif self.hybrid_batching_mode == "combined":
            grad = alpha_t * actor_grad + (1.0 - alpha_t) * critic_grad
            self._accumulate_point_model_grad(grad)
        else:
            self._accumulate_point_actor_grad(alpha_t * actor_grad)
            grad = (1.0 - alpha_t) * critic_grad
            actor_batch_grad = self._consume_point_actor_grad_if_ready()
            if actor_batch_grad is not None:
                grad = grad + actor_batch_grad
            self._apply_point_model_grad(grad)
        self._update_point_utility_baseline(reward)
        self._update_adaptive_alpha(reward, adaptive_pred)
        self._record_step_diagnostics(
            alpha_t=alpha_t,
            scalar_feedback=reward,
            actor_grad_raw=actor_grad_raw,
            critic_grad_raw=critic_grad_raw,
            actor_grad=actor_grad,
            critic_grad=critic_grad,
            combined_grad=grad,
            critic_prediction=adaptive_pred,
        )

    def update(self, context, action, reward, info=None):
        self.last_step_diagnostics = {}
        if self.hybrid_distribution_mode == "gaussian":
            self._update_gaussian_hybrid(context, action, reward, info=info)
        else:
            self._update_generative_hybrid(context, action, reward, info=info)
