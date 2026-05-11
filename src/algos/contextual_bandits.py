"""Contextual bandit baselines."""

from __future__ import annotations

import numpy as np

from src.algos.base_algo import BaseAlgo
from src.algos.batch_updates import RoundBatchAccumulator
from src.algos.lr_schedules import build_lr_schedule
from src.common.models import create_model


def _coalesce(value, default):
    return default if value is None else value


GENERATIVE_MODEL_TYPES = {"diffusion", "cnf", "shared_diffusion", "shared_cnf"}


class ContextualBanditBase(BaseAlgo):
    """Shared point-model bandit-feedback update for contextual bandit baselines."""

    def __init__(self, config, oracle):
        super().__init__(config, oracle)
        self.p = config["p"]
        self.q = getattr(oracle, "d", None)
        if self.q is None:
            self.q = getattr(oracle, "q")

        model_type = config.get("model_type", "linear")
        canonical_model_type = str(model_type).lower()
        if canonical_model_type in GENERATIVE_MODEL_TYPES:
            raise ValueError(
                "Contextual bandit baselines only support point model types; "
                f"got model_type='{model_type}'."
            )
        self.model = create_model(model_type, self.p, self.q, seed=config.get("seed", 42), config=config)
        self.theta = self.model.initialize_params()
        self.model_family = self.model.model_family
        if self.model_family != "point":
            raise ValueError(
                "Contextual bandit baselines only support point model types; "
                f"got model_family='{self.model_family}'."
            )

        self.theta_lr_schedule = build_lr_schedule(config, "theta", default_lr=0.01)
        self.theta_lr = self.theta_lr_schedule.initial_lr
        self.lambda_reg = float(_coalesce(config.get("lambda_reg"), 0.0))
        self.grad_clip_norm = float(_coalesce(config.get("grad_clip_norm"), 10.0))
        self.model_update_batch_rounds = max(1, int(config.get("model_update_batch_rounds", 1)))
        self.model_batch_accumulator = RoundBatchAccumulator(self.model_update_batch_rounds)

        self.rng = np.random.RandomState(config.get("seed", 42))
        self.t_curr = 0
        self.last_c_hat = None
        self.last_c_tilde = None
        self.last_random_exploration = False
        self.last_oracle_context = None

    def predict_mu(self, x, theta=None):
        if theta is None:
            theta = self.theta
        return self.model.predict(x, theta)

    def _oracle_solve(self, score, oracle_context=None):
        return self._cached_oracle_solve(score, oracle_context=oracle_context)

    def _flush_point_model_batch(self):
        grad = self.model_batch_accumulator.consume_numpy_mean()
        if grad is None:
            return
        if self.lambda_reg > 0:
            grad += 2.0 * self.lambda_reg * self.theta

        grad_norm = np.linalg.norm(grad)
        if grad_norm > self.grad_clip_norm:
            grad = grad * (self.grad_clip_norm / grad_norm)

        lr = self.theta_lr_schedule.next()
        self.theta_lr = lr
        self.theta = self.theta - lr * grad

    def _accumulate_point_model_grad(self, grad):
        self.model_batch_accumulator.add_numpy(grad)
        if self.model_batch_accumulator.should_flush(is_final_round=self._is_final_round()):
            self._flush_point_model_batch()

    def _start_round(self):
        self.t_curr += 1
        self._reset_oracle_solution_cache()

    def _solve_and_record(self, c_hat, action_score, oracle_context=None):
        action = self._oracle_solve(action_score, oracle_context=oracle_context)
        self.last_c_hat = np.asarray(c_hat, dtype=float).copy()
        self.last_c_tilde = np.asarray(action_score, dtype=float).copy()
        self.last_oracle_context = oracle_context
        return action

    def update(self, context, action, reward, info=None) -> None:
        mu_hat = self.predict_mu(context)
        observed, mask = self._extract_observed_score_feedback(info)
        if observed is not None and mask is not None:
            if observed.shape[0] != self.q or mask.shape[0] != self.q:
                raise ValueError(
                    "Observed semi-bandit vector and mask must match score dimension "
                    f"{self.q}; got {observed.shape} and {mask.shape}."
                )
            denom = max(1.0, float(mask.sum()))
            w = mask * (mu_hat - observed) / denom
            grad = self.model.compute_grad_r(context, w, self.theta)
            self._accumulate_point_model_grad(grad)
            return

        observation_matrix, observation_values = self._extract_linear_observation_feedback(info)
        if observation_matrix is not None and observation_values is not None:
            if observation_matrix.shape[1] != self.q:
                raise ValueError(
                    "Linear semi-bandit observation matrix must have one column per score dimension "
                    f"{self.q}; got shape {observation_matrix.shape}."
                )
            denom = max(1.0, float(observation_matrix.shape[0]))
            residual = (observation_matrix @ mu_hat - observation_values) / denom
            w = observation_matrix.T @ residual
            grad = self.model.compute_grad_r(context, w, self.theta)
            self._accumulate_point_model_grad(grad)
            return

        action_arr = np.asarray(action, dtype=float).reshape(-1)
        pred_scalar = float(np.dot(mu_hat, action_arr))
        target = float(reward)
        grad = 2.0 * (pred_scalar - target) * self.model.compute_grad_r(context, action_arr, self.theta)
        self._accumulate_point_model_grad(grad)


class GreedyContextualBandit(ContextualBanditBase):
    """Pure-exploitation contextual bandit baseline."""

    def select_action(self, context, oracle_context=None, true_reward=None):
        del true_reward
        self._start_round()
        c_hat = self.predict_mu(context)
        self.last_random_exploration = False
        return self._solve_and_record(c_hat, c_hat, oracle_context=oracle_context)


class EpsilonGreedyContextualBandit(ContextualBanditBase):
    """Epsilon-greedy contextual bandit baseline over oracle scores."""

    def __init__(self, config, oracle):
        super().__init__(config, oracle)
        epsilon = config.get("epsilon_greedy_epsilon", config.get("exploration_epsilon", 0.1))
        self.epsilon_greedy_epsilon = float(_coalesce(epsilon, 0.1))
        if not 0.0 <= self.epsilon_greedy_epsilon <= 1.0:
            raise ValueError("epsilon_greedy_epsilon must be in [0, 1]")

    def select_action(self, context, oracle_context=None, true_reward=None):
        del true_reward
        self._start_round()
        c_hat = self.predict_mu(context)
        self.last_random_exploration = self.rng.rand() < self.epsilon_greedy_epsilon
        if self.last_random_exploration:
            action_score = self.rng.randn(self.q)
        else:
            action_score = c_hat
        return self._solve_and_record(c_hat, action_score, oracle_context=oracle_context)


class ThompsonSamplingContextualBandit(ContextualBanditBase):
    """Plug-in Gaussian Thompson-style contextual bandit baseline."""

    def __init__(self, config, oracle):
        super().__init__(config, oracle)
        resolved_scale = config.get("thompson_sampling_scale", _coalesce(config.get("policy_sampling_scale"), 0.1))
        self.thompson_sampling_scale = float(_coalesce(resolved_scale, 0.1))
        if self.thompson_sampling_scale < 0.0:
            raise ValueError("thompson_sampling_scale must be non-negative")

    def select_action(self, context, oracle_context=None, true_reward=None):
        del true_reward
        self._start_round()
        c_hat = self.predict_mu(context)
        action_score = c_hat + self.thompson_sampling_scale * self.rng.randn(self.q)
        self.last_random_exploration = False
        return self._solve_and_record(c_hat, action_score, oracle_context=oracle_context)
