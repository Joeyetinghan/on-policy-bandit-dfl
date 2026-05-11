"""Decision-focused helpers for online generative actors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.common.point_surrogate_losses import PointSurrogateState, point_surrogate_grad
from src.common.surrogate_loss_names import CANONICAL_POINT_LOSS_TYPES, canonical_surrogate_loss_name
from src.common.surrogate_pool_losses import SolutionPool


_PER_SCENARIO_PREFIX = "per_scenario:"
_PER_SCENARIO_SUFFIX = "_per_scenario"
_SAA_PREFIX = "saa:"
_SAA_SUFFIX = "_saa"
_VALID_AUX_LOSS_TYPES = frozenset(CANONICAL_POINT_LOSS_TYPES)


def _coalesce(value, default):
    return default if value is None else value


def _normalize_aux_mode(aux_mode: str) -> str:
    """Map user-facing aux_mode strings to canonical internal forms.

    Accepts ``per_scenario:<loss>`` / ``<loss>_per_scenario`` for the
    per-scenario gradient path and ``saa:<loss>`` / ``<loss>_saa`` for the
    sample-average approximation path.
    Existing modes (``auto``, ``contrastive_map``) pass through.
    """

    for prefix in (_PER_SCENARIO_PREFIX, _SAA_PREFIX):
        if aux_mode.startswith(prefix):
            loss_part = aux_mode[len(prefix):]
            canonical = canonical_surrogate_loss_name(loss_part)
            return f"{prefix}{canonical}"
    if aux_mode.endswith(_PER_SCENARIO_SUFFIX) and aux_mode != _PER_SCENARIO_SUFFIX.lstrip("_"):
        loss_part = aux_mode[: -len(_PER_SCENARIO_SUFFIX)]
        canonical = canonical_surrogate_loss_name(loss_part)
        return f"{_PER_SCENARIO_PREFIX}{canonical}"
    if aux_mode.endswith(_SAA_SUFFIX) and aux_mode != _SAA_SUFFIX.lstrip("_"):
        loss_part = aux_mode[: -len(_SAA_SUFFIX)]
        canonical = canonical_surrogate_loss_name(loss_part)
        return f"{_SAA_PREFIX}{canonical}"
    return aux_mode


def _is_valid_aux_mode(aux_mode: str) -> bool:
    if aux_mode in {"auto", "contrastive_map"}:
        return True
    if aux_mode.startswith(_PER_SCENARIO_PREFIX):
        loss_part = aux_mode[len(_PER_SCENARIO_PREFIX):]
        return loss_part in _VALID_AUX_LOSS_TYPES
    if aux_mode.startswith(_SAA_PREFIX):
        loss_part = aux_mode[len(_SAA_PREFIX):]
        return loss_part in _VALID_AUX_LOSS_TYPES
    return False


def _per_scenario_loss_type(aux_mode: str) -> str | None:
    if aux_mode.startswith(_PER_SCENARIO_PREFIX):
        return aux_mode[len(_PER_SCENARIO_PREFIX):]
    return None


def _saa_loss_type(aux_mode: str) -> str | None:
    if aux_mode.startswith(_SAA_PREFIX):
        return aux_mode[len(_SAA_PREFIX):]
    return None


@dataclass
class GenerativeDFLConfig:
    lambda_score: float = 0.5
    beta_dfl: float = 1.0
    lambda_gen: float = 0.1
    num_dfl_samples: int = 16
    risk_alpha: float = 1.0
    margin: float = 0.0
    contrastive_hinge: bool = False
    aux_mode: str = "auto"


def generative_dfl_config(config: dict) -> GenerativeDFLConfig:
    """Parse generative decision-focused learning options."""

    lambda_score = float(
        _coalesce(
            config.get("generative_lambda_score"),
            _coalesce(config.get("lambda_score"), 0.5),
        )
    )
    beta_dfl = float(_coalesce(config.get("generative_beta_dfl"), _coalesce(config.get("beta_dfl"), 1.0)))
    lambda_gen = float(
        _coalesce(
            config.get("generative_regularizer_weight"),
            _coalesce(config.get("lambda_gen"), 0.1),
        )
    )
    num_dfl_samples = max(
        1,
        int(
            _coalesce(
                config.get("generative_num_dfl_samples"),
                _coalesce(config.get("K_dfl"), config.get("surrogate_num_samples", 16)),
            )
        ),
    )
    risk_alpha = float(np.clip(_coalesce(config.get("generative_risk_alpha"), config.get("risk_alpha", 1.0)), 0.0, 1.0))
    if risk_alpha <= 0.0:
        risk_alpha = 1.0
    margin = float(_coalesce(config.get("generative_contrastive_margin"), config.get("margin", 0.0)))
    contrastive_hinge = bool(_coalesce(config.get("generative_contrastive_hinge"), False))
    aux_mode = str(_coalesce(config.get("generative_aux_mode"), "auto")).lower().strip()
    aux_mode = _normalize_aux_mode(aux_mode)
    if not _is_valid_aux_mode(aux_mode):
        raise ValueError(
            "generative_aux_mode must be one of {'auto', 'contrastive_map'} "
            f"or 'per_scenario:<loss>' / '<loss>_per_scenario' "
            f"or 'saa:<loss>' / '<loss>_saa' "
            f"for a canonical point loss in {sorted(_VALID_AUX_LOSS_TYPES)}; "
            f"got '{aux_mode}'"
        )
    return GenerativeDFLConfig(
        lambda_score=float(np.clip(lambda_score, 0.0, 1.0)),
        beta_dfl=beta_dfl,
        lambda_gen=lambda_gen,
        num_dfl_samples=num_dfl_samples,
        risk_alpha=risk_alpha,
        margin=margin,
        contrastive_hinge=contrastive_hinge,
        aux_mode=aux_mode,
    )


def generative_update_objective(config: dict) -> str:
    """Return the generative actor training objective for online updates."""

    raw = str(
        _coalesce(
            config.get("generative_update_objective"),
            _coalesce(config.get("generative_training_objective"), "mixed"),
        )
    ).lower()
    aliases = {
        "dfl": "mixed",
        "mixed_dfl": "mixed",
        "score_function": "score",
        "score_only": "score",
        "nll_only": "nll",
        "mle": "nll",
        "flow_nll": "nll",
        "denoising": "nll",
        "denoising_nll": "nll",
    }
    objective = aliases.get(raw, raw)
    if objective not in {"mixed", "score", "nll"}:
        raise ValueError(
            "generative_update_objective must be one of {'mixed', 'score', 'nll'}; "
            f"got '{raw}'"
        )
    return objective


def make_solution_pool(q: int, config: dict) -> SolutionPool:
    max_size = max(
        2,
        int(
            _coalesce(
                config.get("generative_solution_pool_size"),
                _coalesce(config.get("solution_pool_size"), config.get("surrogate_solution_pool_max_size", 128)),
            )
        ),
    )
    return SolutionPool(q=q, max_size=max_size)


def oracle_uses_min_objective(oracle) -> bool:
    return str(getattr(oracle, "objective_sense", "max")).lower() == "min"


def oracle_input_from_cost(oracle, cost_vec):
    """Convert a generated cost/objective vector to the repo's oracle input."""

    arr = np.asarray(cost_vec, dtype=float)
    if oracle_uses_min_objective(oracle):
        return arr
    return -arr


def solve_from_cost(oracle, cost_vec, *, oracle_context=None, oracle_solve=None) -> np.ndarray:
    solve = oracle.solve if oracle_solve is None else oracle_solve
    return np.asarray(solve(oracle_input_from_cost(oracle, cost_vec), oracle_context=oracle_context), dtype=float).reshape(-1)


def tensor_to_2d_scenarios(scenarios: torch.Tensor) -> torch.Tensor:
    if scenarios.ndim == 2:
        return scenarios
    if scenarios.ndim == 3 and scenarios.shape[0] == 1:
        return scenarios.squeeze(0)
    raise ValueError(f"Expected scenario tensor with shape [K, d] or [1, K, d], got {tuple(scenarios.shape)}")


def _tail_mean(values: torch.Tensor, risk_alpha: float, *, largest: bool) -> torch.Tensor:
    flat = values.reshape(-1)
    if flat.numel() == 0:
        return torch.zeros((), dtype=values.dtype, device=values.device)
    if risk_alpha >= 1.0:
        return flat.mean()
    count = max(1, int(np.ceil(float(risk_alpha) * flat.numel())))
    sorted_values, _ = torch.sort(flat)
    tail = sorted_values[-count:] if largest else sorted_values[:count]
    return tail.mean()


def scenario_losses_for_action(C_hat: torch.Tensor, action, oracle) -> torch.Tensor:
    """Return per-scenario losses under a fixed action."""

    action_np = np.asarray(action, dtype=float).reshape(-1)
    if oracle_uses_min_objective(oracle) and hasattr(oracle, "feedback_loss_torch"):
        return oracle.feedback_loss_torch(C_hat, action_np).reshape(-1)
    action_tensor = C_hat.new_tensor(action_np)
    return C_hat.matmul(action_tensor).reshape(-1)


def risk_eval(C_hat: torch.Tensor, action, oracle, risk_alpha: float) -> torch.Tensor:
    losses = scenario_losses_for_action(C_hat, action, oracle)
    # For min-objective oracles, large losses are worst (right tail).
    # For max-objective oracles, small losses (= small gains) are worst (left tail).
    # `_tail_mean` short-circuits to mean() when risk_alpha >= 1.0, so direction
    # only matters for CVaR (alpha < 1.0).
    return _tail_mean(losses, risk_alpha, largest=oracle_uses_min_objective(oracle))


def _update_pool_from_scenarios(
    *,
    C_detached: torch.Tensor,
    oracle,
    oracle_context,
    solution_pool: SolutionPool,
    oracle_solve=None,
) -> None:
    scenario_np = C_detached.detach().cpu().numpy()
    if scenario_np.ndim != 2:
        raise ValueError(f"Expected detached scenarios [K, d], got {scenario_np.shape}")

    if oracle_uses_min_objective(oracle):
        try:
            solution_pool.add(np.asarray(oracle.solve(scenario_np, oracle_context=oracle_context), dtype=float).reshape(-1))
        except TypeError:
            solution_pool.add(np.asarray(oracle.solve(scenario_np), dtype=float).reshape(-1))
        return

    solve = oracle.solve if oracle_solve is None else oracle_solve
    for scenario in scenario_np:
        solution_pool.add(np.asarray(solve(-scenario, oracle_context=oracle_context), dtype=float).reshape(-1))


def contrastive_map_scenario_loss(
    *,
    C_hat: torch.Tensor,
    w_pos,
    solution_pool: SolutionPool,
    oracle,
    risk_alpha: float,
    margin: float,
    hinge: bool = False,
) -> torch.Tensor:
    """Distributional contrastive MAP for one online target solution.

    Online single-context contract:
      - C_hat: K reparameterized scenarios from the actor's predictive distribution
        at one context (shape [K, d]).
      - w_pos: one oracle target solution computed from the imputed cost.
      - solution_pool: |pool| candidate negative solutions accumulated across rounds.

    K serves as the Monte Carlo variance budget: at risk_alpha=1 with linear cost
    `c.w`, `E[c.w_pos]` has a closed-form mean independent of K, but the
    MC estimate `(1/K) sum_k c_k.w_pos` has variance proportional to 1/K. Online
    updates have no replay buffer so K controls per-step gradient noise.

    Differs from offline batch contrastive MAP: scenarios are differentiable
    inputs (the model's posterior), candidate solutions are the fixed pool.
    """

    C_2d = tensor_to_2d_scenarios(C_hat)
    w_pos_np = np.asarray(w_pos, dtype=float).reshape(-1)
    pool = solution_pool.as_array()
    if pool.size == 0:
        return C_2d.sum() * 0.0

    keep = [not np.allclose(sol, w_pos_np, atol=1.0e-8, rtol=1.0e-6) for sol in pool]
    neg_solutions = pool[np.asarray(keep, dtype=bool)] if any(keep) else pool

    pos_cost = risk_eval(C_2d, w_pos_np, oracle, risk_alpha)
    neg_costs = torch.stack([risk_eval(C_2d, sol, oracle, risk_alpha) for sol in neg_solutions])
    violations = float(margin) + pos_cost - neg_costs
    max_violation = torch.max(violations)
    return torch.relu(max_violation) if hinge else max_violation


def cnf_contrastive_map_loss(
    *,
    actor,
    context,
    c_tilde,
    oracle,
    oracle_context,
    solution_pool: SolutionPool,
    cfg: GenerativeDFLConfig,
    oracle_solve=None,
) -> torch.Tensor:
    c_tilde_np = np.asarray(c_tilde.detach().cpu().numpy(), dtype=float).reshape(-1)
    w_pos = solve_from_cost(oracle, c_tilde_np, oracle_context=oracle_context, oracle_solve=oracle_solve)
    C_hat = actor.sample_scenarios(context, K=cfg.num_dfl_samples, keep_graph=True)
    C_2d = tensor_to_2d_scenarios(C_hat)
    _update_pool_from_scenarios(
        C_detached=C_2d.detach(),
        oracle=oracle,
        oracle_context=oracle_context,
        solution_pool=solution_pool,
        oracle_solve=oracle_solve,
    )
    return contrastive_map_scenario_loss(
        C_hat=C_2d,
        w_pos=w_pos,
        solution_pool=solution_pool,
        oracle=oracle,
        risk_alpha=cfg.risk_alpha,
        margin=cfg.margin,
        hinge=cfg.contrastive_hinge,
    )


def per_scenario_point_surrogate_loss(
    *,
    actor,
    context,
    c_tilde,
    oracle,
    oracle_context,
    state: PointSurrogateState,
    loss_type: str,
    cfg: GenerativeDFLConfig,
    oracle_solve=None,
    point_grad_kwargs: dict | None = None,
) -> torch.Tensor:
    """Per-scenario aggregation of an analytical point-surrogate gradient.

    For each of K reparameterized scenarios drawn from the actor's predictive
    distribution at one online context, compute the per-sample gradient
    ``g_k = dL/dc`` of the named point-model surrogate loss against the imputed
    target ``c_tilde``. Build the surrogate ``(1/K) sum_k <c_k, g_k.detach()>``
    and backprop through ``c_k`` to the actor parameters.

    K acts as the Monte Carlo variance budget for the per-step gradient (see
    ``contrastive_map_scenario_loss`` docstring). Internal state held by
    ``PointSurrogateState`` is shared with point-model surrogate updates so the
    cached solution pool, IMLE noise stream, and AIMLE alpha are consistent.
    """

    C_hat = actor.sample_scenarios(context, K=cfg.num_dfl_samples, keep_graph=True)
    C_2d = tensor_to_2d_scenarios(C_hat)
    if C_2d.shape[1] != state.q:
        raise ValueError(
            f"Scenario dim {C_2d.shape[1]} does not match PointSurrogateState.q={state.q}"
        )
    target_np = np.asarray(c_tilde.detach().cpu().numpy(), dtype=float).reshape(-1)
    solve_fn = oracle.solve if oracle_solve is None else oracle_solve
    extra_kwargs = dict(point_grad_kwargs or {})

    grads = []
    K = int(C_2d.shape[0])
    for k in range(K):
        c_np = np.asarray(C_2d[k].detach().cpu().numpy(), dtype=float).reshape(-1)
        grad = point_surrogate_grad(
            loss_type=loss_type,
            pred_vec=c_np,
            target_vec=target_np,
            oracle_solve=solve_fn,
            state=state,
            oracle_context=oracle_context,
            **extra_kwargs,
        )
        grads.append(np.asarray(grad, dtype=float).reshape(-1))
    grad_t = C_2d.new_tensor(np.stack(grads))
    return torch.sum(C_2d * grad_t.detach()) / float(K)


def saa_point_surrogate_loss(
    *,
    actor,
    context,
    c_tilde,
    oracle,
    oracle_context,
    state: PointSurrogateState,
    loss_type: str,
    cfg: GenerativeDFLConfig,
    oracle_solve=None,
    point_grad_kwargs: dict | None = None,
) -> torch.Tensor:
    """SAA aggregation of a point-surrogate gradient.

    Draw K reparameterized scenarios, average them into one differentiable
    empirical expected score, then evaluate the point-surrogate gradient once
    at that mean score.  For risk-neutral linear objectives this matches the
    stochastic-programming SAA decision ``oracle(mean_k c_k)`` while retaining
    gradient flow to every generated scenario through the sample mean.
    """

    C_hat = actor.sample_scenarios(context, K=cfg.num_dfl_samples, keep_graph=True)
    C_2d = tensor_to_2d_scenarios(C_hat)
    if C_2d.shape[1] != state.q:
        raise ValueError(
            f"Scenario dim {C_2d.shape[1]} does not match PointSurrogateState.q={state.q}"
        )
    mean_score = C_2d.mean(dim=0)
    pred_np = np.asarray(mean_score.detach().cpu().numpy(), dtype=float).reshape(-1)
    target_np = np.asarray(c_tilde.detach().cpu().numpy(), dtype=float).reshape(-1)
    solve_fn = oracle.solve if oracle_solve is None else oracle_solve
    extra_kwargs = dict(point_grad_kwargs or {})

    grad = point_surrogate_grad(
        loss_type=loss_type,
        pred_vec=pred_np,
        target_vec=target_np,
        oracle_solve=solve_fn,
        state=state,
        oracle_context=oracle_context,
        **extra_kwargs,
    )
    grad_t = mean_score.new_tensor(np.asarray(grad, dtype=float).reshape(-1))
    return torch.sum(mean_score * grad_t.detach())


def generative_auxiliary_loss(
    *,
    actor,
    context,
    c_tilde,
    oracle,
    oracle_context,
    solution_pool: SolutionPool,
    cfg: GenerativeDFLConfig,
    oracle_solve=None,
    point_state: PointSurrogateState | None = None,
    point_grad_kwargs: dict | None = None,
) -> torch.Tensor:
    per_scenario_loss_type = _per_scenario_loss_type(cfg.aux_mode)
    if per_scenario_loss_type is not None:
        if point_state is None:
            raise RuntimeError(
                f"generative_aux_mode='{cfg.aux_mode}' requires a PointSurrogateState; "
                "pass point_state=... from the caller."
            )
        return per_scenario_point_surrogate_loss(
            actor=actor,
            context=context,
            c_tilde=c_tilde,
            oracle=oracle,
            oracle_context=oracle_context,
            state=point_state,
            loss_type=per_scenario_loss_type,
            cfg=cfg,
            oracle_solve=oracle_solve,
            point_grad_kwargs=point_grad_kwargs,
        )
    saa_loss_type = _saa_loss_type(cfg.aux_mode)
    if saa_loss_type is not None:
        if point_state is None:
            raise RuntimeError(
                f"generative_aux_mode='{cfg.aux_mode}' requires a PointSurrogateState; "
                "pass point_state=... from the caller."
            )
        return saa_point_surrogate_loss(
            actor=actor,
            context=context,
            c_tilde=c_tilde,
            oracle=oracle,
            oracle_context=oracle_context,
            state=point_state,
            loss_type=saa_loss_type,
            cfg=cfg,
            oracle_solve=oracle_solve,
            point_grad_kwargs=point_grad_kwargs,
        )
    return cnf_contrastive_map_loss(
        actor=actor,
        context=context,
        c_tilde=c_tilde,
        oracle=oracle,
        oracle_context=oracle_context,
        solution_pool=solution_pool,
        cfg=cfg,
        oracle_solve=oracle_solve,
    )


def combine_generative_losses(
    *,
    score_loss: torch.Tensor,
    aux_loss: torch.Tensor,
    regularizer_loss: torch.Tensor,
    cfg: GenerativeDFLConfig,
) -> torch.Tensor:
    aux_total = cfg.beta_dfl * aux_loss + cfg.lambda_gen * regularizer_loss
    return cfg.lambda_score * score_loss + (1.0 - cfg.lambda_score) * aux_total
