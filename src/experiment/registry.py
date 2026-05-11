"""Factories for environments, oracles, and algorithms."""

from __future__ import annotations

from typing import Any, Dict

from src.algos.baselines import RandomOracleAlgo, TrueModelAlgo
from src.algos.dfhpg import DFHPG
from src.algos.contextual_bandits import (
    EpsilonGreedyContextualBandit,
    GreedyContextualBandit,
    ThompsonSamplingContextualBandit,
)
from src.envs.energy_env import EnergyEnv
from src.envs.pricing_env import PricingEnv
from src.envs.shortest_path_env import ShortestPathEnv
from src.envs.topk_env import TopKEnv
from src.oracles import (
    EnergyOracle,
    PricingOracle,
    ShortestPathOracle,
    TopKOracle,
)


def create_env(config: Dict[str, Any], seed: int):
    """Create an environment for a benchmark."""
    benchmark = config["benchmark"]
    if benchmark == "topk":
        return TopKEnv(config, seed)
    if benchmark == "shortest_path":
        return ShortestPathEnv(config, seed)
    if benchmark == "energy":
        return EnergyEnv(config, seed)
    if benchmark == "pricing":
        return PricingEnv(config, seed)
    raise ValueError(f"Unknown benchmark: {benchmark}")


def create_oracle(config: Dict[str, Any], env):
    """Create an oracle for an existing environment."""
    benchmark = config["benchmark"]
    solver_backend = str(config.get("solver_backend", "native")).lower()
    if benchmark == "topk":
        return TopKOracle(env.d, env.k, backend=solver_backend)
    if benchmark == "shortest_path":
        return ShortestPathOracle(env.G, env.edges, backend=solver_backend)
    if benchmark == "energy":
        return EnergyOracle(env.instance_data, backend=solver_backend)
    if benchmark == "pricing":
        return PricingOracle(
            n_products=env.n_products,
            num_price_levels=env.num_price_levels,
            promotion_budget=env.promotion_budget,
            pricing_num_low_levels=env.pricing_num_low_levels,
        )
    raise ValueError(f"Unknown benchmark: {benchmark}")


def create_env_and_oracle(config: Dict[str, Any], seed: int):
    """Create environment and corresponding oracle for a benchmark."""
    env = create_env(config, seed)
    oracle = create_oracle(config, env)
    return env, oracle


def create_algorithms(config: Dict[str, Any], oracle) -> Dict[str, Any]:
    """Create algorithm instances to run for a given config."""
    run_greedy_contextual_bandit = bool(config.get("run_greedy_contextual_bandit", False))
    run_epsilon_greedy_contextual_bandit = bool(config.get("run_epsilon_greedy_contextual_bandit", False))
    run_thompson_contextual_bandit = bool(config.get("run_thompson_contextual_bandit", False))
    run_hybrid_bandit = bool(config.get("run_hybrid_bandit", False))
    run_random_oracle = bool(config.get("run_random_oracle", True))
    run_true_model = bool(config.get("run_true_model", True))

    algorithms: Dict[str, Any] = {}
    if run_greedy_contextual_bandit:
        algorithms["GreedyContextualBandit"] = GreedyContextualBandit(config, oracle)
    if run_epsilon_greedy_contextual_bandit:
        algorithms["EpsilonGreedyContextualBandit"] = EpsilonGreedyContextualBandit(config, oracle)
    if run_thompson_contextual_bandit:
        algorithms["ThompsonSamplingContextualBandit"] = ThompsonSamplingContextualBandit(config, oracle)
    if run_hybrid_bandit:
        algorithms["DFHPG"] = DFHPG(config, oracle)
    if run_random_oracle:
        algorithms["RandomOracle"] = RandomOracleAlgo(config, oracle)
    if run_true_model:
        algorithms["TrueModel"] = TrueModelAlgo(config, oracle)

    if not algorithms:
        raise ValueError(
            "No algorithms enabled. Set at least one of: "
            "run_greedy_contextual_bandit, "
            "run_epsilon_greedy_contextual_bandit, run_thompson_contextual_bandit, run_hybrid_bandit, "
            "run_random_oracle, run_true_model."
        )

    return algorithms


def create_env_for_algo_run(config: Dict[str, Any], seed: int):
    """Create a fresh environment for each algorithm run to keep trajectories aligned."""
    return create_env(config, seed)
